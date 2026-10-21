"""GPU 运行前检查：进程组通信，以及一次 DeepSpeed ZeRO-2 更新组的显存与更新核对。

    torchrun --nproc_per_node N -m minitoolrl.preflight --mode smoke --out <dir>
    torchrun --nproc_per_node N -m minitoolrl.preflight --mode memory \
        --model <ckpt> --seq-len 4096 --batch 2 --accum 8 --max-gb 23 --out <dir>

这不是训练算法的一部分，也不产生模型：smoke 核对 all-reduce；memory 用 SFT 相同的
损失与零损失跑 3 次完整更新，报告每卡峰值显存，超过 --max-gb 时非零退出。
范围只到 DeepSpeed 训练环境：不验证 vLLM 采样、权重同步、任务成功率或数据内容，
这些由 GRPO 阶段的协议核对和评测产物证明。run_experiment 要求本检查通过后才开始训练。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from minitoolrl.train import deepspeed_backend as B
from minitoolrl.train.train_sft import make_sft_loss, sft_zero_loss


def smoke(args, out: Path) -> dict:
    device = B.init_process_group(args.device)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    backend = torch.distributed.get_backend()
    t = torch.tensor([float(rank + 1)], device=device)
    started = time.monotonic()
    torch.distributed.all_reduce(t)
    expected = world_size * (world_size + 1) / 2.0
    ok = abs(float(t.item()) - expected) < 1e-6
    torch.distributed.barrier()
    device_name = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'
    device_names = [None] * world_size
    torch.distributed.all_gather_object(device_names, device_name)
    ranks = dict(rank=rank, world_size=world_size,
                 local_rank=int(os.environ.get('LOCAL_RANK', '0')),
                 device=str(device), backend=backend, device_name=device_name)
    record = dict(mode='smoke', ok=ok, world_size=world_size, backend=backend,
                  all_reduce_seconds=round(time.monotonic() - started, 4), ranks=ranks,
                  device_names=device_names)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'ds_smoke.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    if not ok:
        raise SystemExit('all_reduce smoke test failed: got %s expected %s' % (t.item(), expected))
    return record


def memory(args, out: Path) -> dict:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32,
                                                 attn_implementation='sdpa', local_files_only=True)
    model.config.use_cache = False
    trainer = B.DeepSpeedTrainer(model, lr=1e-5, weight_decay=0.01, grad_clip=1.0,
                                 gas_global=args.accum, zero_loss=sft_zero_loss, device='cuda')
    device = next(trainer.module.parameters()).device
    vocab = int(trainer.module.config.vocab_size)
    gen = torch.Generator().manual_seed(17 + trainer.rank)
    group = []
    for _ in range(args.accum):
        ids = torch.randint(0, vocab, (args.batch, args.seq_len), generator=gen)
        labels = ids.clone()
        labels[:, :args.seq_len // 2] = -100
        group.append((ids, labels))
    n_total = sum(int((y[:, 1:] != -100).sum()) for _, y in group)
    peaks = []
    seconds = []
    loss = None
    for _ in range(args.steps):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        started = time.monotonic()
        result = trainer.update(group, make_sft_loss(n_total, device))
        loss = result.loss
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        seconds.append(time.monotonic() - started)
        peaks.append(round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
                     if device.type == 'cuda' else 0.0)
        if not torch.isfinite(torch.tensor(result.grad_norm)):
            raise SystemExit('non-finite gradient in memory probe')
    device_name = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'
    local = dict(rank=trainer.rank, peak_gb=max(peaks), seconds_per_update=seconds,
                 device=device_name)
    per_rank = [None] * trainer.world
    torch.distributed.all_gather_object(per_rank, local)
    worst = max(r['peak_gb'] for r in per_rank)
    record = dict(mode='memory', world_size=trainer.world, batch=args.batch, seq_len=args.seq_len,
                  accum=args.accum, steps=args.steps, n_supervised=n_total, loss_last=loss,
                  per_rank=per_rank, peak_gb_max=worst, max_gb=args.max_gb, ok=worst <= args.max_gb,
                  trainer=trainer.describe())
    if trainer.rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'ds_memory.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(json.dumps(dict(peak_gb_max=worst, ok=record['ok'],
                              seconds_per_update=[round(s, 2) for s in seconds])), flush=True)
    torch.distributed.barrier()
    if not record['ok']:
        raise SystemExit('DeepSpeed memory probe: peak %.2f GB exceeds %.2f GB' % (worst, args.max_gb))
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['smoke', 'memory'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default=None)
    ap.add_argument('--seq-len', type=int, default=4096)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--accum', type=int, default=8, help='global microbatches per probe update')
    ap.add_argument('--steps', type=int, default=3)
    ap.add_argument('--max-gb', type=float, default=23.0)
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    args = ap.parse_args()
    if args.mode == 'memory' and not args.model:
        raise SystemExit('--mode memory needs --model')
    out = Path(args.out)
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    cpu_count = os.cpu_count() or 8 * world_size
    torch.set_num_threads(8 if world_size == 1 else max(2, min(8, cpu_count // world_size)))
    try:
        if args.mode == 'smoke':
            smoke(args, out)
        else:
            memory(args, out)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
