"""GPU-host preflight for the DeepSpeed backend: process-group smoke test and memory probe.

    torchrun --nproc_per_node N -m minitoolrl.train.preflight_deepspeed --mode smoke --out <dir>
    torchrun --nproc_per_node N -m minitoolrl.train.preflight_deepspeed --mode memory \
        --model <ckpt> --seq-len 4096 --batch 2 --accum 8 --max-gb 23 --out <dir>

memory: one ZeRO-2 bf16 update group (grad_accum microbatches at the configured
sequence length) through DeepSpeedTrainer, then the per-rank peak allocation.
Exits non-zero when any rank exceeds --max-gb.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from minitoolrl.train import distributed as D
from minitoolrl.train import deepspeed_backend as B
from minitoolrl.train.train_sft import make_sft_loss, sft_zero_loss


def smoke(ctx: D.DistCtx, out: Path) -> dict:
    t = torch.tensor([float(ctx.rank + 1)], device=D._coll_device(ctx))
    started = time.monotonic()
    if ctx.distributed:
        torch.distributed.all_reduce(t)
    expected = ctx.world_size * (ctx.world_size + 1) / 2.0
    ok = abs(float(t.item()) - expected) < 1e-6
    D.barrier(ctx)
    record = dict(mode='smoke', ok=ok, world_size=ctx.world_size, backend=ctx.backend,
                  all_reduce_seconds=round(time.monotonic() - started, 4), ranks=ctx.describe(),
                  device_names=D.all_gather_object(D.device_name(ctx), ctx))
    if ctx.is_main:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'ds_smoke.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    if not ok:
        raise SystemExit('all_reduce smoke test failed: got %s expected %s' % (t.item(), expected))
    return record


def memory(ctx: D.DistCtx, args, out: Path) -> dict:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32,
                                                 attn_implementation='sdpa', local_files_only=True)
    model.config.use_cache = False
    trainer = B.DeepSpeedTrainer(model, lr=1e-5, weight_decay=0.01, grad_clip=1.0,
                                 gas_global=args.accum, zero_loss=sft_zero_loss, device='cuda')
    device = next(trainer.module.parameters()).device
    vocab = int(trainer.module.config.vocab_size)
    gen = torch.Generator().manual_seed(17 + ctx.rank)
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
        D.synchronize(ctx)
        started = time.monotonic()
        result = trainer.update(group, make_sft_loss(n_total, device))
        loss = result.loss
        D.synchronize(ctx)
        seconds.append(time.monotonic() - started)
        peaks.append(D.peak_gb(ctx))
        if not torch.isfinite(torch.tensor(result.grad_norm)):
            raise SystemExit('non-finite gradient in memory probe')
    per_rank = D.all_gather_object(dict(rank=ctx.rank, peak_gb=max(peaks),
                                        seconds_per_update=seconds,
                                        device=D.device_name(ctx)), ctx)
    worst = max(r['peak_gb'] for r in per_rank)
    record = dict(mode='memory', world_size=ctx.world_size, batch=args.batch, seq_len=args.seq_len,
                  accum=args.accum, steps=args.steps, n_supervised=n_total, loss_last=loss,
                  per_rank=per_rank, peak_gb_max=worst, max_gb=args.max_gb, ok=worst <= args.max_gb,
                  trainer=trainer.describe())
    if ctx.is_main:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'ds_memory.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(json.dumps(dict(peak_gb_max=worst, ok=record['ok'],
                              seconds_per_update=[round(s, 2) for s in seconds])), flush=True)
    D.barrier(ctx)
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
    ctx = D.init(args.device)
    torch.set_num_threads(D.threads_per_rank(ctx))
    try:
        if args.mode == 'smoke':
            smoke(ctx, out)
        else:
            memory(ctx, args, out)
    finally:
        D.cleanup(ctx)


if __name__ == '__main__':
    main()
