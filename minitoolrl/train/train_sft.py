#!/usr/bin/env python3
"""固定预算 SFT：从明确哈希的底座或已训练检查点继续训练。

一次调用只创建一个 AdamW 和一条学习率日程；probe 只跑同一日程的前缀，不产生
主链检查点。参数更新全部经 DeepSpeed ZeRO-2。损失按整个更新组的全局监督 token
数归一化，训练中的验证集指标与开发集任务成功率评测分别由本文件与
eval/rollout_eval.py 负责。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from minitoolrl.data.sft_dataset import build_sequence, collate_sft
from minitoolrl.files import dump_json, load_json, read_jsonl, sha256_file
from minitoolrl.train import deepspeed_backend


def masked_logits(model, ids, keep):
    """只在 keep 为真的位置做 lm_head 词表投影，返回 logits 与对应的目标 token。

    hidden[t] 预测 ids[t+1]，keep 形状为 [B, L-1]，已按目标位置后移一位。题面、
    工具返回等位置仍作为上下文参与 Transformer，只是不展开整词表 logits。
    """
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, :-1]
    return model.lm_head(hidden[keep]).float(), ids[:, 1:][keep]


def sft_zero_loss(module):
    """零损失：只用于对齐 DeepSpeed 累积边界，不产生梯度。"""
    device = next(module.parameters()).device
    ids = torch.tensor([[0, 0]], dtype=torch.long, device=device)
    return module.model(input_ids=ids, use_cache=False).last_hidden_state.sum() * 0.0


def make_sft_loss(n_total: int, device):
    """每个 microbatch 的按全局监督 token 数归一化的损失。"""
    def loss_fn(module, microbatch):
        ids, labels = microbatch
        ids = ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, target = masked_logits(module, ids, labels[:, 1:] != -100)
        loss = F.cross_entropy(logits, target, reduction='sum') / n_total
        return loss, {}
    return loss_fn


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = correct = total = exact = count = 0
    for ids, labels in loader:
        ids, labels = ids.to(device), labels.to(device)
        keep = labels[:, 1:] != -100
        logits, target = masked_logits(model, ids, keep)
        total_loss += F.cross_entropy(logits, target, reduction='sum').item()
        hit = logits.argmax(-1) == target
        correct += hit.sum().item()
        total += target.numel()
        offset = 0
        for n in keep.sum(1).tolist():
            exact += int(bool(hit[offset:offset + n].all()))
            count += 1
            offset += n
    model.train()
    return dict(loss=total_loss / max(1, total), token_acc=correct / max(1, total),
                seq_exact=exact / max(1, count), n_sequences=count, n_tokens=total)


def prepare(path, tokenizer, max_length):
    records = read_jsonl(path)
    items = []
    for rec in records:
        seq = build_sequence(tokenizer, rec)
        if len(seq.ids) > max_length:
            raise RuntimeError(f"SFT would truncate {rec.get('task_id')}: {len(seq.ids)}>{max_length}")
        labels = [i if m else -100 for i, m in zip(seq.ids, seq.policy_mask)]
        if sum(1 for x in labels if x != -100) == 0:
            raise RuntimeError(f"No supervised tokens in {rec.get('task_id')}")
        items.append((torch.tensor(seq.ids), torch.tensor(labels)))
    return items, records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='explicitly hashed initial checkpoint')
    p.add_argument('--output', required=True)
    p.add_argument('--train-data', required=True)
    p.add_argument('--val-data', required=True)
    p.add_argument('--expected-sha', required=True, help='SHA256 of the explicitly selected initial weights')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--grad-accum', type=int, default=8)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--max-length', type=int, default=4096)
    p.add_argument('--max-seconds', type=int, default=1800)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--max-updates', type=int, default=0, help='prefix limit; does not rescale the LR schedule')
    p.add_argument('--probe', action='store_true',
                   help='engineering prefix without validation or saved weights')
    p.add_argument('--update-plan', default='',
                   help='JSON update schedule; disables length-bucket shuffle')
    args = p.parse_args()
    if args.max_updates < 0 or (args.probe and args.max_updates <= 0):
        p.error('--probe requires a positive --max-updates')

    started = time.time()
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    cpu_count = os.cpu_count() or 8 * world_size
    torch.set_num_threads(8 if world_size == 1 else max(2, min(8, cpu_count // world_size)))
    if not torch.cuda.is_available():
        raise RuntimeError('SFT requires CUDA')
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('training output is not empty; use a new output directory: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)

    initial_model_sha256 = sha256_file(Path(args.model) / 'model.safetensors')
    if initial_model_sha256 != args.expected_sha:
        raise RuntimeError(
            f'Initial weights SHA mismatch: {initial_model_sha256} != {args.expected_sha}')

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    # 底座和续训均使用检查点自己的模板；不覆盖 tokenizer 的对话协议。
    train, records = prepare(args.train_data, tokenizer, args.max_length)
    # probe 是工程性能短测：只跑更新前缀，不准备验证集。
    val = None if args.probe else prepare(args.val_data, tokenizer, args.max_length)[0]

    plan = None
    items_by_pos = None
    if args.update_plan:
        plan = load_json(args.update_plan)
        items_by_pos = {rec['train_position']: item for rec, item in zip(records, train)}
    else:
        # 长度分桶只减少批内补齐，不打乱当前输入定义的 microbatch 组成和顺序。
        train.sort(key=lambda item: len(item[0]))
    # 后续只消费 token；计划分支已把 train_position 映射到对应样本。
    del records

    collate = lambda batch: collate_sft(batch, tokenizer.pad_token_id)
    train_loader = None if plan is not None else DataLoader(
        train, batch_size=args.batch_size, collate_fn=collate, pin_memory=True)
    val_loader = None if val is None else DataLoader(
        val, batch_size=args.batch_size, collate_fn=collate, pin_memory=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation='sdpa', local_files_only=True)
    model.config.use_cache = False
    trainer = deepspeed_backend.DeepSpeedTrainer(
        model, lr=args.lr, weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        gas_global=args.grad_accum, zero_loss=sft_zero_loss, device='cuda')
    device = next(trainer.module.parameters()).device

    # 更新前记录本次运行的模型、数据和参数；训练结果写在 summary.json。
    config = dict(vars(args), backend=trainer.backend_name, world_size=trainer.world,
                  initial_model_sha256=initial_model_sha256,
                  chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode('utf-8')).hexdigest(),
                  train_data_sha256=sha256_file(args.train_data),
                  validation_data_sha256=sha256_file(args.val_data),
                  update_plan_sha256=sha256_file(args.update_plan) if args.update_plan else None)
    # 多卡训练时只由 rank 0 写文件；其他 rank 仍执行相同的前向和反向。
    if trainer.rank == 0:
        dump_json(out / 'config.json', config)

    stop = False
    stop_reason = None

    def request_stop(*_):
        nonlocal stop, stop_reason
        stop = True
        stop_reason = 'signal_stop'

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    # 工程短测只测更新；主训练在训练前和每个 epoch 后计算验证指标。
    baseline = (evaluate(trainer.module, val_loader, device)
                if val_loader is not None and trainer.rank == 0 else None)

    if plan is not None:
        total_steps = sum(len(epoch['updates']) for epoch in plan['epochs_plan'])
    else:
        total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    steps = 0
    history = []
    update_log = (open(out / 'updates.jsonl', 'a', buffering=1, encoding='utf-8')
                  if trainer.rank == 0 else None)

    def run_update(group, epoch):
        nonlocal steps
        n_total = sum(int((y[:, 1:] != -100).sum()) for _, y in group)
        n_input = sum(int(x.numel()) for x, _ in group)
        n_examples = sum(int(x.shape[0]) for x, _ in group)
        warmup = max(1, int(total_steps * .03))
        factor = ((steps + 1) / warmup if steps < warmup
                  else .1 + .9 * .5 * (1 + math.cos(math.pi * (steps - warmup) / max(1, total_steps - warmup))))
        lr_now = args.lr * factor
        trainer.set_lr(lr_now)
        result = trainer.update(group, make_sft_loss(n_total, device))
        steps += 1
        row = dict(step=steps, epoch=epoch, loss=result.loss, grad_norm=result.grad_norm,
                   lr=lr_now, n_supervised=n_total, n_input=n_input, n_examples=n_examples,
                   update_seconds=result.seconds, seconds=round(time.time() - started, 1),
                   peak_gb=round(torch.cuda.max_memory_allocated(device) / 1e9, 2))
        if update_log is not None:
            update_log.write(json.dumps(row) + '\n')
        if trainer.rank == 0 and (steps == 1 or steps % 5 == 0 or steps == total_steps):
            print(json.dumps(row), flush=True)

    def run_validation(epoch):
        if val_loader is None or trainer.rank != 0:
            return
        metrics = evaluate(trainer.module, val_loader, device)
        history.append(dict(epoch=epoch, step=steps, **metrics))
        print('VALIDATION', json.dumps(history[-1]), flush=True)

    def update_groups(epoch_spec):
        if plan is not None:
            for update in epoch_spec['updates']:
                group = [collate([items_by_pos[int(position)] for position in microbatch['positions']])
                         for microbatch in update['microbatches']]
                yield group
            return
        group = []
        for batch in iter(train_loader):
            group.append(batch)
            if len(group) == args.grad_accum:
                yield group
                group = []
        if group:
            yield group

    def should_stop():
        nonlocal stop, stop_reason
        prefix_done = bool(args.max_updates and steps >= args.max_updates)
        want = stop or prefix_done or time.time() - started >= args.max_seconds
        if trainer.world > 1:
            flag = torch.tensor([int(want)], dtype=torch.int64, device=device)
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
            want = bool(flag.item())
        if want:
            stop = True
            stop_reason = stop_reason or ('max_updates' if prefix_done else 'time_limit')
        return stop

    epoch_specs = plan['epochs_plan'] if plan is not None else [None] * args.epochs
    for epoch, epoch_spec in enumerate(epoch_specs, 1):
        for group in update_groups(epoch_spec):
            if should_stop():
                break
            run_update(group, epoch)
            if should_stop():
                break
        run_validation(epoch)
        if stop:
            break
    if update_log is not None:
        update_log.close()

    optimizer_updates = trainer.optimizer_updates
    complete_budget = steps == total_steps and optimizer_updates == total_steps
    # 只有完整预算产生后续阶段可使用的模型；部分运行只保留训练记录。
    if complete_budget and not args.probe:
        if trainer.rank == 0:
            trainer.export_model(out / 'endpoint', tokenizer)
            print('SAVED', out / 'endpoint', flush=True)
        torch.distributed.barrier()
    peak_gb = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
    peak_gb_per_rank = [None] * trainer.world
    torch.distributed.all_gather_object(peak_gb_per_rank, peak_gb)
    summary = dict(
        steps=steps, total_steps_planned=total_steps, baseline=baseline, validation=history,
        seconds=time.time() - started,
        stop_reason=stop_reason if stop else 'epochs_complete',
        probe=args.probe, complete_budget=complete_budget,
        optimizer_updates=optimizer_updates, peak_gb_per_rank=peak_gb_per_rank,
    )
    if trainer.rank == 0:
        dump_json(out / 'summary.json', summary)
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
