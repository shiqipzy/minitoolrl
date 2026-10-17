#!/usr/bin/env python3
"""Fixed-budget SFT from an explicitly hashed base or trained checkpoint.

One optimizer and learning-rate schedule span the entire invocation. A probe
runs a prefix of that same schedule without producing a main-chain checkpoint.
Updates go through DeepSpeed ZeRO-2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import signal
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from minitoolrl.data.sft_dataset import build_sequence, collate_sft
from minitoolrl.env.policy import VERSION
from minitoolrl.train import distributed as D
from minitoolrl.train import deepspeed_backend as B
from minitoolrl.train import sft_diagnostics as SD


def masked_logits(model, ids, labels):
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
    keep = labels[:, 1:] != -100
    return model.lm_head(hidden[:, :-1][keep]).float(), labels[:, 1:][keep], keep


def sft_zero_loss(module):
    """零损失：只用于对齐 DeepSpeed 累积边界，不产生梯度。"""
    device = next(module.parameters()).device
    ids = torch.tensor([[0, 0]], dtype=torch.long, device=device)
    return module.model(input_ids=ids, use_cache=False).last_hidden_state.sum() * 0.0


def make_sft_loss(n_total: int, device):
    """每个 microbatch 的按全局监督 token 数归一化的损失。"""
    def loss_fn(module, mb):
        ids, labels = mb
        ids = ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, target, _ = masked_logits(module, ids, labels)
        loss = F.cross_entropy(logits, target, reduction='sum') / n_total
        return loss, {}
    return loss_fn


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = correct = total = exact = count = 0
    for ids, labels in loader:
        ids, labels = ids.to(device), labels.to(device)
        logits, target, keep = masked_logits(model, ids, labels)
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
    records = [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s]
    items = []
    for rec in records:
        seq = build_sequence(tokenizer, rec)
        if len(seq.ids) > max_length:
            raise RuntimeError(f"SFT would truncate {rec.get('task_id')}: {len(seq.ids)}>{max_length}")
        labels = [i if m else -100 for i, m in zip(seq.ids, seq.policy_mask)]
        if sum(1 for x in labels if x != -100) == 0:
            raise RuntimeError(f"No supervised tokens in {rec.get('task_id')}")
        items.append((torch.tensor(seq.ids), torch.tensor(labels)))
    n_supervised = sum(int((lab[1:] != -100).sum()) for _, lab in items)
    n_input = sum(int(ids.numel()) for ids, _ in items)
    return items, records, n_supervised, n_input


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
    p.add_argument('--probe', action='store_true', help='record the update prefix without saving weights')
    p.add_argument('--keep-file-order', action='store_true',
                   help='do not length-bucket shuffle; keep jsonl position alignment')
    p.add_argument('--update-plan', default='',
                   help='JSON update schedule; disables length-bucket shuffle')
    p.add_argument('--diagnostic-steps', default='',
                   help='save weights and same-batch before/after checks at these update numbers')
    args = p.parse_args()
    if args.max_updates < 0 or (args.probe and args.max_updates <= 0):
        p.error('--probe requires a positive --max-updates')
    try:
        diagnostic_steps = SD.parse_steps(args.diagnostic_steps)
    except ValueError as exc:
        p.error(str(exc))
    if diagnostic_steps and (args.probe or not args.max_updates
                             or diagnostic_steps[-1] != args.max_updates):
        p.error('diagnostics require --max-updates equal to the final diagnostic step; no --probe')

    started = time.time()
    ctx = D.init('cuda')
    say = (lambda *a: print(*a, flush=True)) if ctx.is_main else (lambda *a: None)
    say('Policy', VERSION)
    torch.set_num_threads(D.threads_per_rank(ctx))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.output)
    if D.all_reduce_flag(ctx.is_main and out.exists() and any(out.iterdir()), ctx):
        raise RuntimeError('training output is not empty; use a new output directory: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)

    weight_file = Path(args.model) / 'model.safetensors'
    digest = hashlib.sha256()
    with weight_file.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != args.expected_sha:
        raise RuntimeError(f'Initial weights SHA mismatch: {actual} != {args.expected_sha}')
    say('Initial SFT weights SHA256 verified')

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    # 底座和续训均使用检查点自己的模板；不覆盖 tokenizer 的对话协议。
    train, records, n_sup, n_inp = prepare(args.train_data, tokenizer, args.max_length)
    val, _, n_sup_val, n_inp_val = prepare(args.val_data, tokenizer, args.max_length)
    say(f'SFT data: train={len(train)} val={len(val)} '
        f'supervised={n_sup} input={n_inp}')

    lengths = [len(x) for x, _ in train]
    plan = None
    items_by_pos = None
    if args.update_plan:
        plan = json.loads(Path(args.update_plan).read_text(encoding='utf-8'))
        if plan.get('length_bucket_shuffle'):
            raise RuntimeError('update plan must not length-bucket shuffle')
        items_by_pos = {}
        for rec, item in zip(records, train):
            if 'train_position' not in rec:
                raise RuntimeError(f'missing train_position in {rec.get("task_id")}')
            pos = int(rec['train_position'])
            if pos in items_by_pos:
                raise RuntimeError(f'duplicate train_position {pos}')
            items_by_pos[pos] = item
        n_pos = int(plan['n_positions'])
        if sorted(items_by_pos) != list(range(n_pos)):
            raise RuntimeError('train_position must be 0..n_positions-1')
        if int(plan.get('microbatch', args.batch_size)) != args.batch_size:
            raise RuntimeError('update-plan microbatch != --batch-size')
        if int(plan.get('grad_accum', args.grad_accum)) != args.grad_accum:
            raise RuntimeError('update-plan grad_accum != --grad-accum')
        say('UPDATE_PLAN', plan.get('recipe'), 'epochs', plan.get('epochs'),
            'updates_per_epoch', plan.get('updates_per_epoch'))
    elif not args.keep_file_order:
        # 数据顺序只由训练种子决定，不消费诊断或其他代码的随机状态。
        order_rng = random.Random(args.seed)
        order = sorted(range(len(train)), key=lambda i: lengths[i])
        buckets = [order[i:i + args.batch_size * 16] for i in range(0, len(order), args.batch_size * 16)]
        order_rng.shuffle(buckets)
        for b in buckets:
            order_rng.shuffle(b)
        train = [train[i] for b in buckets for i in b]

    collate = lambda batch: collate_sft(batch, tokenizer.pad_token_id)
    dl = None if plan is not None else DataLoader(
        train, batch_size=args.batch_size, collate_fn=collate, pin_memory=ctx.cuda)
    vl = DataLoader(val, batch_size=args.batch_size, collate_fn=collate, pin_memory=ctx.cuda)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation='sdpa', local_files_only=True)
    model.config.use_cache = False
    trainer = B.DeepSpeedTrainer(
        model, lr=args.lr, weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        gas_global=args.grad_accum, zero_loss=sft_zero_loss, device='cuda')
    device = next(trainer.module.parameters()).device
    say('FRESH_ADAMW', type(trainer.optimizer).__name__ if hasattr(trainer, 'optimizer') else 'DeepSpeed',
        'state_loaded=False', D.launcher_note(ctx), 'backend=%s' % trainer.backend_name)

    config = dict(vars(args), warm_start_sha=actual,
                  n_parameters=sum(t.numel() for t in model.parameters()),
                  total_input_tokens=sum(lengths), max_input_length=max(lengths),
                  n_supervised_tokens=n_sup, n_input_tokens=n_inp,
                  n_supervised_tokens_val=n_sup_val, n_input_tokens_val=n_inp_val,
                  policy_version=VERSION, optimizer_class='AdamW',
                  optimizer_state_loaded=False,
                  chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode('utf-8')).hexdigest(),
                  train_sha256=hashlib.sha256(Path(args.train_data).read_bytes()).hexdigest(),
                  val_sha256=hashlib.sha256(Path(args.val_data).read_bytes()).hexdigest(),
                  update_plan_sha256=(hashlib.sha256(Path(args.update_plan).read_bytes()).hexdigest()
                                      if args.update_plan else None),
                  update_plan_recipe=(plan or {}).get('recipe'),
                  trainer_backend=(trainer.describe() if hasattr(trainer, 'describe')
                                   else dict(backend=trainer.backend_name)),
                  distributed=ctx.describe(), launcher=D.launcher_note(ctx),
                  data_parallel_note='global batch and n_total normalisation identical to one process; '
                                     'rank r computes microbatches r, r+N, ...; zero-loss backwards only '
                                     'align the accumulation boundary and carry no gradient',
                  note=f'fresh AdamW; fixed {args.epochs}-epoch endpoint; no checkpoint picking by test score')
    if ctx.is_main:
        (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

    stop = False
    stop_reason = None

    def request_stop(*_):
        nonlocal stop, stop_reason
        stop = True
        stop_reason = 'signal_stop'

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    # 逐步训练记录写 updates.jsonl（report/curves 都读它）；baseline 与逐 epoch
    # 验证写进 summary.json。不额外落一份内容重复的 metrics.jsonl。
    base = evaluate(trainer.module, vl, device) if ctx.is_main else None
    say('BASELINE', json.dumps(base))

    if plan is not None:
        total_steps = int(plan['epochs']) * int(plan['updates_per_epoch'])
        if args.epochs != int(plan['epochs']):
            raise RuntimeError(f'--epochs {args.epochs} != plan {plan["epochs"]}')
    else:
        total_steps = math.ceil(len(dl) / args.grad_accum) * args.epochs
    steps = 0
    seen = 0
    examples_seen = 0
    supervised_seen = 0
    train_seconds = 0.0
    history = []
    if diagnostic_steps and diagnostic_steps[-1] > total_steps:
        raise ValueError('diagnostic prefix exceeds the original learning-rate schedule')
    diagnostics = []
    update_log = D.MainFile(out / 'updates.jsonl', ctx)

    def save(name):
        # rank 0 写盘，其余 rank 在同一个 barrier 上等待；barrier 不在 is_main 分支内，
        # 各 rank 的通信次数才一致（export_model 自身不做集合通信）。
        if ctx.is_main:
            dest = out / name
            trainer.export_model(dest, tokenizer)
            print('SAVED', dest, flush=True)
        D.barrier(ctx)

    def run_update(group, epoch):
        nonlocal steps, seen, examples_seen, supervised_seen, train_seconds
        n_total = sum(int((y[:, 1:] != -100).sum()) for _, y in group)
        if n_total <= 0:
            raise RuntimeError('empty supervised batch')
        seen += sum(int(x.numel()) for x, _ in group)
        examples_seen += sum(int(x.shape[0]) for x, _ in group)
        supervised_seen += n_total
        warmup = max(1, int(total_steps * .03))
        factor = ((steps + 1) / warmup if steps < warmup
                  else .1 + .9 * .5 * (1 + math.cos(math.pi * (steps - warmup) / max(1, total_steps - warmup))))
        lr_now = args.lr * factor
        trainer.set_lr(lr_now)
        inspect = steps + 1 in diagnostic_steps
        if inspect:
            D.barrier(ctx)
            before = SD.score_update_batch(trainer.module, group, device, masked_logits) if ctx.is_main else None
            D.barrier(ctx)
        result = trainer.update(group, make_sft_loss(n_total, device))
        steps += 1
        train_seconds += result.seconds
        row = D.enrich_row(dict(stage='train', step=steps, total_steps=total_steps, epoch=epoch,
                                loss=result.loss, grad_norm=result.grad_norm, lr=lr_now,
                                n_supervised=n_total, n_input=sum(int(x.numel()) for x, _ in group),
                                n_examples=sum(int(x.shape[0]) for x, _ in group),
                                update_seconds=result.seconds,
                                seconds=round(time.time() - started, 1),
                                peak_gb=D.peak_gb(ctx)), ctx)
        update_log.write(json.dumps(row) + '\n')
        if inspect:
            D.barrier(ctx)
            if ctx.is_main:
                after = SD.score_update_batch(trainer.module, group, device, masked_logits)
                if before['batch_sha256'] != after['batch_sha256']:
                    raise RuntimeError('diagnostic update group changed')
                diagnostics.append(dict(step=steps, epoch=epoch, lr=lr_now, before=before, after=after))
            D.barrier(ctx)
            # 保存期间也不消耗训练的随机数；所有 rank 等待同一写盘边界。
            with SD.read_only_model(trainer.module):
                save('diagnostics/step_%06d' % steps)
            if ctx.is_main:
                checkpoint = out / 'diagnostics' / ('step_%06d' % steps)
                diagnostics[-1].update(checkpoint=str(checkpoint),
                    weight_sha256=SD.file_sha(checkpoint / 'model.safetensors'))
                (out / 'diagnostics.json').write_text(
                    json.dumps(dict(parent_sha256=actual, requested_steps=diagnostic_steps,
                                    points=diagnostics), indent=2, allow_nan=False), encoding='utf-8')
            D.barrier(ctx)
        if steps == 1 or steps % 5 == 0 or steps == total_steps:
            say(json.dumps(row))
        return row

    def should_stop():
        nonlocal stop, stop_reason
        prefix_done = bool(args.max_updates and steps >= args.max_updates)
        want = stop or prefix_done or time.time() - started >= args.max_seconds
        if D.all_reduce_flag(want, ctx):
            stop = True
            stop_reason = stop_reason or ('max_updates' if prefix_done else 'time_limit')
        return stop

    if plan is not None:
        for epoch_spec in plan['epochs_plan']:
            epoch = int(epoch_spec['epoch']) + 1
            for upd in epoch_spec['updates']:
                if should_stop():
                    break
                group = []
                for mb in upd['microbatches']:
                    items = [items_by_pos[int(p)] for p in mb['positions']]
                    group.append(collate(items))
                if len(group) != args.grad_accum:
                    raise RuntimeError(f'plan microbatches {len(group)} != grad_accum {args.grad_accum}')
                run_update(group, epoch)
                if should_stop():
                    break
            if ctx.is_main:
                ev = evaluate(trainer.module, vl, device)
                history.append(dict(stage='eval', epoch=epoch, step=steps, **ev))
                print('VALIDATION', json.dumps(history[-1]), flush=True)
            if stop:
                break
    else:
        for epoch in range(args.epochs):
            it = iter(dl)
            while True:
                if should_stop():
                    break
                group = []
                for _ in range(args.grad_accum):
                    batch = next(it, None)
                    if batch is None:
                        break
                    group.append(batch)
                if not group:
                    break
                run_update(group, epoch + 1)
                if should_stop():
                    break
            if ctx.is_main:
                ev = evaluate(trainer.module, vl, device)
                history.append(dict(stage='eval', epoch=epoch + 1, step=steps, **ev))
                print('VALIDATION', json.dumps(history[-1]), flush=True)
            if stop:
                break
    update_log.close()

    # Fixed endpoint: save after the planned epoch count, or on wall-clock stop.
    if not args.probe:
        save('endpoint')
    throughput = D.write_throughput(out / 'throughput.json', ctx, stage='sft_continue', updates=steps,
                                    train_seconds=train_seconds, supervised_tokens=supervised_seen,
                                    input_tokens=seen,
                                    extra=dict(batch_size=args.batch_size, grad_accum=args.grad_accum,
                                               backend=trainer.backend_name,
                                               recipe=(plan or {}).get('recipe')))
    summary = dict(
        steps=steps, total_steps_planned=total_steps, baseline=base, validation=history,
        seconds=time.time() - started, train_examples_seen=examples_seen,
        stop_reason=stop_reason if stop else 'epochs_complete',
        checkpoint=None if args.probe else str(out / 'endpoint'),
        probe=args.probe, complete_budget=steps == total_steps,
        supervised_tokens_seen=supervised_seen, input_tokens_seen=seen,
        train_records=len(train), val_records=len(val),
        optimizer_updates=steps, world_size=ctx.world_size,
        backend=trainer.backend_name,
        train_seconds=round(train_seconds, 1), throughput=throughput,
        diagnostic_steps=diagnostic_steps,
        diagnostic_prefix_complete=(bool(diagnostic_steps) and steps == args.max_updates
                                    and stop_reason == 'max_updates'),
    )
    if ctx.is_main:
        (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    say('SFT_COMPLETE', json.dumps(summary))
    D.cleanup(ctx)


if __name__ == '__main__':
    main()
