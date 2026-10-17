"""DeepSpeed n1/n2 更新核对：完整梯度、FP32 主权重与独立 AdamW 更新。

固定执行同一批 SFT 和 GRPO 更新，全局各 9 个 microbatch，覆盖双 rank 的
5/4 分配与零损失补齐。每个 rank 在最后一次 backward 后读取完整未裁剪梯度，
更新前后读取完整 FP32 主权重；这些集合通信由所有 rank 按同样顺序执行。

比较先严格核对证据、配置、两次更新及全部参数。梯度按张量核对相对 L2 误差；
各侧主权重的实际位移再与独立 PyTorch AdamW 的计算逐元素核对。bf16 梯度
抵消可能导致少数元素的 AdamW 方向不同，这种差异保留为诊断，不能通过放宽
整个参数本值的容差来掩盖错误更新。缺失证据、跳步和错误更新均非零退出。
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from minitoolrl.train import distributed as D
from minitoolrl.train import deepspeed_backend as B
from minitoolrl.train.train_sft import make_sft_loss, sft_zero_loss
from minitoolrl.train.train_grpo import make_grpo_loss

N_GLOBAL = 9
SEED = 4242
PAD_ID = 0
LR = 1e-4
LABELS = ('sft', 'grpo')
EVIDENCE_VERSION = 'full-gradient-adamw-v1'
OPTIMIZER = dict(lr=LR, betas=[0.9, 0.999], eps=1e-8, weight_decay=0.0, grad_clip=1.0)
TOLERANCES = dict(loss=5e-3, grad_norm=3e-2, gradients=3e-2)
# 更新位移的舍入误差取 FP32 参数舍入量；相对误差作用于位移，不作用于参数本值。
UPDATE_RTOL = 5e-4
FP32_ROUNDING_ULPS = 4


def build_tiny_model(vocab: int = 512):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(SEED)
    cfg = Qwen3Config(vocab_size=vocab, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=256, tie_word_embeddings=True,
                      attn_implementation='sdpa')
    model = Qwen3ForCausalLM(cfg)
    model.config.use_cache = False
    return model


def synth_sft_batch(vocab: int, n: int, seq_len: int = 48):
    gen = torch.Generator().manual_seed(SEED + 1)
    group = []
    for _ in range(n):
        ids = torch.randint(1, vocab, (1, seq_len), generator=gen)
        labels = ids.clone()
        labels[:, :seq_len // 2] = -100
        group.append((ids, labels))
    return group, sum(int((y[:, 1:] != -100).sum()) for _, y in group)


def synth_grpo_batch(vocab: int, n: int, seq_len: int = 32):
    """合成 advantage 为 -1/0/1 的轨迹，概率由损失前向计算。"""
    gen = torch.Generator().manual_seed(SEED + 2)
    samples = []
    for i in range(n):
        ids = torch.randint(1, vocab, (1, seq_len), generator=gen)[0].tolist()
        samples.append(dict(ids=ids, mask=[0.0] * 8 + [1.0] * (len(ids) - 8),
                            adv=float((i % 3) - 1)))
    return samples


def snapshot_params(module) -> dict:
    return {name: p.detach().float().cpu().clone() for name, p in module.named_parameters()}


def snapshot_master(module) -> dict:
    from deepspeed.utils import safe_get_full_fp32_param
    return {name: safe_get_full_fp32_param(p).detach().cpu().clone()
            for name, p in module.named_parameters()}


def snapshot_gradients(module) -> dict:
    from deepspeed.utils import safe_get_full_grad
    return {name: safe_get_full_grad(p).detach().float().cpu().clone()
            for name, p in module.named_parameters()}


def reference_gradients(module, batch, loss_fn):
    """在同一低精度参数上逐条求梯度，再用 FP32 按整个更新组直接求和。

    该参照不经过 DeepSpeed、rank 切分或累积步长缩放，核对全局归一化本身。
    """
    parameter = next(module.parameters())
    reference = build_tiny_model().to(device=parameter.device, dtype=parameter.dtype)
    reference.load_state_dict(module.state_dict())
    gradients = {n:torch.zeros_like(p, dtype=torch.float32, device='cpu')
                 for n,p in reference.named_parameters()}
    total = 0.
    for mb in batch:
        reference.zero_grad(set_to_none=True)
        loss, _ = loss_fn(reference, mb)
        loss.backward()
        total += float(loss.detach())
        for name, p in reference.named_parameters():
            if p.grad is not None:
                gradients[name] += p.grad.detach().float().cpu()
    return gradients, total


def update_report(before: dict, after: dict) -> dict:
    diffs = [after[name].double() - before[name].double() for name in before]
    return dict(param_delta_l2=math.sqrt(sum(float((d * d).sum()) for d in diffs)),
                param_max_abs_delta=max(float(d.abs().max()) for d in diffs))


def run_update(trainer, batch, loss_fn, label: str):
    before = snapshot_master(trainer.module)
    reference, reference_loss = reference_gradients(trainer.module, batch, loss_fn)
    gradients = {}
    result = trainer.update(batch, loss_fn, before_step=lambda module:
                            gradients.update(snapshot_gradients(module)))
    after = snapshot_master(trainer.module)
    report = dict(label=label, loss=result.loss, grad_norm=result.grad_norm,
                  seconds=result.seconds,
                  n_microbatches=result.n_microbatches,
                  optimizer_updates=result.applied_updates, aux=dict(result.aux),
                  reference_loss=reference_loss,
                  **update_report(before, after))
    return report, dict(gradients=gradients, reference_gradients=reference, params=after,
                        model_params=snapshot_params(trainer.module))


def run(ctx: D.DistCtx, out: Path) -> dict:
    started = time.monotonic()
    trainer = B.DeepSpeedTrainer(build_tiny_model(), lr=LR, weight_decay=0.0, grad_clip=1.0,
                                 gas_global=N_GLOBAL, zero_loss=sft_zero_loss,
                                 device='cuda' if ctx.cuda else 'cpu')
    if trainer.world_size not in (1, 2):
        raise RuntimeError('correctness probe requires one or two ranks')
    device = str(next(trainer.module.parameters()).device)
    tensors = dict(initial=snapshot_master(trainer.module))
    group, n_total = synth_sft_batch(512, N_GLOBAL)
    sft, tensors['sft'] = run_update(trainer, group, make_sft_loss(n_total, device), 'sft')
    group = synth_grpo_batch(512, N_GLOBAL)
    grpo, tensors['grpo'] = run_update(
        trainer, group, make_grpo_loss(N_GLOBAL, PAD_ID, 0.2, device), 'grpo')
    record = dict(mode='run', evidence_version=EVIDENCE_VERSION,
                  world_size=trainer.world_size, n_global=N_GLOBAL,
                  gas_local=trainer.gas_local, loss_scale=trainer.scale, seed=SEED,
                  optimizer=OPTIMIZER, trainer=trainer.describe(),
                  micro_steps=trainer.micro_steps, updates=[sft, grpo],
                  parameter_shapes={k: list(v.shape) for k, v in tensors['initial'].items()})
    record.update(wall_seconds=time.monotonic() - started,
                  gpu_count=trainer.world_size if ctx.cuda else 0)
    record['gpu_seconds'] = record['wall_seconds'] * record['gpu_count']
    if ctx.is_main:
        out.mkdir(parents=True, exist_ok=True)
        (out / ('ds_correctness_n%d.json' % trainer.world_size)).write_text(
            json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
        torch.save(tensors, out / ('ds_correctness_n%d_params.pt' % trainer.world_size))
        print(json.dumps(dict(world_size=trainer.world_size, updates=record['updates'])), flush=True)
    D.barrier(ctx)
    return record


def _close(field, update, a, b, tol, *, absolute_floor=1e-12):
    a, b = float(a), float(b)
    diff = abs(a - b)
    scale = max(absolute_floor, abs(a), abs(b))
    return dict(field=field, update=update,
                ok=math.isfinite(a) and math.isfinite(b) and diff <= tol * scale,
                n1=a, n2=b, abs_diff=diff, rel_diff=diff / scale, tolerance=tol)


def _validate(record, tensors, world):
    """完整证据必须含固定两次更新、全部模型参数与有限 FP32 张量。"""
    expected_model = build_tiny_model()
    expected_shapes = {name: list(p.shape) for name, p in expected_model.named_parameters()}
    gas = B.local_gas(N_GLOBAL, world)
    expected = dict(mode='run', evidence_version=EVIDENCE_VERSION, world_size=world,
                    n_global=N_GLOBAL, gas_local=gas, loss_scale=gas * world,
                    seed=SEED, optimizer=OPTIMIZER, micro_steps=gas * len(LABELS),
                    parameter_shapes=expected_shapes)
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError('n%d invalid or missing %s' % (world, key))
    updates = record.get('updates', [])
    if [u.get('label') for u in updates] != list(LABELS):
        raise ValueError('n%d requires exactly SFT then GRPO updates' % world)
    if set(tensors) != {'initial', *LABELS}:
        raise ValueError('n%d missing or unexpected tensor stages' % world)
    collections = [tensors['initial']]
    for label in LABELS:
        if set(tensors[label]) != {'gradients', 'reference_gradients', 'params', 'model_params'}:
            raise ValueError('n%d incomplete %s tensors' % (world, label))
        collections.extend(tensors[label].values())
    for values in collections:
        if set(values) != set(expected_shapes):
            raise ValueError('n%d missing or unexpected parameter names' % world)
        for name, tensor in values.items():
            if (not isinstance(tensor, torch.Tensor) or list(tensor.shape) != expected_shapes[name]
                    or tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all())):
                raise ValueError('n%d invalid tensor %s' % (world, name))
    for name, parameter in expected_model.named_parameters():
        if not torch.equal(tensors['initial'][name], parameter.detach().bfloat16().float()):
            raise ValueError('n%d initial weights do not match the fixed seed: %s' % (world, name))
    for u in updates:
        if u.get('optimizer_updates') != 1 or u.get('n_microbatches') != N_GLOBAL:
            raise ValueError('n%d incomplete optimizer update' % world)
        for key in ('loss', 'reference_loss', 'grad_norm', 'param_delta_l2', 'param_max_abs_delta'):
            value = u.get(key)
            if value is None or not math.isfinite(float(value)) or (key not in ('loss', 'reference_loss') and value <= 0):
                raise ValueError('n%d invalid %s' % (world, key))


def _check_adamw(record, tensors, side):
    """用独立 CPU AdamW 重算各侧两次更新，逐元素核对 FP32 主权重位移。"""
    names = sorted(tensors['initial'])
    reference = [torch.nn.Parameter(tensors['initial'][name].clone()) for name in names]
    opt = torch.optim.AdamW(reference, lr=LR, betas=tuple(OPTIMIZER['betas']),
                            eps=OPTIMIZER['eps'], weight_decay=0., foreach=False, fused=False)
    checks = []
    previous = tensors['initial']
    for u in record['updates']:
        label = u['label']
        evidence = tensors[label]
        checks.append(_close('reference_loss:' + side, label, u['loss'], u['reference_loss'],
                             TOLERANCES['loss'], absolute_floor=1.))
        # 第二次参照从本侧第一次实际主权重出发，优化器动量仍连续保留。
        for name, p in zip(names, reference):
            p.data.copy_(previous[name])
            p.grad = evidence['gradients'][name].clone()
        norm = float(torch.nn.utils.clip_grad_norm_(reference, OPTIMIZER['grad_clip']))
        if not math.isfinite(norm):
            raise ValueError('non-finite captured gradient norm: ' + side)
        checks.append(_close('captured_grad_norm:' + side, label, norm, u['grad_norm'],
                             TOLERANCES['grad_norm']))
        opt.step()
        measured = update_report(previous, evidence['params'])
        for key in measured:
            checks.append(_close(key + ':' + side, label, measured[key], u[key], 1e-6))
        for name, p in zip(names, reference):
            actual_grad = evidence['gradients'][name].double()
            reference_grad = evidence['reference_gradients'][name].double()
            error = float(torch.linalg.vector_norm(actual_grad - reference_grad))
            scale = max(float(torch.linalg.vector_norm(actual_grad)),
                        float(torch.linalg.vector_norm(reference_grad)), 1e-12)
            checks.append(dict(field='reference_gradient:' + side + ':' + name, update=label,
                ok=error <= TOLERANCES['gradients'] * scale, rel_diff=error / scale,
                tolerance=TOLERANCES['gradients']))
            actual = evidence['params'][name] - previous[name]
            expected = p.detach() - previous[name]
            magnitude = float(previous[name].abs().max())
            atol = FP32_ROUNDING_ULPS * torch.finfo(torch.float32).eps * max(magnitude, LR)
            diff = (actual - expected).abs()
            checks.append(dict(field='adamw_update:' + side + ':' + name, update=label,
                ok=bool(torch.allclose(actual, expected, rtol=UPDATE_RTOL, atol=atol)),
                elements=actual.numel(), max_abs_diff=float(diff.max()), atol=atol, rtol=UPDATE_RTOL))
            # 低精度模型必须确实从本次主权重刷新，否则保存和下一批采样用的是旧模型。
            checks.append(dict(field='model_refresh:' + side + ':' + name, update=label,
                ok=torch.equal(evidence['params'][name].bfloat16().float(),
                               evidence['model_params'][name])))
        previous = evidence['params']
    return checks


def compare_records(n1, n2, tensors1, tensors2):
    """供文件入口和反例测试共用，不在缺失证据上构造成功结果。"""
    _validate(n1, tensors1, 1)
    _validate(n2, tensors2, 2)
    checks = []
    for name in tensors1['initial']:
        if not torch.equal(tensors1['initial'][name], tensors2['initial'][name]):
            raise ValueError('initial FP32 parameters differ: ' + name)
    checks.extend(_check_adamw(n1, tensors1, 'n1'))
    checks.extend(_check_adamw(n2, tensors2, 'n2'))
    changed = elements = 0
    max_abs = 0.
    for u1, u2 in zip(n1['updates'], n2['updates']):
        label = u1['label']
        # on-policy GRPO 合成优势均值为零，损失接近零；该项用绝对容差，梯度独立核对。
        checks.append(_close('loss', label, u1['loss'], u2['loss'],
                             TOLERANCES['loss'], absolute_floor=1.))
        checks.append(_close('grad_norm', label, u1['grad_norm'], u2['grad_norm'],
                             TOLERANCES['grad_norm']))
        for name in tensors1['initial']:
            left = tensors1[label]['gradients'][name].double()
            right = tensors2[label]['gradients'][name].double()
            delta = float(torch.linalg.vector_norm(left - right))
            scale = max(float(torch.linalg.vector_norm(left)),
                        float(torch.linalg.vector_norm(right)), 1e-12)
            checks.append(dict(field='gradients:' + name, update=label,
                ok=delta <= TOLERANCES['gradients'] * scale, rel_diff=delta / scale,
                tolerance=TOLERANCES['gradients'], elements=left.numel()))
            diff = (tensors1[label]['params'][name] - tensors2[label]['params'][name]).abs()
            changed += int((diff > 0).sum())
            elements += diff.numel()
            max_abs = max(max_abs, float(diff.max()))
    return dict(mode='compare', evidence_version=EVIDENCE_VERSION, n_global=N_GLOBAL,
                updates_per_side=2, lr=LR, n1={k:n1[k] for k in ('gas_local','loss_scale','micro_steps')},
                n2={k:n2[k] for k in ('gas_local','loss_scale','micro_steps')},
                zero_pad_microbatches=1, param_elements=elements, param_changed=changed,
                parameter_names=sorted(tensors1['initial']),
                param_max_abs_diff=max_abs, checks=checks, ok=all(c['ok'] for c in checks),
                note='Complete pre-clip gradients are compared across ranks; each side must '
                     'match an independent AdamW update of FP32 master weights. Cross-side '
                     'parameter differences are diagnostics, never a blanket lr-scaled allowance.')


def validate_comparison(record):
    """报告只接受当前协议的完整成功检查，不能只读取顶层 ok。"""
    model = build_tiny_model()
    names = sorted(name for name, _ in model.named_parameters())
    elements = 2 * sum(p.numel() for p in model.parameters())
    expected = set()
    for label in LABELS:
        fields = {'loss', 'grad_norm'}
        for side in ('n1', 'n2'):
            fields.update(k + ':' + side for k in (
                'captured_grad_norm', 'reference_loss', 'param_delta_l2', 'param_max_abs_delta'))
            for name in names:
                fields.update(('adamw_update:' + side + ':' + name,
                               'reference_gradient:' + side + ':' + name,
                               'model_refresh:' + side + ':' + name))
        fields.update('gradients:' + name for name in names)
        expected.update((label, field) for field in fields)
    checks = record.get('checks', [])
    actual = [(c.get('update'), c.get('field')) for c in checks]
    if (record.get('evidence_version') != EVIDENCE_VERSION or record.get('ok') is not True
            or record.get('parameter_names') != names or record.get('updates_per_side') != 2
            or record.get('param_elements') != elements
            or len(actual) != len(expected) or set(actual) != expected
            or any(c.get('ok') is not True for c in checks)
            or any(not math.isfinite(float(c[k])) for c in checks
                   for k in ('rel_diff', 'max_abs_diff', 'abs_diff') if k in c)):
        raise ValueError('missing, failed or incomplete full-gradient AdamW comparison')


def compare(out: Path):
    destination = out / 'ds_correctness.json'
    try:
        records = [json.loads((out / ('ds_correctness_n%d.json' % n)).read_text(encoding='utf-8'))
                   for n in (1, 2)]
        tensors = [torch.load(out / ('ds_correctness_n%d_params.pt' % n),
                              map_location='cpu', weights_only=True) for n in (1, 2)]
        result = compare_records(*records, *tensors)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        result = dict(mode='compare', evidence_version=EVIDENCE_VERSION, ok=False, error=str(exc))
    destination.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(ok=result['ok'], error=result.get('error'),
                          failed=sum(not c['ok'] for c in result.get('checks', [])))), flush=True)
    if not result['ok']:
        raise SystemExit('ds_correctness failed: ' + str(result.get('error') or
                         [c for c in result['checks'] if not c['ok']][:4]))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['run', 'compare'], default='run')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda',
                    help='cpu only for real DeepSpeed/gloo verification; production probe uses cuda')
    args = ap.parse_args()
    out = Path(args.out)
    if args.mode == 'compare':
        compare(out)
        return
    ctx = D.init(args.device)
    torch.set_num_threads(D.threads_per_rank(ctx))
    try:
        run(ctx, out)
    finally:
        D.cleanup(ctx)


if __name__ == '__main__':
    main()
