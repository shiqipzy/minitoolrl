"""汇总 SFT 单/双 rank 性能测量及 vLLM 训推分离性能测量。

只读已有产物。SFT 固定全局工作量比较单/双 rank 的吞吐和更新漂移；
GRPO 记录 DeepSpeed 训练卡与 vLLM 采样卡部署的吞吐、耗时与显存。
每个测量给出墙钟秒数和 GPU 秒数，排除预热迭代；完整作业用时见 budget.json。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from minitoolrl import files

GRPO_DEPLOYMENTS = ('vllm_separated',)
SFT_MATCH_KEYS = ('initial_model_sha256', 'train_data_sha256', 'validation_data_sha256',
                  'chat_template_sha256', 'seed',
                  'batch_size', 'grad_accum', 'epochs', 'lr', 'weight_decay', 'grad_clip',
                  'max_length', 'max_updates', 'probe', 'update_plan_sha256')
GRPO_MATCH_KEYS = ('init_weights', 'chat_template_sha256', 'train_pool_sha256', 'max_iterations',
                   'rl', 'episode_budget', 'student_samples')


def load_rows(path: Path):
    return files.read_jsonl(path) if Path(path).exists() else []


def training_rows(path: Path) -> dict:
    """逐步更新记录按 step 建索引；重复 step 说明日志被写坏。"""
    result = {}
    for row in load_rows(path):
        if 'step' not in row:
            continue
        step = int(row['step'])
        if step in result:
            raise ValueError('duplicate update in ' + str(path))
        result[step] = row
    return result


def loss_delta(rows_a: dict, rows_b: dict):
    """同一更新前缀上的 loss 与梯度差；用于核对单/双 rank 是否走了同一次更新。"""
    common = sorted(set(rows_a) & set(rows_b))
    if not common:
        return None
    abs_d = [abs(float(rows_a[s]['loss']) - float(rows_b[s]['loss'])) for s in common]
    rel_d = [d / max(1e-9, abs(float(rows_a[s]['loss']))) for s, d in zip(common, abs_d)]
    norm_d = [abs(float(rows_a[s].get('grad_norm', 0.0)) - float(rows_b[s].get('grad_norm', 0.0)))
              for s in common]
    return dict(steps_compared=len(common), first_step=common[0], last_step=common[-1],
                loss_abs_delta_max=round(max(abs_d), 6),
                loss_abs_delta_mean=round(sum(abs_d) / len(abs_d), 6),
                loss_rel_delta_max=round(max(rel_d), 6),
                grad_norm_abs_delta_max=round(max(norm_d), 6))


def _collect_sft(run, spec, sizes, discard, result):
    configs, logs = {}, {}
    for n in sizes:
        folder = run / 'engineering/sft' / ('n%d' % n)
        configs[n] = files.load_json(folder / 'config.json') or {}
        logs[n] = training_rows(folder / 'updates.jsonl')
        expected = spec['sft_updates']
        summary = files.load_json(folder / 'summary.json') or {}
        complete = summary.get('steps') == expected and sorted(logs[n]) == list(range(1, expected + 1))
        if (not all(k in configs[n] for k in SFT_MATCH_KEYS)
                or summary.get('probe') is not True
                or summary.get('stop_reason') != 'max_updates'
                or summary.get('optimizer_updates') != expected
                or len(summary.get('peak_gb_per_rank', [])) != n
                or configs[n].get('backend') != 'deepspeed_zero2'
                or any(not math.isfinite(float(r.get(k, float('nan'))))
                       for r in logs[n].values() for k in ('loss', 'grad_norm', 'lr', 'update_seconds'))
                or any(r.get('update_seconds', 0) <= 0 for r in logs[n].values())):
            complete = False
        if not complete:
            result['issues'].append('incomplete SFT probe n%d' % n)
        rows = [logs[n][i] for i in sorted(logs[n]) if i > discard]
        seconds = sum(r.get('update_seconds', 0) for r in rows)
        if seconds <= 0:
            result['issues'].append('missing SFT measured time n%d' % n)
        result['sft'].append(dict(world_size=n, complete=complete, updates=summary.get('steps'),
            backend=configs[n].get('backend'),
            measured_updates=len(rows), measured_seconds=round(seconds, 3),
            gpu_seconds=round(seconds * n, 3), gpu_count=n,
            updates_per_second=len(rows) / seconds if seconds > 0 else None,
            supervised_tokens_per_second=sum(r['n_supervised'] for r in rows) / seconds if seconds > 0 else None,
            input_tokens_per_second=sum(r['n_input'] for r in rows) / seconds if seconds > 0 else None,
            peak_gb_per_rank=summary.get('peak_gb_per_rank')))
    if not sizes:
        return
    base = sizes[0]
    for n in sizes[1:]:
        changed = [k for k in SFT_MATCH_KEYS if configs[base].get(k) != configs[n].get(k)]
        if changed:
            result['issues'].append('SFT n%d mismatched config: %s' % (n, ', '.join(changed)))
        for step in sorted(set(logs[base]) & set(logs[n])):
            if any(logs[base][step].get(k) != logs[n][step].get(k)
                   for k in ('n_input', 'n_supervised', 'n_examples', 'lr')):
                result['issues'].append('SFT n%d data or LR mismatch at step %d' % (n, step))
                break
    if len(sizes) > 1:
        result['sft_update_drift'] = loss_delta(logs[base], logs[sizes[-1]])


def _collect_grpo(run, spec, discard, result):
    configs = {}
    for name in GRPO_DEPLOYMENTS:
        folder = run / 'engineering/grpo' / name
        configs[name] = files.load_json(folder / 'protocol.json') or {}
        stats = files.load_json(folder / 'grpo_cost.json') or {}
        rows = load_rows(folder / 'grpo_train.jsonl')
        wanted = spec['grpo_iterations']
        valid = (stats.get('iterations') == wanted and
                 [r.get('iteration') for r in rows] == list(range(1, wanted + 1)) and
                 sum(len(group['rewards']) for row in rows for group in row['groups']) == wanted * 8)
        expected_sampler = 'vllm'
        expected_gpus = 2
        if (not all(k in configs[name] for k in GRPO_MATCH_KEYS)
                or configs[name].get('sampler') != expected_sampler or stats.get('gpus_in_use') != expected_gpus
                or any(not math.isfinite(float(r.get('iteration_seconds', float('nan'))))
                       or r.get('iteration_seconds', 0) <= 0 for r in rows)):
            valid = False
        if not valid:
            result['issues'].append('incomplete GRPO probe %s' % name)
        measured = rows[discard:]
        seconds = sum(r.get('iteration_seconds', 0) for r in measured)
        tokens = sum(row['generated_tokens'] for row in measured)
        if seconds <= 0:
            result['issues'].append('missing GRPO measured time %s' % name)
        gpus = 2
        result['grpo'].append(dict(
            deployment=name, sampler=configs[name].get('sampler'), complete=valid,
            iterations=stats.get('iterations'), measured_iterations=len(measured),
            measured_seconds=round(seconds, 3), gpu_count=gpus,
            gpu_seconds=round(seconds * gpus, 3),
            episodes_per_iteration=8,
            episodes_per_hour=len(measured) * 8 / seconds * 3600 if seconds > 0 else None,
            generated_tokens_per_second=tokens / seconds if seconds > 0 else None,
            updates=stats.get('updates'), zero_signal_iterations=stats.get('zero_signal_iterations'),
            peak_gb_per_rank=stats.get('peak_memory_gb')))


def collect(run):
    run = Path(run)
    protocol = files.load_json(run / 'protocol.json') or {}
    spec = protocol.get('engineering', {})
    sizes = spec.get('world_sizes', [])
    discard = spec.get('exclude_first', 2)
    result = dict(complete=True, issues=[], sft=[], grpo=[],
                  sft_update_drift=None,
                  scope='SFT prefix drift and deployment throughput; '
                        'no full-endpoint equivalence claim.')
    if (not isinstance(discard, int) or discard < 0 or sizes != [1, 2]
            or any(not isinstance(spec.get(k), int) or spec[k] <= discard
                   for k in ('sft_updates', 'grpo_iterations'))):
        result['issues'].append('missing or invalid paired engineering plan')
        result['complete'] = False
        return result
    _collect_sft(run, spec, sizes, discard, result)
    _collect_grpo(run, spec, discard, result)
    result['complete'] = not result['issues']
    # SFT 的单/双 rank 证据完整时才计算加速比。
    if result['complete'] and len(sizes) > 1:
        ref = result['sft'][0]
        for row in result['sft']:
            row['speedup'] = row['updates_per_second'] / ref['updates_per_second']
            row['efficiency'] = row['speedup'] / (row['world_size'] / ref['world_size'])
    return result


def _fmt(value):
    if value is None:
        return '-'
    return '%.4f' % value if isinstance(value, float) else str(value)


def render(report):
    lines = ['# 分布式训练性能测量', '', '测量完整：' + str(report['complete']), '',
        'SFT 固定全局批量和学习率日程，比较相同更新前缀的单 rank 与双 rank（强扩展）；'
        'GRPO 使用 DeepSpeed 训练卡 + vLLM 采样卡，记录该部署的实际吞吐与显存。'
        '两类短测均排除前 2 次预热计时，同时给出墙钟秒数与 GPU 秒数；'
        '训练计时不含加载、验证、保存和报告，完整子任务用时见 budget.json。', '']
    lines += ['## SFT（单 rank vs 双 rank）', '', '| rank 数 | 后端 | 测量更新数 | 更新/秒 | 加速比 | 效率 | GPU 秒 | 每卡峰值 GB |',
              '|---:|---|---:|---:|---:|---:|---:|---|']
    for r in report['sft']:
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (
            r['world_size'], _fmt(r.get('backend')), r.get('measured_updates'),
            _fmt(r.get('updates_per_second')), _fmt(r.get('speedup')), _fmt(r.get('efficiency')),
            _fmt(r.get('gpu_seconds')), _fmt(r.get('peak_gb_per_rank'))))
    lines += ['', '## GRPO（DeepSpeed + vLLM 训推分离）', '', '| 部署 | 采样器 | 占用 GPU | 测量迭代数 | 轨迹/小时 | GPU 秒 | 训练卡峰值 GB |',
              '|---|---|---:|---:|---:|---:|---|']
    for r in report['grpo']:
        lines.append('| %s | %s | %s | %s | %s | %s | %s |' % (
            r['deployment'], _fmt(r.get('sampler')), _fmt(r.get('gpu_count')),
            r.get('measured_iterations'), _fmt(r.get('episodes_per_hour')),
            _fmt(r.get('gpu_seconds')),
            _fmt(r.get('peak_gb_per_rank'))))
    lines += ['', '## SFT 更新差异与测量缺项', '', '```json', json.dumps(dict(
        sft_update_drift=report['sft_update_drift'], issues=report['issues']),
        ensure_ascii=False, indent=2), '```', '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--require-complete', action='store_true')
    args = ap.parse_args()
    run = Path(args.run_dir)
    report = collect(run)
    out = run / 'engineering'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'report.md').write_text(render(report), encoding='utf-8')
    if args.require_complete and not report['complete']:
        raise SystemExit('Engineering evidence incomplete: ' + '; '.join(report['issues']))
    print(json.dumps(dict(complete=report['complete'], report=str(out / 'report.md'))))


if __name__ == '__main__':
    main()
