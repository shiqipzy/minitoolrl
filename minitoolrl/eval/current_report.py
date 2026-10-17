"""Summarize the current five-model evaluation, training budgets and diagnostics."""
import json
from collections import Counter, defaultdict
from pathlib import Path

from minitoolrl.eval.train_curves import collect as collect_curves


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_report(run, models, summaries):
    result = dict(run_id=load(run / 'protocol.json')['run_id'], suites=summaries,
                  training={}, failures={})
    lines = ['# MiniToolRL 运行结果', '',
             '组合指标及按世界配对的置信区间见 [组合报告](comp_suite/final_report.json)。'
             '分布式性能测量见 [工程报告](../engineering/report.md)。', '',
             '| 模型 | 组合成功/360 | 通用成功/372 | 双订单退款成功/60 |', '|---|---:|---:|---:|']
    for tag in models:
        lines.append('| %s | %s | %s | %s |' % (tag, *[summaries[s+'_'+tag]['success'] for s in ('comp_suite', 'c_recipe', 'long')]))
        failures = {}
        for suite in ('comp_suite', 'c_recipe', 'long'):
            rows = load(run / 'eval' / suite / 'run' / tag / 'greedy' / (tag + '_analysis.json'))
            by_family = defaultdict(list)
            for r in rows:
                by_family[r['family']].append(r)
            failures[suite] = {family: dict(n=len(rs), success=sum(r['success'] for r in rs),
                missing=dict(Counter(x for r in rs for x in r.get('missing', []))),
                extra_writes=dict(Counter(x for r in rs for x in r.get('extra_writes', []))),
                parse_error_tasks=sum(bool(r.get('parse_errors')) for r in rs),
                tool_error_tasks=sum(bool(r.get('tool_errors')) for r in rs)) for family, rs in by_family.items()}
        result['failures'][tag] = failures
    result['training']['SFT'] = load(run / 'sft/summary.json')
    result['training']['G0'] = load(run / 'grpo/main/grpo_cost.json')
    n = load(run / 'protocol.json')['config']['world_size']
    for tag in models[2:]:
        result['training'][tag] = load(run / 'arms' / tag / ('n%d' % n) / 'train/summary.json')
    result['engineering'] = load(run / 'engineering/report.json')
    curves = collect_curves(run)
    (run / 'eval/train_curves.json').write_text(json.dumps(curves, ensure_ascii=False, indent=2), encoding='utf-8')
    lines += ['', '[训练曲线数据](train_curves.json) 分别保存主训练与工程短测的逐步 loss、梯度、'
              '学习率、采样成功率及开发集任务成功率。']
    lines += ['', '每族成功计数、漏做/多做操作、解析和工具错误、实际训练预算保存在 eval_summary.json。'
              '所有数值来自本次已完成的评测；训练与短测检查点不按测试成绩挑选。', '']
    (run / 'eval/eval_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (run / 'eval/report.md').write_text('\n'.join(lines), encoding='utf-8')
