"""Behavioural metrics on full autonomous trajectories.

token accuracy on a correct prefix is not evidence that the agent can carry a
task; these counters look only at what the model actually did and what the
executor ended up with: policy coverage, whether compensation was really
granted, amount correctness, repeated-query loops and missing steps.
"""
from __future__ import annotations

import json
from collections import Counter

from minitoolrl.env import policy as P

COMPENSATION = ('tier_compensation', 'refund_with_credit')
BUDGET_STOPS = ('turn_budget', 'call_budget', 'context_budget')


def calls_of(row):
    out = []
    for turn in row.get('trace', []):
        for call in turn.get('calls', []):
            out.append(call)
    return out


def cell_of(inst):
    if inst is None or inst.family not in COMPENSATION:
        return None
    tier = inst.params.get('tier')
    credit = inst.params.get('credit_cents')
    capped = bool(credit == P.TIER_CREDIT_CAP_CENTS.get(tier))
    return tier + '|capped' if capped else tier + '|uncapped'


def expected_records(inst, kind):
    for assertion in inst.assertions or []:
        if assertion.get('kind') == kind:
            return assertion.get('expected', [])
    return []


def credit_ok(row, inst):
    if inst is None or 'credit_cents' not in inst.params:
        return None
    want = (inst.params['customer_id'], inst.params['credit_cents'])
    got = [(c.get('customer_id'), c.get('amount_cents')) for c in row['final_world'].get('credits', [])]
    return got.count(want) == 1 and len(got) == 1


def refund_ok(row, inst):
    if inst is None:
        return None
    want = expected_records(inst, 'refunds_exact')
    got = row['final_world'].get('refunds', [])
    if not want:
        return None
    return len(got) == len(want) and all(
        any(all(g.get(k) == v for k, v in exp.items()) for g in got) for exp in want)


def behaviour(row, inst):
    calls = calls_of(row)
    names = Counter(c['name'] for c in calls)
    repeats = Counter((c['name'], json.dumps(c['arguments'], sort_keys=True, ensure_ascii=False))
                      for c in calls)
    comp = row['family'] in COMPENSATION
    loop = bool(repeats) and repeats.most_common(1)[0][1] >= 3 and row.get('stop_reason') in BUDGET_STOPS
    out = dict(task_id=row['task_id'], family=row['family'], cell=cell_of(inst),
               success=bool(row['success']), n_turns=row['n_turns'], n_calls=row['n_calls'],
               stop_reason=row.get('stop_reason'), failure_tags=list(row.get('failure_tags', [])),
               tool_errors=row['n_tool_errors'], parse_errors=row['n_parse_errors'],
               max_repeat=repeats.most_common(1)[0][1] if repeats else 0,
               repeat_query_loop=loop, truncated=bool(row.get('stop_reason') in BUDGET_STOPS),
               no_tool_call=sum(names.values()) == 0)
    if comp:
        policy_calls = [c for c in calls if c['name'] == 'get_policy'
                        and str(c['arguments'].get('topic', '')).lower() == 'compensation']
        credited = names.get('apply_credit', 0) > 0
        out['policy_query_compensation'] = bool(policy_calls)
        out['apply_credit_called'] = credited
        out['credit_amount_correct'] = credit_ok(row, inst)
        out['refunds_expected'] = len(expected_records(inst, 'refunds_exact'))
        out['refunds_final'] = len(row['final_world'].get('refunds', []))
        out['missing_refund'] = out['refunds_final'] < out['refunds_expected']
    return out


def rate(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return dict(n=len(vals), correct=sum(bool(v) for v in vals),
                rate=(round(sum(bool(v) for v in vals) / len(vals), 4) if vals else None))


def success_rate(rows):
    return dict(n=len(rows), correct=sum(r['success'] for r in rows),
                rate=round(sum(r['success'] for r in rows) / len(rows), 4) if rows else None)


def summarize(transcripts, instances):
    rows = [behaviour(row, instances.get(row['task_id'])) for row in transcripts]
    comp = [r for r in rows if r['family'] in COMPENSATION]
    tier = [r for r in comp if r['family'] == 'tier_compensation']
    composite = [r for r in comp if r['family'] == 'refund_with_credit']
    ordinary = [r for r in rows if r['family'] not in COMPENSATION]
    out = dict(overall=success_rate(rows), ordinary=success_rate(ordinary),
               compensation=success_rate(comp), tier_compensation=success_rate(tier),
               refund_with_credit=success_rate(composite),
               by_family={f: success_rate([r for r in rows if r['family'] == f])
                          for f in sorted({r['family'] for r in rows})},
               by_cell={c: dict(success=success_rate([r for r in comp if r['cell'] == c]),
                                policy_query=rate(comp, 'policy_query_compensation'))
                        for c in sorted({r['cell'] for r in comp if r['cell']})},
               compensation_flow=dict(
                   policy_query=rate(comp, 'policy_query_compensation'),
                   apply_credit_called=rate(comp, 'apply_credit_called'),
                   credit_amount_correct_given_credit=rate(
                       [r for r in comp if r['apply_credit_called']], 'credit_amount_correct'),
                   credit_amount_correct_all=rate(comp, 'credit_amount_correct'),
                   repeat_query_loop=rate(comp, 'repeat_query_loop'),
                   truncated=rate(rows, 'truncated'),
                   tool_error_mean=round(sum(r['tool_errors'] for r in rows) / max(1, len(rows)), 3),
                   parse_error_tasks=sum(1 for r in rows if r['parse_errors'])),
               composite_flow=dict(missing_refund=rate(composite, 'missing_refund'),
                                   apply_credit_called=rate(composite, 'apply_credit_called'),
                                   credit_amount_correct_all=rate(composite, 'credit_amount_correct')),
               failure_tags=dict(sorted(Counter(tag for r in rows for tag in r['failure_tags']).items())),
               stop_reasons=dict(sorted(Counter(str(r['stop_reason']) for r in rows).items())),
               per_task=rows)
    return out
