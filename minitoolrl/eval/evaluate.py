#!/usr/bin/env python3
"""五个模型终点的组合、通用和双订单退款 greedy 评测。

退款单和信用补偿按实体、金额核对；任务要求的回库操作计入完成条件。
已完成轨迹用于补齐评测配额，实际任务执行量写入结果。
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer

from minitoolrl.data.continuation_data import finance_amount_stats
from minitoolrl.env.policy import VERSION
from minitoolrl.env.session import EpisodeBudget
from minitoolrl.env.tasks import TaskInstance
from minitoolrl.train.experiment_config import EPISODE_BUDGET
from minitoolrl.eval.runtime import (EVAL_BATCH_SIZE, VISIBLE_TOOLS, bind_evaluation_context,
                                   close_sampler, greedy_rollouts,
                                   load_done, make_sampler, sha, tools_for_order)


def to_inst(row: dict) -> TaskInstance:
    keys = ['task_id', 'family', 'split_group', 'difficulty', 'seed', 'prompt', 'world',
            'params', 'tool_signature', 'gold_calls', 'assertions']
    return TaskInstance.from_json({k: row[k] for k in keys})


def analyze(row: dict, meta: dict) -> dict:
    stats = finance_amount_stats(row)
    tags = list(row.get('failure_tags') or [])
    trace = row.get('trace') or []
    attempted = Counter()
    succeeded = Counter()
    queried_customer = queried_policy = False
    first_finance = None
    finance_order = []
    for turn in trace:
        for call in turn.get('calls') or []:
            name = call.get('name')
            attempted[name] += 1
            result = call.get('result') or {}
            if not (isinstance(result, dict) and 'error' in result):
                succeeded[name] += 1
            args = call.get('arguments') or {}
            if name == 'get_customer':
                queried_customer = True
            if name == 'get_policy' and args.get('topic') == 'compensation':
                queried_policy = True
            if name in ('create_refund', 'apply_credit'):
                finance_order.append(name)
                if first_finance is None:
                    first_finance = name
    return dict(
        task_id=row['task_id'],
        eval_tag=meta.get('eval_tag') or row.get('eval_tag'),
        variant=meta.get('variant') or row.get('variant'),
        family=row.get('family'),
        seed=row.get('seed'),
        world_id=meta.get('world_id') or row.get('world_id'),
        stratum=meta.get('stratum') or row.get('stratum'),
        sample_index=row.get('sample_index'),
        success=bool(row.get('success')),
        failure_tags=tags,
        n_turns=row.get('n_turns'),
        n_calls=row.get('n_calls'),
        refund_attempted=attempted.get('create_refund', 0) > 0,
        credit_attempted=attempted.get('apply_credit', 0) > 0,
        queried_customer=queried_customer,
        queried_policy_compensation=queried_policy,
        first_finance=first_finance,
        finance_order=finance_order,
        parse_errors=int(row.get('n_parse_errors') or 0),
        tool_errors=int(row.get('n_tool_errors') or 0),
        stop_reason=row.get('stop_reason'),
        **stats,
    )


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def summarize(analyses: List[dict], tag: str, quota: int, filled: int,
              incomplete: bool, stop_reason, decode, batch_log) -> dict:
    summary = dict(
        tag=tag, n_analyses=len(analyses), quota=quota, filled=filled,
        incomplete=incomplete or filled < quota,
        stop_reason=stop_reason if (incomplete or filled < quota) else 'complete',
        success=sum(a['success'] for a in analyses),
        decode=decode, batch_log=batch_log, by_tag={},
    )
    by_tag = defaultdict(list)
    for a in analyses:
        by_tag[a.get('eval_tag')].append(a)
    for t, rows in by_tag.items():
        cells = defaultdict(list)
        for r in rows:
            cell = (r.get('stratum') or {}).get('cell') or '?'
            cells[cell].append(r)
        variants = defaultdict(list)
        for r in rows:
            variants[r.get('variant')].append(r)
        summary['by_tag'][t] = dict(
            n=len(rows),
            success=sum(r['success'] for r in rows),
            refund_written=sum(r['refund_written'] for r in rows),
            refund_amount_ok=sum(r['refund_amount_ok'] for r in rows),
            credit_written=sum(r['credit_written'] for r in rows),
            credit_amount_ok=sum(r['credit_amount_ok'] for r in rows),
            extra_ticket=sum('open_ticket' in r['extra_writes'] for r in rows),
            extra_notify=sum('notify_customer' in r['extra_writes'] for r in rows),
            extra_restock=sum('restock_item' in r['extra_writes'] for r in rows),
            missing_refund=sum('missing_refund' in r['missing'] for r in rows),
            missing_credit=sum('missing_credit' in r['missing'] for r in rows),
            queried_customer=sum(r['queried_customer'] for r in rows),
            queried_policy=sum(r['queried_policy_compensation'] for r in rows),
            parse_errors=sum(r['parse_errors'] > 0 for r in rows),
            tool_errors=sum(r['tool_errors'] > 0 for r in rows),
            truncated=sum(r.get('stop_reason') in ('turn_budget', 'call_budget', 'context_budget')
                          for r in rows),
            by_family=dict(Counter(r['family'] for r in rows if r['success'])),
            by_variant={k: dict(n=len(v), success=sum(x['success'] for x in v))
                        for k, v in variants.items() if k is not None},
            by_cell={c: dict(n=len(rs), success=sum(x['success'] for x in rs),
                             credit_amount_ok=sum(x['credit_amount_ok'] for x in rs),
                             refund_amount_ok=sum(x['refund_amount_ok'] for x in rs))
                     for c, rs in cells.items()},
        )
    return summary


def remaining_ok(started: float, max_seconds: float, pad: float = 20.0) -> bool:
    return time.time() - started + pad < max_seconds


def run_greedy(sampler, tok, pool: List[dict], budget, out_dir: Path, tag: str,
               temperature: float, top_p: float, top_k: int, batch_size: int,
               max_seconds: float, started: float, *, identity=None, visible_tools=None) -> dict:
    if temperature != 0.0:
        raise ValueError('evaluation requires greedy temperature=0')
    rollouts_path = out_dir / f'{tag}_rollouts.jsonl'
    bind_evaluation_context(rollouts_path, tok, [to_inst(r) for r in pool], budget,
                            top_p=top_p, top_k=top_k, batch_size=batch_size, identity=identity,
                            visible_tools=visible_tools)
    done = load_done(rollouts_path)
    expected = {(rec['task_id'], None) for rec in pool}
    if len(expected) != len(pool) or not set(done).issubset(expected):
        raise ValueError('duplicate input tasks or saved rollouts outside the greedy quota')
    analyses = []
    batch_log = []
    decode = dict(temperature=temperature, top_p=top_p, top_k=top_k,
                  do_sample=temperature > 0)
    pending = [r for r in pool if (r['task_id'], None) not in done]
    for rec in pool:
        if (rec['task_id'], None) in done:
            analyses.append(analyze(done[(rec['task_id'], None)], rec))
    filled = len(analyses)
    quota = len(pool)
    incomplete = False
    stop_reason = None
    print(json.dumps(dict(event='greedy_resume', tag=tag, already=filled, pending=len(pending))),
          flush=True)
    with torch.inference_mode():
        for start in range(0, len(pending), batch_size):
            if not remaining_ok(started, max_seconds):
                incomplete = True
                stop_reason = 'time_budget'
                break
            batch_meta = pending[start:start + batch_size]
            insts = [to_inst(r) for r in batch_meta]
            bseed = 17 + start
            torch.manual_seed(bseed)
            random.seed(bseed)
            batch_log.append(dict(batch_seed=bseed, start=start,
                                  task_ids=[r['task_id'] for r in batch_meta]))
            group = greedy_rollouts(
                sampler, tok, insts, budget, top_p=top_p, top_k=top_k, seed_base=bseed,
                visible_tools=visible_tools,
            )
            for rec, ep in zip(batch_meta, group.episodes):
                row = ep.transcript()
                row['eval_tag'] = rec.get('eval_tag')
                row['variant'] = rec.get('variant')
                row['world_id'] = rec.get('world_id')
                row['stratum'] = rec.get('stratum')
                row['decode'] = decode
                row['assertions'] = rec.get('assertions')
                row['gold_calls'] = rec.get('gold_calls')
                row['world'] = rec.get('world')
                append_row(rollouts_path, row)
                analyses.append(analyze(row, rec))
                filled += 1
            print(json.dumps(dict(
                event='eval_progress', tag=tag, done=filled, quota=quota,
                success=sum(a['success'] for a in analyses),
                seconds=round(time.time() - started, 1),
            ), ensure_ascii=False), flush=True)
    return dict(summary=summarize(analyses, tag, quota, filled, incomplete, stop_reason,
                                  decode, batch_log),
                analyses=analyses)


def write_eval_outputs(out: Path, tag: str, meta: dict, result: dict) -> None:
    meta = dict(meta, summary=result['summary'])
    (out / f'{tag}_run.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                         encoding='utf-8')
    (out / f'{tag}_analysis.json').write_text(
        json.dumps(result['analyses'], ensure_ascii=False, indent=2), encoding='utf-8')
    (out / f'{tag}_summary.json').write_text(
        json.dumps(result['summary'], ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--pool', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--mode', choices=['greedy'], default='greedy')
    ap.add_argument('--temperature', type=float, choices=[0.0], default=0.0)
    ap.add_argument('--top-p', type=float, default=0.95)
    ap.add_argument('--top-k', type=int, default=0)
    ap.add_argument('--batch-size', type=int, default=EVAL_BATCH_SIZE)
    ap.add_argument('--tool-order', choices=['registry', 'alphabetical'], default='registry',
                    help='match frozen training: SFT/G0 registry; existing continuation arms alphabetical')
    ap.add_argument('--max-seconds', type=float, default=900)
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    args = ap.parse_args()

    started = time.time()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool = [json.loads(l) for l in Path(args.pool).read_text(encoding='utf-8').splitlines()
            if l.strip()]
    budget = EpisodeBudget(**EPISODE_BUDGET)
    temperature = args.temperature
    visible_tools = tools_for_order(args.tool_order)
    meta = dict(
        tag=args.tag, model=args.model, mode=args.mode, backend='vllm',
        weight_sha256=sha(Path(args.model) / 'model.safetensors'),
        pool=args.pool, pool_sha256=sha(args.pool), policy_version=VERSION,
        n_tasks=len(pool), sample_n=1,
        temperature=temperature, top_p=args.top_p, top_k=args.top_k,
        episode_budget=dict(EPISODE_BUDGET), visible_tools=list(visible_tools),
        tool_order=args.tool_order,
        resume=True,
        note='full autonomous; exact entity/amount match; required restock not extra',
    )
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    identity = dict(weight_sha256=meta['weight_sha256'], pool_sha256=meta['pool_sha256'])
    context = bind_evaluation_context(out / f'{args.tag}_rollouts.jsonl', tok,
        [to_inst(r) for r in pool], budget, top_p=args.top_p, top_k=args.top_k,
        batch_size=args.batch_size, identity=identity, visible_tools=visible_tools)
    meta.update(evaluation_context=context['context'],
                evaluation_context_sha256=context['context_sha256'])
    sampler = make_sampler(args.model,
                           gpu_memory_utilization=args.gpu_memory_utilization)
    try:
        result = run_greedy(
            sampler, tok, pool, budget, out, args.tag, temperature, args.top_p,
            args.top_k, args.batch_size, args.max_seconds, started, identity=identity,
            visible_tools=visible_tools)
    finally:
        close_sampler(sampler)

    meta.update(seconds=round(time.time() - started, 1),
                peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
    write_eval_outputs(out, args.tag, meta, result)
    if result['summary']['incomplete']:
        raise SystemExit('evaluation quota incomplete')
    print('EVAL_COMPLETE', json.dumps(dict(
        tag=args.tag, mode=args.mode, seconds=meta['seconds'],
        filled=result['summary']['filled'], quota=result['summary']['quota'],
        incomplete=result['summary']['incomplete'],
        success=result['summary']['success'],
    ), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
