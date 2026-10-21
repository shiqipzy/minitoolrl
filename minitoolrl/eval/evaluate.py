#!/usr/bin/env python3
"""正式评测：执行与续跑、逐题分析、组合比较和完整运行报告。

五个模型终点各执行组合 360、通用 372 和双订单退款 60 条 greedy 任务。
run_experiment 的 evals 阶段跑齐全部任务后，调用本模块核对配额并生成报告。
SFT 开发集检查复用这里的 greedy 调度；模型与工具的多轮交互由 train/rollout.py 执行。
GPU 依赖只在实际执行任务时加载，已有轨迹的统计与报告可在 CPU 主机运行。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from minitoolrl import files
from minitoolrl.data.composition_suite import (
    BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, TARGET_FAMILIES, VARIANTS,
)
from minitoolrl.data.continuation_data import finance_amount_stats
from minitoolrl.env import policy as P, tools as T
from minitoolrl.env.session import EpisodeBudget
from minitoolrl.env.tasks import TaskInstance
from minitoolrl.train import experiment_config as C


# 任务执行：工具与评测条件、采样器、轨迹落盘和续跑。
BUDGET = EpisodeBudget(**C.EPISODE_BUDGET)
# 基础 SFT 未指定 exposed_tools，GRPO 采样也使用工具注册顺序。
VISIBLE_TOOLS = tuple(T.ALL_TOOL_NAMES)
EVAL_BATCH_SIZE = 8
SEED_BASE = 17


def tools_for_order(order):
    """SFT、GRPO、三组续训和评测统一使用工具注册顺序。"""
    if order != 'registry':
        raise ValueError('the current input bundle requires registry tool order')
    return VISIBLE_TOOLS


def resolved_tools(visible_tools=None):
    names = tuple(VISIBLE_TOOLS if visible_tools is None else visible_tools)
    if names != VISIBLE_TOOLS:
        raise ValueError('evaluation must expose every tool in registry order')
    return names


def content_sha(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def evaluation_context(tok, budget=BUDGET, *, top_p=0.95, top_k=0, batch_size=EVAL_BATCH_SIZE,
                       visible_tools=None):
    """记录实际使用的工具顺序、说明、模板、规则和采样条件。"""
    if asdict(budget) != C.EPISODE_BUDGET:
        raise ValueError('evaluation budget differs from the shared episode budget')
    names = resolved_tools(visible_tools)
    return dict(
        visible_tools=list(names),
        tool_schema_sha256=content_sha(T.tool_schemas(names)),
        chat_template_sha256=hashlib.sha256(tok.chat_template.encode('utf-8')).hexdigest(),
        system_prompt_sha256=content_sha(P.OPERATING_RULES),
        policy_docs_sha256=content_sha(P.POLICY_DOCS),
        enable_thinking=False, episode_budget=asdict(budget),
        temperature=0.0, top_p=top_p, top_k=top_k, seed=SEED_BASE, batch_size=batch_size,
    )


def bind_evaluation_context(rollouts_path, tok, instances, budget=BUDGET, *,
                            top_p=0.95, top_k=0, batch_size=EVAL_BATCH_SIZE, identity=None,
                            visible_tools=None):
    """续跑前核对输入条件；缺少旧上下文记录时拒绝混入已有轨迹。"""
    rollouts_path = Path(rollouts_path)
    context = evaluation_context(tok, budget, top_p=top_p, top_k=top_k, batch_size=batch_size,
                                 visible_tools=visible_tools)
    record = dict(context=context, context_sha256=content_sha(context),
                  task_pool_sha256=content_sha([inst.to_json() for inst in instances]),
                  identity=identity)
    path = rollouts_path.with_suffix('.context.json')
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != record:
            raise ValueError('evaluation context or inputs changed; use a new output directory')
    else:
        if rollouts_path.exists():
            raise ValueError('saved rollouts have no evaluation context; use a new output directory')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return record


def greedy_rollouts(sampler, tok, instances, budget=BUDGET, *, top_p=0.95, top_k=0,
                    seed_base=SEED_BASE, visible_tools=None):
    """正式评测与开发集检查共用同一工具列表、顺序和 greedy 调度。"""
    if asdict(budget) != C.EPISODE_BUDGET:
        raise ValueError('evaluation budget differs from the shared episode budget')
    from minitoolrl.train.rollout import rollout_pool
    return rollout_pool(sampler, tok, instances, budget, temperature=0.0,
                        visible_tools=resolved_tools(visible_tools), top_p=top_p, top_k=top_k,
                        seed_base=seed_base)


def load_done(path):
    """只读已有任务轨迹；重复执行记录不能当成已完成配额。"""
    path = Path(path)
    done = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row['task_id'], row.get('sample_index'))
            if key in done:
                raise ValueError('duplicate saved rollout: ' + str(key))
            done[key] = row
    return done


def make_sampler(model_path, max_model_len=4096, gpu_memory_utilization=0.85):
    """评测统一使用进程内 vLLM 实例。"""
    from minitoolrl.train.vllm_backend import LocalVLLM  # 懒加载：无 GPU 环境仍可导入
    return LocalVLLM(model_path, max_model_len=max_model_len,
                     gpu_memory_utilization=gpu_memory_utilization)


def close_sampler(sampler) -> None:
    """关闭 vLLM 实例并释放显存。"""
    import torch
    sampler.close()
    import gc
    gc.collect()
    torch.cuda.empty_cache()


@dataclass
class PoolRun:
    """一次 greedy 执行的原始记录与完成情况；成绩由调用方计算。"""
    rows: list
    incomplete: bool
    stop_reason: Optional[str]
    context: dict
    batch_log: list = field(default_factory=list)
    resumed: int = 0


def run_greedy_pool(sampler, tok, instances, rollouts_path, *, budget=BUDGET, top_p=0.95,
                    top_k=0, batch_size=EVAL_BATCH_SIZE, max_seconds=None, started=None,
                    identity=None, visible_tools=None, enrich=None) -> PoolRun:
    """按任务顺序执行 greedy 轨迹，已有记录用于补足配额。

    每个批次固定 seed = 17 + 批内起始位置，并按池内任务顺序追加写入 JSONL。
    max_seconds 为 None 时不在批次之间检查时间；给定时超时返回 incomplete，
    由调用方决定是否拒绝该结果。enrich(row, instance) 可补充评测池元数据。
    """
    import torch

    path = Path(rollouts_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    context = bind_evaluation_context(path, tok, instances, budget, top_p=top_p, top_k=top_k,
                                      batch_size=batch_size, identity=identity,
                                      visible_tools=visible_tools)
    done = load_done(path)
    expected = {(inst.task_id, None) for inst in instances}
    if len(expected) != len(instances) or not set(done).issubset(expected):
        raise ValueError('duplicate input tasks or saved rollouts outside the greedy quota')
    rows = [done[(inst.task_id, None)] for inst in instances if (inst.task_id, None) in done]
    pending = [inst for inst in instances if (inst.task_id, None) not in done]
    print(json.dumps(dict(event='greedy_resume', file=path.name,
                          already=len(rows), pending=len(pending))), flush=True)
    batch_log = []
    incomplete = False
    stop_reason = None
    clock_start = started if started is not None else time.time()
    with torch.inference_mode():
        for start in range(0, len(pending), batch_size):
            if max_seconds is not None and time.time() - clock_start + 20.0 >= max_seconds:
                incomplete = True
                stop_reason = 'time_budget'
                break
            batch = pending[start:start + batch_size]
            seed = SEED_BASE + start
            batch_log.append(dict(batch_seed=seed, start=start,
                                  task_ids=[inst.task_id for inst in batch]))
            torch.manual_seed(seed)
            random.seed(seed)
            group = greedy_rollouts(sampler, tok, batch, budget, top_p=top_p, top_k=top_k,
                                    seed_base=seed, visible_tools=visible_tools)
            for inst, episode in zip(batch, group.episodes):
                row = episode.transcript()
                if enrich is not None:
                    enrich(row, inst)
                with path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                rows.append(row)
            print(json.dumps(dict(event='greedy_progress', file=path.name, done=len(rows),
                                  total=len(instances),
                                  success=sum(bool(row.get('success')) for row in rows)),
                             ensure_ascii=False), flush=True)
    return PoolRun(rows=rows, incomplete=incomplete, stop_reason=stop_reason, context=context,
                   batch_log=batch_log, resumed=len(instances) - len(pending))


# 单个评测池：逐题业务分析、成功计数和配额验收。
def to_inst(row: dict) -> TaskInstance:
    keys = ['task_id', 'family', 'split_group', 'difficulty', 'seed', 'prompt', 'world',
            'params', 'tool_signature', 'gold_calls', 'assertions']
    return TaskInstance.from_json({k: row[k] for k in keys})


def analyze(row: dict, meta: dict) -> dict:
    stats = finance_amount_stats(row)
    tags = list(row.get('failure_tags') or [])
    trace = row.get('trace') or []
    attempted = Counter()
    queried_customer = queried_policy = False
    first_finance = None
    finance_order = []
    for turn in trace:
        for call in turn.get('calls') or []:
            name = call.get('name')
            attempted[name] += 1
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


def run_greedy(sampler, tok, pool: List[dict], budget, out_dir: Path, tag: str,
               temperature: float, top_p: float, top_k: int, batch_size: int,
               max_seconds: float, started: float, *, identity=None,
               visible_tools=None) -> dict:
    """执行一个评测池并汇总；已有轨迹复用，成绩按池内任务顺序计算。"""
    if temperature != 0.0:
        raise ValueError('evaluation requires greedy temperature=0')
    decode = dict(temperature=temperature, top_p=top_p, top_k=top_k,
                  do_sample=temperature > 0)
    records = {rec['task_id']: rec for rec in pool}

    def attach(row, inst):
        """把评测池的标签、分层和参考实现写进轨迹，供失败分析读取。"""
        rec = records[inst.task_id]
        row['eval_tag'] = rec.get('eval_tag')
        row['variant'] = rec.get('variant')
        row['world_id'] = rec.get('world_id')
        row['stratum'] = rec.get('stratum')
        row['decode'] = decode
        row['assertions'] = rec.get('assertions')
        row['gold_calls'] = rec.get('gold_calls')
        row['world'] = rec.get('world')

    run = run_greedy_pool(
        sampler, tok, [to_inst(rec) for rec in pool], Path(out_dir) / f'{tag}_rollouts.jsonl',
        budget=budget, top_p=top_p, top_k=top_k, batch_size=batch_size,
        max_seconds=max_seconds, started=started, identity=identity,
        visible_tools=visible_tools, enrich=attach)
    by_task = {row['task_id']: row for row in run.rows}
    analyses = [analyze(by_task[rec['task_id']], rec) for rec in pool
                if rec['task_id'] in by_task]
    return dict(summary=summarize(analyses, tag, len(pool), len(analyses), run.incomplete,
                                  run.stop_reason, decode, run.batch_log),
                analyses=analyses, context=run.context)


def write_eval_outputs(out: Path, tag: str, meta: dict, result: dict) -> None:
    meta = dict(meta, summary=result['summary'])
    files.dump_json(out / f'{tag}_run.json', meta)
    files.dump_json(out / f'{tag}_analysis.json', result['analyses'])
    files.dump_json(out / f'{tag}_summary.json', result['summary'])


def verify_evaluations(jobs):
    """核对五个终点各套任务的轨迹、逐题分析和汇总，拒绝缺题或重复记录。"""
    summaries = {}
    for job in jobs:
        pool = files.read_jsonl(job['pool'])
        quota = len(pool)
        expected = {(row['task_id'], None) for row in pool}
        tag, out = job['tag'], job['out']
        row = files.load_json(out / (tag + '_summary.json'), {})
        done = load_done(out / (tag + '_rollouts.jsonl'))
        analyses = files.load_json(out / (tag + '_analysis.json'), [])
        keys = [(r['task_id'], r.get('sample_index')) for r in analyses]
        if (len(expected) != quota or set(done) != expected or len(keys) != quota
                or set(keys) != expected or row.get('filled') != quota or row.get('quota') != quota
                or row.get('incomplete', True)
                or any(r.get('success') != done[k].get('success') for r, k in zip(analyses, keys))
                or row.get('success') != sum(bool(r.get('success')) for r in done.values())):
            raise RuntimeError('evaluation missing, incomplete or inconsistent: ' + job['name'])
        summaries[job['name']] = row
    return summaries


# 组合任务比较：同场景 A/B 表达和按场景配对的 Bootstrap。
def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot take percentile of an empty sequence")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    fraction = index - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def paired_bootstrap(
    left: Mapping[Tuple[str, str], float],
    right: Mapping[Tuple[str, str], float],
    worlds_by_family: Mapping[str, Sequence[str]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    families = list(TARGET_FAMILIES)
    keys = {(family, world) for family in families for world in worlds_by_family[family]}
    if set(left) != keys or set(right) != keys:
        raise RuntimeError("Bootstrap inputs do not match the frozen family/world keys")

    point_by_family = {
        family: sum(
            left[(family, world)] - right[(family, world)]
            for world in worlds_by_family[family]
        )
        / len(worlds_by_family[family])
        for family in families
    }
    point = sum(point_by_family.values()) / len(families)
    rng = random.Random(seed)
    samples: List[float] = []
    for _ in range(draws):
        family_deltas = []
        for family in families:
            worlds = worlds_by_family[family]
            delta = 0.0
            for _ in worlds:
                world = worlds[rng.randrange(len(worlds))]
                delta += left[(family, world)] - right[(family, world)]
            family_deltas.append(delta / len(worlds))
        samples.append(sum(family_deltas) / len(families))
    return {
        "point": point,
        "ci_low": percentile(samples, 0.025),
        "ci_high": percentile(samples, 0.975),
        "draws": draws,
        "seed": seed,
        "point_by_family": point_by_family,
    }


def validate_and_index(
    analyses: Sequence[Mapping[str, Any]],
    expected: Mapping[str, Mapping[str, Any]],
    model: str,
) -> Dict[Tuple[str, str, str], Mapping[str, Any]]:
    by_task: Dict[str, Mapping[str, Any]] = {}
    for row in analyses:
        task_id = row.get("task_id")
        if task_id in by_task:
            raise RuntimeError(f"{model}: duplicate analysis row {task_id}")
        by_task[task_id] = row
    missing = sorted(set(expected) - set(by_task))
    extra = sorted(set(by_task) - set(expected))
    if missing or extra:
        raise RuntimeError(
            f"{model}: task mismatch missing={missing[:3]} extra={extra[:3]} "
            f"counts={len(missing)}/{len(extra)}"
        )

    indexed: Dict[Tuple[str, str, str], Mapping[str, Any]] = {}
    for task_id, meta in expected.items():
        row = by_task[task_id]
        family = meta["family"]
        world_id = meta["world_id"]
        variant = meta["variant"]
        for key, want in (("family", family), ("world_id", world_id), ("variant", variant)):
            if row.get(key) != want:
                raise RuntimeError(
                    f"{model}: {task_id} {key}={row.get(key)!r}, expected {want!r}"
                )
        indexed[(family, world_id, variant)] = row
    return indexed


def summarize_model(
    indexed: Mapping[Tuple[str, str, str], Mapping[str, Any]],
    worlds_by_family: Mapping[str, Sequence[str]],
) -> Dict[str, Any]:
    per_family: Dict[str, Any] = {}
    avg_scores: Dict[Tuple[str, str], float] = {}
    both_scores: Dict[Tuple[str, str], float] = {}
    failure_tags: Dict[str, Any] = {}

    for family in TARGET_FAMILIES:
        worlds = worlds_by_family[family]
        by_variant = {}
        for variant in VARIANTS:
            rows = [indexed[(family, world, variant)] for world in worlds]
            by_variant[variant] = {
                "success": sum(bool(row.get("success")) for row in rows),
                "n": len(rows),
            }
        both = 0
        either = 0
        for world in worlds:
            a = float(bool(indexed[(family, world, "A")].get("success")))
            b = float(bool(indexed[(family, world, "B")].get("success")))
            avg_scores[(family, world)] = (a + b) / 2.0
            both_scores[(family, world)] = min(a, b)
            both += int(a == 1.0 and b == 1.0)
            either += int(a == 1.0 or b == 1.0)
        avg_rate = sum(avg_scores[(family, world)] for world in worlds) / len(worlds)
        per_family[family] = {
            "worlds": len(worlds),
            "by_variant": by_variant,
            "mean_single_prompt_success": avg_rate,
            "both_wordings_success": both,
            "both_wordings_rate": both / len(worlds),
            "either_wording_success": either,
            "either_wording_rate": either / len(worlds),
        }
        failure_tags[family] = {
            variant: dict(
                Counter(
                    tag
                    for world in worlds
                    for tag in indexed[(family, world, variant)].get("failure_tags", [])
                )
            )
            for variant in VARIANTS
        }

    return {
        "per_family": per_family,
        "macro_mean_single_prompt_success": sum(
            row["mean_single_prompt_success"] for row in per_family.values()
        )
        / len(per_family),
        "macro_both_wordings_rate": sum(
            row["both_wordings_rate"] for row in per_family.values()
        )
        / len(per_family),
        "avg_world_scores": avg_scores,
        "both_world_scores": both_scores,
        "failure_tags": failure_tags,
    }


def serializable_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in summary.items()
        if key not in ("avg_world_scores", "both_world_scores")
    }


def write_composition_report(root, models, comparisons):
    """按冻结场景核对组合成绩，统计 A/B 表达与模型差异的配对区间。"""
    pairs = []
    for item in comparisons:
        item = item.strip()
        if not item:
            continue
        left, right = item.split("-", 1)
        if left not in models or right not in models:
            raise RuntimeError(f"comparison {item} names a model outside the evaluated models")
        pairs.append((left, right))
    root = Path(root).resolve()
    protocol = read_json(root / "protocol.json")
    if tuple(protocol.get("families", [])) != TARGET_FAMILIES:
        raise RuntimeError(f"Unexpected family freeze: {protocol.get('families')}")
    if "bulk_refund" in protocol.get("families", []):
        raise RuntimeError("bulk_refund is outside this analysis protocol")

    pool_path = root / protocol["artifacts"]["pool"]["path"]
    got_pool_sha = files.sha256_file(pool_path)
    if got_pool_sha != protocol["artifacts"]["pool"]["sha256"]:
        raise RuntimeError("Frozen pool SHA256 mismatch")
    pool = files.read_jsonl(pool_path)
    expected = {row["task_id"]: row for row in pool}
    if len(expected) != protocol["n_records"]:
        raise RuntimeError(f"Pool count mismatch: {len(expected)}")

    worlds_by_family: Dict[str, List[str]] = {}
    for family in TARGET_FAMILIES:
        worlds = sorted({row["world_id"] for row in pool if row["family"] == family})
        if len(worlds) != protocol["n_worlds_per_family"]:
            raise RuntimeError(f"{family}: expected 60 worlds, got {len(worlds)}")
        worlds_by_family[family] = worlds

    summaries: Dict[str, Dict[str, Any]] = {}
    run_checks: Dict[str, Any] = {}
    for model in models:
        run_dir = root / "run" / model / "greedy"
        analysis_path = run_dir / f"{model}_analysis.json"
        summary_path = run_dir / f"{model}_summary.json"
        run_path = run_dir / f"{model}_run.json"
        analyses = read_json(analysis_path)
        run_summary = read_json(summary_path)
        run_meta = read_json(run_path)
        if run_summary.get("incomplete") or run_summary.get("filled") != len(pool):
            raise RuntimeError(f"{model}: incomplete run {run_summary}")
        expected_sha = protocol["models"][model]["sha256"]
        if run_meta.get("weight_sha256") != expected_sha:
            raise RuntimeError(
                f"{model}: weight SHA mismatch {run_meta.get('weight_sha256')} != {expected_sha}"
            )
        if run_meta.get("pool_sha256") != got_pool_sha:
            raise RuntimeError(f"{model}: pool SHA mismatch")
        indexed = validate_and_index(analyses, expected, model)
        summaries[model] = summarize_model(indexed, worlds_by_family)
        run_checks[model] = {
            "analysis_sha256": files.sha256_file(analysis_path),
            "summary_sha256": files.sha256_file(summary_path),
            "run_sha256": files.sha256_file(run_path),
            "filled": run_summary["filled"],
            "quota": run_summary["quota"],
            "incomplete": run_summary["incomplete"],
            "weight_sha256": run_meta["weight_sha256"],
        }

    comparisons = {}
    for left, right in pairs:
        label = f"{left}-{right}"
        comparisons[label] = {
            "mean_single_prompt_success": paired_bootstrap(
                summaries[left]["avg_world_scores"],
                summaries[right]["avg_world_scores"],
                worlds_by_family,
            ),
            "both_wordings_rate": paired_bootstrap(
                summaries[left]["both_world_scores"],
                summaries[right]["both_world_scores"],
                worlds_by_family,
                seed=BOOTSTRAP_SEED + 1,
            ),
        }

    report = {
        "status": "complete",
        "run_id": protocol["run_id"],
        "pool_sha256": got_pool_sha,
        "models": {model: serializable_summary(summary) for model, summary in summaries.items()},
        "comparisons": comparisons,
        "run_checks": run_checks,
        "scope": {
            "families": list(TARGET_FAMILIES),
            "worlds_per_family": protocol["n_worlds_per_family"],
            "variants": list(VARIANTS),
            "bulk_refund_included": False,
            "training_seed_uncertainty_measured": False,
        },
    }
    out_path = root / "final_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


# 完整运行报告：五个终点、三套评测、失败原因和逐步训练记录。
def read_json(path):
    """读取汇总输入；缺失即报错，不能把缺产物当成空结果。"""
    value = files.load_json(path)
    if value is None:
        raise FileNotFoundError(path)
    return value


def grpo_rows(path):
    """GRPO 逐迭代记录：补出训练采样的成功率与混合组比例。"""
    rows = files.read_jsonl(path)
    for row in rows:
        groups = [group['rewards'] for group in row['groups']]
        row['success_rate'] = sum(sum(rewards) for rewards in groups) / sum(map(len, groups))
        row['mixed_group_rate'] = sum(0 < sum(rewards) < len(rewards) for rewards in groups) / len(groups)
    return rows


def collect_curves(run):
    """保留所有更新/采样迭代，包括 GRPO 没有学习信号的迭代。

    分别保存 SFT loss、GRPO 训练采样成功率和工程短测记录，不把不同阶段的
    数据合并成同预算组间比较。
    """
    run = Path(run)
    protocol = read_json(run / 'protocol.json')
    n = protocol['config']['world_size']
    training = dict(SFT=files.read_jsonl(run / 'sft/updates.jsonl'),
                    G0=grpo_rows(run / 'grpo/main/grpo_train.jsonl'))
    for tag in ('C_blocked', 'C_mixed', 'U_mixed'):
        training[tag] = files.read_jsonl(run / 'arms' / tag / f'n{n}/train/updates.jsonl')
    engineering = {}
    for size in protocol['engineering']['world_sizes']:
        engineering[f'sft_n{size}'] = files.read_jsonl(run / f'engineering/sft/n{size}/updates.jsonl')
    engineering['grpo_vllm_separated'] = grpo_rows(
        run / 'engineering/grpo/vllm_separated/grpo_train.jsonl')
    return dict(run_id=protocol['run_id'], training=training, engineering=engineering)


def write_reports(run, models, summaries, comparisons):
    """配额验收后生成组合比较、统一结果页、失败统计和训练曲线。"""
    run = Path(run)
    write_composition_report(run / 'eval/comp_suite', models, comparisons)
    result = dict(run_id=read_json(run / 'protocol.json')['run_id'], suites=summaries,
                  training={}, failures={})
    lines = ['# MiniToolRL 运行结果', '',
             '组合指标及按世界配对的置信区间见 [组合报告](comp_suite/final_report.json)。'
             '分布式性能测量见 [工程报告](../engineering/report.md)。', '',
             '| 模型 | 组合成功/360 | 通用成功/372 | 双订单退款成功/60 |', '|---|---:|---:|---:|']
    for tag in models:
        lines.append('| %s | %s | %s | %s |' % (
            tag, *[summaries[s + '_' + tag]['success'] for s in ('comp_suite', 'c_recipe', 'long')]))
        failures = {}
        for suite in ('comp_suite', 'c_recipe', 'long'):
            rows = read_json(run / 'eval' / suite / 'run' / tag / 'greedy'
                                   / (tag + '_analysis.json'))
            by_family = defaultdict(list)
            for r in rows:
                by_family[r['family']].append(r)
            failures[suite] = {family: dict(
                n=len(rs), success=sum(r['success'] for r in rs),
                missing=dict(Counter(x for r in rs for x in r.get('missing', []))),
                extra_writes=dict(Counter(x for r in rs for x in r.get('extra_writes', []))),
                parse_error_tasks=sum(bool(r.get('parse_errors')) for r in rs),
                tool_error_tasks=sum(bool(r.get('tool_errors')) for r in rs))
                for family, rs in by_family.items()}
        result['failures'][tag] = failures
    result['training']['SFT'] = read_json(run / 'sft/summary.json')
    result['training']['G0'] = read_json(run / 'grpo/main/grpo_cost.json')
    n = read_json(run / 'protocol.json')['config']['world_size']
    for tag in models[2:]:
        result['training'][tag] = read_json(run / 'arms' / tag / ('n%d' % n) / 'train/summary.json')
    result['engineering'] = read_json(run / 'engineering/report.json')
    curves = collect_curves(run)
    files.dump_json(run / 'eval/train_curves.json', curves)
    lines += ['', '[训练曲线数据](train_curves.json) 分别保存主训练与工程短测的逐步 loss、梯度、'
              '学习率及训练采样成功率。']
    lines += ['', '每族成功计数、漏做/多做操作、解析和工具错误、实际训练预算保存在 eval_summary.json。'
              '所有数值来自本次已完成的评测；训练与短测检查点不按测试成绩挑选。', '']
    files.dump_json(run / 'eval/eval_summary.json', result)
    (run / 'eval/report.md').write_text('\n'.join(lines), encoding='utf-8')


# 每个模型、每个任务池的 GPU 子进程入口，由 run_experiment 调度。
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
    ap.add_argument('--tool-order', choices=['registry'], default='registry',
                    help='use the same registry order as SFT, GRPO and continuation training')
    ap.add_argument('--max-seconds', type=float, default=900)
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    started = time.time()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool = files.read_jsonl(args.pool)
    budget = EpisodeBudget(**C.EPISODE_BUDGET)
    temperature = args.temperature
    visible_tools = tools_for_order(args.tool_order)
    meta = dict(
        tag=args.tag, model=args.model, mode=args.mode, backend='vllm',
        weight_sha256=files.sha256_file(Path(args.model) / 'model.safetensors'),
        pool=args.pool, pool_sha256=files.sha256_file(args.pool), policy_version=P.VERSION,
        n_tasks=len(pool), sample_n=1,
        temperature=temperature, top_p=args.top_p, top_k=args.top_k,
        episode_budget=dict(C.EPISODE_BUDGET), visible_tools=list(visible_tools),
        tool_order=args.tool_order,
        resume=True,
        note='full autonomous; exact entity/amount match; required restock not extra',
    )
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    identity = dict(weight_sha256=meta['weight_sha256'], pool_sha256=meta['pool_sha256'])
    sampler = make_sampler(args.model, gpu_memory_utilization=args.gpu_memory_utilization)
    try:
        result = run_greedy(
            sampler, tok, pool, budget, out, args.tag, temperature, args.top_p,
            args.top_k, args.batch_size, args.max_seconds, started, identity=identity,
            visible_tools=visible_tools)
    finally:
        close_sampler(sampler)

    context = result.pop('context')
    meta.update(evaluation_context=context['context'],
                evaluation_context_sha256=context['context_sha256'],
                seconds=round(time.time() - started, 1),
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
