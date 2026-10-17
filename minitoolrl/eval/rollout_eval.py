"""Autonomous rollout diagnostics with shared task and policy semantics.

One greedy full rollout per task with the same chat template, policy documents,
tools, episode budget and parser for every arm. No correct prefix is provided:
this measures task-level autonomy, not teacher-forced accuracy.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from minitoolrl.env.policy import VERSION
from minitoolrl.env.tasks import TaskInstance
from minitoolrl.eval import trajectory_metrics as M
from minitoolrl.train import experiment_config as C
from minitoolrl.eval.runtime import (bind_evaluation_context, close_sampler, free_rollouts,
                                   make_sampler, sha, tools_for_order)


def evaluate_pool(sampler, tokenizer, instances, jsonl_path, *, identity=None, visible_tools=None):
    """Run greedy rollouts, keep every transcript (including legal failures), return metrics."""
    path = Path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = free_rollouts(sampler, tokenizer, instances, path, identity=identity,
                         visible_tools=visible_tools)
    metrics = M.summarize(rows, {inst.task_id: inst for inst in instances})
    C.dump_json(path.with_suffix('.metrics.json'), metrics)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--pool', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--note', default='')
    ap.add_argument('--tool-order', choices=['registry', 'alphabetical'], default='registry',
                    help='match frozen training: SFT/G0 registry; existing continuation arms alphabetical')
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    args = ap.parse_args()
    visible_tools = tools_for_order(args.tool_order)
    started = time.monotonic()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool = [TaskInstance.from_json(row) for row in C.read_jsonl(args.pool)]
    if args.limit:
        pool = pool[:args.limit]
    checkpoint = Path(args.model)
    template = checkpoint / 'chat_template.jinja'
    record = dict(tag=args.tag, model=str(checkpoint), weight_sha256=sha(checkpoint / 'model.safetensors'),
                  chat_template_sha256=sha(template) if template.exists() else None,
                  pool=str(args.pool), pool_sha256=sha(args.pool), policy_version=VERSION,
                  n_tasks=len(pool), temperature=0.0, episode_budget=dict(C.EPISODE_BUDGET),
                  backend='vllm', note=args.note, tool_order=args.tool_order)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rollouts_path = out / (args.tag + '_rollouts.jsonl')
    identity = dict(weight_sha256=record['weight_sha256'], pool_sha256=record['pool_sha256'])
    context = bind_evaluation_context(rollouts_path, tokenizer, pool, identity=identity,
                                      visible_tools=visible_tools)
    record.update(evaluation_context=context['context'],
                  evaluation_context_sha256=context['context_sha256'],
                  visible_tools=context['context']['visible_tools'])
    sampler = make_sampler(args.model,
                           gpu_memory_utilization=args.gpu_memory_utilization)
    try:
        metrics = evaluate_pool(sampler, tokenizer, pool, rollouts_path, identity=identity,
                                visible_tools=visible_tools)
    finally:
        close_sampler(sampler)
    C.dump_json(out / (args.tag + '_metrics.json'), metrics)
    record.update(seconds=round(time.monotonic() - started, 1),
                  peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                  correct=metrics['overall']['correct'], rate=metrics['overall']['rate'])
    C.dump_json(out / (args.tag + '_run.json'), record)
    print('EVAL_COMPLETE ' + json.dumps(dict(tag=args.tag, n=len(pool), correct=record['correct'],
                                            rate=record['rate'], seconds=record['seconds']),
                                        ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
