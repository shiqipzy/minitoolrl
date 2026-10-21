#!/usr/bin/env python3
"""SFT 终点的开发集任务成功检查：每条任务一次自主 greedy 执行。

与正式评测使用同一聊天模板、政策文档、工具、回合上限和解析器；不提供正确前缀，
衡量的是任务级自主完成，而不是 teacher-forcing 的 token 准确率。执行调度见
eval/evaluate.py，指标见 trajectory_metrics.py。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from minitoolrl import files
from minitoolrl.env.policy import VERSION
from minitoolrl.env.tasks import TaskInstance
from minitoolrl.eval import trajectory_metrics as M
from minitoolrl.eval.evaluate import close_sampler, make_sampler, run_greedy_pool, tools_for_order
from minitoolrl.train import experiment_config as C


def evaluate_pool(sampler, tokenizer, instances, jsonl_path, *, identity=None,
                  visible_tools=None):
    """执行开发集轨迹并返回任务指标；合法失败的轨迹同样保留。"""
    path = Path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    run = run_greedy_pool(sampler, tokenizer, instances, path, identity=identity,
                          visible_tools=visible_tools)
    metrics = M.summarize(run.rows, {inst.task_id: inst for inst in instances})
    files.dump_json(path.with_suffix('.metrics.json'), metrics)
    return run, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--pool', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--note', default='')
    ap.add_argument('--tool-order', choices=['registry'], default='registry',
                    help='use the same registry order as SFT, GRPO and continuation training')
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    args = ap.parse_args()
    visible_tools = tools_for_order(args.tool_order)
    started = time.monotonic()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool = [TaskInstance.from_json(row) for row in files.read_jsonl(args.pool)]
    if args.limit:
        pool = pool[:args.limit]
    checkpoint = Path(args.model)
    template = checkpoint / 'chat_template.jinja'
    record = dict(tag=args.tag, model=str(checkpoint),
                  weight_sha256=files.sha256_file(checkpoint / 'model.safetensors'),
                  chat_template_sha256=(files.sha256_file(template) if template.exists() else None),
                  pool=str(args.pool), pool_sha256=files.sha256_file(args.pool),
                  policy_version=VERSION, n_tasks=len(pool), temperature=0.0,
                  episode_budget=dict(C.EPISODE_BUDGET), backend='vllm', note=args.note,
                  tool_order=args.tool_order)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rollouts_path = out / (args.tag + '_rollouts.jsonl')
    identity = dict(weight_sha256=record['weight_sha256'], pool_sha256=record['pool_sha256'])
    sampler = make_sampler(args.model, gpu_memory_utilization=args.gpu_memory_utilization)
    try:
        run, metrics = evaluate_pool(sampler, tokenizer, pool, rollouts_path, identity=identity,
                                     visible_tools=visible_tools)
    finally:
        close_sampler(sampler)
    record.update(evaluation_context=run.context['context'],
                  evaluation_context_sha256=run.context['context_sha256'],
                  visible_tools=run.context['context']['visible_tools'])
    files.dump_json(out / (args.tag + '_metrics.json'), metrics)
    record.update(seconds=round(time.monotonic() - started, 1),
                  peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                  correct=metrics['overall']['correct'], rate=metrics['overall']['rate'])
    files.dump_json(out / (args.tag + '_run.json'), record)
    print('EVAL_COMPLETE ' + json.dumps(dict(tag=args.tag, n=len(pool), correct=record['correct'],
                                            rate=record['rate'], seconds=record['seconds']),
                                        ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
