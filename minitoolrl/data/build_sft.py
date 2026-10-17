"""执行任务中写好的工具调用，组成 SFT 轨迹；main 将每条轨迹写成一行 JSONL。

输入 TaskInstance 已包含题目、初始场景和 gold_calls。本模块复制初始场景，
按顺序直接调用工具，把调用和真实返回保存在 turns 里。
构造过程不判分、不加载 tokenizer、不比较聊天模板。任务规则由 env/tasks.py
或 decomposition_tasks.py 定义；训练文本由 sft_dataset.py 使用聊天模板生成。

python -m minitoolrl.data.build_sft --split sft --pool-dir /path/to/pools --out-dir /path/to/new_trajectories
"""
import argparse
import json
import os
import random
from collections import Counter
from typing import Any, Dict, List, Optional


from minitoolrl.env import tools as T
from minitoolrl.env.session import EpisodeBudget
from minitoolrl.env.state import WorldState
from minitoolrl.env.tasks import TaskInstance, instance_key

def load_pool(path: str) -> List[TaskInstance]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(TaskInstance.from_json(json.loads(line)))
    return out


def chunk_calls(calls: List[Dict[str, Any]], calls_per_turn: int) -> List[List[Dict[str, Any]]]:
    """把参考调用切成若干轮。写操作之间保持顺序，不重排。"""
    if calls_per_turn <= 1:
        return [[c] for c in calls]
    return [calls[i:i + calls_per_turn] for i in range(0, len(calls), calls_per_turn)]


FINAL_LINE = "已完成上述操作。"

# 纠错轨迹先调用不存在的实体 ID，记录工具报错，然后用正确参数重试。
BAD_IDS = {"order_id": "O-9999", "sku": "SKU-9999", "customer_id": "C-9999"}


def corrupt_args(args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """把参数里的实体 ID 换成不存在的 ID；没有可换的键就返回 None。"""
    for k, bad in BAD_IDS.items():
        if isinstance(args.get(k), str) and args[k] != bad:
            return {**args, k: bad}
    return None


def inject_recovery(state: WorldState, chunk: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """记录一次错误 ID 调用的实际返回；这轮只作为上下文，不作为学习目标。"""
    bad = corrupt_args(chunk[0]["arguments"])
    if bad is None:
        return None
    call = {"name": chunk[0]["name"], "arguments": bad}
    result = T.execute(state, call["name"], call["arguments"])
    return {"calls": [call], "observations": [json.dumps(result, ensure_ascii=False)],
            "recovery": True}


def build_trajectory(inst: TaskInstance, calls_per_turn: int, budget: EpisodeBudget,
                     add_final_turn: bool = True, recover_at: int = -1,
                     visible_tools: Optional[List[str]] = None) -> Dict[str, Any]:
    """顺序执行 gold_calls 并记录返回；工具报错或超出保存预算时返回失败。

    recover_at 指定在哪一轮前插入一次错误 ID 调用，-1 表示不插入。
    成功返回的 record 是结构化对话，还没有渲染成训练 token。
    """
    state = WorldState.from_snapshot(inst.world)
    chunks = chunk_calls(inst.gold_calls, calls_per_turn)
    n_inject = int(0 <= recover_at < len(chunks)
                   and corrupt_args(chunks[recover_at][0]["arguments"]) is not None)
    if len(chunks) + n_inject + int(add_final_turn) > budget.max_turns:
        return {"ok": False, "reason": "turns_over_budget", "record": None}
    if len(inst.gold_calls) + n_inject > budget.max_calls:
        return {"ok": False, "reason": "calls_over_budget", "record": None}

    turns: List[Dict[str, Any]] = []
    n_recovery = 0
    for ci, chunk in enumerate(chunks):
        if ci == recover_at:
            recovery = inject_recovery(state, chunk)
            if recovery is not None:
                turns.append(recovery)
                n_recovery += 1
        observations = []
        for call in chunk:
            result = T.execute(state, call["name"], call["arguments"])
            if "error" in result:
                return {"ok": False, "reason": "tool_error:" + call["name"], "record": None}
            observations.append(json.dumps(result, ensure_ascii=False))
        turns.append({"calls": chunk, "observations": observations})

    record = {
        "task_id": inst.task_id, "family": inst.family, "split_group": inst.split_group,
        "difficulty": inst.difficulty, "seed": inst.seed, "instance_key": instance_key(inst),
        "tool_signature": inst.tool_signature, "prompt": inst.prompt,
        "n_turns": len(turns) + int(add_final_turn),
        "n_calls": len(inst.gold_calls) + n_recovery, "n_recovery": n_recovery,
        "turns": turns, "final_line": FINAL_LINE if add_final_turn else None,
    }
    if visible_tools is not None:
        record['exposed_tools'] = list(visible_tools)
    return {"ok": True, "reason": "", "record": record}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["sft", "train", "val"],
                    help="只允许 sft/train/val：测试集永远不做训练轨迹数据")
    ap.add_argument("--pool-dir", required=True, help="明确指定任务池目录")
    ap.add_argument("--out-dir", required=True, help="新的轨迹输出目录，不覆盖已有数据")
    ap.add_argument("--calls-per-turn", type=int, default=1)
    ap.add_argument("--max-per-family", type=int, default=0, help="0 = 不限")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--max-calls", type=int, default=16)
    ap.add_argument("--no-final-turn", action="store_true")
    ap.add_argument("--recovery-ratio", type=float, default=0.0,
                    help="多大比例的训练轨迹里插入一次真实的工具报错+改正，0 = 不插")
    ap.add_argument("--recovery-seed", type=int, default=7)
    args = ap.parse_args()
    if os.path.exists(args.out_dir):
        ap.error('--out-dir must be a new directory')

    pool = load_pool(os.path.join(args.pool_dir, f"{args.split}.jsonl"))
    rng = random.Random(args.recovery_seed)
    budget = EpisodeBudget(max_turns=args.max_turns, max_calls=args.max_calls)
    per_family: Counter = Counter()
    kept: List[Dict[str, Any]] = []
    skipped_by_quota = 0
    for inst in pool:
        if args.max_per_family and per_family[inst.family] >= args.max_per_family:
            skipped_by_quota += 1
            continue
        n_chunks = len(chunk_calls(inst.gold_calls, args.calls_per_turn))
        # 注入位置从第 2 个 chunk 起：第一步通常是只读确认，让报错落在写操作上更有意义
        recover_at = rng.randrange(1, n_chunks) if (
            args.recovery_ratio > 0 and n_chunks >= 2
            and rng.random() < args.recovery_ratio) else -1
        r = build_trajectory(inst, args.calls_per_turn, budget,
                             add_final_turn=not args.no_final_turn, recover_at=recover_at)
        if not r["ok"]:
            raise RuntimeError(f"轨迹构造失败 {inst.task_id}: {r['reason']}")
        kept.append(r["record"])
        per_family[inst.family] += 1

    os.makedirs(args.out_dir, exist_ok=False)
    out_path = os.path.join(args.out_dir, f"sft_{args.split}.jsonl")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = {
        "split": args.split, "n_pool": len(pool), "n_kept": len(kept),
        "calls_per_turn": args.calls_per_turn, "add_final_turn": not args.no_final_turn,
        "budget": {"max_turns": args.max_turns, "max_calls": args.max_calls},
        "per_family": dict(sorted(per_family.items())),
        "difficulty": dict(sorted(Counter(r["difficulty"] for r in kept).items())),
        "n_turns_hist": dict(sorted(Counter(r["n_turns"] for r in kept).items())),
        "n_calls_hist": dict(sorted(Counter(r["n_calls"] for r in kept).items())),
        "skipped_by_quota": skipped_by_quota,
        "recovery": {"ratio": args.recovery_ratio, "seed": args.recovery_seed,
                     "n_with_recovery": sum(1 for r in kept if r.get("n_recovery")),
                     "note": "错误 ID 调用的工具返回保存在上下文，正确重试参与学习。"},
        "note": "按任务中的调用列表执行工具，tool 消息保存实际返回。",
    }
    with open(os.path.join(args.out_dir, f"manifest_{args.split}.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[{args.split}] {len(kept)} 条训练轨迹；按数量上限跳过 {skipped_by_quota} 条")
    print(f"  按族: {dict(sorted(per_family.items()))}")
    print(f"  难度: {manifest['difficulty']}  轮数: {manifest['n_turns_hist']}")
    print("  已写入", out_path)


if __name__ == "__main__":
    main()
