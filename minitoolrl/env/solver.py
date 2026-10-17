"""参考求解器 + 反例（perturbation）检查。

作用有两个：
1. 提供 gold 轨迹的真实执行结果，用于生成 SFT 示范和"任务可解"的上界；
2. 生成一批必须判为失败的变体。如果某个变体也被判成功，说明裁判太松，
   任务实例直接作废。这是方案里"反例检查 → 建立可信的成功率定义"的落地。
"""
from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional

from minitoolrl.env import tools as T
from minitoolrl.env.grader import grade, mutation_summary
from minitoolrl.env.state import WorldState
from minitoolrl.env.tasks import TaskInstance

WRITE_TOOLS = set(T.WRITE_TOOLS)


def run_calls(world_snapshot: Dict[str, Any], calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """在快照的独立副本上顺序执行调用，返回最终状态与每步结果。"""
    st = WorldState.from_snapshot(world_snapshot)
    results, errors = [], []
    for c in calls:
        out = T.execute(st, c["name"], c.get("arguments", {}))
        results.append({"name": c["name"], "arguments": c.get("arguments", {}), "result": out})
        if isinstance(out, dict) and "error" in out:
            errors.append({"name": c["name"], "error": out["error"]})
    return {"final": st.snapshot(), "results": results, "errors": errors}


def solve(inst: TaskInstance) -> Dict[str, Any]:
    """执行 gold 计划并判分。正常情况下必须 success=True 且无工具报错。"""
    run = run_calls(inst.world, inst.gold_calls)
    g = grade(inst.world, run["final"], inst.assertions)
    return {"task_id": inst.task_id, "success": g.success, "grade": g.to_json(),
            "errors": run["errors"], "results": run["results"],
            "mutations": mutation_summary(run["final"])}


# ================================ 反例构造 ================================
def _write_idx(calls: List[Dict[str, Any]]) -> List[int]:
    return [i for i, c in enumerate(calls) if c["name"] in WRITE_TOOLS]


def _other_order(inst: TaskInstance, order_id: str) -> Optional[str]:
    ids = sorted(i for i in inst.world["orders"] if i != order_id)
    return ids[0] if ids else None


def _other_sku(inst: TaskInstance, sku: str) -> Optional[str]:
    ids = sorted(s for s in inst.world["inventory"]
                 if s != sku and inst.world["inventory"][s]["on_hand"] >= 0)
    return ids[0] if ids else None


def p_drop_write(inst, rng):
    """漏掉一个写操作。"""
    idx = _write_idx(inst.gold_calls)
    if not idx:
        return None
    drop = rng.choice(idx)
    return [c for i, c in enumerate(inst.gold_calls) if i != drop]


def p_wrong_amount(inst, rng):
    """金额算错（差 1 分或差一位）。"""
    calls = copy.deepcopy(inst.gold_calls)
    hit = [c for c in calls if c["name"] in ("create_refund", "apply_credit")]
    if not hit:
        return None
    c = rng.choice(hit)
    old = c["arguments"]["amount_cents"]
    c["arguments"]["amount_cents"] = max(1, old - 1) if old > 1 else old + 1
    return calls


def p_wrong_quantity(inst, rng):
    """数量算错。"""
    calls = copy.deepcopy(inst.gold_calls)
    hit = [c for c in calls if c["name"] in ("restock_item", "release_reservation")]
    if not hit:
        return None
    c = rng.choice(hit)
    c["arguments"]["quantity"] = c["arguments"]["quantity"] + 1
    return calls


def p_wrong_target(inst, rng):
    """操作到别的订单/商品上（检查反例断言是否真的拦得住）。"""
    calls = copy.deepcopy(inst.gold_calls)
    for c in calls:
        if c["name"] in WRITE_TOOLS and "order_id" in c["arguments"]:
            alt = _other_order(inst, c["arguments"]["order_id"])
            if alt:
                c["arguments"]["order_id"] = alt
                return calls
        if c["name"] in WRITE_TOOLS and "sku" in c["arguments"]:
            alt = _other_sku(inst, c["arguments"]["sku"])
            if alt:
                c["arguments"]["sku"] = alt
                return calls
    return None


def p_extra_refund(inst, rng):
    """多做一次没让做的退款。"""
    calls = copy.deepcopy(inst.gold_calls)
    oid = None
    for c in calls:
        if "order_id" in c.get("arguments", {}):
            oid = c["arguments"]["order_id"]
            break
    if oid is None:
        return None
    calls.append({"name": "create_refund",
                  "arguments": {"order_id": oid, "amount_cents": 1, "reason": "多余退款"}})
    return calls


def p_extra_restock(inst, rng):
    """多做一次没让做的库存回补（用于抓"只退钱不退货也回补库存"这类套路误用）。"""
    calls = copy.deepcopy(inst.gold_calls)
    skus = [s for s in sorted(inst.world["inventory"]) if inst.world["inventory"][s]["restockable"]]
    if not skus:
        return None
    calls.append({"name": "restock_item", "arguments": {"sku": skus[0], "quantity": 1}})
    return calls


def p_wrong_status(inst, rng):
    """状态写错。"""
    calls = copy.deepcopy(inst.gold_calls)
    hit = [c for c in calls if c["name"] == "update_order_status"]
    if not hit:
        return None
    c = rng.choice(hit)
    alt = "cancelled" if c["arguments"]["status"] != "cancelled" else "refunded"
    c["arguments"]["status"] = alt
    return calls


def p_reorder_writes(inst, rng):
    """把写操作顺序倒过来（依赖前置条件的任务会因此失败）。"""
    idx = _write_idx(inst.gold_calls)
    if len(idx) < 2:
        return None
    calls = copy.deepcopy(inst.gold_calls)
    writes = [calls[i] for i in idx][::-1]
    for i, c in zip(idx, writes):
        calls[i] = c
    return calls


def p_reads_only(inst, rng):
    """只读不写：必须失败（否则说明任务没有真实状态要求）。"""
    return [c for c in inst.gold_calls if c["name"] not in WRITE_TOOLS]


PERTURBATIONS = [
    ("drop_write", p_drop_write),
    ("wrong_amount", p_wrong_amount),
    ("wrong_quantity", p_wrong_quantity),
    ("wrong_target", p_wrong_target),
    ("extra_refund", p_extra_refund),
    ("extra_restock", p_extra_restock),
    ("wrong_status", p_wrong_status),
    ("reorder_writes", p_reorder_writes),
    ("reads_only", p_reads_only),
]


# ================================ 实例自检 ================================
def _fp(snapshot: Dict[str, Any]) -> str:
    """与 WorldState.fingerprint 一致：忽略审计日志，只看实际状态。"""
    import json as _json
    return _json.dumps({k: v for k, v in snapshot.items() if k != "mutations"},
                       sort_keys=True, ensure_ascii=False)


def validate_instance(inst: TaskInstance, seed: int = 0) -> Dict[str, Any]:
    """gold 必须成功；所有"真的改变了最终状态"的反例必须失败。

    注意：如果某个变体的所有写操作都被工具前置条件挡下来（例如在已全额退款的订单上
    再退 1 分会触发 REFUND_EXCEEDS_REMAINING），最终状态与 gold 完全相同。
    这种变体是无效反例（no_effect），不能算裁判过松——判同一个状态得到同一个结论是正确的。
    """
    rng = random.Random(f"perturb|{inst.task_id}|{seed}")
    gold = solve(inst)
    gold_fp = _fp(run_calls(inst.world, inst.gold_calls)["final"])
    problems: List[str] = []
    if not gold["success"]:
        problems.append(f"gold 未通过: {gold['grade']['failures']}")
    if gold["errors"]:
        problems.append(f"gold 执行报错: {gold['errors']}")

    checked, skipped, no_effect = [], [], []
    for name, fn in PERTURBATIONS:
        calls = fn(inst, rng)
        if calls is None or calls == inst.gold_calls:
            skipped.append(name)
            continue
        run = run_calls(inst.world, calls)
        if _fp(run["final"]) == gold_fp:
            no_effect.append(name)
            continue
        g = grade(inst.world, run["final"], inst.assertions)
        checked.append({"name": name, "success": g.success, "tags": g.failure_tags})
        if g.success:
            problems.append(f"反例 {name} 竟然通过了判分（裁判过松）")
    if not checked:
        problems.append("没有任何有效反例，无法证明判分严格")
    return {"task_id": inst.task_id, "family": inst.family, "ok": not problems,
            "problems": problems, "gold": {"success": gold["success"],
                                           "n_assertions": gold["grade"]["n_total"],
                                           "mutations": gold["mutations"]},
            "perturbations": checked, "skipped": skipped, "no_effect": no_effect}


def validate_pool(pool: List[TaskInstance], verbose: bool = False) -> Dict[str, Any]:
    n_ok, bad = 0, []
    tag_hist: Dict[str, int] = {}
    ne_hist: Dict[str, int] = {}
    n_perturb = 0
    for inst in pool:
        rep = validate_instance(inst)
        if rep["ok"]:
            n_ok += 1
        else:
            bad.append(rep)
        n_perturb += len(rep["perturbations"])
        for p in rep["perturbations"]:
            for t in p["tags"]:
                tag_hist[t] = tag_hist.get(t, 0) + 1
        for name in rep["no_effect"]:
            ne_hist[name] = ne_hist.get(name, 0) + 1
        if verbose and not rep["ok"]:
            print(rep["task_id"], rep["problems"])
    return {"n": len(pool), "n_ok": n_ok, "n_bad": len(bad), "n_perturbations": n_perturb,
            "bad": bad[:10], "perturbation_tag_hist": dict(sorted(tag_hist.items())),
            "no_effect_hist": dict(sorted(ne_hist.items()))}


if __name__ == "__main__":
    from minitoolrl.env.tasks import (COMP_FAMILIES, LONG_FAMILIES, TRAIN_FAMILIES,
                                      generate_pool, difficulty_histogram)

    for label, fams, lo, hi, n in [("train", TRAIN_FAMILIES, 0, 600, 30),
                                   ("comp", COMP_FAMILIES, 900000, 900600, 30),
                                   ("long", LONG_FAMILIES, 950000, 950600, 30)]:
        pool = generate_pool(fams, n, lo, hi)
        rep = validate_pool(pool, verbose=True)
        print(f"[{label}] 实例={rep['n']} 难度={difficulty_histogram(pool)} "
              f"gold+反例全部正确={rep['n_ok']}/{rep['n']} 有效反例={rep['n_perturbations']}")
        print(f"        反例失败原因分布={rep['perturbation_tag_hist']}")
        print(f"        无效反例（被工具前置条件挡下，最终状态与 gold 相同）={rep['no_effect_hist']}")
        if rep["n_bad"]:
            for b in rep["bad"][:3]:
                print("        BAD", b["task_id"], b["problems"])
