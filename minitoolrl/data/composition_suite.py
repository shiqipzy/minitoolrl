"""Current three-family evaluation definitions, prompt pairs and CPU checks.

This module defines composition tasks; it does not launch or freeze
a separate experiment.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from minitoolrl.data.decomposition_tasks import held_out_reason
from minitoolrl.env import policy as P
from minitoolrl.env.tasks import TaskInstance, generate_instance


TARGET_FAMILIES: Tuple[str, ...] = (
    "refund_with_credit",
    "partial_refund_restock_ticket",
    "cancel_refund_release",
)
MODEL_TAGS: Tuple[str, ...] = ("SFT", "G0", "C_blocked", "C_mixed", "U_mixed")
VARIANTS: Tuple[str, ...] = ("A", "B")
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260922
SEED_START = {
    "refund_with_credit": 70_000_000,
    "partial_refund_restock_ticket": 71_000_000,
    "cancel_refund_release": 72_000_000,
}
SEED_SCAN_LIMIT = 200_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_world_hash(world: Mapping[str, Any]) -> str:
    cleaned = {k: v for k, v in world.items() if k not in ("_seed", "task_id", "prompt")}
    payload = json.dumps(cleaned, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_prompt_pair(inst: TaskInstance) -> Dict[str, str]:
    """Return two equivalent prompts while keeping world, calls, and assertions fixed."""
    family = inst.family
    params = inst.params
    order_id = params["order_id"]

    if family == "refund_with_credit":
        customer_id = params["customer_id"]
        customer = inst.world["customers"][customer_id]
        return {
            "A": (
                f"客户{customer['name']}（{customer_id}）的订单 {order_id} 需要全额退款，"
                "另外按补偿政策再给一笔店铺信用，最后把订单状态置为 refunded。"
                "商品不退回仓库。"
            ),
            "B": (
                f"订单 {order_id} 的商品有严重问题，客户{customer['name']}（{customer_id}）"
                "不退货但要求全额退款，公司同意再按补偿政策给他等级对应的店铺信用。"
                "请办理全额退款、发放信用，并把订单状态改成 refunded。"
            ),
        }

    if family == "partial_refund_restock_ticket":
        sku = params["sku"]
        quantity = params["quantity"]
        return {
            "A": (
                f"客户把订单 {order_id} 里 {quantity} 件 {sku} 退回了仓库。"
                "请按这些商品的金额退款，把退回的数量补回库存，开一条 refund 类工单，"
                "并把订单状态改成 partially_refunded。"
            ),
            "B": (
                f"订单 {order_id} 有 {quantity} 件 {sku} 已退回入库。"
                "请办理对应金额退款、补回库存、开 refund 工单，并把订单状态置为 "
                "partially_refunded。"
            ),
        }

    if family == "cancel_refund_release":
        sku = params["sku"]
        quantity = params["quantity"]
        order = inst.world["orders"][order_id]
        customer_id = order["customer_id"]
        customer = inst.world["customers"][customer_id]
        return {
            "A": (
                f"客户{customer['name']}（{customer_id}）取消了还没发货的订单 {order_id} "
                f"并要求退款。请取消运单、全额退款、释放该订单占用的 {sku} 预留"
                f"（{quantity} 件），最后把订单状态改成 cancelled。"
            ),
            "B": (
                f"订单 {order_id} 未发货，客户要求取消并退全款。请依次处理运单取消、"
                f"退款、释放 {sku} 的 {quantity} 件预留，并把订单状态置为 cancelled。"
            ),
        }

    raise ValueError(f"Unsupported composition family: {family}")


PROMPT_MARKERS: Dict[str, Tuple[str, ...]] = {
    "refund_with_credit": ("全额退款", "信用", "refunded"),
    "partial_refund_restock_ticket": ("退款", "库存", "工单", "partially_refunded"),
    "cancel_refund_release": ("取消", "退款", "预留", "cancelled"),
}


def make_variants(base: TaskInstance) -> List[TaskInstance]:
    prompts = render_prompt_pair(base)
    variants: List[TaskInstance] = []
    for variant in VARIANTS:
        payload = base.to_json()
        payload["task_id"] = f"{base.task_id}#{variant}"
        payload["prompt"] = prompts[variant]
        payload["params"] = copy.deepcopy(payload["params"])
        payload["params"]["variant"] = variant
        variants.append(TaskInstance.from_json(payload))
    return variants


def credit_cell(inst: TaskInstance) -> str:
    tier = inst.params["tier"]
    amount = inst.params["credit_cents"]
    bound = "capped" if amount == P.TIER_CREDIT_CAP_CENTS[tier] else "uncapped"
    return f"{tier}|{bound}"


def collect_family(
    family: str,
    n: int,
    seen_seeds: set[int],
    seen_worlds: set[str],
) -> Tuple[List[TaskInstance], Dict[str, Any]]:
    if family not in TARGET_FAMILIES:
        raise ValueError(f"Family is not in the frozen target set: {family}")
    if family == "refund_with_credit" and n % 6:
        raise ValueError("refund_with_credit count must be divisible by six strata")

    need = None
    if family == "refund_with_credit":
        need = {
            f"{tier}|{bound}": n // 6
            for tier in P.TIERS
            for bound in ("capped", "uncapped")
        }

    selected: List[TaskInstance] = []
    rejected: Counter[str] = Counter()
    start = SEED_START[family]
    attempted = 0
    for seed in range(start, start + SEED_SCAN_LIMIT):
        if len(selected) >= n:
            break
        attempted += 1
        if seed in seen_seeds:
            rejected["used_seed"] += 1
            continue
        inst = generate_instance(family, seed)
        if inst is None:
            rejected["ineligible_world"] += 1
            continue
        world_id = canonical_world_hash(inst.world)
        if world_id in seen_worlds:
            rejected["used_world"] += 1
            continue
        if need is not None:
            cell = credit_cell(inst)
            if need[cell] <= 0:
                rejected["stratum_full"] += 1
                continue
        reason = held_out_reason(inst)
        if not reason:
            raise RuntimeError(f"Held-out guard did not recognize {inst.task_id}")
        selected.append(inst)
        seen_seeds.add(seed)
        seen_worlds.add(world_id)
        if need is not None:
            need[credit_cell(inst)] -= 1

    if len(selected) != n:
        raise RuntimeError(
            f"{family}: collected {len(selected)}/{n}; rejected={dict(rejected)}; need={need}"
        )
    if need is not None and any(need.values()):
        raise RuntimeError(f"{family}: incomplete strata {need}")
    return selected, {
        "seed_start": start,
        "attempted": attempted,
        "rejected": dict(rejected),
        "strata": None if need is None else {cell: n // 6 for cell in need},
    }


def build_run_plan(checkpoints: Mapping[str, Mapping[str, str]]) -> Dict[str, Any]:
    jobs = []
    for tag in MODEL_TAGS:
        env_name = f"{tag.upper()}_MODEL"
        jobs.append(
            {
                "tag": tag,
                "model_env": env_name,
                "expected_weight_sha256": checkpoints[tag]["sha256"],
                "recorded_model_path": checkpoints[tag]["recorded_path"],
                "argv": [
                    "python",
                    "-m",
                    "minitoolrl.eval.evaluate",
                    "--model",
                    f"${{{env_name}}}",
                    "--pool",
                    "${EXP_DIR}/test_comp_suite.jsonl",
                    "--out-dir",
                    f"${{EXP_DIR}}/run/{tag}/greedy",
                    "--tag",
                    tag,
                    "--mode",
                    "greedy",
                    "--batch-size",
                    "8",
                    "--max-seconds",
                    "1800",
                ],
            }
        )
    return {
        "required_environment": ["EXP_DIR", *(f"{tag.upper()}_MODEL" for tag in MODEL_TAGS)],
        "jobs": jobs,
        "after_jobs": [
            "python",
            "-m",
            "minitoolrl.eval.analyze_comp_suite",
            "--root",
            "${EXP_DIR}",
        ],
        "execution_order": list(MODEL_TAGS),
        "parallel_safe": False,
        "note": "Run each model once to keep GPU memory and logs isolated; resume is supported.",
    }
