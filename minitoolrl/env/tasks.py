"""任务族定义、程序化实例生成、难度分桶。

每个任务实例自带：
- world：初始状态快照（rollout 独立初始化，可回放）
- prompt：用户指令（含全部必须遵守的具体要求；比例类政策放在 get_policy 里）
- gold_calls：参考求解器的调用序列（只用于 SFT 与预算基线，评测不使用）
- assertions：最终状态断言（对模型隐藏，由 grader 评估）

组合泛化的划分方式：按"任务族的工具集合签名"隔离。
comp 族用到的每个工具都在 train 族里出现过，但这个组合从未在训练中出现。
"""
from __future__ import annotations

import hashlib
import json
import random
import dataclasses
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional

from minitoolrl.env import policy as P
from minitoolrl.env.state import make_world


@dataclass
class TaskInstance:
    task_id: str
    family: str
    split_group: str          # train / comp / long
    difficulty: str           # d1..d4
    seed: int
    prompt: str
    world: Dict[str, Any]
    params: Dict[str, Any]
    tool_signature: List[str]
    gold_calls: List[Dict[str, Any]]
    assertions: List[Dict[str, Any]]

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "TaskInstance":
        # 输入包的 pool 记录带 instance_key / input_version / policy_version 等运行元数据，
        # 而 to_json 只写 dataclass 字段。按字段过滤与 to_json 对称，避免加载时直接
        # 报 "unexpected keyword argument"；训练语义与数据内容均不变。
        names = {f.name for f in dataclasses.fields(TaskInstance)}
        return TaskInstance(**{k: v for k, v in d.items() if k in names})


def difficulty_of(n_calls: int, arithmetic: bool, policy_lookup: bool) -> str:
    score = n_calls + int(arithmetic) + int(policy_lookup)
    if score <= 3:
        return "d1"
    if score == 4:
        return "d2"
    if score <= 5:
        return "d3"
    return "d4"


# ================================ 生成辅助 ================================
def _orders(w: Dict[str, Any]) -> List[Dict[str, Any]]:
    return sorted(w["orders"].values(), key=lambda o: o["order_id"])


def _pick_order(w: Dict[str, Any], rng: random.Random,
                pred: Callable[[Dict[str, Any]], bool]) -> Optional[Dict[str, Any]]:
    cands = [o for o in _orders(w) if pred(o)]
    return rng.choice(cands) if cands else None


def _call(name: str, **kwargs) -> Dict[str, Any]:
    return {"name": name, "arguments": kwargs}


def _sig(calls: List[Dict[str, Any]]) -> List[str]:
    return sorted({c["name"] for c in calls})


def _restockable_items(order: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [it for it in order["items"] if it["restockable"]]


def restock_order_plan(world: Dict[str, Any], order: Dict[str, Any]):
    """按订单清单生成全部回库调用及库存要求；不可回库商品的库存保持原值。"""
    items = _restockable_items(order)
    calls = [_call("restock_item", sku=item["sku"], quantity=item["quantity"])
             for item in items]
    inventory = [{"kind": "inventory_on_hand", "sku": item["sku"],
                  "value": world["inventory"][item["sku"]]["on_hand"]
                  + (item["quantity"] if item["restockable"] else 0)}
                 for item in order["items"]]
    return calls, inventory, [item["sku"] for item in items]


def _base_negatives(order_ids: List[str], skus: List[str]) -> List[Dict[str, Any]]:
    return [
        {"kind": "unchanged_orders", "except": sorted(set(order_ids))},
        {"kind": "unchanged_inventory", "except": sorted(set(skus))},
        {"kind": "unchanged_shipments", "except": sorted(set(order_ids))},
    ]


# ================================ 训练侧任务族 ================================
def build_refund_restock(rng: random.Random, w: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """T1 全额退款 + 回补库存 + 改状态。"""
    o = _pick_order(w, rng, lambda o: o["status"] == "delivered" and _restockable_items(o))
    if o is None:
        return None
    restock_calls, inventory_assertions, restock_skus = restock_order_plan(w, o)
    cust = w["customers"][o["customer_id"]]
    calls = [_call("get_order", order_id=o["order_id"]),
             _call("create_refund", order_id=o["order_id"], amount_cents=o["total_cents"],
                   reason="客户退货全额退款")]
    calls.extend(restock_calls)
    calls.append(_call("update_order_status", order_id=o["order_id"], status="refunded"))

    # 告诉判分器最终应有什么：一张金额正确的退款单、正确的订单状态和已退金额。
    # 下方 expected=[] 表示不应出现补偿、工单或通知；这些断言本身不执行操作。
    asserts = [
        {"kind": "refunds_exact", "expected": [{"order_id": o["order_id"], "amount_cents": o["total_cents"]}]},
        {"kind": "order_status", "order_id": o["order_id"], "status": "refunded"},
        {"kind": "order_refunded_cents", "order_id": o["order_id"], "value": o["total_cents"]},
        {"kind": "credits_exact", "expected": []},
        {"kind": "tickets_exact", "expected": []},
        {"kind": "notifications_exact", "expected": []},
    ]
    asserts += inventory_assertions
    asserts += _base_negatives([o["order_id"]], restock_skus)

    lines = [
        f"客户{cust['name']}（{o['customer_id']}）已经把订单 {o['order_id']} 的商品退回仓库。"
        f"请办理全额退款，把可回补的商品补回库存，并把订单状态改成 refunded。",
        f"订单 {o['order_id']} 的退货已入库，请给客户{cust['name']}全额退款，"
        f"把退回的可回补商品加回库存，然后把订单状态置为 refunded。",
    ]
    return {"prompt": rng.choice(lines), "calls": calls, "assertions": asserts,
            "params": {"order_id": o["order_id"], "customer_id": o["customer_id"],
                       "restock_skus": restock_skus},
            "arithmetic": False, "policy_lookup": False}


def build_cancel_unshipped(rng: random.Random, w: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """T2 取消未发货订单：取消运单 + 改状态。"""
    o = _pick_order(w, rng, lambda o: o["status"] == "paid")
    if o is None:
        return None
    w["shipments"][o["order_id"]]["status"] = "pending"
    cust = w["customers"][o["customer_id"]]
    calls = [_call("get_order", order_id=o["order_id"]),
             _call("cancel_shipment", order_id=o["order_id"]),
             _call("update_order_status", order_id=o["order_id"], status="cancelled")]
    asserts = [
        {"kind": "shipment_status", "order_id": o["order_id"], "status": "cancelled"},
        {"kind": "order_status", "order_id": o["order_id"], "status": "cancelled"},
        {"kind": "refunds_exact", "expected": []},
        {"kind": "credits_exact", "expected": []},
        {"kind": "tickets_exact", "expected": []},
        {"kind": "notifications_exact", "expected": []},
    ] + _base_negatives([o["order_id"]], [])
    lines = [
        f"客户{cust['name']}（{o['customer_id']}）要取消还没发货的订单 {o['order_id']}。"
        f"请取消运单，并把订单状态改成 cancelled。",
        f"订单 {o['order_id']} 尚未发货，客户临时不要了。请取消它的运单，"
        f"再把订单状态置为 cancelled。",
    ]
    return {"prompt": rng.choice(lines), "calls": calls, "assertions": asserts,
            "params": {"order_id": o["order_id"], "customer_id": o["customer_id"]},
            "arithmetic": False, "policy_lookup": False}


def build_partial_refund_notify(rng: random.Random, w: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """T3 部分退款（金额需自己算）+ 改状态 + 通知，不退货所以不回补库存。"""
    o = _pick_order(w, rng, lambda o: o["status"] in ("shipped", "delivered") and o["items"])
    if o is None:
        return None
    it = rng.choice(o["items"])
    qty = rng.randint(1, it["quantity"])
    amount = qty * it["unit_price_cents"]
    if amount >= o["total_cents"]:
        qty = 1
        amount = it["unit_price_cents"]
    cust = w["customers"][o["customer_id"]]
    calls = [_call("get_order", order_id=o["order_id"]),
             _call("create_refund", order_id=o["order_id"], amount_cents=amount, reason="部分商品质量问题"),
             _call("update_order_status", order_id=o["order_id"], status="partially_refunded"),
             _call("notify_customer", customer_id=o["customer_id"], template="refund_processed",
                   order_id=o["order_id"])]
    asserts = [
        {"kind": "refunds_exact", "expected": [{"order_id": o["order_id"], "amount_cents": amount}]},
        {"kind": "order_status", "order_id": o["order_id"], "status": "partially_refunded"},
        {"kind": "order_refunded_cents", "order_id": o["order_id"], "value": amount},
        {"kind": "notifications_exact", "expected": [
            {"customer_id": o["customer_id"], "template": "refund_processed", "order_id": o["order_id"]}]},
        {"kind": "credits_exact", "expected": []},
        {"kind": "tickets_exact", "expected": []},
        # 关键反例：只退钱不退货，库存不能动（用来检查模型是否把 T1 的套路乱套过来）
        {"kind": "inventory_on_hand", "sku": it["sku"], "value": w["inventory"][it["sku"]]["on_hand"]},
    ] + _base_negatives([o["order_id"]], [])
    lines = [
        f"客户{cust['name']}（{o['customer_id']}）反馈订单 {o['order_id']} 里 {it['sku']} 有 {qty} 件质量问题，"
        f"商品不退回。请按这些商品的金额办理部分退款，把订单状态改成 partially_refunded，"
        f"并用 refund_processed 模板通知客户。",
        f"订单 {o['order_id']} 中 {qty} 件 {it['sku']} 存在质量问题，客户不退货只要退款。"
        f"请退掉这部分金额，订单状态置为 partially_refunded，然后用 refund_processed 模板通知本人。",
    ]
    return {"prompt": rng.choice(lines), "calls": calls, "assertions": asserts,
            "params": {"order_id": o["order_id"], "customer_id": o["customer_id"],
                       "sku": it["sku"], "quantity": qty, "amount_cents": amount},
            "arithmetic": True, "policy_lookup": False}


def build_tier_compensation(rng: random.Random, w: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """T4 按等级补偿：查订单 + 查客户等级 + 查政策 + 发信用 + 开工单。"""
    o = _pick_order(w, rng, lambda o: o["status"] in ("shipped", "delivered"))
    if o is None:
        return None
    cust = w["customers"][o["customer_id"]]
    credit = P.tier_credit_cents(o["total_cents"], cust["tier"])
    if credit <= 0:
        return None
    calls = [_call("get_order", order_id=o["order_id"]),
             _call("get_customer", customer_id=o["customer_id"]),
             _call("get_policy", topic="compensation"),
             _call("apply_credit", customer_id=o["customer_id"], amount_cents=credit,
                   reason=f"订单 {o['order_id']} 延迟补偿"),
             _call("open_ticket", category="compensation", order_id=o["order_id"],
                   note=f"已按等级政策补偿 {credit} 分")]
    asserts = [
        {"kind": "credits_exact", "expected": [{"customer_id": o["customer_id"], "amount_cents": credit}]},
        {"kind": "credit_balance", "customer_id": o["customer_id"], "value": credit},
        {"kind": "tickets_exact", "expected": [{"category": "compensation", "order_id": o["order_id"]}]},
        # 补偿不是退款
        {"kind": "refunds_exact", "expected": []},
        {"kind": "order_status", "order_id": o["order_id"], "status": o["status"]},
        {"kind": "notifications_exact", "expected": []},
    ] + _base_negatives([], [])
    lines = [
        f"订单 {o['order_id']} 严重延迟，客户{cust['name']}（{o['customer_id']}）要求补偿。"
        f"请按补偿政策给他发放店铺信用，并开一条 compensation 类工单记录这件事。",
        f"客户{cust['name']}（{o['customer_id']}）因为订单 {o['order_id']} 延迟要求补偿。"
        f"请查一下补偿政策，按他的等级发放店铺信用，然后开 compensation 工单归档。",
    ]
    return {"prompt": rng.choice(lines), "calls": calls, "assertions": asserts,
            "params": {"order_id": o["order_id"], "customer_id": o["customer_id"],
                       "tier": cust["tier"], "credit_cents": credit},
            "arithmetic": True, "policy_lookup": True}


def build_inventory_correction(rng: random.Random, w: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """T5 盘点纠正：增量 = 实际 - 系统，可能为负。"""
    sku = rng.choice([s for s in w["inventory"] if w["inventory"][s]["restockable"]])
    inv = w["inventory"][sku]
    delta = rng.choice([-3, -2, -1, 2, 3, 5, 7])
    actual = inv["on_hand"] + delta
    if actual < 0:
        actual = inv["on_hand"] + abs(delta)
        delta = abs(delta)
    calls = [_call("get_inventory", sku=sku),
             _call("restock_item", sku=sku, quantity=delta),
             _call("open_ticket", category="inventory", sku=sku, note=f"盘点纠正 {delta}")]
    asserts = [
        {"kind": "inventory_on_hand", "sku": sku, "value": actual},
        {"kind": "inventory_reserved", "sku": sku, "value": inv["reserved"]},
        {"kind": "tickets_exact", "expected": [{"category": "inventory", "sku": sku}]},
        {"kind": "refunds_exact", "expected": []},
        {"kind": "credits_exact", "expected": []},
        {"kind": "notifications_exact", "expected": []},
    ] + _base_negatives([], [sku])
    lines = [
        f"仓库盘点发现 {sku} 的实际库存是 {actual} 件，和系统数量不一致。"
        f"请把系统库存调整到实际数量，并开一条 inventory 类工单记录这次盘点。",
        f"{sku} 盘点结果为 {actual} 件。请据此纠正系统里的库存数量，然后开 inventory 工单归档。",
    ]
    return {"prompt": rng.choice(lines), "calls": calls, "assertions": asserts,
            "params": {"sku": sku, "actual": actual, "delta": delta},
            "arithmetic": True, "policy_lookup": False}


def build_stuck_reservation(rng: random.Random, w: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """T6 已取消订单的预留没释放：查订单 + 释放预留 + 开工单。"""
    o = _pick_order(w, rng, lambda o: o["items"])
    if o is None:
        return None
    it = o["items"][0]
    o["status"] = "cancelled"
    w["shipments"][o["order_id"]]["status"] = "cancelled"
    inv = w["inventory"][it["sku"]]
    inv["reserved"] = it["quantity"] + rng.randint(0, 2)
    calls = [_call("get_order", order_id=o["order_id"]),
             _call("release_reservation", sku=it["sku"], quantity=it["quantity"]),
             _call("open_ticket", category="inventory", sku=it["sku"],
                   note=f"释放订单 {o['order_id']} 的预留")]
    asserts = [
        {"kind": "inventory_reserved", "sku": it["sku"], "value": inv["reserved"] - it["quantity"]},
        {"kind": "inventory_on_hand", "sku": it["sku"], "value": inv["on_hand"]},
        {"kind": "tickets_exact", "expected": [{"category": "inventory", "sku": it["sku"]}]},
        {"kind": "order_status", "order_id": o["order_id"], "status": "cancelled"},
        {"kind": "refunds_exact", "expected": []},
        {"kind": "credits_exact", "expected": []},
        {"kind": "notifications_exact", "expected": []},
    ] + _base_negatives([], [it["sku"]])
    lines = [
        f"订单 {o['order_id']} 已经取消，但它占用的 {it['sku']} 预留库存还没释放。"
        f"请按该订单里这个商品的数量释放预留，并开一条 inventory 工单。",
        f"已取消的订单 {o['order_id']} 还占着 {it['sku']} 的预留。请把这部分预留释放掉，"
        f"然后开 inventory 工单说明原因。",
    ]
    return {"prompt": rng.choice(lines), "calls": calls, "assertions": asserts,
            "params": {"order_id": o["order_id"], "sku": it["sku"], "quantity": it["quantity"]},
            "arithmetic": False, "policy_lookup": False}


# ================================ 组合泛化测试族（训练不出现） ================================
def build_refund_with_credit(rng: random.Random, w: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """C1 全额退款 + 按等级补偿信用 + 改状态。工具都见过，这个组合没见过。"""
    o = _pick_order(w, rng, lambda o: o["status"] == "delivered")
    if o is None:
        return None
    cust = w["customers"][o["customer_id"]]
    credit = P.tier_credit_cents(o["total_cents"], cust["tier"])
    if credit <= 0:
        return None
    calls = [_call("get_order", order_id=o["order_id"]),
             _call("get_customer", customer_id=o["customer_id"]),
             _call("get_policy", topic="compensation"),
             _call("create_refund", order_id=o["order_id"], amount_cents=o["total_cents"],
                   reason="商品严重问题全额退款"),
             _call("apply_credit", customer_id=o["customer_id"], amount_cents=credit,
                   reason=f"订单 {o['order_id']} 额外补偿"),
             _call("update_order_status", order_id=o["order_id"], status="refunded")]
    asserts = [
        {"kind": "refunds_exact", "expected": [{"order_id": o["order_id"], "amount_cents": o["total_cents"]}]},
        {"kind": "order_status", "order_id": o["order_id"], "status": "refunded"},
        {"kind": "order_refunded_cents", "order_id": o["order_id"], "value": o["total_cents"]},
        {"kind": "credits_exact", "expected": [{"customer_id": o["customer_id"], "amount_cents": credit}]},
        {"kind": "credit_balance", "customer_id": o["customer_id"], "value": credit},
        {"kind": "tickets_exact", "expected": []},
        {"kind": "notifications_exact", "expected": []},
    ] + _base_negatives([o["order_id"]], [])
    lines = [
        f"订单 {o['order_id']} 的商品有严重问题，客户{cust['name']}（{o['customer_id']}）不退货但要求全额退款，"
        f"公司同意再按补偿政策给他等级对应的店铺信用。请办理全额退款、发放信用，并把订单状态改成 refunded。",
        f"客户{cust['name']}（{o['customer_id']}）的订单 {o['order_id']} 需要全额退款，"
        f"另外按补偿政策再给一笔店铺信用，最后把订单状态置为 refunded。商品不退回仓库。",
    ]
    return {"prompt": rng.choice(lines), "calls": calls, "assertions": asserts,
            "params": {"order_id": o["order_id"], "customer_id": o["customer_id"],
                       "tier": cust["tier"], "credit_cents": credit},
            "arithmetic": True, "policy_lookup": True}


def build_partial_refund_restock_ticket(rng: random.Random, w: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """C2 部分退款 + 商品退回所以要回补库存 + 开工单。"""
    o = _pick_order(w, rng, lambda o: o["status"] == "delivered" and _restockable_items(o))
    if o is None:
        return None
    it = rng.choice(_restockable_items(o))
    qty = rng.randint(1, it["quantity"])
    amount = qty * it["unit_price_cents"]
    if amount >= o["total_cents"]:
        qty = 1
        amount = it["unit_price_cents"]
        if amount >= o["total_cents"]:
            return None
    inv_before = w["inventory"][it["sku"]]["on_hand"]
    calls = [_call("get_order", order_id=o["order_id"]),
             _call("create_refund", order_id=o["order_id"], amount_cents=amount, reason="部分退货退款"),
             _call("restock_item", sku=it["sku"], quantity=qty),
             _call("open_ticket", category="refund", order_id=o["order_id"],
                   note=f"部分退货退款 {amount} 分"),
             _call("update_order_status", order_id=o["order_id"], status="partially_refunded")]
    asserts = [
        {"kind": "refunds_exact", "expected": [{"order_id": o["order_id"], "amount_cents": amount}]},
        {"kind": "order_refunded_cents", "order_id": o["order_id"], "value": amount},
        {"kind": "order_status", "order_id": o["order_id"], "status": "partially_refunded"},
        {"kind": "inventory_on_hand", "sku": it["sku"], "value": inv_before + qty},
        {"kind": "tickets_exact", "expected": [{"category": "refund", "order_id": o["order_id"]}]},
        {"kind": "credits_exact", "expected": []},
        {"kind": "notifications_exact", "expected": []},
    ] + _base_negatives([o["order_id"]], [it["sku"]])
    lines = [
        f"客户把订单 {o['order_id']} 里 {qty} 件 {it['sku']} 退回了仓库。请按这些商品的金额退款，"
        f"把退回的数量补回库存，开一条 refund 类工单，并把订单状态改成 partially_refunded。",
        f"订单 {o['order_id']} 有 {qty} 件 {it['sku']} 已退回入库。请退掉对应金额、补回库存、"
        f"开 refund 工单，并把订单状态置为 partially_refunded。",
    ]
    return {"prompt": rng.choice(lines), "calls": calls, "assertions": asserts,
            "params": {"order_id": o["order_id"], "sku": it["sku"], "quantity": qty,
                       "amount_cents": amount},
            "arithmetic": True, "policy_lookup": False}


def build_cancel_refund_release(rng: random.Random, w: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """C3 未发货订单：取消运单 + 全额退款 + 释放预留 + 改状态。"""
    o = _pick_order(w, rng, lambda o: o["status"] == "paid" and o["items"])
    if o is None:
        return None
    it = o["items"][0]
    w["shipments"][o["order_id"]]["status"] = "pending"
    inv = w["inventory"][it["sku"]]
    inv["reserved"] = it["quantity"] + rng.randint(0, 1)
    cust = w["customers"][o["customer_id"]]
    calls = [_call("get_order", order_id=o["order_id"]),
             _call("cancel_shipment", order_id=o["order_id"]),
             _call("create_refund", order_id=o["order_id"], amount_cents=o["total_cents"],
                   reason="取消订单全额退款"),
             _call("release_reservation", sku=it["sku"], quantity=it["quantity"]),
             _call("update_order_status", order_id=o["order_id"], status="cancelled")]
    asserts = [
        {"kind": "shipment_status", "order_id": o["order_id"], "status": "cancelled"},
        {"kind": "refunds_exact", "expected": [{"order_id": o["order_id"], "amount_cents": o["total_cents"]}]},
        {"kind": "order_refunded_cents", "order_id": o["order_id"], "value": o["total_cents"]},
        {"kind": "order_status", "order_id": o["order_id"], "status": "cancelled"},
        {"kind": "inventory_reserved", "sku": it["sku"], "value": inv["reserved"] - it["quantity"]},
        {"kind": "inventory_on_hand", "sku": it["sku"], "value": inv["on_hand"]},
        {"kind": "credits_exact", "expected": []},
        {"kind": "tickets_exact", "expected": []},
        {"kind": "notifications_exact", "expected": []},
    ] + _base_negatives([o["order_id"]], [it["sku"]])
    lines = [
        f"客户{cust['name']}（{o['customer_id']}）取消了还没发货的订单 {o['order_id']} 并要求退款。"
        f"请取消运单、全额退款、释放该订单占用的 {it['sku']} 预留（{it['quantity']} 件），"
        f"最后把订单状态改成 cancelled。",
        f"订单 {o['order_id']} 未发货，客户要求取消并退全款。请依次处理运单取消、退款、"
        f"释放 {it['sku']} 的 {it['quantity']} 件预留，并把订单状态置为 cancelled。",
    ]
    return {"prompt": rng.choice(lines), "calls": calls, "assertions": asserts,
            "params": {"order_id": o["order_id"], "sku": it["sku"], "quantity": it["quantity"]},
            "arithmetic": False, "policy_lookup": False}


# ================================ 长链外推族（工具与训练相同，链更长） ================================
def build_bulk_refund(rng: random.Random, w: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """L1 同一客户两个订单各自全额退款并改状态：只增加链长，不引入新工具。"""
    by_cust: Dict[str, List[Dict[str, Any]]] = {}
    for o in _orders(w):
        if o["status"] in ("shipped", "delivered"):
            by_cust.setdefault(o["customer_id"], []).append(o)
    cands = sorted([(c, os_) for c, os_ in by_cust.items() if len(os_) >= 2], key=lambda kv: kv[0])
    if not cands:
        return None
    cid, os_ = rng.choice(cands)
    targets = os_[:2]
    cust = w["customers"][cid]
    calls: List[Dict[str, Any]] = [_call("get_order", order_id=o["order_id"]) for o in targets]
    asserts: List[Dict[str, Any]] = []
    expected_refunds = []
    for o in targets:
        calls.append(_call("create_refund", order_id=o["order_id"], amount_cents=o["total_cents"],
                           reason="批量全额退款"))
        calls.append(_call("update_order_status", order_id=o["order_id"], status="refunded"))
        expected_refunds.append({"order_id": o["order_id"], "amount_cents": o["total_cents"]})
        asserts += [
            {"kind": "order_status", "order_id": o["order_id"], "status": "refunded"},
            {"kind": "order_refunded_cents", "order_id": o["order_id"], "value": o["total_cents"]},
        ]
    asserts += [
        {"kind": "refunds_exact", "expected": expected_refunds},
        {"kind": "credits_exact", "expected": []},
        {"kind": "tickets_exact", "expected": []},
        {"kind": "notifications_exact", "expected": []},
    ] + _base_negatives([o["order_id"] for o in targets], [])
    ids = " 和 ".join(o["order_id"] for o in targets)
    prompt = (f"客户{cust['name']}（{cid}）的订单 {ids} 都要全额退款，商品都不退回仓库。"
              f"请逐个查清金额后分别办理退款，并把这两个订单的状态都改成 refunded。")
    return {"prompt": prompt, "calls": calls, "assertions": asserts,
            "params": {"customer_id": cid, "order_ids": [o["order_id"] for o in targets]},
            "arithmetic": False, "policy_lookup": False}


# ================================ 任务族注册表 ================================
# split_group:
#   train —— 训练/验证/IID 测试都从这些族出（IID 测试用不同的 seed 区间）
#   comp  —— 只出现在 test_comp：工具全部见过，工具集合组合没见过
#   long  —— 只出现在 test_long：工具集合是训练的子集，链更长（外推）
FAMILIES: Dict[str, Dict[str, Any]] = {
    "refund_restock": {
        "builder": build_refund_restock, "split_group": "train",
        "tools": ["create_refund", "get_order", "restock_item", "update_order_status"]},
    "cancel_unshipped": {
        "builder": build_cancel_unshipped, "split_group": "train",
        "tools": ["cancel_shipment", "get_order", "update_order_status"]},
    "partial_refund_notify": {
        "builder": build_partial_refund_notify, "split_group": "train",
        "tools": ["create_refund", "get_order", "notify_customer", "update_order_status"]},
    "tier_compensation": {
        "builder": build_tier_compensation, "split_group": "train",
        "tools": ["apply_credit", "get_customer", "get_order", "get_policy", "open_ticket"]},
    "inventory_correction": {
        "builder": build_inventory_correction, "split_group": "train",
        "tools": ["get_inventory", "open_ticket", "restock_item"]},
    "stuck_reservation": {
        "builder": build_stuck_reservation, "split_group": "train",
        "tools": ["get_order", "open_ticket", "release_reservation"]},
    "refund_with_credit": {
        "builder": build_refund_with_credit, "split_group": "comp",
        "tools": ["apply_credit", "create_refund", "get_customer", "get_order", "get_policy",
                  "update_order_status"]},
    "partial_refund_restock_ticket": {
        "builder": build_partial_refund_restock_ticket, "split_group": "comp",
        "tools": ["create_refund", "get_order", "open_ticket", "restock_item",
                  "update_order_status"]},
    "cancel_refund_release": {
        "builder": build_cancel_refund_release, "split_group": "comp",
        "tools": ["cancel_shipment", "create_refund", "get_order", "release_reservation",
                  "update_order_status"]},
    "bulk_refund": {
        "builder": build_bulk_refund, "split_group": "long",
        "tools": ["create_refund", "get_order", "update_order_status"]},
}

TRAIN_FAMILIES = [k for k, v in FAMILIES.items() if v["split_group"] == "train"]
COMP_FAMILIES = [k for k, v in FAMILIES.items() if v["split_group"] == "comp"]
LONG_FAMILIES = [k for k, v in FAMILIES.items() if v["split_group"] == "long"]
# 始终出现在工具列表里但没有任何任务需要的干扰工具（防止"工具列表泄漏正确计划"）
DISTRACTOR_TOOLS = ["search_orders"]


# ================================ 实例生成 ================================
def instance_key(inst: "TaskInstance") -> str:
    """用于去重与跨划分泄漏检查的内容指纹（不含 task_id / seed）。"""
    payload = json.dumps({"family": inst.family, "prompt": inst.prompt, "params": inst.params,
                          "calls": inst.gold_calls, "assertions": inst.assertions,
                          "world": inst.world}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def generate_instance(family: str, seed: int) -> Optional[TaskInstance]:
    """确定性生成；世界无法承载该任务族时返回 None（调用方换 seed）。"""
    meta = FAMILIES[family]
    rng = random.Random(f"{family}|{seed}")
    world = make_world(seed)
    built = meta["builder"](rng, world)
    if built is None:
        return None
    calls = built["calls"]
    sig = _sig(calls)
    inst = TaskInstance(
        task_id=f"{family}#{seed}",
        family=family,
        split_group=meta["split_group"],
        difficulty=difficulty_of(len(calls), built["arithmetic"], built["policy_lookup"]),
        seed=seed,
        prompt=built["prompt"],
        world=world,
        params=built["params"],
        tool_signature=sig,
        gold_calls=calls,
        assertions=built["assertions"],
    )
    return inst


def generate_pool(families: List[str], n_per_family: int, seed_lo: int, seed_hi: int,
                  strict: bool = True) -> List[TaskInstance]:
    """在 [seed_lo, seed_hi) 的 seed 区间内为每个任务族取 n_per_family 个实例。

    不同划分使用互不相交的 seed 区间，因此世界与实例天然不重叠。
    """
    out: List[TaskInstance] = []
    for fam in families:
        got, seen = 0, set()
        for seed in range(seed_lo, seed_hi):
            if got >= n_per_family:
                break
            inst = generate_instance(fam, seed)
            if inst is None:
                continue
            key = instance_key(inst)
            if key in seen:
                continue
            seen.add(key)
            out.append(inst)
            got += 1
        if got < n_per_family and strict:
            raise RuntimeError(f"{fam}: seed 区间 [{seed_lo},{seed_hi}) 只生成了 {got}/{n_per_family} 个实例")
    return out


def difficulty_histogram(pool: List[TaskInstance]) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for inst in pool:
        hist[inst.difficulty] = hist.get(inst.difficulty, 0) + 1
    return dict(sorted(hist.items()))


def check_split_isolation() -> Dict[str, Any]:
    """核对组合泛化的隔离条件：
    1) 每个 comp 族的工具集合都没有在任何 train 族出现过；
    2) comp 族用到的每个工具都在某个 train 族出现过（否则是"没见过的工具"而不是"没见过的组合"）；
    3) long 族的工具集合是训练工具的子集。
    """
    train_sigs = {tuple(sorted(FAMILIES[f]["tools"])) for f in TRAIN_FAMILIES}
    train_tools = {t for f in TRAIN_FAMILIES for t in FAMILIES[f]["tools"]}
    report: Dict[str, Any] = {"train_signatures": sorted(train_sigs), "train_tools": sorted(train_tools),
                              "violations": []}
    for f in COMP_FAMILIES:
        sig = tuple(sorted(FAMILIES[f]["tools"]))
        if sig in train_sigs:
            report["violations"].append(f"comp 族 {f} 的工具集合在训练中出现过")
        unseen = sorted(set(sig) - train_tools)
        if unseen:
            report["violations"].append(f"comp 族 {f} 含训练未见工具 {unseen}")
    for f in LONG_FAMILIES:
        unseen = sorted(set(FAMILIES[f]["tools"]) - train_tools)
        if unseen:
            report["violations"].append(f"long 族 {f} 含训练未见工具 {unseen}")
    report["ok"] = not report["violations"]
    return report


if __name__ == "__main__":
    rep = check_split_isolation()
    print("划分隔离检查:", "通过" if rep["ok"] else rep["violations"])
    print("训练侧工具并集:", rep["train_tools"])
    print("干扰工具:", DISTRACTOR_TOOLS)
    print()
    for fam, meta in FAMILIES.items():
        inst = None
        for seed in range(0, 200):
            inst = generate_instance(fam, seed)
            if inst is not None:
                break
        assert inst is not None, fam
        print(f"[{meta['split_group']:5s}] {fam:30s} {inst.difficulty} "
              f"calls={len(inst.gold_calls)} seed={inst.seed}")
        print(f"        tools={inst.tool_signature}")
        print(f"        prompt={inst.prompt[:88]}")
    print()
    pool = generate_pool(TRAIN_FAMILIES, 8, 0, 400)
    print("训练池样例:", len(pool), difficulty_histogram(pool))
    comp = generate_pool(COMP_FAMILIES, 8, 900000, 900400)
    print("comp 池样例:", len(comp), difficulty_histogram(comp))
    long_ = generate_pool(LONG_FAMILIES, 8, 950000, 950400)
    print("long 池样例:", len(long_), difficulty_histogram(long_))
    keys = {instance_key(i) for i in pool} & {instance_key(i) for i in comp + long_}
    print("训练/测试实例指纹交集:", len(keys))
