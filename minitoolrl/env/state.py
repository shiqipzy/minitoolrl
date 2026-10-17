"""可回放的订单、库存、退款单等后台状态。

设计约束：
- 状态可 JSON 序列化，每条 rollout 独立初始化，保存初始状态即可完整回放。
- 金额一律整数分。
- 所有写操作都记进 mutations 操作日志，供统计和排查使用；判分比较实际业务状态。
"""
from __future__ import annotations

import copy
import json
import random
from typing import Any, Dict, List

from minitoolrl.env.policy import TIERS

# ======== 商品目录（固定，保证不同 seed 之间 SKU 语义一致） ========
CATALOG = [
    ("SKU-1001", "无线鼠标", 12900, True),
    ("SKU-1002", "机械键盘", 39900, True),
    ("SKU-1003", "USB-C 数据线", 4900, True),
    ("SKU-1004", "显示器支架", 18500, True),
    ("SKU-1005", "降噪耳机", 79900, True),
    ("SKU-1006", "定制刻字礼盒", 25900, False),   # 定制品不可回补
    ("SKU-1007", "打印机墨盒", 9900, True),
    ("SKU-1008", "一次性清洁湿巾", 2900, False),  # 一次性商品不可回补
]
# 注意：不要再建 SKU -> 语义 的全局表。编号是按世界抽的，只能从 world['inventory'] 里查。

CUSTOMER_NAMES = ["林岚", "赵岩", "陈默", "周洲", "吴桐", "郑迪", "钱屿", "孙澈"]
CITIES = ["杭州", "成都", "南京", "武汉", "西安", "广州"]
CARRIERS = ["顺丰", "京东物流", "中通"]


class WorldState:
    """可变的世界状态。写操作只应由 tools.py 调用。"""

    def __init__(self, data: Dict[str, Any]):
        self.data = data

    # ---- 序列化 ----
    @staticmethod
    def from_snapshot(snapshot: Dict[str, Any]) -> "WorldState":
        return WorldState(copy.deepcopy(snapshot))

    def snapshot(self) -> Dict[str, Any]:
        return copy.deepcopy(self.data)

    def fingerprint(self) -> str:
        """用于判断状态是否被改动（不含审计日志）。"""
        payload = {k: v for k, v in self.data.items() if k != "mutations"}
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)

    # ---- 便捷访问 ----
    @property
    def orders(self) -> Dict[str, Any]:
        return self.data["orders"]

    @property
    def customers(self) -> Dict[str, Any]:
        return self.data["customers"]

    @property
    def inventory(self) -> Dict[str, Any]:
        return self.data["inventory"]

    @property
    def shipments(self) -> Dict[str, Any]:
        return self.data["shipments"]

    @property
    def refunds(self) -> List[Dict[str, Any]]:
        """退款单列表：每张记录一次退款的编号、订单、金额和原因。"""
        return self.data["refunds"]

    @property
    def credits(self) -> List[Dict[str, Any]]:
        return self.data["credits"]

    @property
    def tickets(self) -> List[Dict[str, Any]]:
        """工单列表：仅在执行 open_ticket 时添加处理事项。"""
        return self.data["tickets"]

    @property
    def notifications(self) -> List[Dict[str, Any]]:
        return self.data["notifications"]

    @property
    def mutations(self) -> List[Dict[str, Any]]:
        return self.data["mutations"]

    # ---- 审计 ----
    def log_mutation(self, tool: str, args: Dict[str, Any], target: str) -> None:
        """记录写工具、参数和操作对象，用于统计和排查，不直接参与训练损失。"""
        self.data["mutations"].append({
            "seq": len(self.data["mutations"]) + 1,
            "tool": tool,
            "args": args,
            "target": target,
        })

    def next_id(self, prefix: str) -> str:
        self.data["_counters"][prefix] = self.data["_counters"].get(prefix, 0) + 1
        return f"{prefix}-{self.data['_counters'][prefix]:03d}"


def make_world(seed: int) -> Dict[str, Any]:
    """按 seed 确定性生成一个世界快照。

    实体编号**每个世界重新抽**：订单 O-2000~2999、客户 C-001~999、SKU-1000~1999。
    早先所有世界都用 O-2001..O-2006 这 6 个订单号，实测后果是模型在 ID 位置直接背下
    边缘分布（恒定输出训练集里最常见的那个数字，反事实跟随率 0），条件抄写电路根本不形成；
    换成按世界抽号之后同一个 ID 只在少数世界里出现过，只能从上下文抄。

    SKU 的语义（名称、价格、可否回补）仍与 CATALOG 一一对应，只是编号随世界变化；
    因此判断 restockable 之类的属性一律从 world["inventory"] 里查，不要用全局表。
    """
    rng = random.Random(seed)
    n_customers = rng.randint(3, 4)
    names = rng.sample(CUSTOMER_NAMES, n_customers)
    cust_nums = rng.sample(range(1, 1000), n_customers)
    customers: Dict[str, Any] = {}
    for num, name in zip(cust_nums, names):
        cid = f"C-{num:03d}"
        customers[cid] = {
            "customer_id": cid,
            "name": name,
            "tier": rng.choice(TIERS),
            "city": rng.choice(CITIES),
            "credit_balance_cents": 0,
        }

    inventory: Dict[str, Any] = {}
    sku_nums = rng.sample(range(1000, 2000), len(CATALOG))
    for num, (_, name, price, restockable) in zip(sku_nums, CATALOG):
        sku = f"SKU-{num}"
        inventory[sku] = {
            "sku": sku,
            "name": name,
            "price_cents": price,
            "restockable": restockable,
            "on_hand": rng.randint(6, 80),
            "reserved": rng.randint(0, 4),
        }

    orders: Dict[str, Any] = {}
    shipments: Dict[str, Any] = {}
    n_orders = rng.randint(4, 6)
    order_nums = rng.sample(range(2000, 3000), n_orders)
    for num in order_nums:
        oid = f"O-{num}"
        cid = rng.choice(list(customers.keys()))
        n_items = rng.randint(1, 3)
        skus = rng.sample(list(inventory.keys()), n_items)
        items = []
        for sku in skus:
            qty = rng.randint(1, 3)
            items.append({
                "sku": sku,
                "name": inventory[sku]["name"],
                "quantity": qty,
                "unit_price_cents": inventory[sku]["price_cents"],
                "restockable": inventory[sku]["restockable"],
            })
        total = sum(it["quantity"] * it["unit_price_cents"] for it in items)
        status = rng.choice(["paid", "shipped", "delivered"])
        orders[oid] = {
            "order_id": oid,
            "customer_id": cid,
            "status": status,
            "created_at": f"2026-0{rng.randint(1, 8)}-{rng.randint(10, 28)}",
            "items": items,
            "total_cents": total,
            "refunded_cents": 0,
        }
        shipments[oid] = {
            "order_id": oid,
            "carrier": rng.choice(CARRIERS),
            "tracking_no": f"{rng.randint(10 ** 9, 10 ** 10 - 1)}",
            "status": {"paid": "pending", "shipped": "in_transit", "delivered": "delivered"}[status],
        }

    return {
        "customers": customers,
        "inventory": inventory,
        "orders": orders,
        "shipments": shipments,
        "refunds": [],
        "credits": [],
        "tickets": [],
        "notifications": [],
        "mutations": [],
        "_counters": {},
        "_seed": seed,
    }


