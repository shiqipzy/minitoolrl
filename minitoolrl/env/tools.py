"""工具定义与执行器。

约束（对应方案里"环境不能替模型完成跨服务操作"）：
- 每个工具只做一件事，没有任何级联副作用。
- 参数非法或前置条件不满足时抛 ToolError，不产生任何状态改动。
- 报错信息要能说明"哪里不对"，但不能直接给出答案。
"""
from __future__ import annotations

from typing import Any, Dict, List

from minitoolrl.env import policy as P
from minitoolrl.env.state import WorldState


class ToolError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message

    def payload(self) -> Dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message}}


# ================================ 参数校验 ================================
def _req(args: Dict[str, Any], key: str) -> Any:
    if key not in args or args[key] is None or (isinstance(args[key], str) and not args[key].strip()):
        raise ToolError("MISSING_ARG", f"缺少参数 {key}")
    return args[key]


def _req_str(args: Dict[str, Any], key: str) -> str:
    v = _req(args, key)
    if not isinstance(v, str):
        raise ToolError("BAD_TYPE", f"参数 {key} 应为字符串，收到 {type(v).__name__}")
    return v.strip()


def _req_int(args: Dict[str, Any], key: str) -> int:
    v = _req(args, key)
    if isinstance(v, bool):
        raise ToolError("BAD_TYPE", f"参数 {key} 应为整数")
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if v.is_integer():
            return int(v)
        raise ToolError("AMOUNT_NOT_INTEGER", f"参数 {key} 必须是整数（金额单位为分），收到 {v}")
    if isinstance(v, str):
        s = v.strip().replace(",", "")
        try:
            return int(s)
        except ValueError:
            raise ToolError("AMOUNT_NOT_INTEGER", f"参数 {key} 必须是整数（金额单位为分），收到 {v!r}")
    raise ToolError("BAD_TYPE", f"参数 {key} 应为整数，收到 {type(v).__name__}")


def _get_order(state: WorldState, order_id: str) -> Dict[str, Any]:
    o = state.orders.get(order_id)
    if o is None:
        raise ToolError("ORDER_NOT_FOUND", f"订单 {order_id} 不存在")
    return o


def _get_customer(state: WorldState, customer_id: str) -> Dict[str, Any]:
    c = state.customers.get(customer_id)
    if c is None:
        raise ToolError("CUSTOMER_NOT_FOUND", f"客户 {customer_id} 不存在")
    return c


def _get_sku(state: WorldState, sku: str) -> Dict[str, Any]:
    s = state.inventory.get(sku)
    if s is None:
        raise ToolError("SKU_NOT_FOUND", f"商品 {sku} 不存在")
    return s


# ================================ 只读工具 ================================
def t_get_order(state: WorldState, args: Dict[str, Any]) -> Dict[str, Any]:
    o = _get_order(state, _req_str(args, "order_id"))
    ship = state.shipments.get(o["order_id"], {})
    return {
        "order_id": o["order_id"],
        "customer_id": o["customer_id"],
        "status": o["status"],
        "total_cents": o["total_cents"],
        "refunded_cents": o["refunded_cents"],
        "items": [
            {"sku": it["sku"], "quantity": it["quantity"],
             "unit_price_cents": it["unit_price_cents"], "restockable": it["restockable"]}
            for it in o["items"]
        ],
        "shipment_status": ship.get("status"),
    }


def t_get_customer(state: WorldState, args: Dict[str, Any]) -> Dict[str, Any]:
    c = _get_customer(state, _req_str(args, "customer_id"))
    return {"customer_id": c["customer_id"], "name": c["name"], "tier": c["tier"],
            "credit_balance_cents": c["credit_balance_cents"]}


def t_get_inventory(state: WorldState, args: Dict[str, Any]) -> Dict[str, Any]:
    s = _get_sku(state, _req_str(args, "sku"))
    return {"sku": s["sku"], "on_hand": s["on_hand"], "reserved": s["reserved"],
            "restockable": s["restockable"], "price_cents": s["price_cents"]}


def t_get_policy(state: WorldState, args: Dict[str, Any]) -> Dict[str, Any]:
    topic = _req_str(args, "topic").lower()
    if topic not in P.POLICY_DOCS:
        raise ToolError("UNKNOWN_POLICY_TOPIC", f"未知主题 {topic}，可选：{', '.join(P.POLICY_TOPICS)}")
    return {"topic": topic, "content": P.POLICY_DOCS[topic]}


def t_search_orders(state: WorldState, args: Dict[str, Any]) -> Dict[str, Any]:
    cid = _req_str(args, "customer_id")
    _get_customer(state, cid)
    status = args.get("status")
    rows = [
        {"order_id": o["order_id"], "status": o["status"], "total_cents": o["total_cents"]}
        for o in state.orders.values()
        if o["customer_id"] == cid and (not status or o["status"] == status)
    ]
    return {"customer_id": cid, "count": len(rows), "orders": sorted(rows, key=lambda r: r["order_id"])}


# ================================ 写工具 ================================
def t_create_refund(state: WorldState, args: Dict[str, Any]) -> Dict[str, Any]:
    """执行退款：更新订单已退金额，生成退款单，并自动记录操作日志。"""
    order_id = _req_str(args, "order_id")
    amount = _req_int(args, "amount_cents")
    reason = _req_str(args, "reason")
    o = _get_order(state, order_id)
    if o["status"] not in P.REFUNDABLE_ORDER_STATUSES:
        raise ToolError("NOT_REFUNDABLE", f"订单状态 {o['status']} 不可退款")
    if amount <= 0:
        raise ToolError("AMOUNT_MUST_BE_POSITIVE", "退款金额必须为正整数分")
    remaining = o["total_cents"] - o["refunded_cents"]
    if amount > remaining:
        raise ToolError("REFUND_EXCEEDS_REMAINING", f"退款金额超过可退余额 {remaining} 分")
    refund_id = state.next_id("R")
    o["refunded_cents"] += amount
    # 退款单保存本次退款的订单、金额和原因，供退款断言核验。
    state.refunds.append({"refund_id": refund_id, "order_id": order_id,
                          "amount_cents": amount, "reason": reason})
    # 操作日志用于排查工具执行；工单由 open_ticket 单独创建。
    state.log_mutation("create_refund", {"order_id": order_id, "amount_cents": amount}, order_id)
    return {"refund_id": refund_id, "order_id": order_id, "amount_cents": amount,
            "order_refunded_cents": o["refunded_cents"]}


def t_restock_item(state: WorldState, args: Dict[str, Any]) -> Dict[str, Any]:
    sku = _req_str(args, "sku")
    qty = _req_int(args, "quantity")
    s = _get_sku(state, sku)
    if qty == 0:
        raise ToolError("QUANTITY_ZERO", "quantity 是增量，不能为 0")
    if qty > 0 and not s["restockable"]:
        raise ToolError("NOT_RESTOCKABLE", f"商品 {sku} 不可回补库存")
    if s["on_hand"] + qty < 0:
        raise ToolError("STOCK_NEGATIVE", f"调整后库存为负（当前 {s['on_hand']}）")
    s["on_hand"] += qty
    state.log_mutation("restock_item", {"sku": sku, "quantity": qty}, sku)
    return {"sku": sku, "on_hand": s["on_hand"], "delta": qty}


def t_release_reservation(state: WorldState, args: Dict[str, Any]) -> Dict[str, Any]:
    sku = _req_str(args, "sku")
    qty = _req_int(args, "quantity")
    s = _get_sku(state, sku)
    if qty <= 0:
        raise ToolError("AMOUNT_MUST_BE_POSITIVE", "释放数量必须为正整数")
    if qty > s["reserved"]:
        raise ToolError("RESERVATION_TOO_LARGE", f"释放数量超过当前预留 {s['reserved']}")
    s["reserved"] -= qty
    state.log_mutation("release_reservation", {"sku": sku, "quantity": qty}, sku)
    return {"sku": sku, "reserved": s["reserved"], "released": qty}


def t_update_order_status(state: WorldState, args: Dict[str, Any]) -> Dict[str, Any]:
    order_id = _req_str(args, "order_id")
    status = _req_str(args, "status").lower()
    o = _get_order(state, order_id)
    if status not in P.ORDER_STATUSES:
        raise ToolError("INVALID_STATUS", f"非法状态 {status}，可选：{', '.join(P.ORDER_STATUSES)}")
    o["status"] = status
    state.log_mutation("update_order_status", {"order_id": order_id, "status": status}, order_id)
    return {"order_id": order_id, "status": status}


def t_cancel_shipment(state: WorldState, args: Dict[str, Any]) -> Dict[str, Any]:
    order_id = _req_str(args, "order_id")
    o = _get_order(state, order_id)
    ship = state.shipments.get(order_id)
    if ship is None:
        raise ToolError("SHIPMENT_NOT_FOUND", f"订单 {order_id} 没有运单")
    if o["status"] not in P.CANCELLABLE_ORDER_STATUSES:
        raise ToolError("NOT_CANCELLABLE", f"订单状态 {o['status']} 不可取消")
    if ship["status"] not in P.CANCELLABLE_SHIPMENT_STATUSES:
        raise ToolError("NOT_CANCELLABLE", f"运单状态 {ship['status']} 不可取消")
    ship["status"] = "cancelled"
    state.log_mutation("cancel_shipment", {"order_id": order_id}, order_id)
    return {"order_id": order_id, "shipment_status": "cancelled"}


def t_apply_credit(state: WorldState, args: Dict[str, Any]) -> Dict[str, Any]:
    cid = _req_str(args, "customer_id")
    amount = _req_int(args, "amount_cents")
    reason = _req_str(args, "reason")
    c = _get_customer(state, cid)
    if amount <= 0:
        raise ToolError("AMOUNT_MUST_BE_POSITIVE", "信用金额必须为正整数分")
    c["credit_balance_cents"] += amount
    state.credits.append({"customer_id": cid, "amount_cents": amount, "reason": reason})
    state.log_mutation("apply_credit", {"customer_id": cid, "amount_cents": amount}, cid)
    return {"customer_id": cid, "granted_cents": amount,
            "credit_balance_cents": c["credit_balance_cents"]}


def t_open_ticket(state: WorldState, args: Dict[str, Any]) -> Dict[str, Any]:
    """记录任务要求的已处理事项，工单创建后标记为 completed。"""
    category = _req_str(args, "category").lower()
    note = _req_str(args, "note")
    order_id = args.get("order_id") or None
    sku = args.get("sku") or None
    if category not in P.TICKET_CATEGORIES:
        raise ToolError("BAD_CATEGORY", f"非法类别 {category}，可选：{', '.join(P.TICKET_CATEGORIES)}")
    if not order_id and not sku:
        raise ToolError("MISSING_ARG", "order_id 与 sku 至少要填一个")
    if order_id:
        order_id = str(order_id).strip()
        _get_order(state, order_id)
    if sku:
        sku = str(sku).strip()
        _get_sku(state, sku)
    ticket_id = state.next_id("T")
    state.tickets.append({"ticket_id": ticket_id, "category": category, "note": note,
                          "order_id": order_id, "sku": sku, "status": "completed"})
    state.log_mutation("open_ticket", {"category": category, "order_id": order_id, "sku": sku},
                       order_id or sku)
    return {"ticket_id": ticket_id, "category": category, "status": "completed"}


def t_notify_customer(state: WorldState, args: Dict[str, Any]) -> Dict[str, Any]:
    cid = _req_str(args, "customer_id")
    template = _req_str(args, "template").lower()
    order_id = _req_str(args, "order_id")
    c = _get_customer(state, cid)
    if template not in P.NOTIFY_TEMPLATES:
        raise ToolError("BAD_TEMPLATE", f"非法模板 {template}，可选：{', '.join(P.NOTIFY_TEMPLATES)}")
    o = _get_order(state, order_id)
    if o["customer_id"] != c["customer_id"]:
        raise ToolError("CUSTOMER_MISMATCH", f"订单 {order_id} 不属于客户 {cid}")
    state.notifications.append({"customer_id": cid, "template": template, "order_id": order_id})
    state.log_mutation("notify_customer",
                       {"customer_id": cid, "template": template, "order_id": order_id}, cid)
    return {"sent": True, "customer_id": cid, "template": template}


# ================================ 注册表 ================================
def _schema(props: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {"type": "object", "properties": props, "required": required}


_STR = {"type": "string"}
_INT = {"type": "integer"}

SPECS: List[Dict[str, Any]] = [
    {"name": "get_order", "kind": "read", "fn": t_get_order,
     "description": "按订单号查询订单：状态、金额(分)、已退金额、商品明细、运单状态。",
     "parameters": _schema({"order_id": _STR}, ["order_id"])},
    {"name": "get_customer", "kind": "read", "fn": t_get_customer,
     "description": "按客户号查询客户：姓名、等级(regular/silver/gold)、店铺信用余额(分)。",
     "parameters": _schema({"customer_id": _STR}, ["customer_id"])},
    {"name": "get_inventory", "kind": "read", "fn": t_get_inventory,
     "description": "查询商品库存：on_hand 实际库存、reserved 预留数、restockable 是否可回补。",
     "parameters": _schema({"sku": _STR}, ["sku"])},
    {"name": "get_policy", "kind": "read", "fn": t_get_policy,
     "description": "查询业务政策细则，topic 取 refund/cancel/restock/compensation/notification。",
     "parameters": _schema({"topic": _STR}, ["topic"])},
    {"name": "search_orders", "kind": "read", "fn": t_search_orders,
     "description": "按客户号列出订单，可用 status 过滤。",
     "parameters": _schema({"customer_id": _STR, "status": _STR}, ["customer_id"])},
    {"name": "create_refund", "kind": "write", "fn": t_create_refund,
     "description": "对订单创建一笔退款，amount_cents 为整数分。只退钱，不改订单状态、不回补库存。",
     "parameters": _schema({"order_id": _STR, "amount_cents": _INT, "reason": _STR},
                           ["order_id", "amount_cents", "reason"])},
    {"name": "restock_item", "kind": "write", "fn": t_restock_item,
     "description": "调整商品实际库存，quantity 是增量（可为负），不是目标值。",
     "parameters": _schema({"sku": _STR, "quantity": _INT}, ["sku", "quantity"])},
    {"name": "release_reservation", "kind": "write", "fn": t_release_reservation,
     "description": "释放商品预留数量，只减少 reserved，不改 on_hand。",
     "parameters": _schema({"sku": _STR, "quantity": _INT}, ["sku", "quantity"])},
    {"name": "update_order_status", "kind": "write", "fn": t_update_order_status,
     "description": "设置订单状态，可选 paid/shipped/delivered/cancelled/refunded/partially_refunded。",
     "parameters": _schema({"order_id": _STR, "status": _STR}, ["order_id", "status"])},
    {"name": "cancel_shipment", "kind": "write", "fn": t_cancel_shipment,
     "description": "取消订单运单，仅 status=paid 且运单 pending 时可用。不改订单状态、不释放预留。",
     "parameters": _schema({"order_id": _STR}, ["order_id"])},
    {"name": "apply_credit", "kind": "write", "fn": t_apply_credit,
     "description": "给客户发放店铺信用，amount_cents 为整数分。属于补偿，不是退款。",
     "parameters": _schema({"customer_id": _STR, "amount_cents": _INT, "reason": _STR},
                           ["customer_id", "amount_cents", "reason"])},
    {"name": "open_ticket", "kind": "write", "fn": t_open_ticket,
     "description": "开工单，category 取 refund/logistics/inventory/compensation，order_id 与 sku 至少填一个。",
     "parameters": _schema({"category": _STR, "note": _STR, "order_id": _STR, "sku": _STR},
                           ["category", "note"])},
    {"name": "notify_customer", "kind": "write", "fn": t_notify_customer,
     "description": "按模板通知客户，template 取 refund_processed/order_cancelled/delay_apology/restock_done。",
     "parameters": _schema({"customer_id": _STR, "template": _STR, "order_id": _STR},
                           ["customer_id", "template", "order_id"])},
]

REGISTRY: Dict[str, Dict[str, Any]] = {s["name"]: s for s in SPECS}
ALL_TOOL_NAMES: List[str] = [s["name"] for s in SPECS]
WRITE_TOOLS = [s["name"] for s in SPECS if s["kind"] == "write"]


def tool_schemas(names: List[str] | None = None) -> List[Dict[str, Any]]:
    """返回 OpenAI function 格式的工具清单，供 apply_chat_template(tools=...) 使用。"""
    names = names or ALL_TOOL_NAMES
    out = []
    for n in names:
        s = REGISTRY[n]
        out.append({"type": "function", "function": {
            "name": s["name"], "description": s["description"], "parameters": s["parameters"]}})
    return out


def execute(state: WorldState, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """分派工具调用；参数或业务拒绝返回 error，程序自身异常直接抛出。

    各写工具先完成参数和业务检查，再修改状态。
    """
    spec = REGISTRY.get(name)
    if spec is None:
        return {"error": {"code": "UNKNOWN_TOOL",
                          "message": f"没有工具 {name}，可用：{', '.join(ALL_TOOL_NAMES)}"}}
    if not isinstance(args, dict):
        return {"error": {"code": "BAD_ARGUMENTS", "message": "arguments 必须是 JSON 对象"}}
    try:
        return spec["fn"](state, args)
    except ToolError as e:
        return e.payload()
