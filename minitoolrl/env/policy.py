"""业务规则的唯一来源。

政策文档（模型可查询）、参考求解器、评分断言都从本模块取常量。
这样可以保证：不会出现"规则只存在于评分器里、模型无法获知"的题目。
金额一律用整数分表示，禁止浮点比较。
"""

# ======== 状态机 ========
ORDER_STATUSES = ("paid", "shipped", "delivered", "cancelled", "refunded", "partially_refunded")
CANCELLABLE_ORDER_STATUSES = ("paid",)          # 只有已付款未发货可取消
CANCELLABLE_SHIPMENT_STATUSES = ("pending",)
REFUNDABLE_ORDER_STATUSES = ("paid", "shipped", "delivered", "partially_refunded")

# ======== 补偿政策（按客户等级） ========
TIERS = ("regular", "silver", "gold")
TIER_CREDIT_PERCENT = {"regular": 2, "silver": 5, "gold": 10}      # 百分比，整数
TIER_CREDIT_CAP_CENTS = {"regular": 2000, "silver": 5000, "gold": 10000}

# ======== 枚举 ========
TICKET_CATEGORIES = ("refund", "logistics", "inventory", "compensation")
NOTIFY_TEMPLATES = ("refund_processed", "order_cancelled", "delay_apology", "restock_done")


def tier_credit_cents(order_total_cents: int, tier: str) -> int:
    """按等级计算店铺信用：订单金额 * 百分比，向下取整到分，再按等级上限截断。"""
    pct = TIER_CREDIT_PERCENT[tier]
    cap = TIER_CREDIT_CAP_CENTS[tier]
    return min(order_total_cents * pct // 100, cap)


# ======== 对模型可见的政策文档 ========
# 由 get_policy 工具返回。所有评分需要的数值规则都必须写在这里或写在任务指令里。
POLICY_DOCS = {
    "refund": (
        "退款政策：\n"
        "1. 可退款订单状态：paid / shipped / delivered / partially_refunded。\n"
        "2. 单笔退款金额不得超过 订单金额 - 已退金额，金额单位为整数分。\n"
        "3. 订单状态按任务明确要求更新：普通全额退款为 refunded，部分退款为 partially_refunded，取消退款为 cancelled。\n"
        "4. 退款本身不会自动回补库存，也不会自动改订单状态，需要分别调用对应工具。"
    ),
    "cancel": (
        "取消政策：\n"
        "1. 只有 status=paid 且运单 status=pending 的订单可以取消。\n"
        "2. 取消运单后需把订单状态改为 cancelled。\n"
        "3. 取消运单不会自动释放预留库存；任务要求释放时需要单独调用 release_reservation。"
    ),
    "restock": (
        "库存政策：\n"
        "1. 客户退回实物商品时才回补库存，仅退款不退货不回补。\n"
        "2. restock_item 的 quantity 是增量，不是目标值。\n"
        "3. 只有 restockable=true 的商品可以回补，一次性商品不回补。\n"
        "4. 盘点纠正时，增量 = 实际库存 - 系统库存，可以为负。"
    ),
    "compensation": (
        "补偿政策：\n"
        "1. 店铺信用 = 订单金额 * 等级比例，向下取整到分，再按等级上限截断。\n"
        f"2. 等级比例：regular {TIER_CREDIT_PERCENT['regular']}%、"
        f"silver {TIER_CREDIT_PERCENT['silver']}%、gold {TIER_CREDIT_PERCENT['gold']}%。\n"
        f"3. 等级上限：regular {TIER_CREDIT_CAP_CENTS['regular']} 分、"
        f"silver {TIER_CREDIT_CAP_CENTS['silver']} 分、gold {TIER_CREDIT_CAP_CENTS['gold']} 分。\n"
        "4. 信用补偿和退款是独立操作；任务明确要求两者时分别执行，否则不要自行追加。\n"
        "5. 仅在任务明确要求工单时开指定类别工单，不因发放补偿自动追加工单。"
    ),
    "notification": (
        "通知政策：\n"
        f"1. 可用模板：{', '.join(NOTIFY_TEMPLATES)}。\n"
        "2. 退款完成用 refund_processed，取消订单用 order_cancelled，"
        "延迟致歉用 delay_apology，补货完成用 restock_done。\n"
        "3. 通知必须发给该订单的下单客户。"
    ),
}

POLICY_TOPICS = tuple(POLICY_DOCS.keys())

# ======== 系统提示中始终可见的操作规则摘要 ========
OPERATING_RULES = (
    "你是电商后台运营助手，只能通过工具修改系统状态。工作规则：\n"
    "1. 先用只读工具确认事实（订单金额、商品、库存、客户等级），再执行写操作，不要凭猜测填参数。\n"
    "2. 每个工具只做一件事：退款不会自动回补库存或改订单状态，取消运单不会自动释放预留库存。\n"
    "3. 金额参数一律是整数分（例如 129 元写 12900），不要带小数点。\n"
    "4. 不要修改任务没有要求的订单、商品或客户。\n"
    "5. 细则可以用 get_policy 查询，主题：" + ", ".join(POLICY_TOPICS) + "。\n"
    "6. 全部完成后用一句话说明你做了什么，不要再输出工具调用。"
)

# Policy identity recorded in the frozen input manifests.
VERSION = 'minitoolrl-policy-v2-20260908'
