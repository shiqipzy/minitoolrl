"""最终状态裁判：只看世界状态，不看模型说了什么。

判分语义（对模型隐藏，但模型能看到的指令里已包含全部要求）：
- 把期望的退款单、信用补偿、工单和通知记录，与最终实际生成的记录逐条核对。
  数量必须相同，顺序可以不同；只检查期望中指定的字段，未指定的 reason/note 等字段不判。
- 状态型字段（订单状态、库存、预留、运单）用精确值判。
- unchanged_* 是反例检查：除白名单外，其它订单/库存/运单一律不许动。
- success 只有在全部断言通过时才为 True；单条通过率只作诊断（方案要求二值奖励）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

# 断言名称 ->（世界状态中保存记录的列表名，失败标签使用的名称）
RECORD_KINDS = {
    "refunds_exact": ("refunds", "refund"),
    "credits_exact": ("credits", "credit"),
    "tickets_exact": ("tickets", "ticket"),
    "notifications_exact": ("notifications", "notification"),
}


def _match_records(actual: List[Dict[str, Any]], expected: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """核验退款单等条目：把期望的记录逐条匹配到最终实际生成的记录。

    expected 来自任务断言，actual 来自工具执行后的后台状态。
    两边数量必须相同，排列顺序不限；只核对期望中指定的字段。
    每条实际记录只能匹配一次，避免把同一张退款单重复计数。

    当前的多记录任务是两个不同订单分别退款，按订单号和金额核验两张退款单；
    信用补偿、工单和通知各要求零条或一条记录。
    """
    if len(actual) != len(expected):
        return False, f"条数 实际={len(actual)} 期望={len(expected)}"

    unmatched_records = list(actual)
    for expected_record in expected:
        matched = False
        for actual_record in unmatched_records:
            # 只核对断言指定的字段，例如退款单的订单号和金额。
            if all(actual_record.get(field) == value for field, value in expected_record.items()):
                # 匹配过的记录取出，避免同一条实际记录被重复使用。
                unmatched_records.remove(actual_record)
                matched = True
                break
        if not matched:
            return False, f"字段不匹配 实际={actual} 期望={expected}"
    return True, "ok"


def _cmp_dicts(initial: Dict[str, Any], final: Dict[str, Any], skip: List[str],
               label: str) -> Tuple[bool, str]:
    """跳过任务允许修改的对象，其余对象必须和初始状态一样。"""
    skip_set = set(skip)
    diffs = []
    for key, before in initial.items():
        if key in skip_set:
            continue
        after = final.get(key)
        if after != before:
            # 已确定整个对象不同，再逐字段找出改动，写进失败说明。
            # 对象被删除时 after 为 None，按空字典列出缺失字段。
            after_fields = after or {}
            changed = []
            for attribute in set(before) | set(after_fields):
                if before.get(attribute) != after_fields.get(attribute):
                    changed.append(attribute)
            diffs.append(f"{key}:{sorted(changed)}")
    for key in final:
        if key not in initial and key not in skip_set:
            diffs.append(f"{key}:新增")
    if diffs:
        return False, f"{label} 被改动 {diffs}"
    return True, "ok"


def _eval_one(initial: Dict[str, Any], final: Dict[str, Any],
              a: Dict[str, Any]) -> Tuple[bool, str, str]:
    """返回 (ok, detail, tag)。tag 用于失败归类统计。"""
    kind = a["kind"]

    if kind in RECORD_KINDS:
        bucket, noun = RECORD_KINDS[kind]
        actual = final.get(bucket, [])
        expected = a["expected"]
        ok, detail = _match_records(actual, expected)
        if ok:
            return True, "ok", ""
        if len(actual) < len(expected):
            return False, f"{kind}: {detail}", f"missing_{noun}"
        if len(actual) > len(expected):
            return False, f"{kind}: {detail}", f"extra_{noun}"
        return False, f"{kind}: {detail}", f"wrong_{noun}"

    if kind == "order_status":
        got = final["orders"].get(a["order_id"], {}).get("status")
        ok = got == a["status"]
        return ok, f"order_status[{a['order_id']}] 实际={got} 期望={a['status']}", "" if ok else "wrong_order_status"

    if kind == "order_refunded_cents":
        got = final["orders"].get(a["order_id"], {}).get("refunded_cents")
        ok = got == a["value"]
        return ok, f"refunded_cents[{a['order_id']}] 实际={got} 期望={a['value']}", "" if ok else "wrong_amount"

    if kind == "shipment_status":
        got = final["shipments"].get(a["order_id"], {}).get("status")
        ok = got == a["status"]
        return ok, f"shipment[{a['order_id']}] 实际={got} 期望={a['status']}", "" if ok else "wrong_shipment_status"

    if kind == "inventory_on_hand":
        got = final["inventory"].get(a["sku"], {}).get("on_hand")
        ok = got == a["value"]
        return ok, f"on_hand[{a['sku']}] 实际={got} 期望={a['value']}", "" if ok else "wrong_inventory"

    if kind == "inventory_reserved":
        got = final["inventory"].get(a["sku"], {}).get("reserved")
        ok = got == a["value"]
        return ok, f"reserved[{a['sku']}] 实际={got} 期望={a['value']}", "" if ok else "wrong_reservation"

    if kind == "credit_balance":
        got = final["customers"].get(a["customer_id"], {}).get("credit_balance_cents")
        ok = got == a["value"]
        return ok, f"credit_balance[{a['customer_id']}] 实际={got} 期望={a['value']}", "" if ok else "wrong_amount"

    if kind == "unchanged_orders":
        ok, detail = _cmp_dicts(initial["orders"], final["orders"], a.get("except", []), "其它订单")
        return ok, detail, "" if ok else "collateral_order"

    if kind == "unchanged_inventory":
        ok, detail = _cmp_dicts(initial["inventory"], final["inventory"], a.get("except", []), "其它库存")
        return ok, detail, "" if ok else "collateral_inventory"

    if kind == "unchanged_shipments":
        ok, detail = _cmp_dicts(initial["shipments"], final["shipments"], a.get("except", []), "其它运单")
        return ok, detail, "" if ok else "collateral_shipment"

    raise ValueError(f"未知断言类型: {kind}")


@dataclass
class GradeResult:
    success: bool
    n_pass: int
    n_total: int
    pass_ratio: float                      # 只作诊断，不参与奖励
    failures: List[Dict[str, Any]] = field(default_factory=list)
    failure_tags: List[str] = field(default_factory=list)
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {"success": self.success, "n_pass": self.n_pass, "n_total": self.n_total,
                "pass_ratio": round(self.pass_ratio, 4), "failure_tags": self.failure_tags,
                "failures": self.failures}


def grade(initial: Dict[str, Any], final: Dict[str, Any],
          assertions: List[Dict[str, Any]]) -> GradeResult:
    details, failures, tags = [], [], []
    n_pass = 0
    for a in assertions:
        ok, detail, tag = _eval_one(initial, final, a)
        details.append({"kind": a["kind"], "ok": ok, "detail": detail})
        if ok:
            n_pass += 1
        else:
            failures.append({"kind": a["kind"], "detail": detail, "tag": tag})
            if tag and tag not in tags:
                tags.append(tag)
    n_total = len(assertions)
    return GradeResult(success=(n_pass == n_total and n_total > 0), n_pass=n_pass, n_total=n_total,
                       pass_ratio=(n_pass / n_total if n_total else 0.0),
                       failures=failures, failure_tags=tags, details=details)


def mutation_summary(final: Dict[str, Any]) -> Dict[str, Any]:
    """写操作审计摘要，用于诊断（不参与判分）。"""
    by_tool: Dict[str, int] = {}
    for m in final.get("mutations", []):
        by_tool[m["tool"]] = by_tool.get(m["tool"], 0) + 1
    return {"n_mutations": len(final.get("mutations", [])), "by_tool": dict(sorted(by_tool.items()))}


if __name__ == "__main__":
    from minitoolrl.env.state import WorldState
    from minitoolrl.env import tools as T
    from minitoolrl.env.tasks import generate_instance

    inst = generate_instance("refund_restock", 0)
    st = WorldState.from_snapshot(inst.world)
    for c in inst.gold_calls:
        T.execute(st, c["name"], c["arguments"])
    r = grade(inst.world, st.snapshot(), inst.assertions)
    print("gold:", r.to_json(), mutation_summary(st.snapshot()))

    st2 = WorldState.from_snapshot(inst.world)
    for c in inst.gold_calls[:-1]:
        T.execute(st2, c["name"], c["arguments"])
    r2 = grade(inst.world, st2.snapshot(), inst.assertions)
    print("漏最后一步:", r2.success, r2.failure_tags)
