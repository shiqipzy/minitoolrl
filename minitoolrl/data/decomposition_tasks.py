"""定义 C_blocked、C_mixed 使用的分解任务，并构造正式 SFT 训练轨迹。

分解任务分别要求信用补偿与部分退款，保存题目、正确工具调用和真实工具返回。
每个初始场景生成四条记录：两条“等级信用补偿＋补偿工单”，两条
“部分退款＋退款通知”。四条记录分别复制同一初始场景，各自独立执行。

C_blocked 与 C_mixed 每轮都使用 576 条分解任务轨迹和 576 条基础任务回放。
本文件定义前者；基础 SFT、基础任务回放和 U_mixed 的六类任务由 env/tasks.py
定义，参考轨迹通过 data/build_sft.py 的 build_trajectory 执行并记录。
两组 C 数据内容相同，只改变训练顺序。

阅读顺序：pick_kr_payload 选订单，build_decomposition_block 生成四个任务，
gold_and_assertions 定义正确调用和评分要求，build_record 执行调用并保存轨迹。

本文件提供函数，不启动模型训练，也不自行写入训练文件。当前输入包已经准备好，
训练入口直接读取其中的 JSONL；GRPO 使用模型现场生成的调用轨迹。
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from minitoolrl.data.build_sft import build_trajectory
from minitoolrl.env import policy as P
from minitoolrl.env import tools as T
from minitoolrl.env.session import EpisodeBudget
from minitoolrl.env.tasks import TaskInstance, _base_negatives, _call, difficulty_of, instance_key

ALL_VISIBLE = list(T.ALL_TOOL_NAMES)
EPISODE_BUDGET = dict(max_turns=8, max_calls=12, max_new_tokens=160, max_context_tokens=3584)
GOLD_BUDGET = EpisodeBudget(**EPISODE_BUDGET)

# 以下三种写操作组合留给组合评测，SFT 训练记录中不能出现。
TARGET_HELD_OUT = frozenset({'create_refund', 'apply_credit'})
WRITE_SUBSET_BANS = (
    frozenset({'create_refund', 'restock_item', 'open_ticket'}),
    frozenset({'cancel_shipment', 'create_refund', 'release_reservation'}),
)

# K=等级信用补偿，R=部分退款，T=工单，N=通知。
# 两种题目各保留两个训练位置；重复记录计入训练量，不去重。
DECOMPOSITION_SLOTS = (('K', 'T'), ('K', 'T'), ('R', 'N'), ('R', 'N'))

SENT_K = '订单 {oid} 严重延迟，请按补偿政策给客户{name}（{cid}）发放店铺信用。'
SENT_R = ('订单 {oid} 中 {qty} 件 {sku} 存在质量问题，商品不退回。'
          '请退掉这部分金额，并把订单状态改为 partially_refunded。')
SENT_T_K = '并开一条 compensation 类工单记录这件事。'
SENT_T_R = '并开一条 refund 类工单绑定该订单。'
SENT_N_K = '并用 delay_apology 模板通知下单客户。'
SENT_N_R = '并用 refund_processed 模板通知下单客户。'
SENT_REFUND_ONLY_GUARD = '不要发放补偿，不要回补库存，不要开工单。'


def world_hash(world: Dict[str, Any]) -> str:
    """给初始电商场景生成标识，便于识别共享同一场景的题目。"""
    payload = json.dumps(world, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


def gold_names(inst: TaskInstance) -> set:
    """取出任务的参考调用工具名，用于检查评测组合是否混入训练。"""
    return {c['name'] for c in inst.gold_calls}


def banned_write_subset(names: Iterable[str]) -> str:
    """返回调用中出现的留出组合，例如同一任务同时退款和发放信用。"""
    s = set(names)
    if TARGET_HELD_OUT.issubset(s):
        return 'create_refund+apply_credit'
    for ban in WRITE_SUBSET_BANS:
        if ban.issubset(s):
            return '+'.join(sorted(ban))
    return ''


def structured_requires_refund_and_credit(inst: TaskInstance) -> bool:
    """从需求字段和评分要求检查是否同时要求退款、信用，防止参考调用漏写。"""
    mods = list(inst.params.get('modules') or [])
    if 'K' in mods and 'R' in mods:
        return True
    has_refund = any(a.get('kind') == 'refunds_exact' and a.get('expected') for a in inst.assertions)
    has_credit = any(a.get('kind') == 'credits_exact' and a.get('expected') for a in inst.assertions)
    req = inst.params.get('requirements') or {}
    if req.get('credit') and req.get('partial_refund'):
        return True
    return bool(has_refund and has_credit)


def held_out_reason(inst: TaskInstance, record: Optional[Dict[str, Any]] = None) -> str:
    """检查留给组合评测的操作组合是否进入 SFT，返回原因或空字符串。

    分别核对参考调用、题目需求和参与训练的调用，避免某处漏写导致检查失效。
    recovery 标记的故意错误调用不参与 SFT 损失，因此不计入训练操作组合。
    """
    reason = banned_write_subset(gold_names(inst))
    if reason:
        return 'gold:' + reason
    if structured_requires_refund_and_credit(inst):
        return 'requirements:refund+credit'
    if record is not None:
        loss_names = []
        for turn in record.get('turns') or []:
            if turn.get('recovery'):
                continue
            for call in turn.get('calls') or []:
                loss_names.append(call['name'])
        reason = banned_write_subset(loss_names)
        if reason:
            return 'loss_calls:' + reason
    return ''


def cell_of_credit(total_cents: int, tier: str) -> Tuple[str, str]:
    """按客户等级、补偿是否达到金额上限分组，供数据数量统计。"""
    amount = P.tier_credit_cents(total_cents, tier)
    bound = 'capped' if amount == P.TIER_CREDIT_CAP_CENTS[tier] else 'uncapped'
    return tier, bound


def pick_kr_payload(world: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """选一个既能出信用补偿题、又能出部分退款题的订单。

    两种题使用同一订单事实；部分退款金额必须大于零且小于订单总额。
    返回订单、客户、补偿金额、退款商品与数量，以及操作前的库存。
    """
    orders = sorted(world['orders'].values(), key=lambda o: o['order_id'])
    for order in orders:
        if order['status'] not in ('shipped', 'delivered') or not order['items']:
            continue
        cust = world['customers'][order['customer_id']]
        credit = P.tier_credit_cents(order['total_cents'], cust['tier'])
        if credit <= 0:
            continue
        partial = None
        for item in sorted(order['items'], key=lambda it: it['sku']):
            for qty in range(1, item['quantity'] + 1):
                amount = qty * item['unit_price_cents']
                if 0 < amount < order['total_cents']:
                    partial = dict(item=item, qty=qty, amount=amount)
                    break
            if partial:
                break
        if partial is None:
            continue
        tier, bound = cell_of_credit(order['total_cents'], cust['tier'])
        return dict(
            order=order, customer=cust, credit=credit, tier=tier, bound=bound,
            sku=partial['item']['sku'], quantity=partial['qty'],
            amount_cents=partial['amount'],
            on_hand=world['inventory'][partial['item']['sku']]['on_hand'],
            reserved=world['inventory'][partial['item']['sku']]['reserved'],
        )
    return None


def _credit_invariants(world: Dict[str, Any], cid: str, value: int) -> List[Dict[str, Any]]:
    """核对目标客户的最终信用余额，并要求其他客户余额不变。"""
    out = []
    for other, cust in sorted(world['customers'].items()):
        expected = value if other == cid else cust['credit_balance_cents']
        out.append({'kind': 'credit_balance', 'customer_id': other, 'value': expected})
    return out


def _reserved_invariants(world: Dict[str, Any]) -> List[Dict[str, Any]]:
    """信用补偿和不退货的部分退款都不能改变商品预留库存。"""
    return [{'kind': 'inventory_reserved', 'sku': sku, 'value': inv['reserved']}
            for sku, inv in sorted(world['inventory'].items())]


def render_prompt(modules: Sequence[str], payload: Dict[str, Any], *,
                  refund_only_guard: bool = False) -> str:
    """按信用补偿或部分退款需求生成题面，明确是否要求工单、通知。"""
    order = payload['order']
    cust = payload['customer']
    oid, cid = order['order_id'], cust['customer_id']
    parts: List[str] = []
    if 'K' in modules:
        parts.append(SENT_K.format(oid=oid, name=cust['name'], cid=cid))
    elif 'R' in modules:
        parts.append(SENT_R.format(oid=oid, qty=payload['quantity'], sku=payload['sku']))
    else:
        raise ValueError('modules must include K or R')
    if 'T' in modules:
        parts.append(SENT_T_K if 'K' in modules else SENT_T_R)
    if 'N' in modules:
        parts.append(SENT_N_K if 'K' in modules else SENT_N_R)
    if refund_only_guard:
        parts.append(SENT_REFUND_ONLY_GUARD)
    return ''.join(parts)


def gold_and_assertions(modules: Sequence[str], payload: Dict[str, Any],
                        world: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """返回参考工具调用及最终后台状态的评分要求。

    信用补偿按客户等级计算金额，并保持订单、退款单和库存不变。
    部分退款按指定商品数量计算金额，生成一张退款单并修改订单状态；
    商品不退回，库存不变。工单、通知只在题目明确要求时生成。
    """
    order = payload['order']
    cust = payload['customer']
    oid, cid = order['order_id'], cust['customer_id']
    sku = payload['sku']
    calls: List[Dict[str, Any]] = []
    asserts: List[Dict[str, Any]] = []
    if 'K' in modules:
        credit = payload['credit']
        calls.extend([
            _call('get_order', order_id=oid),
            _call('get_customer', customer_id=cid),
            _call('get_policy', topic='compensation'),
            _call('apply_credit', customer_id=cid, amount_cents=credit,
                  reason=f'订单 {oid} 延迟补偿'),
        ])
        if 'T' in modules:
            calls.append(_call('open_ticket', category='compensation', order_id=oid,
                               note=f'已按等级政策补偿 {credit} 分'))
        if 'N' in modules:
            calls.append(_call('notify_customer', customer_id=cid, template='delay_apology',
                               order_id=oid))
        asserts.extend([
            {'kind': 'credits_exact', 'expected': [{'customer_id': cid, 'amount_cents': credit}]},
            {'kind': 'refunds_exact', 'expected': []},
            {'kind': 'order_status', 'order_id': oid, 'status': order['status']},
            {'kind': 'order_refunded_cents', 'order_id': oid, 'value': order.get('refunded_cents', 0)},
            {'kind': 'inventory_on_hand', 'sku': sku, 'value': payload['on_hand']},
        ])
        asserts.extend(_credit_invariants(world, cid, credit))
        if 'T' in modules:
            asserts.append({'kind': 'tickets_exact',
                            'expected': [{'category': 'compensation', 'order_id': oid}]})
        else:
            asserts.append({'kind': 'tickets_exact', 'expected': []})
        if 'N' in modules:
            asserts.append({'kind': 'notifications_exact',
                            'expected': [{'customer_id': cid, 'template': 'delay_apology',
                                          'order_id': oid}]})
        else:
            asserts.append({'kind': 'notifications_exact', 'expected': []})
        asserts.extend(_reserved_invariants(world))
        asserts.extend(_base_negatives([], []))
    elif 'R' in modules:
        amount = payload['amount_cents']
        calls.extend([
            _call('get_order', order_id=oid),
            _call('create_refund', order_id=oid, amount_cents=amount, reason='部分商品质量问题'),
            _call('update_order_status', order_id=oid, status='partially_refunded'),
        ])
        if 'T' in modules:
            calls.append(_call('open_ticket', category='refund', order_id=oid,
                               note=f'部分退款 {amount} 分'))
        if 'N' in modules:
            calls.append(_call('notify_customer', customer_id=cid, template='refund_processed',
                               order_id=oid))
        asserts.extend([
            {'kind': 'refunds_exact', 'expected': [{'order_id': oid, 'amount_cents': amount}]},
            {'kind': 'order_status', 'order_id': oid, 'status': 'partially_refunded'},
            {'kind': 'order_refunded_cents', 'order_id': oid, 'value': amount},
            {'kind': 'credits_exact', 'expected': []},
            {'kind': 'inventory_on_hand', 'sku': sku, 'value': payload['on_hand']},
        ])
        asserts.extend(_credit_invariants(world, cid, cust['credit_balance_cents']))
        if 'T' in modules:
            asserts.append({'kind': 'tickets_exact',
                            'expected': [{'category': 'refund', 'order_id': oid}]})
        else:
            asserts.append({'kind': 'tickets_exact', 'expected': []})
        if 'N' in modules:
            asserts.append({'kind': 'notifications_exact',
                            'expected': [{'customer_id': cid, 'template': 'refund_processed',
                                          'order_id': oid}]})
        else:
            asserts.append({'kind': 'notifications_exact', 'expected': []})
        asserts.extend(_reserved_invariants(world))
        asserts.extend(_base_negatives([oid], []))
    else:
        raise ValueError(modules)
    return calls, asserts


def source_instance_key(prompt: str, calls: List[Dict[str, Any]],
                        assertions: List[Dict[str, Any]], world: Dict[str, Any]) -> str:
    """按题目、调用、评分和场景识别内容；相同内容的重复训练位置共享此标识。"""
    payload = json.dumps({'prompt': prompt, 'calls': calls, 'assertions': assertions,
                          'world': world}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]


def build_decomposition_instance(
        world: Dict[str, Any], payload: Dict[str, Any], modules: Sequence[str], *,
        block_id: str, occurrence_id: int, seed: int) -> TaskInstance:
    """把选定订单的一个需求装成任务，并复制初始场景。

    gold_calls 保存正确调用，assertions 保存评分要求；此时还没有工具返回。
    block_id 标识四记录组，occurrence_id 区分组内四个训练位置。
    """
    modules = tuple(modules)
    prompt = render_prompt(modules, payload)
    calls, asserts = gold_and_assertions(modules, payload, world)
    src = source_instance_key(prompt, calls, asserts, world)
    family = 'debinding_credit' if 'K' in modules else 'debinding_partial_refund'
    order = payload['order']
    cust = payload['customer']
    params = dict(
        order_id=order['order_id'], customer_id=cust['customer_id'],
        sku=payload['sku'], quantity=payload['quantity'],
        amount_cents=payload['amount_cents'], credit_cents=payload['credit'],
        tier=payload['tier'], bound=payload['bound'],
        modules=list(modules), arm='C', block_id=block_id,
        occurrence_id=occurrence_id, source_instance_key=src,
        world_id=world_hash(world),
        requirements=dict(
            credit='K' in modules, partial_refund='R' in modules,
            ticket={'category': 'compensation' if 'K' in modules else 'refund'} if 'T' in modules else None,
            notify={'template': 'delay_apology' if 'K' in modules else 'refund_processed'} if 'N' in modules else None,
        ),
    )
    inst = TaskInstance(
        task_id=f'{family}#C#{block_id}#{occurrence_id}',
        family=family, split_group='train',
        difficulty=difficulty_of(len(calls), True, 'K' in modules),
        seed=seed, prompt=prompt, world=copy.deepcopy(world), params=params,
        tool_signature=list(ALL_VISIBLE), gold_calls=calls, assertions=asserts,
    )
    return inst


def module_counts(insts: Sequence[TaskInstance]) -> Counter:
    """统计信用补偿、部分退款、工单、通知分别出现多少次。"""
    c: Counter = Counter()
    for inst in insts:
        for m in inst.params.get('modules') or []:
            c[m] += 1
    return c


def build_record(inst: TaskInstance) -> Dict[str, Any]:
    """实际执行参考调用，将真实工具返回保存为正式 SFT 训练记录。

    每轮一个调用，保存到 turns 的 observations；SFT 读取这些已保存的文本。
    构造器处理执行预算；这里另限定训练中使用的操作组合。
    """
    built = build_trajectory(inst, calls_per_turn=1, budget=GOLD_BUDGET,
                            add_final_turn=True, visible_tools=ALL_VISIBLE)
    if not built['ok']:
        return built
    rec = built['record']
    rec['world_id'] = world_hash(inst.world)
    rec['params'] = inst.params
    rec['source_instance_key'] = inst.params.get('source_instance_key') or instance_key(inst)
    rec['occurrence_id'] = inst.params.get('occurrence_id', 0)
    rec['block_id'] = inst.params.get('block_id')
    rec['arm'] = inst.params.get('arm')
    rec['modules'] = list(inst.params.get('modules') or [])
    rec['stratum'] = {'tier': inst.params.get('tier'), 'bound': inst.params.get('bound'),
                      'cell': f"{inst.params.get('tier')}|{inst.params.get('bound')}"}
    leak = held_out_reason(inst, rec)
    if leak:
        return dict(ok=False, reason='held_out:' + leak, record=None)
    return built


def sequence_token_stats(tokenizer, record: Dict[str, Any]) -> Dict[str, int]:
    """按 SFT 实际模板统计输入长度和学习目标长度，供 C/U 预算比较。"""
    from minitoolrl.data.sft_dataset import build_sequence
    seq = build_sequence(tokenizer, record)
    labels = [i if m else -100 for i, m in zip(seq.ids, seq.policy_mask)]
    supervised = sum(1 for x in labels[1:] if x != -100)
    return dict(n_input=len(seq.ids), n_supervised=supervised, n_ids=len(seq.ids))


def decomposition_pairs_expected(insts: Sequence[TaskInstance]) -> bool:
    """核对 C 数据始终要求信用补偿配工单、部分退款配通知，且不同时退款和补偿。"""
    for inst in insts:
        mods = set(inst.params.get('modules') or [])
        names = gold_names(inst)
        if 'K' in mods:
            if 'open_ticket' not in names or 'T' not in mods:
                return False
            if 'create_refund' in names:
                return False
        if 'R' in mods:
            if 'notify_customer' not in names or 'N' not in mods:
                return False
            if 'apply_credit' in names:
                return False
    return True


def build_decomposition_block(world: Dict[str, Any], payload: Dict[str, Any], *,
                            block_id: str, seed: int) -> List[TaskInstance]:
    """从同一订单生成两个补偿工单任务、两个部分退款通知任务。

    每条都有独立的初始场景副本。相同题目各出现两次是正式训练安排，
    此函数返回四个任务；调用 build_record 后才得到带工具返回的训练记录。
    """
    return [build_decomposition_instance(
        world, payload, modules,
        block_id=block_id, occurrence_id=occ, seed=seed)
        for occ, modules in enumerate(DECOMPOSITION_SLOTS)]
