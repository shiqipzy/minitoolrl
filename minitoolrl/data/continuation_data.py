"""三组续训的数据安排、通用评测题目和退款单、信用补偿统计。

C_mixed 与 U_mixed 的更新顺序在这里生成；C_blocked 直接使用文件顺序。
通用评测包括同一场景的三种需求表述和六类基础任务。
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence, Tuple

from minitoolrl.data.decomposition_tasks import ALL_VISIBLE, world_hash as raw_world_hash
from minitoolrl.env import policy as P
from minitoolrl.env.tasks import TaskInstance, _base_negatives, _call, difficulty_of

TRAIN_SEED = 17
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_DRAWS = 10_000

PROMPT_A = (
    '客户{name}（{cid}）的订单 {oid} 需要全额退款，'
    '另外按补偿政策再给一笔店铺信用，最后把订单状态置为 refunded。商品不退回仓库。'
)
PROMPT_B = (
    '订单 {oid} 的商品有严重问题，客户{name}（{cid}）不退货但要求全额退款，'
    '公司同意再按补偿政策给他等级对应的店铺信用。请办理全额退款、发放信用，并把订单状态改成 refunded。'
)
PROMPT_P = (
    '请处理订单 {oid}：将该订单全款退还客户{name}（{cid}），'
    '同时依据其等级适用的补偿政策向账户发放店铺信用，并将订单标为 refunded；商品不退回。'
)
VARIANT_PROMPTS = {'A': PROMPT_A, 'B': PROMPT_B, 'P': PROMPT_P}

REPLAY_FAMS = [
    'refund_restock',
    'cancel_unshipped',
    'partial_refund_notify',
    'inventory_correction',
    'stuck_reservation',
    'tier_compensation',
]
U_FAMILY_ORDER = list(REPLAY_FAMS)
EXTRA_PAIRS = [
    ('refund_restock', 'cancel_unshipped'),
    ('partial_refund_notify', 'inventory_correction'),
    ('stuck_reservation', 'tier_compensation'),
]
TIERS = ('regular', 'silver', 'gold')
BOUNDS = ('capped', 'uncapped')
CELLS = [(t, b) for t in TIERS for b in BOUNDS]

C_POSITIONS = 1152
C_UPDATES_PER_EPOCH = 72
C_MICROBATCH = 2
C_GRAD_ACCUM = 8
C_EFFECTIVE_BATCH = 16
U_PER_FAM = 216
U_PER_EPOCH = 1296
U_UPDATES_PER_EPOCH = 72
U_MICROBATCH = 2
U_GRAD_ACCUM = 9
U_EFFECTIVE_BATCH = 18
N_EPOCHS = 2


P_REQUIRED_MARKERS = ('全款退还', '补偿政策', '店铺信用', 'refunded', '商品不退回')
P_FORBIDDEN_MARKERS = (
    '工单', '通知', '回补库存', '计算公式', '第一步', '第二步',
    '2%', '5%', '10%', '比例 //', 'weak skill', '操作计划',
)
SHARED_REQUIRED = ('退', '补偿政策', '店铺信用', 'refunded')
SHARED_FORBIDDEN = ('工单', '通知', '回补库存', '计算公式', '第一步')

WORLD_HASH_DROP = ('_seed', 'task_id', 'prompt')


def canonical_world(world: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in world.items() if k not in WORLD_HASH_DROP}


def canonical_world_hash(world: Dict[str, Any]) -> str:
    payload = json.dumps(canonical_world(world), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


def world_identities(world: Dict[str, Any]) -> Dict[str, str]:
    return dict(raw=raw_world_hash(world), canonical=canonical_world_hash(world))


def cell_of_order(world: Dict[str, Any], order: Dict[str, Any]) -> Tuple[str, str]:
    cust = world['customers'][order['customer_id']]
    amount = P.tier_credit_cents(order['total_cents'], cust['tier'])
    bound = 'capped' if amount == P.TIER_CREDIT_CAP_CENTS[cust['tier']] else 'uncapped'
    return cust['tier'], bound


def render_variant(variant: str, order: Dict[str, Any], customer: Dict[str, Any]) -> str:
    if variant not in VARIANT_PROMPTS:
        raise ValueError(variant)
    return VARIANT_PROMPTS[variant].format(
        oid=order['order_id'], name=customer['name'], cid=customer['customer_id'])


def refund_with_credit_payload(world: Dict[str, Any], order: Dict[str, Any]) -> Dict[str, Any]:
    cust = world['customers'][order['customer_id']]
    credit = P.tier_credit_cents(order['total_cents'], cust['tier'])
    calls = [
        _call('get_order', order_id=order['order_id']),
        _call('get_customer', customer_id=cust['customer_id']),
        _call('get_policy', topic='compensation'),
        _call('create_refund', order_id=order['order_id'], amount_cents=order['total_cents'],
              reason='商品严重问题全额退款'),
        _call('apply_credit', customer_id=cust['customer_id'], amount_cents=credit,
              reason=f'订单 {order["order_id"]} 额外补偿'),
        _call('update_order_status', order_id=order['order_id'], status='refunded'),
    ]
    asserts = [
        {'kind': 'refunds_exact', 'expected': [{'order_id': order['order_id'],
                                               'amount_cents': order['total_cents']}]},
        {'kind': 'order_status', 'order_id': order['order_id'], 'status': 'refunded'},
        {'kind': 'order_refunded_cents', 'order_id': order['order_id'],
         'value': order['total_cents']},
        {'kind': 'credits_exact', 'expected': [{'customer_id': cust['customer_id'],
                                               'amount_cents': credit}]},
        {'kind': 'credit_balance', 'customer_id': cust['customer_id'], 'value': credit},
        {'kind': 'tickets_exact', 'expected': []},
        {'kind': 'notifications_exact', 'expected': []},
    ] + _base_negatives([order['order_id']], [])
    tier, bound = cell_of_order(world, order)
    return dict(calls=calls, assertions=asserts, credit=credit, customer=cust,
                tier=tier, bound=bound)


def eligible_target_orders(world: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for order in sorted(world['orders'].values(), key=lambda o: o['order_id']):
        if order['status'] != 'delivered':
            continue
        cust = world['customers'][order['customer_id']]
        if P.tier_credit_cents(order['total_cents'], cust['tier']) <= 0:
            continue
        out.append(order)
    return out


def build_target_variants(world: Dict[str, Any], order: Dict[str, Any], *,
                          seed: int, world_index: int) -> List[TaskInstance]:
    payload = refund_with_credit_payload(world, order)
    cust = payload['customer']
    ids = world_identities(world)
    insts = []
    for variant in ('A', 'B', 'P'):
        prompt = render_variant(variant, order, cust)
        params = dict(
            order_id=order['order_id'], customer_id=cust['customer_id'],
            tier=payload['tier'], bound=payload['bound'],
            credit_cents=payload['credit'], variant=variant,
            world_id=ids['canonical'], raw_world_id=ids['raw'],
            world_index=world_index,
        )
        inst = TaskInstance(
            task_id=f'refund_with_credit#{seed}#{variant}',
            family='refund_with_credit', split_group='comp',
            difficulty=difficulty_of(len(payload['calls']), True, True),
            seed=seed, prompt=prompt, world=copy.deepcopy(world), params=params,
            tool_signature=list(ALL_VISIBLE),
            gold_calls=copy.deepcopy(payload['calls']),
            assertions=copy.deepcopy(payload['assertions']),
        )
        insts.append(inst)
    return insts


def family_positions(records: Sequence[Dict[str, Any]]) -> Dict[str, List[int]]:
    by: Dict[str, List[int]] = defaultdict(list)
    for rec in records:
        pos = rec['train_position']
        if pos < 576:
            by[rec['family']].append(pos)
    for fam in REPLAY_FAMS:
        by[fam].sort()
        if len(by[fam]) != 96:
            raise RuntimeError(f'{fam} replay count {len(by[fam])}')
    return dict(by)


def decomposition_blocks(records: Sequence[Dict[str, Any]]) -> List[List[int]]:
    """按训练位置取出 144 个分解任务组，每组两个补偿工单、两个部分退款通知。"""
    decomposition_positions = sorted(r['train_position'] for r in records if r['train_position'] >= 576)
    if decomposition_positions != list(range(576, C_POSITIONS)):
        raise RuntimeError('分解任务的训练位置必须为 576..1151')
    blocks = [list(range(576 + 4 * i, 576 + 4 * (i + 1))) for i in range(144)]
    by_pos = {r['train_position']: r for r in records}
    for block in blocks:
        mods = [tuple(by_pos[p].get('modules') or []) for p in block]
        if mods != [('K', 'T'), ('K', 'T'), ('R', 'N'), ('R', 'N')]:
            raise RuntimeError(f'block modules {mods} at {block}')
    return blocks


def _microbatches(positions: Sequence[int], microbatch: int) -> List[Dict[str, Any]]:
    if len(positions) % microbatch != 0:
        raise RuntimeError('positions not divisible by microbatch')
    out = []
    for i in range(0, len(positions), microbatch):
        chunk = list(positions[i:i + microbatch])
        out.append(dict(microbatch=len(out), positions=chunk))
    return out


def build_c_mixed_schedule(records: Sequence[Dict[str, Any]], *,
                           seed: int = TRAIN_SEED) -> Dict[str, Any]:
    if len(records) != C_POSITIONS:
        raise RuntimeError(f'C records {len(records)} != {C_POSITIONS}')
    fam_pos = family_positions(records)
    blocks = decomposition_blocks(records)
    epochs = []
    for epoch in range(N_EPOCHS):
        fam_base, fam_extra = {}, {}
        for fam, poss in fam_pos.items():
            rng = random.Random(f'{seed}|{epoch}|replay|{fam}')
            shuffled = list(poss)
            rng.shuffle(shuffled)
            fam_base[fam] = shuffled[:72]
            fam_extra[fam] = shuffled[72:]
        # 随机种子字符串保留输入包使用的原值，确保生成相同的分解任务顺序。
        rng_b = random.Random(f'{seed}|{epoch}|controlled_blocks')
        block_order = list(range(144))
        rng_b.shuffle(block_order)
        updates = []
        used = []
        for u in range(C_UPDATES_PER_EPOCH):
            pair = EXTRA_PAIRS[u % 3]
            extra_i = u // 3
            replay = [fam_base[fam][u] for fam in REPLAY_FAMS]
            replay.append(fam_extra[pair[0]][extra_i])
            replay.append(fam_extra[pair[1]][extra_i])
            b0 = blocks[block_order[2 * u]]
            b1 = blocks[block_order[2 * u + 1]]
            positions = list(replay) + list(b0) + list(b1)
            used.extend(positions)
            recs = {r['train_position']: r for r in records}
            updates.append(dict(
                update=u,
                global_update=epoch * C_UPDATES_PER_EPOCH + u + 1,
                positions=positions,
                microbatches=_microbatches(positions, C_MICROBATCH),
                replay_families=list(REPLAY_FAMS) + list(pair),
                extra_pair=list(pair),
                occurrence_ids=[recs[p].get('occurrence_id') for p in positions],
                families=[recs[p].get('family') for p in positions],
            ))
        epochs.append(dict(epoch=epoch, updates=updates, permutation=used))
    return dict(
        recipe='C_mixed', seed=seed, epochs=N_EPOCHS,
        updates_per_epoch=C_UPDATES_PER_EPOCH, microbatch=C_MICROBATCH,
        grad_accum=C_GRAD_ACCUM, effective_batch=C_EFFECTIVE_BATCH,
        n_positions=C_POSITIONS, extra_pairs=EXTRA_PAIRS,
        keep_duplicates=True, length_bucket_shuffle=False,
        epochs_plan=epochs,
    )


def build_u_mixed_schedule(*, seed: int = TRAIN_SEED) -> Dict[str, Any]:
    epochs = []
    for epoch in range(N_EPOCHS):
        fam_order = {}
        for i, fam in enumerate(U_FAMILY_ORDER):
            rng = random.Random(f'{seed}|{epoch}|U|{fam}')
            poss = list(range(i * U_PER_FAM, (i + 1) * U_PER_FAM))
            rng.shuffle(poss)
            fam_order[fam] = poss
        updates = []
        used = []
        for u in range(U_UPDATES_PER_EPOCH):
            positions = []
            for fam in U_FAMILY_ORDER:
                positions.extend(fam_order[fam][3 * u:3 * u + 3])
            used.extend(positions)
            updates.append(dict(
                update=u,
                global_update=epoch * U_UPDATES_PER_EPOCH + u + 1,
                positions=positions,
                microbatches=_microbatches(positions, U_MICROBATCH),
                families=U_FAMILY_ORDER,
                per_family=3,
            ))
        epochs.append(dict(epoch=epoch, updates=updates, permutation=used))
    return dict(
        recipe='U_mixed', seed=seed, epochs=N_EPOCHS,
        updates_per_epoch=U_UPDATES_PER_EPOCH, microbatch=U_MICROBATCH,
        grad_accum=U_GRAD_ACCUM, effective_batch=U_EFFECTIVE_BATCH,
        n_positions=U_PER_EPOCH, family_order=U_FAMILY_ORDER,
        length_bucket_shuffle=False, epochs_plan=epochs,
    )


def exact_record_match(actual: Sequence[Dict[str, Any]],
                       expected: Sequence[Dict[str, Any]],
                       keys: Sequence[str]) -> bool:
    def sig(row: Dict[str, Any]) -> Tuple:
        return tuple(row.get(k) for k in keys)

    return Counter(sig(x) for x in actual) == Counter(sig(x) for x in expected)


def finance_amount_stats(row: Dict[str, Any]) -> Dict[str, Any]:
    """Entity+amount multiset match against assertions, not 'any equal amount'."""
    assertions = row.get('assertions') or []
    final = row.get('final_world') or {}
    exp_refunds, exp_credits = [], []
    tickets_required = False
    notify_required = False
    restock_required = False
    for a in assertions:
        kind = a.get('kind')
        if kind == 'refunds_exact':
            exp_refunds = list(a.get('expected') or [])
        elif kind == 'credits_exact':
            exp_credits = list(a.get('expected') or [])
        elif kind == 'tickets_exact':
            tickets_required = bool(a.get('expected'))
        elif kind == 'notifications_exact':
            notify_required = bool(a.get('expected'))
        elif kind == 'inventory_on_hand':
            sku = a.get('sku')
            initial = ((row.get('world') or {}).get('inventory') or {}).get(sku, {})
            if sku and a.get('value') != initial.get('on_hand'):
                restock_required = True
    gold_writes = {c['name'] for c in (row.get('gold_calls') or [])}
    if 'restock_item' in gold_writes:
        restock_required = True
    refunds = list(final.get('refunds') or [])
    credits = list(final.get('credits') or [])
    tickets = list(final.get('tickets') or [])
    notifications = list(final.get('notifications') or [])
    refund_ok = exact_record_match(refunds, exp_refunds, ('order_id', 'amount_cents'))
    credit_ok = exact_record_match(credits, exp_credits, ('customer_id', 'amount_cents'))
    extra = []
    if tickets and not tickets_required:
        extra.append('open_ticket')
    if notifications and not notify_required:
        extra.append('notify_customer')
    if any(t.get('name') == 'restock_item' for turn in (row.get('trace') or [])
           for t in (turn.get('calls') or [])) and not restock_required:
        extra.append('restock_item')
    missing = []
    if exp_refunds and not refunds:
        missing.append('missing_refund')
    if exp_credits and not credits:
        missing.append('missing_credit')
    return dict(
        refund_written=bool(refunds),
        credit_written=bool(credits),
        refund_amount_ok=refund_ok,
        credit_amount_ok=credit_ok,
        extra_writes=extra,
        missing=missing,
        n_tickets=len(tickets),
        n_notifications=len(notifications),
    )


def s_orig_from_worlds(success: Dict[Tuple[str, str], int], world_ids: Sequence[str]) -> float:
    vals = []
    for wid in world_ids:
        vals.append((success[(wid, 'A')] + success[(wid, 'B')]) / 2.0)
    return sum(vals) / max(1, len(vals))


def s_new_from_worlds(success: Dict[Tuple[str, str], int], world_ids: Sequence[str]) -> float:
    vals = [success[(wid, 'P')] for wid in world_ids]
    return sum(vals) / max(1, len(vals))


def s_all3_from_worlds(success: Dict[Tuple[str, str], int], world_ids: Sequence[str]) -> float:
    vals = [min(success[(wid, 'A')], success[(wid, 'B')], success[(wid, 'P')])
            for wid in world_ids]
    return sum(vals) / max(1, len(vals))


def bootstrap_stratified_delta(
        left: Dict[str, float], right: Dict[str, float],
        worlds_by_cell: Dict[str, List[str]], *,
        draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> Dict[str, Any]:
    rng = random.Random(seed)
    cells = sorted(worlds_by_cell)
    observed_ids = [wid for cell in cells for wid in worlds_by_cell[cell]]
    if not observed_ids:
        return dict(mean=None, lo=None, hi=None, n=0, degenerate=True)
    obs = sum(left[i] - right[i] for i in observed_ids) / len(observed_ids)
    diffs = []
    for _ in range(draws):
        sample = []
        for cell in cells:
            pool = worlds_by_cell[cell]
            sample.extend(pool[rng.randrange(len(pool))] for _ in range(len(pool)))
        diffs.append(sum(left[i] - right[i] for i in sample) / len(sample))
    diffs.sort()
    return dict(
        mean=obs, lo=diffs[int(0.025 * draws)], hi=diffs[int(0.975 * draws)],
        n=len(observed_ids), draws=draws, seed=seed,
        degenerate=all(left[i] == right[i] for i in observed_ids),
    )
