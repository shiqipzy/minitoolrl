"""重建已有整包数据，保留任务来源和训练顺序，写到新的独立目录。

从零生成初始 SFT 使用 build_splits.py 和 build_sft.py，不需要本模块。

调用顺序：build_bundle → rebuild_task → rebuild_record → build_trajectory。
任务构造完成后，build_trajectory 执行工具并保存返回；本模块补充任务字段、
计算训练长度并逐行写 JSONL。GRPO 和评测只保存题目与初始场景。
已准备好的输入包可直接训练；核对现有文件使用 input_bundle --verify。
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from collections import Counter
from dataclasses import fields
from pathlib import Path

from minitoolrl.data.build_sft import build_trajectory
from minitoolrl.data.input_bundle import (
    VERSION, INPUT_PACKAGE_ID, MARKER, MANIFEST, REQUIRED_FILES, sha, verify_bundle,
)
from minitoolrl.env.policy import VERSION as POLICY_VERSION
from minitoolrl.env.session import EpisodeBudget
from minitoolrl.env.tasks import (TaskInstance, difficulty_of, generate_instance,
                                 instance_key, restock_order_plan)

RESTOCK_REQUIREMENT = '请回补订单中全部可回补商品，不限定商品种类数。'
INSTANCE_FIELDS = {f.name for f in fields(TaskInstance)}


def read_rows(path):
    """读取 JSONL 中的正式任务或训练记录，每行一条。"""
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def write_json(path, value):
    """构建新输入包时保存协议、清单或校验结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def rebuild_task(row):
    """读取源任务的初始场景，为新输入包生成符合完整退款回库规则的任务。

    有场景快照就直接使用；紧凑记录只有种子时，按种子重建任务。
    退款回库的调用和最终库存都从订单商品清单推导，保持原始场景不变。
    其他任务保留源需求。此函数只在显式构建新输入包时调用。
    """
    if INSTANCE_FIELDS <= row.keys():
        inst = TaskInstance.from_json({k: copy.deepcopy(row[k]) for k in INSTANCE_FIELDS})
    else:
        inst = generate_instance(row['family'], row['seed'])
        if inst is None:
            raise ValueError(f"Cannot reconstruct task {row['task_id']}")
    if inst.family != 'refund_restock':
        return inst
    order = inst.world['orders'][inst.params['order_id']]
    restock_calls, inventory_assertions, restock_skus = restock_order_plan(inst.world, order)
    calls = [c for c in inst.gold_calls if c['name'] != 'restock_item']
    insert_at = next(i for i, c in enumerate(calls) if c['name'] == 'update_order_status')
    calls[insert_at:insert_at] = restock_calls
    inst.gold_calls = calls
    inst.assertions = [a for a in inst.assertions if a['kind'] != 'inventory_on_hand']
    inst.assertions.extend(inventory_assertions)
    for assertion in inst.assertions:
        if assertion['kind'] == 'unchanged_inventory':
            assertion['except'] = sorted(restock_skus)
    inst.params['restock_skus'] = restock_skus
    inst.difficulty = difficulty_of(len(calls), False, False)
    if RESTOCK_REQUIREMENT not in inst.prompt:
        inst.prompt += RESTOCK_REQUIREMENT
    return inst


def rebuild_record(row, inst):
    """执行新任务的完整调用，重新保存工具返回，保留源记录的纠错位置。

    调用改变后，工具返回、会话文本和长度统计也必须一起更新。
    本函数返回新记录；源记录不被修改，长度统计由 build_bundle 重新计算。
    """
    recover_at = -1
    for i, turn in enumerate(row['turns']):
        if turn.get('recovery'):
            following = row['turns'][i + 1]['calls'][0]
            recover_at = inst.gold_calls.index(following)
    built = build_trajectory(inst, calls_per_turn=1, budget=EpisodeBudget(),
                             add_final_turn=row.get('final_line') is not None,
                             recover_at=recover_at, visible_tools=row.get('exposed_tools'))
    if not built['ok']:
        raise ValueError(f"Cannot replay {inst.task_id}: {built['reason']}")
    result = copy.deepcopy(row)
    result.update(built['record'])
    result.update(inst.to_json())
    result['instance_key'] = instance_key(inst)
    result['source_instance_key'] = inst.params.get('source_instance_key') or instance_key(inst)
    result.pop('observations_changed', None)
    result.pop('token_stats', None)
    result.pop('assertion_pass_ratio', None)
    return result


def build_bundle(source, output, tokenizer_path):
    """从明确指定的源包构建独立输入，保存生成统计和文件身份。

    带 turns 的记录要重放参考操作并重新统计训练文本长度；GRPO 和评测记录
    保存任务本身，运行时由模型自主执行。既有输入包、模型和成绩都不覆盖。
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Output must be independent of the frozen source')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    source_manifest = json.loads((source / 'inputs_manifest.json').read_text(encoding='utf-8'))
    if set(source_manifest['files']) != REQUIRED_FILES:
        raise ValueError('Unexpected source input set')
    for rel, entry in source_manifest['files'].items():
        if sha(source / rel) != entry['sha256']:
            raise ValueError(f'Frozen source mismatch: {rel}')
    from transformers import AutoTokenizer
    from minitoolrl.data.decomposition_tasks import sequence_token_stats
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    output.mkdir(parents=True)
    reports, token_cache = {}, {}
    mirrors = {'inputs/sft_rec/sft_sft.jsonl': 'data/sft_train_v2.jsonl',
               'inputs/sft_rec/sft_val.jsonl': 'data/sft_val_v2.jsonl'}
    for rel in source_manifest['files']:
        if rel in mirrors:
            continue
        dest = output / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not rel.endswith('.jsonl'):
            shutil.copy2(source / rel, dest)
            continue
        rows = read_rows(source / rel)
        report = Counter(records=len(rows))
        with dest.open('w', encoding='utf-8', newline='\n') as handle:
            for row in rows:
                inst = rebuild_task(row)
                if 'turns' in row:
                    updated = rebuild_record(row, inst)
                    key = json.dumps({k: updated.get(k) for k in
                        ('prompt', 'turns', 'final_line', 'exposed_tools')}, sort_keys=True, ensure_ascii=False)
                    if key not in token_cache:
                        token_cache[key] = sequence_token_stats(tokenizer, updated)
                    updated['token_stats'] = token_cache[key]
                    if updated['token_stats']['n_input'] > 4096:
                        raise ValueError(f"Would truncate SFT: {inst.task_id}")
                    report['input_tokens'] += updated['token_stats']['n_input']
                    report['supervised_tokens'] += updated['token_stats']['n_supervised']
                    report['max_input_tokens'] = max(report['max_input_tokens'], updated['token_stats']['n_input'])
                else:
                    updated = copy.deepcopy(row)
                    updated.update(inst.to_json())
                    updated['instance_key'] = instance_key(inst)
                if inst.family == 'refund_restock':
                    report['refund_restock'] += 1
                    if len(inst.params['restock_skus']) == 3:
                        report['three_item_full_restock'] += 1
                updated['input_version'] = VERSION
                updated['policy_version'] = POLICY_VERSION
                handle.write(json.dumps(updated, ensure_ascii=False) + '\n')
        reports[rel] = dict(report)
        print(rel, dict(report), flush=True)
    for rel, original in mirrors.items():
        dest = output / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output / original, dest)
        reports[rel] = dict(reports[original])
    protocol = dict(input_version=VERSION, run_id=INPUT_PACKAGE_ID, policy_version=POLICY_VERSION,
        status='inputs_prepared_not_trained', source_run=source_manifest['run_id'],
        rule='refund_restock returns every restockable item in the requested order',
        sft='One pre-GRPO SFT stage; inputs/sft_rec mirrors the main SFT data',
        training='Start from official Qwen; do not reuse old SFT/G0/arm weights',
        evaluation='Generate fresh model trajectories after retraining; old scores are not new-version results',
        tokenizer_sha256=sha(Path(tokenizer_path) / 'tokenizer.json'),
        reports=reports)
    write_json(output / MARKER, protocol)
    manifest = dict(input_version=VERSION, protocol_sha256=sha(output / MARKER), files={
        rel: dict(bytes=(output / rel).stat().st_size, sha256=sha(output / rel))
        for rel in source_manifest['files']})
    write_json(output / MANIFEST, manifest)
    result = verify_bundle(output, require_validation=False)
    write_json(output / 'validation.json', dict(result,
        scope='input_file_identity; SFT trajectories execute during construction'))
    manifest['validation_sha256'] = sha(output / 'validation.json')
    write_json(output / MANIFEST, manifest)
    return result


def reorder_tools_bundle(source, output, tokenizer_path):
    """仅把续训记录的工具列表改为 SFT/GRPO 注册顺序，另存输入包。

    题目、世界、参考调用、工具返回及样本日程逐项保留；重新计算受上下文影响的
    token 统计。基础 SFT、GRPO 和全部评测题文件逐字节复制，便于复用已有 G0。
    """
    from transformers import AutoTokenizer
    from minitoolrl.env import tools as T
    from minitoolrl.data.decomposition_tasks import sequence_token_stats

    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Output must be independent of the frozen source')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    verify_bundle(source)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    names = list(T.ALL_TOOL_NAMES)
    changed = {'inputs/control/train_control.jsonl', 'inputs/c_recipe/train_C.jsonl',
               'inputs/c_recipe/train_U.jsonl', 'inputs/c_recipe/val_common.jsonl'}
    report, cache = {}, {}
    output.mkdir(parents=True)
    for rel in sorted(REQUIRED_FILES):
        dest = output / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if rel not in changed:
            shutil.copy2(source / rel, dest)
            assert sha(dest) == sha(source / rel)
            continue
        records = read_rows(source / rel)
        with dest.open('w', encoding='utf-8', newline='\n') as handle:
            for row in records:
                if len(row.get('exposed_tools', [])) != len(names) or set(row['exposed_tools']) != set(names):
                    raise ValueError('unexpected visible tool set: ' + row['task_id'])
                updated = copy.deepcopy(row)
                updated['exposed_tools'] = names
                key = json.dumps({k: updated.get(k) for k in
                    ('prompt', 'turns', 'final_line', 'exposed_tools')}, sort_keys=True, ensure_ascii=False)
                if key not in cache:
                    cache[key] = sequence_token_stats(tokenizer, updated)
                updated['token_stats'] = cache[key]
                if updated['token_stats']['n_input'] > 4096:
                    raise ValueError('would truncate: ' + row['task_id'])
                if {k:v for k,v in updated.items() if k not in ('exposed_tools', 'token_stats')} != {
                        k:v for k,v in row.items() if k not in ('exposed_tools', 'token_stats')}:
                    raise ValueError('task or trajectory changed')
                handle.write(json.dumps(updated, ensure_ascii=False) + '\n')
        report[rel] = dict(records=len(records), source_sha256=sha(source / rel),
                           output_sha256=sha(dest), changed_fields=['exposed_tools', 'token_stats'])
        print('REORDERED', rel, len(records), flush=True)
    protocol = json.loads((source / MARKER).read_text(encoding='utf-8'))
    protocol.update(status='inputs_prepared_not_trained', tool_order='registry',
        visible_tools=names, source_manifest_sha256=sha(source / MANIFEST),
        transform='Only continuation exposed_tools and derived token_stats change',
        training='Reuse original G0; three independent continuation runs',
        reports=report)
    write_json(output / MARKER, protocol)
    manifest = dict(input_version=VERSION, protocol_sha256=sha(output / MARKER), files={
        rel: dict(bytes=(output / rel).stat().st_size, sha256=sha(output / rel)) for rel in sorted(REQUIRED_FILES)})
    write_json(output / MANIFEST, manifest)
    result = verify_bundle(output, require_validation=False)
    write_json(output / 'validation.json', dict(result, tool_order='registry',
        scope='All task and trajectory fields preserved; only tool order and token statistics changed',
        changed_files=report, unchanged_files=sorted(REQUIRED_FILES - changed)))
    manifest['validation_sha256'] = sha(output / 'validation.json')
    write_json(output / MANIFEST, manifest)
    verify_bundle(source)
    return verify_bundle(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='源任务包目录')
    parser.add_argument('--output', type=Path, required=True, help='新的输出目录')
    parser.add_argument('--tokenizer', type=Path, required=True, help='本地 tokenizer 目录')
    parser.add_argument('--tool-order-only', choices=['registry'],
                        help='仅调整续训工具顺序，保留现有任务、轨迹及训练样本日程')
    args = parser.parse_args()
    build = reorder_tools_bundle if args.tool_order_only else build_bundle
    print(json.dumps(build(args.source, args.output, args.tokenizer),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
