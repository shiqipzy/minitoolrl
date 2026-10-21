"""核对已经准备好的输入包：版本、文件清单和文件内容身份。

本模块不生成题目、不执行工具、不构造训练轨迹。训练和打包入口调用
verify_bundle，防止使用被替换的运行输入。已有整包数据的重建见 rebuild_inputs.py。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from minitoolrl.paths import ARTIFACTS_ROOT

VERSION = 'all-restock-v1-20261011'
INPUT_PACKAGE_ID = 'all_restock_registry_20261016'
DEFAULT_ROOT = ARTIFACTS_ROOT / 'inputs'
CURRENT_MANIFEST_SHA256 = '864682a89e62af6aa0770d889670c725a20465cc00807697b8270984886ea576'
MARKER = 'input_protocol.json'
# 已有输入包内的文件名属于存储格式，源码改名不改写已有数据。
MANIFEST = 'consistent_inputs_manifest.json'
# 输入包同时包含训练记录、GRPO 任务、评测题目和更新顺序，并非全是训练轨迹。
REQUIRED_FILES = {
    'data/sft_train_v2.jsonl', 'data/sft_val_v2.jsonl', 'data/dev.jsonl',
    'data/dev_curve.jsonl', 'data/gate_spec.json', 'data/rl_train.jsonl',
    'inputs/sft_rec/sft_sft.jsonl', 'inputs/sft_rec/sft_val.jsonl',
    'inputs/c_recipe/train_C.jsonl', 'inputs/c_recipe/train_U.jsonl',
    'inputs/c_recipe/val_common.jsonl', 'inputs/c_recipe/eval_variants.jsonl',
    'inputs/c_recipe/sample_plan.json', 'inputs/c_recipe/schedules/C_mixed.json',
    'inputs/c_recipe/schedules/U_mixed.json', 'inputs/control/train_control.jsonl',
    'inputs/comp_suite/test_comp_suite.jsonl', 'inputs/comp_suite/protocol_template.json',
    'inputs/long/test_long.jsonl', 'inputs/long/analyze_long_supplement.py',
}


def sha(path):
    """计算文件内容哈希，确认输入没有被替换或修改。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_bundle(root, *, require_validation=True, expected_manifest_sha256=None):
    """供打包和启动使用：核对版本、文件身份、准备记录与 SFT 镜像。

    不读取逐条任务，也不重复执行生成时的工具调用。
    require_validation=False 用于新包写入准备记录前的文件核对。
    训练、打包与 CLI 传入固定 manifest 哈希；重建入口只核对源包自身的完整性。
    """
    root = Path(root)
    if expected_manifest_sha256 is not None and sha(root / MANIFEST) != expected_manifest_sha256:
        raise ValueError('Input bundle is not the pinned registry bundle: expected manifest SHA256 '
                         + expected_manifest_sha256)
    protocol = json.loads((root / MARKER).read_text(encoding='utf-8'))
    manifest = json.loads((root / MANIFEST).read_text(encoding='utf-8'))
    if protocol['input_version'] != VERSION or manifest['input_version'] != VERSION:
        raise ValueError('Wrong input version')
    if set(manifest['files']) != REQUIRED_FILES:
        raise ValueError('Incomplete input manifest')
    if require_validation:
        if manifest.get('validation_sha256') != sha(root / 'validation.json'):
            raise ValueError('Missing or changed completed validation')
        if not json.loads((root / 'validation.json').read_text(encoding='utf-8')).get('ok'):
            raise ValueError('Input validation did not pass')
    if manifest['protocol_sha256'] != sha(root / MARKER):
        raise ValueError('Input protocol changed')
    file_hashes = {rel: sha(root / rel) for rel in manifest['files']}
    for rel, entry in manifest['files'].items():
        if file_hashes[rel] != entry['sha256']:
            raise ValueError(f'Input hash mismatch: {rel}')
    for phase in ('train', 'val'):
        # 两个路径是同一份基础 SFT 数据的镜像；不代表要执行两段基础 SFT。
        base = 'inputs/sft_rec/' + ('sft_sft.jsonl' if phase == 'train' else 'sft_val.jsonl')
        training = f'data/sft_{phase}_v2.jsonl'
        if file_hashes[base] != file_hashes[training]:
            raise ValueError(f'SFT bundle copies disagree: {phase}')
    return dict(ok=True, input_version=VERSION, files=len(manifest['files']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT, help='当前固定 registry 输入包目录')
    parser.add_argument('--verify', action='store_true', required=True)
    args = parser.parse_args()
    result = verify_bundle(args.root, expected_manifest_sha256=CURRENT_MANIFEST_SHA256)
    print(json.dumps(dict(result, manifest_sha256=CURRENT_MANIFEST_SHA256, tool_order='registry'),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
