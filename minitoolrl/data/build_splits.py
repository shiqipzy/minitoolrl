"""从种子和任务规则生成任务池，写入一个新的目录。

build_split 调用 tasks.generate_pool；后者调用 generate_instance 和 state.make_world。
本模块只生成并落盘题目、初始场景和参考调用，不执行工具或按评分筛选任务。
划分使用固定且不相交的种子区间，任务类别直接由 SPLIT_FAMILIES 指定。

例如：python -m minitoolrl.data.build_splits --splits sft val --out-dir /path/to/new_pools
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from minitoolrl.env.tasks import (
    COMP_FAMILIES, LONG_FAMILIES, TRAIN_FAMILIES, difficulty_histogram, generate_pool,
)

SEED_RANGES = {
    "train":     (0,      100000),
    "sft":       (100000, 200000),   # 只用来生成基础 SFT 轨迹，与 RL 训练池的实例不重叠
    "val":       (200000, 250000),
    "test_iid":  (300000, 350000),
    "test_comp": (900000, 930000),
    "test_long": (950000, 980000),
}
SPLIT_FAMILIES = {
    "train": TRAIN_FAMILIES, "sft": TRAIN_FAMILIES, "val": TRAIN_FAMILIES,
    "test_iid": TRAIN_FAMILIES,
    "test_comp": COMP_FAMILIES, "test_long": LONG_FAMILIES,
}


def build_split(split: str, n_per_family: int):
    """在此划分的种子区间内，为每个任务类别生成指定数量的任务。"""
    lo, hi = SEED_RANGES[split]
    return generate_pool(SPLIT_FAMILIES[split], n_per_family, lo, hi)


def write_jsonl(path, instances):
    with Path(path).open('w', encoding='utf-8', newline='\n') as handle:
        for inst in instances:
            handle.write(json.dumps(inst.to_json(), ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True, help='新的任务池输出目录')
    parser.add_argument('--splits', nargs='+', choices=tuple(SEED_RANGES),
                        default=list(SEED_RANGES), help='本次要生成的划分')
    defaults = dict(train=250, sft=750, val=40, test_iid=50, test_comp=60, test_long=60)
    for split, count in defaults.items():
        parser.add_argument('--n-' + split.replace('_', '-'), type=int, default=count,
                            help='此划分每个任务类别的数量')
    args = parser.parse_args()
    sizes = {split: getattr(args, 'n_' + split) for split in args.splits}
    if any(n <= 0 for n in sizes.values()):
        parser.error('每个选中划分的任务数量必须大于零')
    args.out_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(seed_ranges={split: SEED_RANGES[split] for split in sizes},
                    sizes={}, per_family={}, difficulty={})
    for split, count in sizes.items():
        instances = build_split(split, count)
        write_jsonl(args.out_dir / f'{split}.jsonl', instances)
        manifest['sizes'][split] = len(instances)
        manifest['per_family'][split] = {family: count for family in SPLIT_FAMILIES[split]}
        manifest['difficulty'][split] = difficulty_histogram(instances)
        print(f'{split}: {len(instances)} 条任务，每类 {count} 条')
    (args.out_dir / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('已写入', args.out_dir)


if __name__ == '__main__':
    main()
