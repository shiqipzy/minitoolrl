"""读取本次主训练和工程短测的逐步曲线；不重新执行训练或采样。"""
import argparse
import json
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def grpo_rows(path):
    rows = read_rows(path)
    for row in rows:
        groups = [group['rewards'] for group in row['groups']]
        row['success_rate'] = sum(sum(rewards) for rewards in groups) / sum(map(len, groups))
        row['mixed_group_rate'] = sum(0 < sum(rewards) < len(rewards) for rewards in groups) / len(groups)
    return rows


def collect(run):
    """保留所有更新/采样迭代，包括 GRPO 没有学习信号的迭代。

    主训练与工程短测分开记录。SFT loss、GRPO 训练采样成功率和开发集任务
    成功率各有自己的曲线，不把不同阶段的数据合并成同预算组间比较。
    """
    protocol = json.loads((run / 'protocol.json').read_text(encoding='utf-8'))
    n = protocol['config']['world_size']
    training = dict(SFT=read_rows(run / 'sft/updates.jsonl'),
                    G0=grpo_rows(run / 'grpo/main/grpo_train.jsonl'))
    for tag in ('C_blocked', 'C_mixed', 'U_mixed'):
        training[tag] = read_rows(run / 'arms' / tag / f'n{n}/train/updates.jsonl')
    engineering = {}
    for size in protocol['engineering']['world_sizes']:
        engineering[f'sft_n{size}'] = read_rows(run / f'engineering/sft/n{size}/updates.jsonl')
    for deployment in ('vllm_separated',):
        engineering[f'grpo_{deployment}'] = grpo_rows(
            run / f'engineering/grpo/{deployment}/grpo_train.jsonl')
    dev = [json.loads((run / f'grpo/main/curve_{tag}.json').read_text(encoding='utf-8'))
           for tag in ('third', 'two_thirds', 'final')]
    return dict(run_id=protocol['run_id'], training=training, engineering=engineering, grpo_dev=dev)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    curves = collect(args.run_dir)
    out = args.out or args.run_dir / 'eval/train_curves.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(curves, ensure_ascii=False, indent=2), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
