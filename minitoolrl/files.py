"""运行期文件读写：文件哈希与 JSON/JSONL 的唯一实现。

阶段编排、训练、评测和汇总都用这里落盘与核对文件身份；不保存张量、优化器
状态或中间计算结果。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


def now() -> str:
    """写进运行清单与阶段标记的本地时间。"""
    return time.strftime('%Y-%m-%d %H:%M:%S')


def sha256_file(path) -> str:
    """流式计算文件内容哈希，用于核对权重、输入包和评测记录。"""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding='utf-8'))


def dump_json(path, value) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return str(path)


def read_jsonl(path) -> list:
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def write_jsonl(path, rows) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    return sha256_file(path)
