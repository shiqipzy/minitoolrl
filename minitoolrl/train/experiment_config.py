"""Shared episode budget, GRPO hyperparameters and artifact serialization."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from minitoolrl.paths import CURRENT_RUN_ID

RUN_ID = CURRENT_RUN_ID

EPISODE_BUDGET = dict(max_turns=8, max_calls=12, max_new_tokens=160, max_context_tokens=3584)

RL = dict(seed=17, lr=1e-6, weight_decay=0.01, grad_clip=1.0, group_size=4, beta=0.0,
          entropy_coefficient=0.0, clip=0.2, prompts_per_iteration=2,
          temperature=1.0,
          normalization='per-trajectory policy-token mean',
          loss='on-policy clipped GRPO; no KL or entropy regularization')


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    out = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def write_jsonl(path, rows) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    return sha256_file(path)


def dump_json(path, value) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return str(path)


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding='utf-8'))


def world_hash(world) -> str:
    payload = json.dumps(world, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]
