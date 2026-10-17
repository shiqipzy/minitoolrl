"""Read update logs and compare paired training prefixes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def load_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def training_rows(path: Path) -> Dict[int, dict]:
    result = {}
    for row in load_rows(path):
        if row.get('stage') != 'train' or 'step' not in row:
            continue
        step = int(row['step'])
        if step in result:
            raise ValueError('duplicate update in ' + str(path))
        result[step] = row
    return result


def loss_delta(rows_a: Dict[int, dict], rows_b: Dict[int, dict]) -> Optional[dict]:
    common = sorted(set(rows_a) & set(rows_b))
    if not common:
        return None
    abs_d = [abs(float(rows_a[s]['loss']) - float(rows_b[s]['loss'])) for s in common]
    rel_d = [d / max(1e-9, abs(float(rows_a[s]['loss']))) for s, d in zip(common, abs_d)]
    norm_d = [abs(float(rows_a[s].get('grad_norm', 0.0)) - float(rows_b[s].get('grad_norm', 0.0))) for s in common]
    return dict(steps_compared=len(common), first_step=common[0], last_step=common[-1],
                loss_abs_delta_max=round(max(abs_d), 6), loss_abs_delta_mean=round(sum(abs_d) / len(abs_d), 6),
                loss_rel_delta_max=round(max(rel_d), 6), grad_norm_abs_delta_max=round(max(norm_d), 6))
