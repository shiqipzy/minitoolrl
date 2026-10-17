"""根据同一任务各条轨迹的二值成功奖励，计算 GRPO 组内优势。"""
from __future__ import annotations

from statistics import mean
from typing import List, Sequence


def group_advantages(rewards: Sequence[float], std_normalize: bool = True,
                     eps: float = 1e-4) -> List[float]:
    """GRPO 组内优势。全对/全错组返回全 0（没有任何梯度信号）。"""
    n = len(rewards)
    if n == 0:
        return []
    m = mean(rewards)
    centered = [r - m for r in rewards]
    if not std_normalize:
        return centered
    var = sum(c * c for c in centered) / n
    sd = var ** 0.5
    if sd < eps:
        return [0.0] * n
    return [c / sd for c in centered]


