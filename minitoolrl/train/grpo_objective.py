"""GRPO 目标的两块计算：组内优势与裁剪后的策略目标。

同一任务的 4 条轨迹按二值成功奖励算出组内优势；策略位置的对数概率由
train_sft.masked_logits 在损失前向中给出，再用裁剪后的比值乘优势得到要最小化的
目标。Qwen3-0.6B 是 dense 模型，不需要 MoE 辅助损失。
"""
from __future__ import annotations

from statistics import mean
from typing import List, Sequence

import torch


# ------------------------------------------------------------------ 组内优势
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


# ------------------------------------------------------------------ 策略目标
def grpo_loss(logps, old_logps, advantages, mask, epsilon=0.2):
    """on-policy 裁剪目标：按每条轨迹的策略 token 取均值，再对组内样本取均值。"""
    if logps.shape != old_logps.shape or mask.shape != logps.shape:
        raise ValueError('token shapes differ')
    if advantages.shape != (logps.shape[0],):
        raise ValueError('row shapes differ')
    adv = advantages.detach().unsqueeze(1)
    # 先按 mask 归零再取指数：prompt、工具返回和 padding 不参与。
    lp = torch.where(mask.bool(), logps, torch.zeros_like(logps))
    old = torch.where(mask.bool(), old_logps.detach(), torch.zeros_like(old_logps))
    ratio = (lp - old).exp()
    objective = torch.minimum(ratio * adv, ratio.clamp(1 - epsilon, 1 + epsilon) * adv)
    counts = mask.sum(1)
    if not bool((counts > 0).all()):
        raise ValueError('empty policy row')
    return -((objective * mask).sum(1) / counts).mean()
