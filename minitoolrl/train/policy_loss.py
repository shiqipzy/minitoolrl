"""多轮序列的 padding、逐 token 对数概率和策略位置对齐。

当前 GRPO 目标在 grpo_loss.py 中；这里只负责给已有序列计算概率。
Qwen3-0.6B 是 dense 模型，不需要 MoE 辅助损失。
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Sequence

import torch
import torch.nn.functional as F


def pad_batch(id_lists: Sequence[Sequence[int]], mask_lists: Sequence[Sequence[int]],
              pad_id: int, device: str = "cpu"):
    """右 padding。返回 ids[B,L]、attn[B,L]、policy_mask[B,L]。"""
    n = max(len(x) for x in id_lists)
    b = len(id_lists)
    ids = torch.full((b, n), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((b, n), dtype=torch.long, device=device)
    pmask = torch.zeros((b, n), dtype=torch.float, device=device)
    for i, (x, m) in enumerate(zip(id_lists, mask_lists)):
        L = len(x)
        ids[i, :L] = torch.tensor(list(x), dtype=torch.long, device=device)
        attn[i, :L] = 1
        pmask[i, :L] = torch.tensor(list(m), dtype=torch.float, device=device)
    return ids, attn, pmask


def per_token_logps(model, ids: torch.Tensor, attn: torch.Tensor,
                    autocast_ctx=None, chunk: int = 0, policy_mask=None):
    """给已有序列打分，返回 [B, L-1] 的对数概率，不生成新 token。

    ids 是 [B, L] 的 token ID；结果位置 t 对应 ids[t+1] 的条件 log 概率。
    chunk 控制一次前向处理几条序列。Qwen 提供 policy_mask 时只在策略位置
    计算词表投影，其余输出位置填 0；这些位置仍作为上下文参与 Transformer。
    """
    ctx = autocast_ctx if autocast_ctx is not None else nullcontext()
    log_probs_parts = []
    batch_size = ids.size(0)
    micro_batch_size = chunk if chunk and chunk > 0 else batch_size
    for start in range(0, batch_size, micro_batch_size):
        batch_ids = ids[start:start + micro_batch_size]
        batch_attn = attn[start:start + micro_batch_size]
        if getattr(getattr(model, "config", None), "model_type", "") == "qwen3" and policy_mask is not None:
            with ctx:
                hidden = model.model(batch_ids, attention_mask=batch_attn,
                                     use_cache=False).last_hidden_state[:, :-1]
                # 位置 t 的 hidden 预测位置 t+1 的 token，mask 和目标都向后取一位。
                keep = policy_mask[start:start + micro_batch_size, 1:].bool()
                logits = model.lm_head(hidden[keep]).float()
                targets = batch_ids[:, 1:][keep]
                # CE = -log p(实际 token)，取负后就是所需的 log 概率。
                selected = -F.cross_entropy(logits, targets, reduction="none")
            log_probs_parts.append(
                torch.zeros_like(keep, dtype=selected.dtype).masked_scatter(keep, selected))
            continue
        with ctx:
            outputs = model(batch_ids, attention_mask=batch_attn)
        logits = outputs.logits[:, :-1, :].float()
        targets = batch_ids[:, 1:]
        # 最后一维是整个词表；gather 取出实际 token 对应的 log 概率。
        log_probs_parts.append(
            torch.gather(F.log_softmax(logits, dim=-1), 2, targets.unsqueeze(-1)).squeeze(-1))
    return torch.cat(log_probs_parts, dim=0)


def shift_mask(pmask: torch.Tensor) -> torch.Tensor:
    """policy_mask 对齐到 per_token_logps 的下标：预测 ids[t+1] 用 mask[t+1]。"""
    return pmask[:, 1:]

