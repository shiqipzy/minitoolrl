"""把已落盘的 SFT 对话转换为 token 和学习目标。

对话格式来自 tokenizer 的聊天模板，与模型执行共用 MultiTurnSequence。
正确 assistant 调用和最终答复参与损失；题目、工具返回、纠错过程的错误调用
只进入上下文。mask 在追加各段时确定，不搜索内容中的角色标记。
"""
from __future__ import annotations

from typing import Any, Dict

import torch

from minitoolrl.chat_format import MultiTurnSequence
from minitoolrl.env import policy as P
from minitoolrl.env import tools as T


def build_sequence(tokenizer, record: Dict[str, Any]) -> MultiTurnSequence:
    """将一条已保存的训练轨迹转换为 token 序列，并标记监督位置。

    policy_mask=1 表示该位置参与训练损失，包括正确的 assistant
    工具调用、最终答复及其结束符；错误调用和上下文标为 0。
    """
    seq = MultiTurnSequence(tokenizer)
    seq.set_prompt(
        [{"role": "system", "content": P.OPERATING_RULES},
         {"role": "user", "content": record["prompt"]}],
        tools=T.tool_schemas(record.get("exposed_tools")))
    turns = record["turns"]
    has_final = record.get("final_line") is not None
    for i, turn in enumerate(turns):
        # 纠错轨迹中的失败调用只作上下文；后续改正调用参与训练。
        seq.append_assistant(calls=turn["calls"], supervised=not turn.get("recovery"))
        if has_final or i < len(turns) - 1:
            seq.append_observation(turn["observations"], assistant_closed=True)
    if has_final:
        seq.append_assistant(content=record["final_line"])
    return seq


def collate_sft(batch, pad_id: int):
    """用 PAD 在同批短序列末尾补齐至该批最长长度，填充位置的 labels 为 -100。"""
    n = max(x.size(0) for x, _ in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, :x.size(0)] = x
        labels[i, :y.size(0)] = y
    return ids, labels
