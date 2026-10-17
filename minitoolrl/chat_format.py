"""SFT 和模型执行共用的聊天文本、token 序列与监督位置。

ChatFormat 按 tokenizer 模板渲染各段；MultiTurnSequence 统一编码、追加 token 和监督标记。
模型生成的 token 原样保留，工具返回作为上下文，不搜索返回文字来定位学习目标。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence


class ChatFormat:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if not tokenizer.chat_template:
            raise ValueError('The checkpoint tokenizer must include its official chat template')

    def _render(self, messages, *, tools=None, generation=False):
        return self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=generation,
            enable_thinking=False)

    def render_model_input(self, messages, tools):
        """渲染系统规则、用户题目和工具说明，并追加模型续写所需的助手开头。"""
        return self._render(messages, tools=tools, generation=True)

    def render_assistant_output(self, *, calls=None, content=''):
        """渲染真实的工具调用或回复，返回模型需要续写的正文及 EOS。"""
        message = dict(role='assistant', content=content, reasoning_content='')
        if calls:
            message['tool_calls'] = calls
        # 用官方模板的同一用户前缀求出待生成部分，不复制 Jinja 中的工具格式。
        prompt = [dict(role='user', content='')]
        prefix = self._render(prompt, generation=True)
        complete = self._render([*prompt, message])
        if not complete.startswith(prefix):
            raise ValueError('Chat template does not support Qwen3 non-thinking continuation')
        body = complete[len(prefix):]
        ending = self.tokenizer.eos_token + '\n'
        if not body.endswith(ending):
            raise ValueError('Expected the Qwen3 assistant EOS and message separator')
        # generate 在 EOS 停止；其后的换行由下一轮上下文追加，不参与损失。
        return body[:-1]

    def render_tool_results(self, observations, assistant_closed):
        """追加工具返回和下一轮 assistant 开头；截断的调用先补结束符。"""
        messages = [dict(role='tool', content=obs) for obs in observations]
        if not messages:
            raise ValueError('A continuing tool turn must include an observation')
        separator = '\n' if assistant_closed else self.tokenizer.eos_token + '\n'
        return separator + self._render(messages, generation=True)


@dataclass
class MultiTurnSequence:
    """保存多轮对话的 token、监督 mask 和计数；mask=1 的位置参与训练损失。"""
    tokenizer: Any
    ids: List[int] = field(default_factory=list)
    policy_mask: List[int] = field(default_factory=list)
    n_unsup_tokens: int = 0          # 纠错轨迹中的错误调用只进上下文
    n_prompt_tokens: int = 0
    n_policy_tokens: int = 0
    n_obs_tokens: int = 0

    def __post_init__(self):
        self.chat = ChatFormat(self.tokenizer)

    def _encode(self, s: str) -> List[int]:
        return self.tokenizer(s, add_special_tokens=False).input_ids

    def set_prompt(self, messages, tools) -> None:
        """把系统规则、用户题目和工具说明渲染、编码为初始上下文。"""
        text = self.chat.render_model_input(messages, tools)
        self.ids = self._encode(text)
        self.policy_mask = [0] * len(self.ids)
        self.n_prompt_tokens = len(self.ids)

    def append_assistant(self, *, calls=None, content='', supervised: bool = True) -> None:
        """把记录中的参考调用或答复渲染、编码并追加，错误调用可排除监督。"""
        text = self.chat.render_assistant_output(calls=calls, content=content)
        self.append_generated(self._encode(text), supervised=supervised)

    def append_generated(self, gen_ids: Sequence[int], supervised: bool = True) -> None:
        """保存原始生成 token；纠错轨迹的错误调用使用 supervised=False。"""
        gen_ids = list(gen_ids)
        self.ids += gen_ids
        self.policy_mask += [1 if supervised else 0] * len(gen_ids)
        if supervised:
            self.n_policy_tokens += len(gen_ids)
        else:
            self.n_unsup_tokens += len(gen_ids)

    def append_observation(self, observations: Sequence[str], assistant_closed: bool) -> None:
        """工具返回和下一轮角色开头由同一个模板生成，全部作为上下文。"""
        delta = self.chat.render_tool_results(observations, assistant_closed)
        delta_ids = self._encode(delta)
        self.ids += delta_ids
        self.policy_mask += [0] * len(delta_ids)
        self.n_obs_tokens += len(delta_ids)

    def stats(self) -> Dict[str, int]:
        return {"n_tokens": len(self.ids), "n_prompt": self.n_prompt_tokens,
                "n_policy": self.n_policy_tokens, "n_obs": self.n_obs_tokens,
                "n_unsup": self.n_unsup_tokens}
