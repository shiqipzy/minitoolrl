"""多轮任务调度：连接模型输出与 Episode，是项目唯一的一套多轮执行逻辑。

rollout_pool 接受任意实现了 generate(requests, logprobs=...) -> [Generation]
的采样器：生产采样后端是 vllm_backend（训练侧的 VLLMRollout、评测侧的
LocalVLLM）。
工具返回由 chat_format.MultiTurnSequence 追加进上下文，工具响应不参与策略损失。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

from minitoolrl.chat_format import MultiTurnSequence
from minitoolrl.env.session import Episode
from minitoolrl.train.vllm_backend import GenRequest, Generation


# ================================ 统一多轮驱动 ================================
@dataclass
class GroupRollout:
    """一批轨迹的 token 序列、任务回合及生成、工具调用和上下文截断计数。"""
    sequences: List[MultiTurnSequence]
    episodes: List[Episode]
    n_gen_tokens: int
    n_env_steps: int
    n_truncated_context: int
    policy_version: int = 0


def rollout_pool(sampler, tokenizer, insts, budget, *, temperature: float,
                 top_p: float = 0.95, top_k: int = 0, visible_tools=None,
                 logprobs: bool = False, seed_base: Optional[int] = None) -> GroupRollout:
    """批量执行任务列表，每个条目创建独立的世界状态和 token 序列。

    同一任务可在列表中重复出现，用于 GRPO 组内采样或重复评测。sampler 是
    VLLMRollout 或 LocalVLLM；训练采样和评测复用这一套调度。
    """
    eps = [Episode(inst, budget, visible_tools) for inst in insts]
    seqs = []
    for ep in eps:
        s = MultiTurnSequence(tokenizer)
        s.set_prompt(ep.messages, ep.tools)
        s.sampler_logprobs = [] if logprobs else None   # 与策略 token 对齐的采样概率
        seqs.append(s)

    n_gen_tokens = 0
    n_env_steps = 0
    n_ctx_trunc = 0
    policy_version = None
    while True:
        active = [i for i, ep in enumerate(eps)
                  if not ep.done and len(seqs[i].ids) < budget.max_context_tokens]
        for i, ep in enumerate(eps):
            if not ep.done and len(seqs[i].ids) >= budget.max_context_tokens:
                ep.done = True
                ep.stop_reason = "context_budget"
                n_ctx_trunc += 1
        if not active:
            break
        requests = []
        for i in active:
            room = max(1, min(budget.max_new_tokens, budget.max_context_tokens - len(seqs[i].ids)))
            requests.append(GenRequest(
                request_id=i, prompt_token_ids=list(seqs[i].ids), max_new_tokens=room,
                temperature=temperature, top_p=top_p, top_k=top_k,
                seed=None if seed_base is None else seed_base + i))
        generations = sampler.generate(requests, logprobs=logprobs)
        by_id = {g.request_id: g for g in generations}
        if len(generations) != len(active) or set(by_id) != set(active):
            raise RuntimeError('sampler returned missing, duplicate or unexpected request IDs')
        rooms = {r.request_id: r.max_new_tokens for r in requests}
        for i in active:
            gen = by_id[i]
            room = rooms[i]
            if len(gen.token_ids) > room:
                raise RuntimeError('sampler exceeded the requested token budget')
            gen_ids = list(gen.token_ids)
            if policy_version is None:
                policy_version = gen.policy_version
            n_gen_tokens += len(gen_ids)
            closed = bool(gen_ids) and gen_ids[-1] == tokenizer.eos_token_id
            seqs[i].append_generated(gen_ids)
            if seqs[i].sampler_logprobs is not None:
                lp = gen.token_logprobs
                if lp is None or len(lp) != len(gen_ids):
                    raise RuntimeError('sampler did not return one logprob per policy token')
                seqs[i].sampler_logprobs.extend(lp)
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            info = eps[i].step(text, n_tokens=len(gen_ids))
            n_env_steps += len(info["calls"])
            if eps[i].done:
                continue
            obs_texts = [json.dumps(c["result"], ensure_ascii=False) for c in info["calls"]]
            seqs[i].append_observation(obs_texts, assistant_closed=closed)
    return GroupRollout(seqs, eps, n_gen_tokens, n_env_steps, n_ctx_trunc,
                        policy_version or 0)
