"""单条任务的执行状态：接收模型回复、执行工具、判断任务是否完成。

Episode.step 处理一轮 assistant 文本，更新世界状态和消息记录。
Episode.result 按任务要求核对最终状态；transcript 整理评测分析和续跑需要的记录。
模型生成与上下文长度检查由 train/rollout.py 驱动。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from minitoolrl.env import policy as P
from minitoolrl.env import tools as T
from minitoolrl.env.grader import grade, mutation_summary
from minitoolrl.env.state import WorldState
from minitoolrl.env.tasks import TaskInstance

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """返回 (调用列表, 解析错误列表)。arguments 允许是对象或 JSON 字符串。"""
    calls, errors = [], []
    blocks = TOOL_CALL_RE.findall(text or "")
    if not blocks and "<tool_call>" in (text or ""):
        errors.append("unclosed_tool_call")
    for raw in blocks:
        raw = raw.strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            errors.append(f"bad_json:{type(e).__name__}")
            continue
        if not isinstance(obj, dict) or "name" not in obj:
            errors.append("missing_name")
            continue
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                errors.append("bad_arguments_json")
                continue
        if not isinstance(args, dict):
            errors.append("arguments_not_object")
            continue
        calls.append({"name": str(obj["name"]).strip(), "arguments": args})
    return calls, errors


@dataclass
class EpisodeBudget:
    max_turns: int = 8
    max_calls: int = 12
    max_new_tokens: int = 160          # 每轮生成上限，由 rollout_pool 检查
    max_context_tokens: int = 4096     # 上下文长度上限，由 rollout_pool 检查


@dataclass
class Episode:
    """一条任务的世界状态、消息和执行记录；接收每轮回复并核对任务结果。"""
    inst: TaskInstance
    budget: EpisodeBudget = field(default_factory=EpisodeBudget)
    visible_tools: Optional[List[str]] = None   # None = 全部工具都可见（默认，避免工具列表泄漏计划）

    def __post_init__(self):
        self.state = WorldState.from_snapshot(self.inst.world)
        self.tools = T.tool_schemas(self.visible_tools)
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": P.OPERATING_RULES},
            {"role": "user", "content": self.inst.prompt},
        ]
        self.turns = 0
        self.n_calls = 0
        self.n_tool_errors = 0
        self.parse_errors: List[str] = []
        self.trace: List[Dict[str, Any]] = []
        self.done = False
        self.stop_reason: Optional[str] = None
        self.gen_tokens = 0

    def step(self, assistant_text: str, n_tokens: int = 0) -> Dict[str, Any]:
        """解析一轮回复、执行工具、记录返回，并检查轮数和工具调用数。"""
        if self.done:
            raise RuntimeError("episode 已结束")
        self.turns += 1
        self.gen_tokens += int(n_tokens)
        calls, errors = parse_tool_calls(assistant_text)
        self.parse_errors += errors
        self.messages.append({"role": "assistant", "content": assistant_text})

        if not calls:
            self.done = True
            self.stop_reason = "final_answer" if not errors else "parse_error"
            self.trace.append({"turn": self.turns, "calls": [], "errors": errors})
            return {"calls": [], "errors": errors, "done": True}

        executed = []
        for c in calls:
            if self.n_calls >= self.budget.max_calls:
                self.done = True
                self.stop_reason = "call_budget"
                break
            out = T.execute(self.state, c["name"], c["arguments"])
            self.n_calls += 1
            if isinstance(out, dict) and "error" in out:
                self.n_tool_errors += 1
            executed.append({"name": c["name"], "arguments": c["arguments"], "result": out})
            # 每次调用的实际返回记录为一条 tool 消息。
            self.messages.append({"role": "tool", "content": json.dumps(out, ensure_ascii=False)})
        self.trace.append({"turn": self.turns, "calls": executed, "errors": errors})

        if not self.done and self.turns >= self.budget.max_turns:
            self.done = True
            self.stop_reason = "turn_budget"
        return {"calls": executed, "errors": errors, "done": self.done}

    def result(self) -> Dict[str, Any]:
        """按最终世界状态判断任务是否完成，返回成功值、奖励和失败原因。"""
        final = self.state.snapshot()
        g = grade(self.inst.world, final, self.inst.assertions)
        tags = list(g.failure_tags)
        if not g.success:
            if self.n_calls == 0:
                tags.append("no_tool_call")
            if self.parse_errors:
                tags.append("format")
            if self.stop_reason in ("turn_budget", "call_budget"):
                tags.append("truncated")
            if self.n_tool_errors:
                tags.append("tool_error")
        return {
            "task_id": self.inst.task_id, "family": self.inst.family,
            "split_group": self.inst.split_group, "difficulty": self.inst.difficulty,
            "success": g.success, "reward": 1.0 if g.success else 0.0,
            "assertion_pass_ratio": round(g.pass_ratio, 4),   # 诊断用，不进奖励
            "failure_tags": tags, "failures": g.failures,
            "n_turns": self.turns, "n_calls": self.n_calls, "n_tool_errors": self.n_tool_errors,
            "n_parse_errors": len(self.parse_errors), "gen_tokens": self.gen_tokens,
            "stop_reason": self.stop_reason, "mutations": mutation_summary(final),
        }

    def transcript(self) -> Dict[str, Any]:
        """返回世界状态、调用和评分等评测记录，供金额、行为分析和评测续跑使用。"""
        out = self.result()
        out.update({"seed": self.inst.seed, "prompt": self.inst.prompt,
                    "initial_world": self.inst.world, "final_world": self.state.snapshot(),
                    "messages": self.messages, "trace": self.trace,
                    "gold_calls": self.inst.gold_calls, "assertions": self.inst.assertions})
        return out
