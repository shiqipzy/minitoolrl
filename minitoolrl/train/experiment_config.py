"""冻结的实验常量：底座版本、回合预算、GRPO 超参数和三组续训日程。

这些值在 init 时写入 protocol.json，之后各阶段读取已保存的配置，不重新推断。
文件读写见 minitoolrl/files.py；运行期参数见 run_experiment.DEFAULTS。
"""
from __future__ import annotations

from minitoolrl.paths import CURRENT_RUN_ID

RUN_ID = CURRENT_RUN_ID

# 官方 Qwen3-0.6B 的固定 revision、权重与聊天模板；fetch_base 逐项核对。
BASE_REPO = 'Qwen/Qwen3-0.6B'
BASE_REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
BASE_TEMPLATE_SHA256 = 'a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8'
BASE_SHA256 = 'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'

# 单条任务的上限：轮数、工具调用数、每轮新 token、上下文 token。
EPISODE_BUDGET = dict(max_turns=8, max_calls=12, max_new_tokens=160, max_context_tokens=3584)

RL = dict(seed=17, lr=1e-6, weight_decay=0.01, grad_clip=1.0, group_size=4, beta=0.0,
          entropy_coefficient=0.0, clip=0.2, prompts_per_iteration=2,
          temperature=1.0,
          normalization='per-trajectory policy-token mean',
          loss='on-policy clipped GRPO; no KL or entropy regularization')

# 三组续训的数据与更新日程；C_blocked 直接按文件顺序训练，没有单独日程文件。
ARMS = {
    'C_mixed': dict(train='inputs/c_recipe/train_C.jsonl',
                    plan='inputs/c_recipe/schedules/C_mixed.json', accum=8),
    'U_mixed': dict(train='inputs/c_recipe/train_U.jsonl',
                    plan='inputs/c_recipe/schedules/U_mixed.json', accum=9),
    'C_blocked': dict(train='inputs/control/train_control.jsonl', plan=None, accum=8),
}
