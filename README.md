# MiniToolRL：多轮工具调用后训练与组合泛化

基于 Qwen3-0.6B，在模拟电商后台中研究小模型的多轮工具调用、SFT / GRPO 后训练，以及未见任务组合上的执行能力。环境提供 13 种读写工具，按最终业务状态与误操作检查评分。

本仓库发布主流程源码与 2026-10-16 实验结果摘要。实验源码的 SHA-256 清单见 [source_files.json](provenance/source_files.json)；该清单标识实际运行时的源码，当前发布源码不等同于实验快照。

## 方法

```text
Qwen3-0.6B → SFT → G0（on-policy GRPO）
                    ├─ C_blocked：分解技能与原任务回放，分块续训
                    ├─ C_mixed：相同数据，混排续训
                    └─ U_mixed：监督 token 预算相近的原任务续训
```

三组从同一 G0 独立开始，学习率 2e-6，各完成 144 次更新。两组 C 使用 1,152 条训练记录。训练与评测统一使用注册工具顺序。SFT 采用 DeepSpeed ZeRO-2，GRPO 使用独立 vLLM 采样进程与逐更新权重同步；损失仅覆盖 assistant 输出，屏蔽工具观测。

## 结果

| 模型 | 三类组合（360 题） | 通用（372 题） | 双订单退款（60 题） |
|---|---:|---:|---:|
| SFT | 165 / 360 | 365 / 372 | 36 / 60 |
| G0 | 164 / 360（45.6%） | 367 / 372 | 37 / 60（61.7%） |
| C_blocked | 178 / 360（49.4%） | 372 / 372 | 43 / 60（71.7%） |
| C_mixed | 175 / 360（48.6%） | 372 / 372 | 37 / 60（61.7%） |
| U_mixed | 181 / 360（50.3%） | 372 / 372 | 15 / 60（25.0%） |

分块方案在双订单退款上优于两个续训对照，组合总分低于原任务续训。结果仅来自一个训练种子，测试题此前已有诊断和结果查看记录；不据此声称整体组合泛化或跨种子稳定性已得到证明。完整统计见 [结果报告](results/20261016/RESULT.md) 与 [analysis.json](results/20261016/analysis.json)。

## 运行

正式实验入口为 `minitoolrl.train.run_experiment`。`preflight` 检查目标 GPU 的通信和显存；`engineering` 测量单/双卡 SFT 与训推分离 GRPO 的速度、耗时和显存。

完整训练的依赖、输入要求与命令见 [运行说明](docs/RUNNING.md)。采样协议与 GRPO 单次更新的开发检查可在修改对应实现时手动运行，不属于正式训练步骤。

## 阅读代码

| 关注点 | 文件 |
|---|---|
| 训练与评测编排 | [run_experiment.py](minitoolrl/train/run_experiment.py) |
| SFT 与 GRPO | [train_sft.py](minitoolrl/train/train_sft.py)、[train_grpo.py](minitoolrl/train/train_grpo.py) |
| 工具、业务政策与评分 | [tools.py](minitoolrl/env/tools.py)、[policy.py](minitoolrl/env/policy.py)、[grader.py](minitoolrl/env/grader.py) |
| 数据构造与三组顺序 | [decomposition_tasks.py](minitoolrl/data/decomposition_tasks.py)、[continuation_data.py](minitoolrl/data/continuation_data.py) |
| 分布式更新、采样与同步 | [deepspeed_backend.py](minitoolrl/train/deepspeed_backend.py)、[vllm_backend.py](minitoolrl/train/vllm_backend.py) |
| 组合评测 | [composition_suite.py](minitoolrl/data/composition_suite.py)、[analyze_comp_suite.py](minitoolrl/eval/analyze_comp_suite.py) |

## 发布范围

源码与结果摘要已发布。模型权重、训练输入包、完整逐题轨迹、运行日志和临时调试脚本未包含，完整 GPU 复现需另行准备输入包与官方底座。结果摘要保留实验身份与模型哈希，不能替代权重或原始轨迹。

代码采用 [Apache-2.0](LICENSE)。项目来源、MiniMind 与 Qwen 的第三方归属见 [NOTICE](NOTICE)。
