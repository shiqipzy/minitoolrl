# MiniToolRL 源码导航

项目目标、训练链和当前状态见上一级 [项目 README](../README.md)。所有模块命令从项目根执行。

| 文档 | 内容 |
|---|---|
| [项目说明](docs/项目说明.md) | 环境、业务政策、基础任务与组合任务 |
| [训练与评测](docs/训练与评测.md) | 数据来源、三组对照、训练预算和评测定义 |
| [运行指南](docs/运行指南.md) | 安装、输入包、GPU 执行和产物读取 |
| [术语表](docs/术语表.md) | 世界、轨迹、监督 token、GRPO、DeepSpeed、训推分离 |
| [数据与产物](docs/运行指南.md#数据目录的用途) | 当前输入、机器日志和阅读入口 |

## 阅读范围与文件职责

理解项目主流程时，先读 `run_experiment.py`，再沿训练、模型执行、环境判分和正式评测的调用往下读。辅助文件按需要展开。

| 文件范围 | 实际作用 | 阅读时机 |
|---|---|---|
| `train_sft.py`、`train_grpo.py`、训练后端、采样后端 | 准备训练数据、生成轨迹、计算损失、更新模型和同步权重 | 理解训练实现时读 |
| `chat_format.py`、`rollout_align.py`、`env/`、`eval/evaluate.py` | 决定模型看到什么、工具如何执行、任务怎样判分并保存评测成绩 | 理解任务运行与结果时读 |
| `experiment_config.py`、`run_support.py`、`distributed.py`、`eval/runtime.py` | 保存运行参数、启动进程、通信、检查点与评测执行支撑 | 查配置、执行或恢复问题时读；这些参与实际运行 |
| `preflight_deepspeed.py`、`engineering_report.py` | GPU 环境检查、工程性能测量汇总 | 运行与性能测量时读 |
| `analyze_comp_suite.py`、`current_report.py`、`train_curves.py`、`report_utils.py` | 从已有结果计算成功率、模型差值与置信区间，汇总失败计数和逐步训练指标 | 查统计口径或结果展示时读；不更新模型、不重新生成任务轨迹 |

## 完整运行：先看阶段如何组织

[run_experiment.py](train/run_experiment.py) 组织 SFT → GRPO → 三组续训 → 评测，以及工程性能测量和分析。当前运行使用已经准备好的 `all-restock-v1-20261011` 输入包。

它使用两类运行辅助：

- [experiment_config.py](train/experiment_config.py)：训练与回合参数。
- [run_support.py](train/run_support.py)：启动子进程、记录检查点、打包和运行产物。

## 阶段实现：再看每个阶段使用什么

### SFT 与三组续训

入口为 [train_sft.py](train/train_sft.py)，基础 SFT 和三组续训复用这个训练器。它读取 JSONL、准备训练数据、执行参数更新并保存模型。

它的下层组件是：

- **把轨迹转成模型输入：[sft_dataset.py](data/sft_dataset.py)。** `build_sequence()` 将一条已保存的轨迹转换为 token 序列和监督位置；`collate_sft()` 用 PAD 在同批短序列的末尾补齐至该批最长序列的长度，每批轨迹条数保持不变。训练器据监督位置生成 `labels`，未监督位置和填充位置设为 `-100`，不参与损失计算。
  - **共用对话文本与序列：[chat_format.py](chat_format.py)。** `ChatFormat` 按 tokenizer 模板渲染模型输入、assistant 正文和工具返回；`MultiTurnSequence` 统一完成文本编码和 token 追加，保存 token、监督 mask 和计数，模型生成的 token 直接追加。SFT 数据读取和模型执行都直接使用这个模块。
- **训练后端：[deepspeed_backend.py](train/deepspeed_backend.py)。** `DeepSpeedTrainer.update()` 是一次全局更新的唯一边界：按 rank 分配 microbatch、以全局监督 token 数归一化损失、不足的累积步用零损失对齐，然后由 DeepSpeed 做一次梯度裁剪和一次优化器步。进程通信与统计（`DistCtx`、集合通信、rank 0 落盘、吞吐记录）在 [distributed.py](train/distributed.py)。

`build_sft.py` 已经构造并保存了完整轨迹：`prompt` 是题目，`turns` 中有工具调用和实际返回，`final_line` 是最后答复。JSONL 是这些内容的保存格式。

记录中的 `prompt` 字段只保存用户题目；初始模型输入还包含系统规则和工具说明，并在末尾预填助手开头。

各类任务都展示全部 13 个工具。`build_sequence(tokenizer, record)` 按记录中 `exposed_tools` 保存的顺序加入工具说明；未保存该字段时使用默认清单。训练准备和长度统计共用这个入口。

训练器的模型前向接收 `input_ids`，即 tokenizer 编码得到的整数序列。比如文件中保存的调用是 `get_order`、参数是 `order_id=O-2935`；读入时，模板把它放进 assistant 的工具调用文本，把已保存的返回放进 tool 文本，再由 tokenizer 编码为数字编号。正确调用和最终答复参与损失，题目和工具返回作为上下文。

这发生在 `train_sft.prepare()` 开头读取训练数据时，每条记录编码一次并保存在内存中，随后供训练批次使用。两个文件的关系是：

```text
保存好的 prompt、turns、final_line
  → sft_dataset.build_sequence：逐轮转换、标记哪些输出参与学习
      → chat_format.py：统一模板渲染、文本编码、token 拼接和监督位置
  → train_sft.prepare：生成 input_ids 与 labels
  → sft_dataset.collate_sft：用 PAD 将同批短序列的长度补齐至该批最长长度
  → SFT 损失与参数更新
```

先看[一条实际训练记录](docs/项目说明.md#一条实际训练记录)。模板使用固定版本 Qwen3 官方 tokenizer，明确关闭思考模式；上下文、生成 token 和监督位置见 [chat_format.py](chat_format.py)。

`chat_format.py` 处理 SFT 与模型执行共用的对话表示；`sft_dataset.py` 读取记录中的 `turns`、决定参考调用的监督标记并整理训练批次；`rollout_align.py` 调用模型生成，将输出交给 `session.py` 执行工具，再把工具返回追加到 token 序列。SFT 数据模块直接引用共用对话模块。

### GRPO

入口为 [train_grpo.py](train/train_grpo.py)。每个任务由当前模型生成多条轨迹，按任务成功的二值奖励计算优势并更新参数。训练是单进程，采样在另一张卡上的常驻 vLLM 进程；主循环只有采样、更新、同步三件事，零信号迭代不更新也不同步。

- **多轮自主执行：[rollout_align.py](train/rollout_align.py)。** `rollout_pool(sampler, ...)` 调度尚未结束的任务，向采样器请求生成，并把工具返回追加为下一轮输入。同一套逻辑同时供训练采样和评测使用。
  - **采样后端：[vllm_backend.py](train/vllm_backend.py)。** `VLLMRollout` 是训推分离的客户端（命令通道 + NCCL 权重同步），`LocalVLLM` 是评测的进程内采样器。
  - **单条任务回合：[session.py](env/session.py)。** 每条轨迹对应一个 `Episode`，负责独立世界状态、消息记录、工具执行、停止条件和最终评分。
  - **共用模型输入：[chat_format.py](chat_format.py)。** `ChatFormat` 渲染模板，`MultiTurnSequence` 保存输入及生成 token 和监督 mask。
- **优势与损失：[reward.py](train/reward.py)、[grpo_loss.py](train/grpo_loss.py)、[policy_loss.py](train/policy_loss.py)。** 分别计算组内优势、GRPO 目标和策略 token 的对数概率。
- **训练后端与权重同步：[deepspeed_backend.py](train/deepspeed_backend.py) + [vllm_backend.py](train/vllm_backend.py)。** 更新完成后调用 vLLM 官方 NCCL 权重传输，清除旧前缀缓存并确认版本号；同步失败即结束本阶段。

`rollout_align.py` 中的对象分别承担以下职责：

| 函数或类 | 处理什么 |
|---|---|
| `rollout_pool()` | 为任务列表创建各自的 `Episode` 和 `MultiTurnSequence`；逐轮选择未结束任务，请求采样器生成，把解码文本交给 `Episode.step()`，再把工具返回追加为上下文；检查生成和上下文长度预算，并断言同一回合内策略版本不变。 |
| `GroupRollout` | 保存这一批的序列、回合对象、生成 token 数、工具调用次数、上下文截断条数和策略版本。GRPO 从序列取训练数据，从回合对象取奖励；评测从回合对象取结果和完整记录。 |

一轮工具调用经过以下步骤：

```text
rollout_pool：取出未结束任务的 token 输入
  → sampler.generate（VLLMRollout / LocalVLLM）：生成新 token 与结束原因
  → MultiTurnSequence.append_generated：保存原始 token 和监督位置
  → tokenizer.decode：得到这一轮 assistant 文本
  → Episode.step：解析调用 → tools.execute → 修改世界、记录工具返回
  → MultiTurnSequence.append_observation：按模板编码返回与下一轮助手开头
  → 下一轮生成；结束后由调用者使用 Episode.result / transcript 取评分和记录
```

### 评测与分析

终点评测入口为 [evaluate.py](eval/evaluate.py)，复用多轮执行逻辑执行组合题、通用题和双订单退款任务；统一通过 vLLM 采样。训练过程的开发任务诊断使用 [rollout_eval.py](eval/rollout_eval.py)，按 [trajectory_metrics.py](eval/trajectory_metrics.py) 汇总成功与失败。检查点加载与采样器构建在 [runtime.py](eval/runtime.py)。

评测完成后，[analyze_comp_suite.py](eval/analyze_comp_suite.py) 做组合指标和配对比较，[current_report.py](eval/current_report.py) 汇总五个终点的结果、失败和预算；[train_curves.py](eval/train_curves.py) 读取本次逐步训练日志。

### 工程性能测量

`run_experiment` 的 `engineering` 阶段执行单/双 rank 的 SFT 成对性能测量，以及 DeepSpeed 训练卡 + vLLM 采样卡的 GRPO 性能测量。[engineering_report.py](eval/engineering_report.py) 汇总更新差异、吞吐、墙钟秒数、GPU 秒数与显存。

## 工具环境：训练与评测调用的后台

单条任务的回合由 [session.py](env/session.py) 管理。`Episode` 接收一轮 assistant 文本，执行工具并记录结果；模型生成循环由上面的 `rollout_pool()` 驱动。

| 函数或类 | 处理什么 |
|---|---|
| `parse_tool_calls()` | 从 assistant 文本中的 `<tool_call>` 提取工具名和参数，记录 JSON 或参数格式错误。 |
| `EpisodeBudget` | 保存最大轮数、工具调用数、每轮生成 token 数和上下文长度。轮数与调用数由 `Episode` 检查，生成与上下文长度由 `rollout_pool()` 检查。 |
| `Episode` | 从任务初始快照创建独立世界，并准备系统规则、用户题目和工具说明；`step()` 执行一轮，`result()` 核对最终状态，`transcript()` 返回世界、消息、调用和评分组成的记录字典。 |

`result()` 给出任务是否完成：成功为 1，失败为 0，供 GRPO 计算奖励和评测统计。GRPO 根据内存中的 token、mask 和奖励更新参数；`grpo_train.jsonl` 保存组内奖励、优势、损失、更新数和耗时，供逐次指标汇总与工程报告读取。

`transcript()` 供评测使用。`evaluate.py`、`runtime.py` 保存记录后，`evaluate.analyze()` 和 `trajectory_metrics.py` 读取工具调用与最终世界状态，计算退款与信用补偿金额是否正确、缺操作、多操作和重复查询等指标；`evaluate.load_done()` 读取已完成任务与采样编号，续跑时补足评测配额。

- [tools.py](env/tools.py)：查询和业务操作，修改 [state.py](env/state.py) 中的订单、客户、库存等状态。
- [policy.py](env/policy.py)：退款、回库、补偿和取消等业务政策。
- [grader.py](env/grader.py)：按任务要求核对最终状态，结果用于评测成功率和 GRPO 奖励。

## 开训前的数据准备

这些入口用于准备数据，准备后的轨迹由 SFT 训练器读取。

### 基础 SFT 数据

[build_splits.py](data/build_splits.py) 调用 [tasks.py](env/tasks.py) 和 [state.py](env/state.py)，生成题目、初始场景、参考调用与评分要求，保存任务池。

[build_sft.py](data/build_sft.py) 再读取任务池，执行 `tools.execute()`，将调用及真实返回保存为 `turns`，写出逐条 JSONL 轨迹。训练时才由上面的 SFT 数据组件转换为 token。

### 分解任务、更新顺序与评测题目

- [decomposition_tasks.py](data/decomposition_tasks.py)：定义 C 两组使用的信用补偿工单、部分退款通知分解任务，并执行参考调用生成训练记录。
- [continuation_data.py](data/continuation_data.py)：生成 C_mixed、U_mixed 更新顺序，以及通用评测题目。C_blocked 直接按文件顺序训练。
- [composition_suite.py](data/composition_suite.py)：构造未见组合任务的评测题目。

### 已有输入包

[input_bundle.py](data/input_bundle.py) 核对已经准备好的文件，供打包和启动使用。需要沿用已有任务重建输入包时，使用 [rebuild_inputs.py](data/rebuild_inputs.py)，写入新的独立目录。运行方法见 [运行指南](docs/运行指南.md)。
