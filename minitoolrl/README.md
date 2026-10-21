# MiniToolRL 源码导航

项目目标、训练链和当前状态见上一级 [项目 README](../README.md)。所有模块命令从项目根执行。

| 文档 | 内容 |
|---|---|
| [项目说明](docs/项目说明.md) | 环境、业务政策、基础任务与组合任务 |
| [训练与评测](docs/训练与评测.md) | 数据来源、三组对照、训练预算和评测定义 |
| [运行指南](docs/运行指南.md) | 安装、输入包、GPU 执行和产物读取 |
| [术语表](docs/术语表.md) | 世界、轨迹、监督 token、GRPO、DeepSpeed、训推分离 |
| [数据与产物](docs/运行指南.md#数据目录的用途) | 当前输入、机器日志和阅读入口 |

## 目录职责

先读 `train/run_experiment.py`，再按阶段向下读。每个位置只负责一件事：

| 位置 | 负责什么 | 阅读时机 |
|---|---|---|
| `train/run_experiment.py` | 唯一实验入口：`--stage` 选择、运行计划、冻结配置、阶段产物核验 | 先读 |
| `train/run_session.py` | 阶段执行：torchrun 命令、子进程与日志、检查点 pin、阶段标记 | 查执行或恢复问题时读 |
| `train/run_artifacts.py` | 输入打包、官方底座获取、回传清单 | 上传与交付时读 |
| `train/train_sft.py`、`train/train_grpo.py` | 两段训练器：数据准备、前向、更新、落盘 | 理解训练实现时读 |
| `train/deepspeed_backend.py` | 唯一训练后端：一次全局更新的边界与 ZeRO-2 配置 | 核对更新语义时读 |
| `train/vllm_backend.py` | 唯一采样后端：进程内评测实例、训推分离客户端与权重同步 | 理解采样与同步时读 |
| `train/rollout.py` | 多轮执行调度，训练采样与评测共用 | 理解模型怎样走完任务时读 |
| `train/grpo_objective.py` | 组内优势、裁剪后的策略目标 | 核对公式时读 |
| `env/` | 世界状态、工具执行、业务政策、最终状态评分 | 理解任务与奖励时读 |
| `chat_format.py`、`data/sft_dataset.py` | 对话模板渲染、token 序列、监督位置、批次整理 | 理解模型输入时读 |
| `eval/evaluate.py` | 正式评测：任务执行与续跑、配额核对、逐题分析、组合比较和完整报告 | 理解评测全流程时读 |
| `eval/rollout_eval.py` | SFT 终点的 60 条开发集任务成功检查 | 理解开发集检查时读 |
| `eval/trajectory_metrics.py` | 轨迹行为指标：政策查询、金额、漏做与多做 | 查指标定义时读 |
| `eval/engineering_report.py` | 单/双卡与训推分离部署的性能汇总 | 查性能数字时读 |
| `preflight.py` | GPU 运行前检查：进程组通信、ZeRO-2 更新与显存 | 上机前读 |
| `files.py` | 文件哈希与 JSON/JSONL 读写，全项目共用一份实现 | 需要落盘时读 |
| `data/` | 任务池、SFT 轨迹、续训顺序、组合题与输入校验 | 改数据时读 |

## 一次运行包含的阶段

| 阶段 | 做什么 | 主要产物 | 读取方 |
|---|---|---|---|
| `plan` / `pack` | 打印预算；把冻结输入与源码打成上传包 | `full_chain_inputs.tar.gz`、源码包 | 人工上传 |
| `init` | 复制输入一次、逐文件核对哈希、冻结配置与源码哈希 | `protocol.json`、`pins.json` | 后续所有阶段 |
| `fetch_base` | 取得固定 revision 的官方底座并核对权重与模板 | `models/Qwen3-0.6B` | SFT、工程短测 |
| `preflight` | 进程组通信 + DeepSpeed 更新/显存检查 | `preflight/ds_*.json`、`gates.json` | `require_preflight` |
| `sft` | 单段 SFT、开发集任务成功检查 | `sft/endpoint`、`sft/dev_eval/` | GRPO、评测 |
| `grpo` | 学生自主采样、组内优势、更新与权重同步 | `grpo/main/models/final`、`grpo_train.jsonl` | 三组续训、评测 |
| `arms` | 三个分支各一次 SFT 续训 | `arms/<组名>/nN/train/endpoint` | 评测 |
| `engineering` | SFT 单/双卡与 GRPO 训推分离短测 | `engineering/*/` | 工程报告 |
| `evals` | 五个终点 × 三个评测池的 greedy 执行，核对配额并生成全部评测报告 | 逐题轨迹、`eval/comp_suite/final_report.json`、`eval/report.md`、`eval/train_curves.json` 等 | 人工阅读 |
| `package` | 列出可回传的文件与权重 | `transfer_manifest.json` | 人工回传 |

## 入口与阶段执行

[run_experiment.py](train/run_experiment.py) 组织 SFT → GRPO → 三组续训 → 评测与报告，并复用同一套训练代码做工程性能短测。它定义运行计划和冻结参数，阶段本身只做“准备、执行、核验”三件事：

- [run_session.py](train/run_session.py)：`Session` 负责构造 `torchrun`/`python -m` 命令、以同一套执行核心串行或并行启动子进程、写日志与 `budget.json`、记录并核对检查点 pin 和 `<stage>_status.json`。子进程失败或超时直接抛 `LaunchError`，不写成功标记。
- [run_artifacts.py](train/run_artifacts.py)：`stage_pack`、`stage_fetch_base`、`stage_package` 只处理文件；不训练、不评测、不改写已完成的运行产物。
- [experiment_config.py](train/experiment_config.py)：底座 revision 与哈希、回合预算、GRPO 超参数、三组续训的数据与日程。

`make_plan()` 只在 `plan/init` 制定计划；之后各阶段读取 `protocol.json` 中冻结的预算，不重新制定计划。配置在 `init` 时固定，之后显式传入不同值会被拒绝；已成功的阶段再次调用时只核验产物后跳过。

## SFT 与三组续训

入口为 [train_sft.py](train/train_sft.py)，基础 SFT 与三组续训复用这个训练器。它读取 JSONL、准备训练数据、执行参数更新并保存模型；一次调用只创建一个 AdamW 和一条学习率日程。

- **把轨迹转成模型输入：[sft_dataset.py](data/sft_dataset.py)。** `build_sequence()` 将一条已保存的轨迹转换为 token 序列和监督位置；`collate_sft()` 用 PAD 在同批短序列末尾补齐，每批轨迹条数不变。训练器据监督位置生成 `labels`，未监督位置与填充位置为 `-100`。
  - **共用对话文本与序列：[chat_format.py](chat_format.py)。** `ChatFormat` 按 tokenizer 模板渲染模型输入、assistant 正文和工具返回；`MultiTurnSequence` 统一完成文本编码与 token 追加，保存 token、监督 mask 和计数。
- **训练后端：[deepspeed_backend.py](train/deepspeed_backend.py)。** `DeepSpeedTrainer.update()` 是一次全局更新的唯一边界：按 rank 分配 microbatch、以全局监督 token 数归一化损失、不足的累积步用零损失对齐，然后由 DeepSpeed 做一次梯度裁剪和一次优化器步。训练进程组也由该后端统一初始化。

`build_sft.py` 已构造并保存完整轨迹：`prompt` 是题目，`turns` 中有工具调用和实际返回，`final_line` 是最后答复。记录中的 `prompt` 只保存用户题目；初始模型输入还包含系统规则和工具说明，并在末尾预填助手开头。

各类任务都展示全部 13 个工具。`build_sequence(tokenizer, record)` 按记录中 `exposed_tools` 保存的顺序加入工具说明；未保存该字段时使用默认清单。

```text
保存好的 prompt、turns、final_line
  → sft_dataset.build_sequence：逐轮转换、标记哪些输出参与学习
      → chat_format.py：统一模板渲染、文本编码、token 拼接和监督位置
  → train_sft.prepare：生成 input_ids 与 labels
  → sft_dataset.collate_sft：用 PAD 补齐同批长度
  → SFT 损失与参数更新
```

SFT 固定预算终点另做一次开发集任务成功检查（[rollout_eval.py](eval/rollout_eval.py)），60 条任务各自自主执行，结果写入 `sft/dev_eval/`。入口按冻结 `data/gate_spec.json` 逐项验收：普通任务成功至少 21/30，补偿任务政策查询至少 24/30、信用调用至少 18/30、重复查询循环最多 6/30；金额正确率只报告、不设门槛。任一不达标直接报错，不登记 SFT pin、不写阶段成功标记，GRPO 启动前再次核对，未通过则不能运行。这是固定终点的进入条件，不用于中途停止、调参或挑选模型。

## GRPO

入口为 [train_grpo.py](train/train_grpo.py)。每个任务由当前模型生成 4 条轨迹，按任务成功的二值奖励计算组内优势并更新参数。训练是单进程，采样在另一张卡上的常驻 vLLM 进程；主循环只有采样、更新、同步三件事，零信号迭代不更新也不同步。

- **多轮自主执行：[rollout.py](train/rollout.py)。** `rollout_pool(sampler, ...)` 调度尚未结束的任务，向采样器请求生成，并把工具返回追加为下一轮输入。训练采样与终点评测共用这一套逻辑。
  - **采样后端：[vllm_backend.py](train/vllm_backend.py)。** `VLLMRollout` 是训推分离的客户端（命令通道 + NCCL 权重同步），`LocalVLLM` 是评测的进程内采样器。
  - **单条任务回合：[session.py](env/session.py)。** 每条轨迹对应一个 `Episode`，负责独立世界状态、消息记录、工具执行、停止条件和最终评分。
  - **共用模型输入：[chat_format.py](chat_format.py)。**
- **目标计算：[grpo_objective.py](train/grpo_objective.py)。** `group_advantages()` 算组内优势，`grpo_loss()` 做裁剪后的策略目标；策略位置的对数概率与 SFT 共用 `train_sft.masked_logits()`，只在监督位置做词表投影。
- **训练后端与权重同步：[deepspeed_backend.py](train/deepspeed_backend.py) + [vllm_backend.py](train/vllm_backend.py)。** 更新完成后调用 vLLM 官方 NCCL 权重传输，清除旧前缀缓存并确认版本号；同步失败即结束本阶段。

| 函数或类 | 处理什么 |
|---|---|
| `rollout_pool()` | 为任务列表创建各自的 `Episode` 和 `MultiTurnSequence`；逐轮选择未结束任务，请求采样器生成，把解码文本交给 `Episode.step()`，再把工具返回追加为上下文；检查生成和上下文长度预算，并断言同一回合内策略版本不变。 |
| `GroupRollout` | 保存这一批的序列、回合对象、生成 token 数、工具调用次数、上下文截断条数和策略版本。 |

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

## 评测

终点评测入口为 [evaluate.py](eval/evaluate.py)，执行五个终点 × 三个评测池的 greedy 轨迹：

| 评测池 | 每模型数量 | 用途 |
|---|---:|---|
| 组合留出 `comp_suite` | 360 | 三族组合泛化、A/B 表达稳定性 |
| 通用集 `c_recipe` | 372 | 基础能力保留、补偿分层 |
| 双订单退款 `long` | 60 | 两张退款单的查询、金额与状态 |

[evaluate.py](eval/evaluate.py) 在一个模块内完成执行、验收与报告，入口只需运行 `--stage evals`：

1. `run_greedy_pool()` 构建固定条件的评测任务批次，调用共用的多轮执行逻辑并保存轨迹；中断后复用已完成任务。工具、模板、政策、模型或输入改变时拒绝混用旧轨迹。
2. `analyze()` 与 `summarize()` 生成逐题业务分析和成功计数；`verify_evaluations()` 核对全部任务的轨迹、分析和配额。
3. `write_reports()` 统计组合任务的分族成功率、A/B 表达表现、按场景配对的置信区间，生成 `eval/comp_suite/final_report.json`、`eval/eval_summary.json`、`eval/report.md` 和 `eval/train_curves.json`。报告直接读取已完成轨迹，不增加任务执行量。

开发集检查复用本模块的 greedy 调度。实际模型与工具的多轮交互统一在 [train/rollout.py](train/rollout.py)。工程性能报告由 `engineering` 阶段生成，在完整结果页中引用。

## 工程性能测量

`run_experiment` 的 `engineering` 阶段复用 [train_sft.py](train/train_sft.py) 与 [train_grpo.py](train/train_grpo.py)：单 rank、双 rank 各 40 次 SFT 更新，以及 DeepSpeed 训练卡 + vLLM 采样卡的 20 次 GRPO 迭代。短测模型不进入主链，也不用于选检查点；[engineering_report.py](eval/engineering_report.py) 在证据完整时才给出加速比与效率。

## 工具环境：训练与评测调用的后台

| 函数或类 | 处理什么 |
|---|---|
| `parse_tool_calls()` | 从 assistant 文本中的 `<tool_call>` 提取工具名和参数，记录 JSON 或参数格式错误。 |
| `EpisodeBudget` | 保存最大轮数、工具调用数、每轮生成 token 数和上下文长度。 |
| `Episode` | 从任务初始快照创建独立世界，准备系统规则、用户题目和工具说明；`step()` 执行一轮，`result()` 核对最终状态，`transcript()` 返回世界、消息、调用和评分组成的记录。 |

- [tools.py](env/tools.py)：查询和业务操作，修改 [state.py](env/state.py) 中的订单、客户、库存等状态。
- [policy.py](env/policy.py)：退款、回库、补偿和取消等业务政策。
- [grader.py](env/grader.py)：按任务要求核对最终状态，结果用于评测成功率和 GRPO 奖励。
- [trajectory_metrics.py](eval/trajectory_metrics.py)：从轨迹统计政策查询、金额正确性、漏做与多做等行为指标。

## 开训前的数据准备

[build_splits.py](data/build_splits.py) 调用 [tasks.py](env/tasks.py) 和 [state.py](env/state.py)，生成题目、初始场景、参考调用与评分要求；[build_sft.py](data/build_sft.py) 再执行 `tools.execute()`，把调用及真实返回保存为 `turns` 并写出 JSONL。

[decomposition_tasks.py](data/decomposition_tasks.py) 定义 C 两组使用的信用补偿工单、部分退款通知分解任务；[continuation_data.py](data/continuation_data.py) 生成 C_mixed、U_mixed 的更新顺序和通用评测题目；[composition_suite.py](data/composition_suite.py) 构造未见组合任务的评测题目。

[input_bundle.py](data/input_bundle.py) 核对已经准备好的文件，供打包与启动使用；需要沿用已有任务重建输入包时用 [rebuild_inputs.py](data/rebuild_inputs.py)，写入新的独立目录。运行方法见 [运行指南](docs/运行指南.md)。
