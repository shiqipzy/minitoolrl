# MiniToolRL：多轮工具调用与组合泛化

MiniToolRL 使用 Qwen3-0.6B，在可执行的模拟电商后台中学习查询订单、退款、补偿、回补库存、取消运单和释放预留。模型通过工具读取事实、执行操作，程序按最终后台状态判断任务是否完成。

项目研究：在训练中分别学过的业务技能，能否组合起来完成未见任务；同一批分解任务轨迹采用不同训练顺序时，组合任务表现怎样变化。

项目同时实现并验证多轮工具环境、SFT/GRPO 后训练与分布式训练：SFT 与续训走 DeepSpeed ZeRO-2 多卡训练，GRPO 把训练卡与 vLLM 采样卡分开部署并在线同步权重。工程阶段通过单/双卡 SFT 和训推分离 GRPO 短测记录速度、耗时与显存。结果统计包括各任务族成功率、同一任务两种表达的成功情况、模型间差值及置信区间、漏做和多做操作等失败计数；另汇总每次更新的损失等训练指标，当前没有自动绘图或训练趋势解释。

输入版本为 `all-restock-v1-20261011`，源码默认运行标识为 `tool_composition_qwen3_20261014_v2`。**已完成的 GPU 实验使用独立运行目录。** 当前简历对应 `tool_composition_qwen3_20261016_registry_lr2e6_full144_r1`，已发布的结果、统计和运行身份见 [2026-10-16 实验结果](https://github.com/shiqipzy/minitoolrl/tree/4bfa90a3f239ec3b032880274cfccf35c5a5efc8/results/20261016)。完整输入、逐题轨迹、日志与源码归档位于本地 `dist/resume_evidence_20261016/`，未随仓库发布。成绩与训练设置按具体运行的协议读取。

## 阅读入口

| 文档                                                            | 内容                                                           |
| --------------------------------------------------------------- | -------------------------------------------------------------- |
| [项目说明](minitoolrl/docs/项目说明.md)                          | 环境、业务规则、工具、基础任务、组合任务和项目目标             |
| [训练与评测](minitoolrl/docs/训练与评测.md)                      | 训练链、数据、三组对照、训练预算与评测定义                     |
| [运行指南](minitoolrl/docs/运行指南.md)                          | 依赖、输入校验、打包、GPU 执行与结果读取                       |
| [术语表](minitoolrl/docs/术语表.md)                              | 世界、任务、轨迹、监督 token、GRPO、DeepSpeed 等概念           |
| [DeepSpeed 与 vLLM 入门](minitoolrl/docs/DeepSpeed与vLLM入门.md) | DeepSpeed 更新机制、vLLM 的 prefill/decode 优化与配置证据，再对照采样和权重同步代码 |

## 没有本地 GPU 时

本地做输入校验和打包。`pack` 生成 `minitoolrl_source.tar.gz` 与 `full_chain_inputs.tar.gz`；将这两个文件上传至 Linux GPU 主机，在那里安装 CUDA 版 PyTorch 与 GPU 依赖（`pip install -e ".[gpu]"`，即 deepspeed 与 vllm）、初始化新运行并逐阶段执行。无需上传整个工作目录或历史模型。具体复制命令见[运行指南](minitoolrl/docs/运行指南.md)。

已有远程交付包及其执行计划位于本地 `dist/remote_deepspeed_vllm_20261014_v4/`，未随仓库发布。运行当前源码需重新执行 `pack`；旧包不包含当前入口的改动。

理解数据先看[实际任务示例](minitoolrl/docs/项目说明.md#一条实际训练记录)，查文件用途看[数据与产物说明](minitoolrl/docs/运行指南.md#数据目录的用途)。

## 训练链

```text
官方 Qwen3-0.6B
  → 一段工具能力 SFT，固定预算终点与开发集诊断
  → G0：学生自主采样的 GRPO
  ├─ C_blocked：分解任务与回放数据，分块训练
  ├─ C_mixed：同一批分解任务与回放数据，混排训练
  └─ U_mixed：六类基础任务，监督 token 预算相近
  → SFT、G0、三个分支共同参加组合、通用与双订单退款评测

工程性能测量：SFT 单/双卡短测＋GRPO 训推分离短测
```

三个分支从同一个 G0 独立开始，均为 SFT 续训。退款回库任务处理订单中全部可回补商品；部分退款任务只处理指令指定的商品与数量。退款、信用补偿、工单等操作按任务的明确要求执行。

默认双卡完整流程包含 5 个主训练任务、3 个工程短测、15 个 greedy 评测任务。三组续训各执行一次；工程性能测量独立记录，不参与主链模型选择。目的、预算与执行量见[训练与评测](minitoolrl/docs/训练与评测.md)。

## 项目目录

```text
minitoolrl/                         项目根：README、依赖与许可证
└── minitoolrl/                     Python 包
    ├── docs/                      当前项目说明与运行指南
    ├── env/                       世界状态、工具、任务、规则和评分
    ├── data/                      训练轨迹、组合划分与训练输入校验
    ├── train/                     SFT、GRPO、DeepSpeed/vLLM 后端与训练编排
    ├── eval/                      自主执行、任务统计与吞吐分析
    └── artifacts/                 训练输入和运行产物
```

所有模块命令从项目根执行。仓库发布源码和文档，训练输入、模型及运行日志保存在本地；克隆后需先准备对应输入包。已有工作区的训练输入位于 `minitoolrl/artifacts/research/full_chain_all_restock_20261011_v1/inputs_bundle/`，可直接使用：

```bash
python -m minitoolrl.data.input_bundle --verify
python -m minitoolrl.train.run_experiment --stage plan
python -m minitoolrl.train.run_experiment --stage pack
```

完整 GPU 命令见[运行指南](minitoolrl/docs/运行指南.md)。

## 代码阅读入口

先读 [run_experiment.py](minitoolrl/train/run_experiment.py)，了解一次完整运行如何组织各阶段，再向下读阶段实现及其使用的组件：

```text
完整运行：run_experiment.py
├── SFT 与三组续训：train_sft.py
│   ├── 把 SFT 轨迹转成模型输入：sft_dataset.py
│   │   └── 共用对话文本与 token 序列：chat_format.py
│   ├── SFT 终点的开发集任务诊断：rollout_eval.py
│   └── 一次全局更新的唯一后端：deepspeed_backend.py（distributed.py 只保留通信与统计）
├── GRPO：train_grpo.py
│   ├── 模型多轮执行：rollout_align.py
│   │   ├── 共用对话文本与 token 序列：chat_format.py
│   │   └── 单条任务回合：session.py
│   │       ├── 世界状态：state.py
│   │       ├── 工具执行与业务政策：tools.py、policy.py
│   │       └── 最终状态评分：grader.py
│   ├── 组内优势与训练损失：reward.py、grpo_loss.py
│   ├── 训练后端：deepspeed_backend.py
│   └── 采样与权重同步：vllm_backend.py
├── 五个模型终点的正式评测：evaluate.py
│   └── 复用 vLLM 模型多轮执行，保存逐题结果并直接汇总成功数与失败统计
├── 工程性能测量：实际速度、耗时与显存
│   ├── 性能与运行短测：SFT 单/双卡各 40 次更新、GRPO 训推分离 20 次迭代
│   │   └── 复用主训练代码：train_sft.py、train_grpo.py；输出写入独立 engineering 目录
│   └── 汇总吞吐、耗时与显存：engineering_report.py
└── 成绩统计与日志汇总
    ├── 分任务族和 A/B 表达统计成功率，计算模型间差值与置信区间：analyze_comp_suite.py
    ├── 汇总五个终点的成绩、漏做/多做操作、解析和工具错误次数：current_report.py
    └── 收集逐次损失、梯度范数、学习率和采样成功率：train_curves.py（输出 JSON，不绘图）
```

40 次 SFT 更新和 20 次 GRPO 迭代是独立工程测量，复用主训练实现，不产生供主链续训的模型；主链预算仍为 SFT 564 次更新、GRPO 240 次迭代、三组续训各 144 次更新。短测通过不能证明完整训练的任务成功率提高。“组合任务成功率比较”使用五个终点在同一批 360 条组合题上的结果，检验 GRPO 前后、分解任务续训与基础任务续训、分块与混排顺序的差异，属于项目研究问题的直接评测。详细控制条件见[训练与评测](minitoolrl/docs/训练与评测.md)。

训练执行时，SFT 与续训逐步写入各自的 `updates.jsonl`，GRPO 写入 `grpo_train.jsonl`，并在训练预算的 1/3、2/3 和终点记录开发集成功率。完整运行的分析阶段将这些记录汇总为运行目录下的 `eval/train_curves.json`。这个文件保存按更新或迭代编号排列的指标；脚本不绘图、不判断是否收敛，也不自动解释分支成绩差异的原因。已完成运行的原始训练日志见上述归档。

`chat_format.py` 集中定义 `ChatFormat` 和 `MultiTurnSequence`，处理共用的模板文本、token 拼接和监督位置。`sft_dataset.py` 把已保存的 SFT 记录交给这个模块，并用 PAD 整理训练批次；`rollout_align.py` 使用同一模块准备模型输入和保存生成 token。

模型自主执行由 [rollout_align.py](minitoolrl/train/rollout_align.py) 和 [session.py](minitoolrl/env/session.py) 共同完成：前者批量调用模型，将解码后的输出交给后者的 `Episode.step()`；后者解析工具调用、执行业务操作、记录返回并管理单条任务的回合。工具返回再编码进下一轮模型输入，最终由 `Episode.result()` 调用评分器核对世界状态。各函数和类的职责见 [源码导航](minitoolrl/README.md)。

开训前的数据准备有自己的入口：`build_splits.py` 调用 `tasks.py`、`state.py` 生成题目、初始场景与参考调用；`build_sft.py` 执行调用，保存真实工具返回并写出 JSONL。C 两组另使用 `decomposition_tasks.py` 定义的信用补偿工单、部分退款通知分解任务；C_mixed、U_mixed 的更新顺序由 `continuation_data.py` 生成，C_blocked 使用文件顺序。

当前运行直接读取已准备的输入包。基础 SFT、C 两组的基础任务回放和 U 组来自六类基础任务；GRPO 的轨迹由模型在训练时自主生成。各阶段的数据来源见 [训练与评测](minitoolrl/docs/训练与评测.md)。

## 来源与许可证

模型、词表与聊天模板均来自固定版本的官方 Qwen3-0.6B。SFT、GRPO 和评测统一使用 `enable_thinking=False`，后续阶段读取检查点保存的 tokenizer。

项目使用 Apache-2.0；见 [LICENSE](LICENSE)、[NOTICE](NOTICE)；[官方 tokenizer 来源](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer_config.json)。
