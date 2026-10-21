# MiniToolRL：多轮工具调用与组合泛化

MiniToolRL 研究小模型能否把分别学过的工具使用技能组合起来，完成未见过的业务任务。项目使用 Qwen3-0.6B，在可执行的模拟电商后台中训练和评测：模型读取订单、商品和政策，调用工具完成退款、回补库存、信用补偿、取消运单或释放预留，评分器按最终后台状态判断任务是否完成。

例如，“退款并按要求补偿”需要查清订单事实、执行退款、读取补偿政策并发放正确金额；只生成一段看似合理的回答不能算成功。环境、实际题面和轨迹示例见[项目说明](minitoolrl/docs/项目说明.md)。

## 实验设计

```text
官方 Qwen3-0.6B
  → SFT：用执行过的工具轨迹做监督微调
  → G0：模型自主采样轨迹，使用 GRPO 训练
  ├─ C_blocked：分解任务与基础任务回放，分块训练
  ├─ C_mixed：同一批分解任务与回放数据，混排训练
  └─ U_mixed：六类基础任务，监督 token 预算相近
  → 五个模型终点共同参加组合、通用和双订单退款评测
```

三个分支均从同一个 G0 独立开始，采用 SFT 续训。C_blocked 与 C_mixed 比较同一批数据的训练顺序；C_mixed 与 U_mixed 比较分解任务训练和基础任务训练。SFT 与 G0 提供续训前的参照。

当前训练实现使用 DeepSpeed ZeRO-2；GRPO 由一张卡训练，另一张卡上的常驻 vLLM 进程采样，每次有效更新后同步权重。独立的工程短测比较单/双卡 SFT，并测量训推分离 GRPO 的吞吐、耗时和显存。

## 查看已完成的实验结果

无需 GPU 或模型权重即可查看公开报告。已完成运行 `tool_composition_qwen3_20261016_registry_lr2e6_full144_r1` 的汇总保存在固定版本的[2026-10-16 结果目录](https://github.com/shiqipzy/minitoolrl/tree/4bfa90a3f239ec3b032880274cfccf35c5a5efc8/results/20261016)，按以下顺序阅读：

1. 打开 [RESULT.md](https://github.com/shiqipzy/minitoolrl/blob/4bfa90a3f239ec3b032880274cfccf35c5a5efc8/results/20261016/RESULT.md)，看五个终点的成功数、三类组合任务的细分表现和执行量。
2. 打开 [composition_report.json](https://github.com/shiqipzy/minitoolrl/blob/4bfa90a3f239ec3b032880274cfccf35c5a5efc8/results/20261016/composition_report.json)，在 `models` 中查看各模型的组合指标，在 `comparisons` 中查看模型间差值及按世界配对的 bootstrap 置信区间。
3. 打开 [analysis.json](https://github.com/shiqipzy/minitoolrl/blob/4bfa90a3f239ec3b032880274cfccf35c5a5efc8/results/20261016/analysis.json)，在 `summary` 中核对三套评测，在 `paired` 中看同题改善/退步，在 `long_per_task` 中看双订单任务逐题检查，在 `cost` 中看实际作业时间。

表中的分子是完整任务成功数，分母是评测任务数；成功要求最终状态满足全部断言。组合评测有三类任务，每类 60 个后台初始场景、每个场景两种表达，共 360 条；同一场景的两种表达相关，置信区间以场景为单位配对重采样。通用评测有 372 条，双订单退款有 60 条。

| 模型终点 | 组合成功 / 360 | 通用成功 / 372 | 双订单退款成功 / 60 |
|---|---:|---:|---:|
| SFT | 165 | 365 | 36 |
| G0 | 164 | 367 | 37 |
| C_blocked | 178 | 372 | 43 |
| C_mixed | 175 | 372 | 37 |
| U_mixed | 181 | 372 | 15 |

这组结果需要同时看三列：U_mixed 的组合总成功数最高，但双订单退款从 G0 的 37/60 降到 15/60；C_blocked 的双订单退款为 43/60。单看组合总分，无法说明某一种续训方式在所有任务上更好。组间差异还应结合分任务族结果和配对区间判断。

该运行的三个续训分支各完成 144 次更新，学习率为 `2e-6`，训练与评测统一使用 registry 工具顺序。它复用已有 SFT/G0 起点；上述成绩来自该次运行，不是当前代码完整重训的成绩。实验只有一个训练种子，评测题已有诊断和查看记录；配对区间不包含重新训练的随机性。

公开目录包含汇总报告、预算与检查点身份；完整输入、逐题轨迹、日志和当时源码保存在未随仓库发布的 `dist/resume_evidence_20261016/results_inputs_logs_source.tar.gz`。仅凭公开汇总不能重新评分逐题轨迹。

## 运行项目

所有模块命令从项目根执行。克隆后可以先查看入口参数，仅需 Python：

```bash
python -m minitoolrl.train.run_experiment --help
```

训练需要 Linux、CUDA GPU、对应依赖和独立准备的冻结输入包。仓库不包含训练输入或模型权重；直接克隆后，输入校验、计划和训练命令尚不能运行。输入包格式、依赖安装、上传和 GPU 执行步骤见[运行指南](minitoolrl/docs/运行指南.md)。

准备好输入后，本地校验并打包：

```bash
python -m minitoolrl.data.input_bundle --verify
python -m minitoolrl.train.run_experiment --stage plan
python -m minitoolrl.train.run_experiment --stage pack
```

默认输入位置为 `minitoolrl/artifacts/inputs/`，格式版本为 `all-restock-v1-20261011`。`pack` 在 `dist/current/` 生成源码包、输入包和哈希清单，可传到 GPU 主机初始化新的运行。

默认新运行标识为 `tool_composition_qwen3_registry_lr2e6`，尚未执行完整 GPU 训练。其计划为 SFT 两轮共 564 次更新、GRPO 240 次迭代、三个续训分支各 144 次更新；工程短测另行记录。五个终点各执行 360 + 372 + 60 条 greedy 评测，共 3,960 次任务执行。

## 查看自己运行的结果

在 GPU 主机按[运行指南](minitoolrl/docs/运行指南.md#执行完整项目)完成训练和 `evals` 阶段后，从所指定运行目录开始阅读：

| 阅读顺序 | 运行目录下的文件 | 看什么 |
|---|---|---|
| 1 | `eval/report.md` | 五个终点在三套任务上的成功数 |
| 2 | `eval/comp_suite/final_report.json` | 分任务族、同场景 A/B 表达、模型间差值和配对区间 |
| 3 | `eval/eval_summary.json` | 漏做/多做操作、解析/工具错误、实际训练预算 |
| 4 | `eval/<套件>/run/<模型>/greedy/<模型>_rollouts.jsonl` | 单题调用、工具返回与最终后台状态；套件名为 `comp_suite`、`c_recipe`、`long` |
| 5 | `eval/train_curves.json`、`engineering/report.md` | 逐步训练指标及独立工程短测的速度、耗时、显存 |

先核对 `protocol.json`、`pins.json` 和完成配额，再比较模型成绩。`train_curves.json` 保存逐步数据，当前不自动绘图或判断收敛；短测速度和训练 loss 也不能替代正式任务成功率。

## 文档与源码

| 文档 | 内容 |
|---|---|
| [项目说明](minitoolrl/docs/项目说明.md) | 业务环境、规则、工具、实际任务与项目目标 |
| [训练与评测](minitoolrl/docs/训练与评测.md) | 数据来源、三组对照、训练预算和评测定义 |
| [运行指南](minitoolrl/docs/运行指南.md) | 依赖、输入准备、打包、GPU 执行、重试与输出 |
| [源码导航](minitoolrl/README.md) | 阶段入口、模块职责和函数阅读顺序 |
| [术语表](minitoolrl/docs/术语表.md) | 世界、任务、轨迹、监督 token、GRPO 等概念 |
| [DeepSpeed 与 vLLM 入门](minitoolrl/docs/DeepSpeed与vLLM入门.md) | 分布式更新、采样和在线权重同步机制 |

```text
minitoolrl/
├── env/             后台状态、工具、任务、业务规则和评分
├── data/            任务划分、训练轨迹、组合任务和输入包校验
├── train/           SFT、GRPO、DeepSpeed/vLLM 后端和阶段编排
├── eval/            自主执行评测、统计和结果报告
├── docs/            设计、训练与运行文档
├── preflight.py     GPU 训练前的通信与显存检查
├── chat_format.py   共用聊天模板与 token 序列
└── files.py         运行期文件哈希与 JSON 读写
```

完整运行入口为 [run_experiment.py](minitoolrl/train/run_experiment.py)；模型自主执行由 [rollout.py](minitoolrl/train/rollout.py) 调用模型，由 [session.py](minitoolrl/env/session.py) 解析并执行工具，最后由评分器核对世界状态。

## 来源与许可证

模型、词表与聊天模板来自固定 revision 的官方 Qwen3-0.6B。SFT、GRPO 和评测统一使用 `enable_thinking=False`，后续阶段读取检查点保存的 tokenizer。

项目使用 Apache-2.0；见 [LICENSE](LICENSE)、[NOTICE](NOTICE) 和[官方 tokenizer 来源](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer_config.json)。
