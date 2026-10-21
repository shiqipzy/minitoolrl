# MiniToolRL 当前项目

默认解释和运行 `minitoolrl/artifacts/inputs/` 中的 registry 输入对应的完整项目，运行标识为 `tool_composition_qwen3_registry_lr2e6`。输入格式版本为 `all-restock-v1-20261011`，固定 manifest SHA256 为 `864682a89e62af6aa0770d889670c725a20465cc00807697b8270984886ea576`。

- 项目入口为根 README；当前文档为 `minitoolrl/docs/项目说明.md`、`训练与评测.md`、`运行指南.md`、`术语表.md`。按完整项目描述目标、设计和执行，不在当前文档穿插修复沿革。
- 当前默认新运行的输入已经校验、完整 GPU 训练尚未执行。已完成实验依据最终压缩归档单独报告，不提前填入新运行成绩或最佳分支。
- 运行入口为 `minitoolrl.train.run_experiment`；输入校验为 `minitoolrl.data.input_bundle --verify`。直接使用已准备的独立输入包。
- 初始 SFT 数据由 `data/build_splits.py` 生成任务池、`data/build_sft.py` 执行工具并落盘；`data/rebuild_inputs.py` 只用于已有输入包重建，写新的独立目录；`build_sft.build_trajectory` 直接执行工具并记录返回，不判分。`input_bundle` 只核对已有文件；SFT 与模型执行的聊天格式统一由 `chat_format.py` 读取固定版本 Qwen3 官方 tokenizer 模板生成，设置 `enable_thinking=False`，不覆盖模板、不重写已生成 token。
- 当前源码按职责命名：`train/run_experiment.py`（唯一入口）、`train/run_session.py`（子进程、阶段记录与检查点身份）、`train/run_artifacts.py`（输入打包、底座与回传清单）、`train_sft.py`、`train_grpo.py`、`train/rollout.py`（多轮执行）、`train/grpo_objective.py`（组内优势、裁剪目标）、`deepspeed_backend.py`（唯一训练后端，ZeRO-2）、`vllm_backend.py`（采样与权重同步）、`preflight.py`（GPU 运行前检查）、`experiment_config.py`（冻结常量）、`files.py`（文件哈希与 JSON 读写）、`eval/evaluate.py`（正式评测的执行、续跑、分析与报告）、`eval/rollout_eval.py`（开发集检查）、`eval/engineering_report.py`（性能测量报告）。政策直接定义于 `env/policy.py`；不添加旧版本跳转模块或运行时政策补丁。
- `preflight.py` 只核对进程组通信与 DeepSpeed 一次更新组的显存/更新，不验证 vLLM 采样、权重同步或任务成功率；训练前必须通过该检查。SFT 只保留正常训练指标、预算完成检查与开发集任务成功检查，不加入更新前后诊断。
- 训练以“一次全局更新”为边界：损失按整个更新组的全局监督 token 数归一化，DeepSpeed 内部只做一次梯度裁剪和一次优化器步，真实 microbatch 不足的 rank 用零损失对齐累积边界（不复制真实样本）。GRPO 训练恒为单进程，采样是另一张卡上的常驻 vLLM 进程；每次有效更新后按四步协议同步权重（传参 → 清前缀缓存 → 确认版本 → 接受下一批）。删除已被替换的 DDP 包装、手写工具轮缓存和重复优化器控制逻辑，不保留兼容层。
- `refunds` 统一称为“退款单”，与 `tickets` 工单、`mutations` 操作日志区分。当前每个订单最多要求一张退款单，双订单批量退款生成两张；信用补偿、工单、通知各最多要求一条。匹配逻辑按此边界直接核对，不为不存在的模糊配对场景加入回溯。
- 注释直接写明本项目在处理什么、核对什么；涉及任务示例时使用实际存在的任务，不用抽象数学术语代替具体业务说明。
- 所有阶段使用完整退款回库规则与 任务明确要求的业务政策；题面、训练轨迹、工具返回、奖励和评分须一致。按世界事实核对预期，不能仅以参考执行通过证明题意正确。
- 三组续训只有 C_blocked、C_mixed、U_mixed，均从同一 G0 独立开始。分解任务轨迹见 `minitoolrl/data/decomposition_tasks.py`；组合评测见 `minitoolrl/data/composition_suite.py`。
- SFT、GRPO、三组续训、开发集检查与正式评测的工具列表统一使用 registry 顺序；三组续训默认学习率为 `2e-6`。blocked/mixed 控制训练样本日程。
- 相同初始场景配不同需求是构造特点。区分具体实现、合理推断和实验证据，不把未经验证的机制解释作为结果。
- 当前入口独立组织单段 SFT → GRPO → 三组各一次续训。SFT 固定两轮、同一优化器和日程，保留终点开发集任务成功检查；开发集验收按冻结 `data/gate_spec.json` 逐项核对普通任务成功、补偿政策查询、信用调用与重复查询循环，任一不达标必须报错，不登记 SFT pin 且阻止后续 GRPO，不能只记录成绩后继续；金额正确率只报告、不设门槛。GRPO 只执行固定预算的采样、更新与同步，终点参加正式评测，不另做训练内开发集评测。
- 完整项目的 `engineering` 只执行性能测量：单卡与双卡 SFT 成对短测、DeepSpeed 训练卡 + vLLM 采样卡的 GRPO 短测。训练、采样与评测只保留 DeepSpeed + vLLM 这一套执行路径；删除 HF 生成、HF 同卡 GRPO、HF/vLLM 推理对照及其报告依赖，不保留可选或兼容入口。不保留独立测试目录、测试专用训练器、模拟接口或小模型更新核对脚本。
- 主训练默认 SFT 564 更新、GRPO 两遍 240 条任务池共 240 迭代、三组各 144 更新；工程默认 SFT 单/双 rank 各 40 更新、GRPO 训推分离 20 迭代。主链与短测分开记录，每个测量同时给墙钟秒数与 GPU 秒数。
- SFT、G0 和三组终点共同参加三族组合 360 条、通用 372 条和双订单退款 60 条 greedy 评测，共 3960 次任务执行。只报告完整配额和实际执行量。
- `make_plan()` 只用于 plan/init 制定计划；运行阶段从 `protocol.json` 读取冻结预算，不重新统计数据或重建计划。运行配置在 init 固定，已有成功阶段跳过；输入包在 init 复制一次并逐文件核对哈希，之后所有阶段只读引用，不改写输入。预算未完成、哈希不符、评测缺失和分析失败必须明确报错，不能写成功标记。检查点统一由 `run_session.py` 的 pin 登记与核对，GRPO 直接读取 SFT pin，不另建起点清单；评测续跑身份、配额核对与结果报告统一在 `eval/evaluate.py`；入口的 `evals` 阶段执行任务后直接生成报告，全部完成才写成功标记；串行与并行子进程共用执行实现，不重复增加校验层。当前流程版本为 `deepspeed-vllm-engineering-qwen3-v8`。
- 不改写已完成运行的输入、模型、日志或分数。新的训练和评测写独立运行目录。
- 当前交付统一写入 `dist/current/`。已完成实验只保留 `dist/resume_evidence_20261016/results_inputs_logs_source.tar.gz` 一份最终压缩归档；本地 SFT/G0 权重在 `minitoolrl/artifacts/models/reference_endpoints/`，不作为默认新运行的起点。当前只依据本文件所列源码、冻结输入和该归档回答。
