# 运行说明

## 执行范围

`preflight` 检查目标 GPU 环境；`engineering` 执行单/双卡 SFT 各 40 次更新和训推分离 GRPO 20 次迭代，记录性能。开发检查独立按需执行。

采样协议开发检查可按需手动运行 `python -m minitoolrl.tests.test_vllm_protocol`。

GRPO 每批轨迹只执行一次优化器更新，所有 microbatch 的前向都发生在更新前。损失直接使用同一次前向的 `lp` 和 `old_lp = lp.detach()`，不单独执行无梯度前向。概率比数值为 1，但分子保留梯度；单次更新中裁剪不触发。采样器返回的 logprob 与该前向保留的概率用于训推差异统计。每次有效更新完成后，将新权重同步给 vLLM，再采样下一批。

修改这段实现时，可手动运行 `python -m minitoolrl.tests.test_grpo_single_update`，核对前向次数、梯度、单步参数更新和训推差异统计。该 CPU 检查不属于正式训练，也不能代替 GPU 性能测量。

## 完整训练前提

需要 Linux、两张支持 BF16 的 NVIDIA GPU、足够显存与磁盘空间，以及对应版本的冻结输入包。GRPO 的训练与 vLLM 采样分别占用一张 GPU。GPU 依赖见 [requirements-gpu.txt](../requirements-gpu.txt) 与 [pyproject.toml](../pyproject.toml)。CPU 和 GPU extras 是互斥环境，不同时安装。

仓库本次仅发布代码与摘要，不包含训练输入包、底座与训练权重。`input_bundle --verify`、`plan` 和完整流程需要输入包，不能在缺失输入时直接运行。`data/` 下的源码提供任务池、轨迹与组合任务构造；`rebuild_inputs` 还要求已有输入清单和 tokenizer，不是从空目录生成完整冻结实验的命令。

准备好对应输入包后，命令形式为：

```bash
python -m pip install -e ".[gpu]"
python -m minitoolrl.train.run_experiment --help
python -m minitoolrl.train.run_experiment --stage init \
  --run-dir /path/to/new-run --inputs-root /path/to/inputs_bundle \
  --world-size 2 --gpus 0,1 --arm-lr 2e-6 --eval-tool-order registry
python -m minitoolrl.train.run_experiment --stage fetch_base --run-dir /path/to/new-run
python -m minitoolrl.train.run_experiment --stage preflight --run-dir /path/to/new-run
python -m minitoolrl.train.run_experiment --stage sft --run-dir /path/to/new-run
python -m minitoolrl.train.run_experiment --stage grpo --run-dir /path/to/new-run
python -m minitoolrl.train.run_experiment --stage arms --run-dir /path/to/new-run
python -m minitoolrl.train.run_experiment --stage engineering --run-dir /path/to/new-run
python -m minitoolrl.train.run_experiment --stage evals --run-dir /path/to/new-run
python -m minitoolrl.train.run_experiment --stage analysis --run-dir /path/to/new-run
```

这是一条新运行的完整流程。已发布的 2026-10-16 三臂实验复用了此前 SFT / G0，再分别完成三组续训与全部评测；不能把该续训运行计时解释为从官方底座训练全部阶段的耗时。新运行不保证逐位复现原成绩。

官方底座是 `Qwen/Qwen3-0.6B`，固定 revision `c1899de289a04d12100db370d81485cdf75e47ca`。输入、协议、模型与结果分别保存；新训练写入新目录，保留已有结果。
