# DeepSpeed 与 vLLM：它们怎样接管训练和生成

本文从一个具体问题开始：**自己写训练循环和生成循环，模型一样能运行，为什么还要引入 DeepSpeed 和 vLLM？引入以后，到底把哪些工作交给了它们？**

理解这件事以后，再看初始化参数和函数调用，才知道这些调用建立了什么、解决了什么问题。本文先讲执行机制，再把它们对应到项目里的对象、进程和一次完整运行。

阅读顺序：第 1–5 节讲 DeepSpeed 更新；第 6–9 节讲 vLLM 的计算、缓存、调度及初始化；第 10–18 节对照项目对象和调用；第 19–21 节讲权重同步；第 22–23 节说明性能与正确性怎样验证。只想理解 prefill/decode 优化时，先读第 6–8 节；其中第 8.2 节集中列出项目配置的证据。

适用范围为输入格式 `all-restock-v1-20261011`、默认新运行 `tool_composition_qwen3_registry_lr2e6`，以及当前 `deepspeed-vllm-engineering-qwen3-v8` 源码。输入已校验，该新运行的完整 GPU 训练尚未执行；文中的加速机制是原理，具体性能收益仍需工程测量。GPU 依赖见 [pyproject.toml](../../pyproject.toml)：DeepSpeed 0.19.7、vLLM 0.31.0。

## 一、先理解它们为什么有用

### 1. 模型负责计算，执行系统负责组织计算

Qwen3 的网络结构和权重，决定了输入 token 怎样变成输出概率。一次前向计算大致经过嵌入、各层注意力与 MLP、词表输出层。

但一个模型对象不会自行决定这些事情：一张卡装不下训练状态怎么办；8 条长短不同的请求怎样一起生成；哪些历史计算要保留；谁负责更新参数；另一进程中的模型什么时候换成新参数。

这些都是围绕模型组织执行的工作。手写代码可以完成，框架也可以完成。两者差别在于执行机制、资源管理，以及需要自己承担的实现量。

| 层次       | 在 MiniToolRL 中回答的问题                        | 负责者                                               |
| ---------- | ------------------------------------------------- | ---------------------------------------------------- |
| 业务和实验 | 退款需要哪些操作？什么算成功？训练哪些任务？      | 任务、工具、评分器、阶段脚本                         |
| 学习目标   | 哪些 token 计入 SFT 损失？GRPO 优势怎样算？       | 项目的损失与奖励代码                                 |
| 训练执行   | 梯度存在哪里？怎样跨卡合并？何时更新参数？        | DeepSpeed 引擎，加上项目的更新组约定                 |
| 生成执行   | 怎样安排 prefill/decode？怎样管理生成缓存和请求？ | vLLM 引擎                                            |
| 接入代码   | 怎样把项目数据和控制请求交给引擎？                | `DeepSpeedTrainer`、`LocalVLLM`、`VLLMRollout` |

最后一行的三个类都是**这个项目自己写的类**。`DeepSpeedEngine`、`vllm.LLM` 才是框架提供的主要对象。读项目封装时，要继续追到它持有或调用的框架对象，才能知道实际工作发生在哪里。

### 2. 手写单卡训练为什么已经能工作

设已经有一个模型、一个 AdamW 优化器和一批 SFT 轨迹。教学用的训练流程可以写成：

```python
optimizer.zero_grad()
for microbatch in update_group:
    loss = compute_loss(model, microbatch)
    loss.backward()
clip_grad_norm_(model.parameters(), max_norm)
optimizer.step()
```

每次 `backward()` 把梯度累加到参数对应的梯度存储中，最后 `optimizer.step()` 根据累积结果修改参数。把一次更新的数据分成几个 microbatch，是为了每次只保留较小批次的中间激活；整个更新组仍只更新一次参数。

这里没有 DeepSpeed，也完全可以学习。代码能工作的前提是模型参数、梯度、优化器状态和当前计算需要的中间结果，都能放进这张卡。

AdamW 除了当前梯度，还要维护梯度的一阶、二阶历史统计。混合精度训练又常需要 fp32 主权重，以免微小更新在低精度权重上被舍入掉。所以“模型权重只有几 GB”并不能推出“训练也只需要几 GB”。

### 3. 增加一张卡，为什么还需要考虑训练状态

最直接的多卡数据并行是让两个进程各拿一份模型、处理不同数据，再合并梯度。两个进程称为 rank 0、rank 1，通信组的 `world_size` 为 2。

```text
GPU 0：完整模型 + 完整梯度 + 完整 AdamW 状态，处理一部分数据
GPU 1：完整模型 + 完整梯度 + 完整 AdamW 状态，处理另一部分数据
                         ↓ 合并梯度
                    两边得到一致的更新
```

这可以分摊计算，却仍在每张卡重复保存完整优化器状态。若单卡显存主要被这些状态占满，仅仅增加一份数据并行副本，未必能解决问题。

ZeRO 的出发点是：既然各 rank 最终要得到同一套更新，就不必让每个 rank 都保存并更新全部优化器状态。ZeRO-1 分片优化器状态；ZeRO-2 再分片规约后的梯度；ZeRO-3 进一步分片模型参数。项目采用 **ZeRO-2，模型参数仍在每个训练 rank 上保留完整副本**。参见 [DeepSpeed 官方 ZeRO 教程](https://www.deepspeed.ai/tutorials/zero/)。

### 4. ZeRO-2 分片之后，参数怎样更新

下面把参数逻辑上分成“前半”和“后半”来理解。真实实现会按张量和通信桶组织，未必沿网络层边界切分。

```text
                         rank 0                         rank 1
前向所用参数：        前半 + 后半                    前半 + 后半
长期维护的优化状态：  前半对应的状态                 后半对应的状态

① 两边各自用完整模型处理自己的数据，产生梯度贡献。
② 通信合并同一参数来自不同 rank 的贡献，并把结果归到负责该分片的一侧。
③ rank 0 更新前半主权重；rank 1 更新后半主权重。
④ 交换更新后的参数分片，让两边的完整模型副本重新一致。
⑤ 进入下一次更新。
```

第②步可以理解为“规约＋分发”，第④步可以理解为“把分片收集回来”。规约是按约定把多个数值相加或平均；收集是让各方拿到其他分片。具体采用的通信实现由框架及配置决定。

你可以自己写这些步骤，但还必须处理分片布局、通信顺序、梯度累积、高低精度状态转换、全局梯度范数和参数刷新。ZeRO 提供这一整套相互配合的训练状态管理。基本方法见 [ZeRO 原始论文](https://arxiv.org/abs/1910.02054)。

因此，`deepspeed.initialize()` 不只是把几个参数存进一个对象。它根据模型、优化器和配置，建立后续反向传播与更新需要的执行状态。初始化后仍能看到原来的模型，但训练状态的管理已经交给引擎。

这里的显存节省主要来自训练状态分片。模型参数和当前 microbatch 的激活仍有开销，通信也需要缓冲；不能把 ZeRO-2 理解成“所有显存自动除以卡数”。

### 5. 为什么 backward 和 step 也交给 DeepSpeed

手写循环里的 `loss.backward()` 表达“沿计算图求导”。在分片训练里，求出的梯度还要进入正确的缓冲、通信和累积流程。项目使用 `engine.backward()` 进入框架管理的反向路径；PyTorch autograd 仍负责求导，DeepSpeed 负责围绕它组织训练状态。

`engine.step()` 则把“目前是不是累积边界、该不该裁剪、该不该调用优化器、更新后怎样刷新模型参数”连接起来。若外层再调用原始 `optimizer.step()`，就可能绕过这套约定或重复更新。

默认托管累积模式下，每个 microbatch 都按 `backward(); step()` 配对调用。`step()` 每次推进微步计数，只有累积边界才执行真正的优化器更新。官方 API 说明见 [DeepSpeed Training API](https://deepspeed.readthedocs.io/en/stable/training.html)。

所以，处理 8 个 microbatch 可以调用 8 次 `engine.step()`，同时只发生 1 次优化器更新。代码里看到 `step`，要先确认它指的是哪种步骤。

引擎并不知道“哪些 token 是工具观察”“哪些输出要学”“这 9 个 microbatch 算一次完整更新”。这些仍由项目决定，框架不会自动修正错误的损失定义。后面会看到，项目还要明确告诉它梯度是求和还是平均。

### 6. Prefill 与 decode：计算和缓存怎样优化

Qwen3 的同一套网络执行 prefill 和 decode。Prefill 处理已知提示，建立各层的 K、V 缓存，并产生预测首个输出 token 所需的结果；decode 把刚生成的 token 送入模型，读取历史 KV，预测下一个 token。这里 decode 指逐步生成阶段；decoder 指网络结构，两者含义不同。

```text
输入提示 → 前向计算 → 选出第一个输出 token
                          ↓
新 token + 已有 KV cache → 前向计算 → 选下一个 token
                          ↓
                    重复直到结束
```

刚选出的 token 要到下一次被送入模型时，才会产生自己的 K、V。缓存省去历史 token 的重复前向计算；新的 query 仍要读取历史 KV。

下面从**已有 KV cache 和普通 batch 的推理循环**出发，解释 vLLM 集成的高效注意力、前缀缓存、连续批处理、分块 prefill、分页 KV 与执行优化。这些机制也可以由其他框架或手写程序实现；比较新增收益时，要看对照实现已经具备哪些能力。本文说明机制，项目没有保留另一套推理路径做性能对照。

#### 6.1 Prefill 和 decode 的瓶颈为什么不同

以一次工具调用前的上下文为例，假设输入已有 1000 个 token，模型接下来要生成工具调用。这些长度只是教学数字：

| 阶段    | 这次前向处理什么                                | 注意力需要做什么                                     | 常见瓶颈                                             |
| ------- | ----------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------- |
| Prefill | 1000 个已知输入 token，可以在每层中成批计算     | 每个位置的 query 关注截至该位置的 K、V，同时建立缓存 | 大矩阵计算，以及注意力中间结果的显存读写             |
| Decode  | 常规自回归下，每条活跃请求每步处理 1 个新 token | 新 query 读取历史 KV 和本位置 KV，预测再下一个 token | 小批次时的权重、KV 读取，以及反复启动 GPU 操作的开销 |

Prefill 能并行处理多个位置，是因为输入 token 已经全部知道。因果 mask 限制每个位置不能读取未来位置，但不要求像生成未知答案那样逐 token 调用模型；网络层之间仍有依赖。处理很多 token 时，线性层可以让同一份权重参与多行输入的矩阵乘法，提高每次读取权重所完成的计算量。

Decode 的单条请求每步只有一行新输入，却仍需经过整套网络。权重常驻显存，只是免去了重新从磁盘加载；GPU 计算单元仍要从显存或片上缓存读取权重。上下文越长，注意力还要读取越多历史 KV。**保存了 KV，不等于后续注意力不再访问历史。**

所以常说 prefill 更偏计算受限、decode 更偏显存带宽受限。这是常见工作区间，不是绝对分类：短提示、小模型可能明显受启动开销影响；decode 批次增大后计算压力也会上升。vLLM 的优化正是围绕这些不同形状的工作安排执行。[vLLM 性能调优说明](https://docs.vllm.ai/en/v0.31.0/configuration/optimization/#chunked-prefill)

#### 6.2 高效注意力：减少计算中的数据搬运

**vLLM 集成 FlashAttention 等注意力后端，承担模型中的注意力计算。** FlashAttention 是其支持的一种实现；是否选择它以及使用哪一代，由硬件、模型、数据类型和配置共同决定。[vLLM 后端选择源码](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/platforms/cuda.py)

**Prefill：避免大中间矩阵反复落入显存。** 对于长度为 L、没有可复用前缀的 prefill，每个注意力头的直接实现可以先算出 `QKᵀ`，将约 `L × L` 的分数矩阵写到显存，再做 softmax 和与 V 相乘。读写这些大中间矩阵会消耗带宽和临时显存。这是解释算法时使用的直接实现，不代表普通 PyTorch 必然这样执行。

FlashAttention 一类算法把 Q、K、V 分块，在较快的片上存储中完成局部计算，并通过在线 softmax 合并结果，避免把完整的注意力分数、概率矩阵落到显存再读回。它保持稠密注意力的数学定义，主要减少 IO 和中间存储；并没有把全长稠密注意力的计算量从平方级变成线性级，浮点执行也不保证逐位相同。[FlashAttention 原论文](https://arxiv.org/abs/2205.14135)

这一优化也适用于训练中的长序列注意力，并且可被多个框架采用。它属于 vLLM 集成的优化；若对照程序已使用同等高效的注意力实现，就不能把它完整计作换用 vLLM 的额外收益。

**Decode：融合单个 query 的注意力计算，必要时沿历史 KV 分工。** 每条请求在每个注意力头上只有一个新 query，面对长度为 L 的 KV，分数只有约 `1 × L`。因此，decode 不能照搬“省去 `L × L` 中间矩阵”的解释；它要高效完成这个 query 对已有 KV 的读取、打分和加权求和。

先看单个请求、当前层的一个注意力头。投影得到当前 token 的 q、k、v 后，新 k、v 写入缓存；注意力计算需要用 q 对包括当前位置在内的所有可见 K 打分，经 softmax 后对 V 加权求和。高效内核可以这样组织：

1. **分块读取并融合计算。** 将一小块 K、V 从显存读入较快的片上存储，计算这一块的 `qKᵀ`，更新 softmax 的统计量和 V 的加权和，再处理下一块。利用在线 softmax，可以边读边累积，无需把完整的 `1 × L` 分数、概率向量作为独立中间张量写回显存再读取。对于分页 KV，内核通过块表定位数据，无需先把整段 KV 复制、拼成连续张量。所有可见 KV 仍需参与计算，这一步减少的是中间数据搬运。
2. **在并行任务不足时，进一步切分 KV 范围。** Prefill 有许多 query 位置可分给 GPU；decode 每头只有一个 query，当 batch 和头数提供的并行工作不足时，部分 GPU 计算单元可能闲置。支持 split-KV 的内核可以把同一 query 对长历史的计算分成多个任务，让不同线程块分别处理不同 KV 段。各段内部仍采用上面的融合计算。
3. **合并各段结果。** 每段输出局部加权结果，以及对应的 softmax 归一化信息（常用 log-sum-exp）。合并内核据此重新缩放各段贡献，得到整个上下文的注意力输出。不能直接平均各段结果，因为各段获得的总注意力权重不同。

例如，一条退款任务的当前层共有 2048 个 KV 位置，示意分成 4 段、每段 512 个位置：

```text
                       同一个 query q
              ┌────────┼────────┼────────┐
              ↓        ↓        ↓        ↓
KV 位置：   0–511   512–1023 1024–1535 1536–2047
计算任务：    ①        ②        ③        ④
              └────────┴────┬───┴────────┘
                   按归一化信息合并
                            ↓
                  当前层这个头的注意力输出
```

这里增加的是**同一张 GPU 上、同一个请求的当前一步内部**的并行工作。计算完成后，还要经过后续投影、MLP 和网络层，才得到预测下一 token 所需的结果；它没有同时生成未来的多个 token。上述按 KV 长度分工的思路常称为 split-KV，Flash-Decoding 是其代表性说明。[Flash-Decoding 原理与合并过程](https://pytorch.org/blog/flash-decoding/)

分段的代价是保存局部结果并增加合并工作。长上下文、小 batch 更可能受益；上下文短或已有足够并行工作时，分更多段可能更慢，所以分段数通常由实现和输入形状决定。计算上的分段与 PagedAttention 的存储块是两个概念，也不等于把新提示分多轮处理的 chunked prefill。

vLLM 0.31.0 的 FlashAttention 后端把 `query`、`key_cache`、`value_cache`、`block_table` 和分段信息传入内核；其 FA3 等支持路径接收 `num_splits`，接口约定 `0` 由内核选择、`1` 不做多段拆分。这说明 vLLM 接入了这种执行能力；不同代际和路径的支持范围有区别，当前项目实际是否采用 split-KV 仍需运行证据。[FlashAttention 后端](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/attention/backends/flash_attn.py)、[分段参数与版本分派](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/vllm_flash_attn/flash_attn_interface.py)

第 7.1 节的连续批处理讨论的是**哪些请求一起进入计算**；这里讨论的是**一个请求的注意力怎样在 GPU 内完成**。第 7.2 节的 CUDA Graph 则减少每一步反复提交 GPU 操作的成本，三个层次可以配合使用。

#### 6.3 前缀缓存：减少重复 prefill

Automatic Prefix Caching（自动前缀缓存）让新请求复用已有请求算出的前缀 KV。例如退款回库任务的一轮工具返回后，下一轮输入包含旧历史和新工具观察；在模型权重不变、前缀 token 完全相同且缓存仍可用时，可以跳过命中前缀的重复前向计算。未命中的后缀仍要计算，其 query 仍需关注旧前缀的 KV。

这不是按“语义相似”复用。相同文字片段如果前面的上下文不同，也不能直接视为相同 KV。vLLM 按前缀关系识别可复用块，并受完整块边界、缓存保留情况等条件限制，不能假定所有相同 token 都免算。前缀缓存主要减少重复 prefill，不会替模型生成后面的未知 token。[自动前缀缓存使用说明](https://docs.vllm.ai/en/v0.31.0/features/automatic_prefix_caching/)、[块识别与复用设计](https://docs.vllm.ai/en/v0.31.0/design/prefix_caching/)

### 7. 请求调度与执行：让 decode 持续推进

#### 7.1 连续批处理：在生成过程中调整批次成员

普通 batch 已经可以一次处理多条请求。**连续批处理（continuous batching）允许在生成过程中，按调度步调整 batch 的成员，而不必等这一批全部生成结束。** 假设 A、B 正在生成，C 已提交但还在等待：

| 时刻                 | 固定批次策略                                                                   | 连续批处理策略                                               |
| -------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------ |
| A、B 都未完成        | A、B 一起推进                                                                  | A、B 一起推进                                                |
| A 先完成，B 尚未完成 | 当前批次只剩 B 有效生成；可能保留 A 的填充位置，也可能缩小 batch，但暂不接纳 C | A 退出；资源允许时安排 C 的 prefill，随后让 B、C 一起 decode |
| B 完成               | 才开始下一批 C                                                                 | C 可能已经推进了若干步                                       |

这里新增的是按生成进度调整批次成员。vLLM 将它与缓存管理配合：分页缓存减少预留与碎片，让同一显存预算更有机会容纳活跃请求；调度器再决定实际接纳哪些请求。这样有机会提高总吞吐、减少等待，但更大的 batch 也可能增加一次前向的耗时，不能据此保证 B 自身的每个 token 都更快。[PagedAttention 与请求调度论文](https://arxiv.org/abs/2309.06180)

#### 7.2 算子融合与 CUDA Graph：减少每一步的执行开销

Decode 会重复执行许多短 GPU 操作。适用时，算子融合减少中间张量读写和 kernel 数量，CUDA Graph 重放减少 CPU 逐个提交操作的成本；它们仍然执行本步的网络计算。这两类机制也能用于适合的 prefill 或混合批次，具体范围由配置和算子支持决定。

以 CUDA Graph 为例：若每步时间主要花在提交许多短操作上，减少提交的相对收益就可能较大；若主要花在长矩阵运算或读取 KV 上，只减少提交的相对收益就可能较小。具体能捕获哪些操作还受注意力实现与配置约束，第 9 节继续解释编译和图捕获。[vLLM CUDA Graph 设计](https://docs.vllm.ai/en/v0.31.0/design/cuda_graphs/)

#### 7.3 分块 prefill：协调长提示与正在生成的请求

新请求需要 prefill，已有请求需要 decode。如果一次执行很长的 prefill，已有请求就可能很久等不到下一次 decode。**分块 prefill 把长提示拆成多次调度，让已有请求在两块之间也能继续生成。** 每块仍会看到此前上下文，不是把提示拆成互不相关的几段。

vLLM V1 在支持的条件下默认启用 chunked prefill。对于已有 decode 请求和等待进入的新提示，常见安排是先推进已有请求，再用剩余 token 预算处理新提示；放不下的长 prefill 分块完成。`max_num_batched_tokens` 约束一轮输入计算的 token 预算，不是所有请求已有 KV 的总长度。[分块 prefill 与调优](https://docs.vllm.ai/en/v0.31.0/configuration/optimization/#chunked-prefill)

具体到 vLLM 0.31.0，调度器先遍历运行中的请求，再考虑等待请求。运行中的请求也可能仍在处理部分 prefill，因此“优先 decode”是常见调度效果，不能理解为任何负载下都先处理所有 decode。实际顺序还受请求队列、token 预算和可用 KV 块限制。[调度器 `schedule()` 源码](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/core/sched/scheduler.py)

假设本轮预算为 10 个 token，两条退款任务轨迹正在 decode，另一条任务有 20 个尚未处理的提示 token。忽略其他限制，示意安排为：

```text
调度轮次     已有轨迹 A、B               新任务 C
   1        各处理 1 个新 token          prefill 前 8 个 token
   2        各处理 1 个新 token          prefill 再 8 个 token
   3        各处理 1 个新 token          prefill 剩余 4 个，得到首 token
   4        A、B、C 各推进一步 decode（假设都未结束）
```

这里的 10 是教学预算，不是项目配置。混合批次让多 token 的 prefill 与单 token 的 decode 一起进入调度，有机会提高 GPU 利用率；这不意味着两者必定在两个独立 CUDA stream 中同时执行，也不意味着 prefill 对 decode 完全没有影响。

可用于 prefill 的预算小一些，通常更容易控制已有请求的 token 间隔；预算大一些，可以减少长提示跨越的调度轮数，也可能提高吞吐。前者关注 ITL（相邻输出 token 的间隔），后者还涉及 TTFT（从提交请求到得到首 token 的时间，包含排队）。分块改变调度粒度，不会自行减少整条提示所需的计算量；最合适的预算需结合负载测量。

#### 7.4 本项目的工具轮仍有等待边界

MiniToolRL 一轮 GRPO 默认有 2 个任务，每任务 4 条轨迹，共 8 条。`rollout_pool()` 把本轮仍活跃的轨迹一次提交给阻塞式 `generate()`；等本次所有请求返回后，才逐条执行工具、追加观察并提交下一轮。[项目多轮执行代码](../train/rollout.py)

因此，vLLM 可以在**本次已经提交的请求**之间调度；但即使 A 先生成完工具调用，项目也要等 B 等其他请求返回后，才执行 A 的工具并提交 A 的下一轮。如果这 8 条已经全部进入执行，之后也没有等待请求，就没有额外请求可补入。连续批处理的实际收益取决于待处理请求和资源限制，不能仅凭使用了 vLLM 就断言这一小批轨迹获得了显著加速。

这里有两层循环：vLLM 安排 token 计算，项目安排工具交互。当前实现也没有让训练更新与下一批采样异步重叠。

### 8. PagedAttention 怎样管理缓存

**可以把它理解为减少 KV 缓存的碎片和预留浪费，但不是压缩 K、V 本身。** 在模型、精度和 token 数相同且不考虑共享时，需要保存的 KV 数据量不会仅因为分页就变小。改变的是这些数据怎样占用、增长和释放缓存空间。

用一个简化例子理解分块。假设每块放 4 个 token 的 KV，某请求已有 9 个位置，就需要 3 个逻辑块；它们可指向显存池中不相邻的存储块。PagedAttention 语境中的“物理块”指缓存池内实际分配的块，不等于 GPU 硬件页表里的物理页：

```text
请求的 token 位置：  [0 1 2 3] [4 5 6 7] [8 _ _ _]
逻辑块编号：              0         1         2
块表指向物理块：         12         3        27
```

注意力计算通过块表找到 K、V。请求继续生成时，先利用最后一块的剩余位置，必要时再分配新块，不要求预先找到一整段容纳最大长度的连续空间。这里的 4 只是教学数字。

**如果提前知道最终长度，能不能连续分配？可以。** 如果只运行一个请求，或者每个请求都能拿到足够大的连续空闲区域，连续分配完全合理，也不必承担分块映射的管理成本。分页不是“推理才能运行”的必要条件。问题在于以下条件往往不能同时满足：

1. **知道上限，不等于知道实际长度。** `max_new_tokens=160` 是最多再生成 160 个 token，不代表一定生成 160 个；模型可能提前结束。按上限给每条请求预留整段空间，会把部分空间留空。
2. **知道最终大小，不等于现在有连续空间。** 假设某个缓存池的空闲情况如下，新请求明确需要 6 块：

   ```text
   [空闲 2 块][占用][空闲 2 块][占用][空闲 2 块]
   ```

   总空闲量足够，但没有连续 6 块。要求整个请求连续存放，就可能需要等待、搬移数据或重新规划；按块分配则可以直接使用这些空闲块。这是外部碎片问题。这里描述的是缓存池中的地址布局，不是在讨论 GPU 底层物理内存是否连续。
3. **提前知道以后会用，不等于现在就值得全部占住。** 即使准确知道最终需要 100 块，当前可能只写到了第 10 块。如果一次把 100 块都划给它，剩余 90 块暂时不能供别的请求使用。按进度分配能让这些空间先服务其他请求；但之后增长仍需要容量，分页并不保证永远不会耗尽缓存。

所以，分页特别适合“多条请求长度不同、不断增长、完成时间不同”的场景。它也不是零浪费：最后一块可能没有填满，上例占了 12 个位置而只用了 9 个；块表和分配管理也有开销。若工作负载完全固定、事先能做好连续布局，连续方案未必差。

这里还容易混淆两层“分配”：**引擎可以在启动时先申请一个较大的 GPU KV 缓存池，再在池内按需给请求分配块。** 因而“请求按需增长”不等于“每生成一个 token 就向 CUDA 申请一次显存”，也不等于进程的显存占用一定随 token 数缓慢增长。分页主要改善已留给引擎的空间如何被各请求使用。

上述示意解释的是 [PagedAttention 原论文](https://arxiv.org/abs/2309.06180)的内存管理思路。官方的 [Paged Attention 设计页](https://docs.vllm.ai/en/v0.31.0/design/paged_attention/)已标注为历史说明，不能把其中的内核布局直接当成当前版本所有注意力后端的实现。项目调用引擎，不自行维护这张块表。

还要区分：KV cache 避免同一序列后续生成时重算历史；前缀缓存让不同请求在满足复用条件时，共享相同前缀已经算出的 KV。多轮工具调用再次提交历史上下文时，后者也可能减少重复 prefill，但不能因此假定本项目每轮都已命中缓存。

#### 8.1 GPU 已有虚拟内存，为什么还需要应用层分页

现代 CUDA GPU 确实使用虚拟地址。程序看到一段连续的 GPU 地址，不代表底层显存物理页必须连续。CUDA 还提供 VMM（Virtual Memory Management）接口，将预留虚拟地址、分配物理存储、建立映射分别控制。参见 [CUDA 虚拟内存管理](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html)。

因此，不能用“GPU 只能分配连续物理显存”解释 PagedAttention。这里同时存在两个层次的映射：

```text
请求中的 token 位置
    ↓ 引擎块表：该请求的第几块，位于缓存池的哪个块
缓存池中的 GPU 虚拟地址
    ↓ GPU 页表：虚拟页对应哪一个物理页
实际 GPU 物理存储
```

底层页表负责地址翻译，不知道某条请求何时结束、某段 KV 是可共享前缀，或者一条轨迹下一步会增加多少 token。引擎知道这些语义，因而可以在请求结束时归还缓存块，在满足条件时共享前缀，在增长时追加块。

例如引擎已经申请一个缓存池，池内空闲块的虚拟地址相互分散。GPU 页表不会因为某请求需要这些块，就自动把它们重新呈现成该请求专属的一段连续虚拟数组。普通张量分配也不等于“只预留地址，未来写到哪里才为哪里提供物理存储”。这与 CPU 已有虚拟内存、程序仍需要分配器和可增长数据结构的关系类似。

PagedAttention 采用的办法是在应用层保留块表，让注意力内核按表访问分散的 KV。这里的内核是 GPU 上执行的计算程序。其系统设计把请求管理、缓存分配、共享和计算访问方式接在一起，并不是重新发明 GPU 硬件分页。

**也可以更直接利用底层虚拟内存。** 例如先为请求预留较大的连续虚拟地址范围，只为已用部分分配和映射物理存储，后续再增长。[vAttention 论文](https://arxiv.org/abs/2405.04437)研究的就是这条路线：保持 KV 虚拟地址连续，通过 CUDA VMM 按需提供物理存储。实现仍要处理映射粒度、映射开销和请求生命周期。

因此，这里有不同实现方案，而不是“已有 GPU 虚拟内存，所以请求级存储管理自然完成了”。虚拟地址空间很大，也不意味着物理显存无限或普通 CUDA 分配都会自动溢出到主机内存。本项目使用 vLLM，不另行实现 VMM 分配器；上述方案比较只用于理解系统层次，不代表本项目做过性能对照。

#### 8.2 项目配置、默认行为与运行证据

**FlashAttention、PagedAttention 和 SDPA 分别描述不同层次。** FlashAttention 是高效注意力算法及实现；PagedAttention 将分页 KV 管理与按块表访问 KV 的计算结合；SDPA 是缩放点积注意力（Scaled Dot-Product Attention）的简称。在本项目的 `attn_implementation='sdpa'` 配置中，SDPA 特指通过 PyTorch 的 `torch.nn.functional.scaled_dot_product_attention` 接口执行注意力。

SDPA 接口可以选择 FlashAttention 或其他适用内核，所以“SDPA”和“FlashAttention”可以同时成立。FA2、FA3 分别是 FlashAttention-2、FlashAttention-3，表示不同代实现；接口名不能证明实际选择了哪一代。GPU 依赖固定为 PyTorch 2.13.0，其接口说明列出了可用实现及自动选择行为。[PyTorch 2.13.0 接口源码与说明](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/nn/functional.py#L6341)

训练和采样的选择分别发生在两条路径中：

```text
训练：Transformers Qwen3 → PyTorch SDPA → 适用的注意力内核
      DeepSpeed 管理反向传播、训练状态与更新

采样：vLLM Qwen3 → vLLM 选择的注意力后端 → 对应内核
      vLLM 管理请求调度与 KV 缓存
```

兼容的 FlashAttention 实现也能读取分页 KV，二者可以配合使用。训练侧的 SDPA 配置不控制采样侧的后端。对照项目的 [train_sft.py](../train/train_sft.py)、[train_grpo.py](../train/train_grpo.py) 和 [vllm_backend.py](../train/vllm_backend.py)，可以确定以下边界：

| 项目                | 源码与固定版本默认规则能确认什么                                                                         | 本运行仍需什么证据                                                      |
| ------------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| 训练侧注意力        | SFT、GRPO 都显式设置 `attn_implementation='sdpa'`；监督续训复用 SFT 路径                               | 实际底层内核需结合运行环境与 profiler 等执行记录确认                    |
| 采样侧注意力        | 两处 `LLM(...)` 均未指定注意力后端，由 vLLM 按运行条件选择                                             | 启动日志/最终配置中的后端；如需具体 FA 代际，还需核对版本选择与执行记录 |
| 前缀缓存            | 未显式覆盖 `enable_prefix_caching`；vLLM 0.31.0 对当前 Qwen3 因果生成模型默认启用                      | 缓存是否命中、命中多少 token、节省多少时间                              |
| 分块 prefill        | 未显式覆盖 `enable_chunked_prefill`；同版本对此模型默认启用；`max_num_batched_tokens` 由默认规则确定 | 最终 token 预算，以及实际请求是否需要拆成多块                           |
| 图执行              | `LocalVLLM` 显式设 `enforce_eager=False`；采样服务沿用 `LLM` 的同名默认值 `False`                | 实际编译、图捕获和重放范围，以及准备时间与显存开销                      |
| 连续批处理与分页 KV | 请求交给 vLLM 引擎调度和管理；项目未配置推测解码                                                         | 本批并发情况、排队与缓存压力，以及相应吞吐和延迟                        |

默认启用的依据是 `EngineArgs._set_default_chunked_prefill_and_prefix_caching_args()`，它读取 `ModelConfig.is_chunked_prefill_supported` 与 `is_prefix_caching_supported`；当前 Qwen3 属于支持两者的因果生成模型。这些默认行为可以从固定版本代码推导，实际运行时再核对最终配置。[参数默认规则](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/engine/arg_utils.py)、[模型支持条件](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/config/model.py)、[LLM 参数默认值](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/entrypoints/llm.py)

启用前缀缓存不保证每轮命中，启用分块 prefill 也不意味着每条提示都会分块。有效训练更新后，项目清除旧前缀缓存并确认新权重版本，旧权重对应的 KV 不能跨更新复用。两张卡分别负责训练和采样；prefill 与 decode 都在采样侧执行。

注意力实现关系到后训练的速度、显存和浮点数值差异，但理解学习过程应先掌握损失、监督位置、奖励与优势、采样概率、更新边界和权重同步。FA 代际的内核细节可在性能与数值排查时继续学习，复现时则应记录实际环境和实现。

当前工程短测记录整体采样、训练、同步等耗时与吞吐，不提供各项优化的独立加速归因。一次阻塞式 `generate()` 的总耗时也不能给出逐请求 TTFT、ITL 或前缀命中收益。默认新运行尚未执行，因此本文不报告这些机制带来的实测增益。

### 9. vLLM 初始化到底比简易推理多做了什么

**加载权重、建立模型、准备 GPU 环境，以及为采用 KV cache 的推理保存 K、V，都是共同成本，不能算作 vLLM 独有的额外开销。** 不用 vLLM，自己写推理循环也需要做这些事。模型常驻后不必反复加载权重，同样不是 vLLM 的专属能力。

比较时先约定基线：同一个模型、相同精度和设备，手写普通 PyTorch eager 推理循环，不额外做图编译或 CUDA Graph 捕获，按固定批次生成。这里 eager 指由程序逐次调用算子执行，而不是提前准备可重放的整段 GPU 执行流程；它仍然使用已有的高效底层算子，不等于所有计算都用 Python 完成。这只是帮助理解的假想基线，不是本项目保留的另一条执行路径。

相对这条基线，vLLM 的准备工作可分成下面几类；具体是否执行、执行多少，取决于版本、配置、模型和缓存状态。

| 准备工作                  | 具体多做什么                                                       | 为什么要做、付出什么                                                       |
| ------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------- |
| 显存探测与容量规划        | 在正式请求前执行模拟前向，估计模型运行的显存需求，规划 KV 缓存容量 | 不仅看权重有多大，还要给运行中的临时张量留空间；代价是启动时先做一部分计算 |
| KV 缓存池与管理结构       | 建立缓存池及块管理资源，而不是只给眼前的一条请求准备张量           | 后续请求能反复分配、回收块；启动时会预留显存，运行时还有块表等管理开销     |
| 编译及预热（启用时）      | 为模型计算准备优化后的执行代码，并运行预热                         | 以准备时间换取后续重复执行效率；已有编译缓存可减少冷启动工作               |
| CUDA Graph 捕获（启用时） | 为选定的批次形状预先捕获 GPU 操作序列，准备重放所需资源            | 后续重放减少 CPU 逐个提交 GPU 操作的开销；捕获耗时，相关资源还会额外占显存 |
| 引擎进程与通信资源        | 按执行配置启动 worker、建立通信和请求管理资源                      | 支撑引擎的运行架构；具体成本依进程布局而变，不是数学上的前向计算所必需     |

在固定的 vLLM 0.31.0 代码中，可以沿 GPU worker 的 `determine_available_memory()`、`initialize_from_config()`、`compile_or_warm_up_model()` 查到显存规划、KV 分配和编译/预热入口。它们是对应准备工作的代码证据，不表示所有配置都按同一条内部路径执行。[官方 GPU worker 源码](https://docs.vllm.ai/en/v0.31.0/api/vllm/v1/worker/gpu_worker/)

“执行结构”这个词需要拆开：普通模型对象本来就有；优化后的可执行代码、捕获后的 CUDA Graph 及配套缓冲区，则是额外选择的执行准备。**编译和 CUDA Graph 不是一回事**：前者准备优化代码，后者记录可重放的 GPU 操作流程；也可以有不依赖编译的图捕获。vLLM 的编译缓存用于复用匹配的编译产物，不能据此认为每次创建引擎都必须完整重编译。[编译说明](https://docs.vllm.ai/en/v0.31.0/design/torch_compile/)、[CUDA Graph 设计](https://docs.vllm.ai/en/v0.31.0/design/cuda_graphs/)

第 8.2 节列出了项目的图执行配置。允许图优化之后，实际捕获哪些图、准备多久仍要由启动记录确认。CUDA Graph 还会占用额外显存，需要和运行速度一起权衡。[显存与 CUDA Graph 配置说明](https://docs.vllm.ai/en/v0.31.0/configuration/conserving_memory/#reduce-cuda-graphs)

手写实现若加入相同的缓存池、编译、图捕获和进程架构，也要承担相应准备与管理成本。vLLM 提供了这些实现；只运行少量、固定形状的请求时，更简单的实现可能有更低的启动成本。选择时要同时看启动时间、实际生成时间、显存和请求规模；有些实现会把准备工作推迟到第一次生成。

**双卡调度也不是 vLLM 独有的需求，而且本项目有两层调度。**

- 项目层：一张卡训练、一张卡采样，决定何时采样、何时更新、何时同步新权重，以及何时执行工具。自己手写同样的训推分离架构，也必须处理这些协作。当前并未把训练与下一批采样异步重叠。
- 推理引擎层：在采样卡上，决定哪些活跃请求进入下一步计算，如何使用 token 预算与 KV 块，何时回收请求资源。即使只有一张推理卡，也有这类问题；手写循环也有调度，只是可以采用“整批完成后再换一批”这样更简单的策略。

本项目的双卡不是“一个 vLLM 模型切到两张推理卡上”。`VLLMRollout` 启动子进程、等待服务就绪，以及首次同步时建立权重传输连接，属于这套训推分离部署的准备；不能全归因于 `LLM(...)` 的初始化，更不能混同于每个生成步骤的请求调度。后续权重传输本身还有按更新发生的成本，不是只支付一次的启动成本。

回到初始化参数，它们不是要求用户手动完成以上工作，而是给引擎提供规划依据：

| 参数                            | 初始化时需要它的原因                                          |
| ------------------------------- | ------------------------------------------------------------- |
| `model=model_path`            | 确定模型结构、配置和初始权重来源                              |
| `dtype='bfloat16'`            | 确定模型执行的数值精度，影响显存与计算                        |
| `max_model_len=4096`          | 确定单请求总序列长度上限，包含输入与输出                      |
| `gpu_memory_utilization=0.85` | 为模型执行与缓存资源规划提供显存预算；不是测得的 GPU 忙碌比例 |
| `weight_transfer_config=...`  | 在需要在线换权重的实例中启用接收参数的能力                    |

字段定义见 [vLLM 引擎参数](https://docs.vllm.ai/en/v0.31.0/configuration/engine_args/)。总序列上限与本项目每轮 `max_new_tokens=160` 不同，项目另外限制单条任务总上下文为 3584。

#### 9.1 编译是什么，Python 为什么也会涉及编译

编译是把一种程序表示转换成另一种程序表示，常伴随优化。目标可以是字节码、CPU 机器码或 GPU 可执行代码；不一定要求用户先手动执行一条编译命令。

这里要分清三件事：

| 位置                             | 实际发生的事                                                                                                                                            |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 普通 CPython                     | Python 源码先变成 Python 字节码，再由解释器执行；导入模块时可能缓存为 `.pyc`。这通常没有把整个程序变成原生机器码。                                    |
| 普通 PyTorch GPU 算子            | Python 调用框架，框架选择已有的底层实现并提交 GPU kernel。矩阵乘法等大量数值运算在底层库和 GPU 上完成；底层代码可以提前编译，也可能包含运行时编译步骤。 |
| `torch.compile` 等模型计算编译 | 捕获可处理的张量运算，分析依赖、融合部分算子、生成或选择执行代码，并缓存符合条件的产物。GPU 后端可以使用 Triton 等代码生成工具，也可调用已有库。        |

所以，Python 写模型，不等于 GPU 正在逐行解释 Python；普通 eager 执行，也不等于底层没有编译好的代码。vLLM 初始化里所说的编译，主要涉及模型计算的优化与执行准备，和生成 `.pyc` 不是同一层。

例如张量表达式 `y = (x + 1) * 2`，简单 eager 执行可先调用加法 kernel，写出中间张量，再调用乘法 kernel。编译器可能把它融合成一个 kernel：每个元素读入后连续完成加法和乘法，再写出结果。这样可以减少 kernel 启动和中间张量读写。它并没有省去数学上所需的加法和乘法。

实际模型不一定整段都能融合或编译。形状变化、动态 Python 控制流、工具调用等可能需要专门处理或留在图外。首次捕获、生成代码和预热会花时间；匹配的编译缓存可以减少后续准备成本。编译是在准备程序的执行方式，训练则是在修改模型参数。

#### 9.2 CUDA Graph 为什么叫图，与缓存有什么关系

CUDA Graph 的节点表示 GPU kernel、内存复制等操作，边表示操作之间的依赖。例如，两份输入各自处理完之后才能合并：

```text
复制输入 A → 计算 A ─┐
                    ├→ 合并结果
复制输入 B → 计算 B ─┘
```

两条分支在依赖上允许并发，是否真的同时执行还取决于资源和调度。如果工作全部串行，`A → B → C` 也仍是一张图，只是形状恰好是一条链。生成 token 的先后顺序，不能推出每次模型前向内部只有一条不可分支的执行链。

普通执行中，CPU 依次向 CUDA 提交操作，GPU 按依赖执行。提交通常是异步的，不是 CPU 每提交一个 kernel 就必须等待它完成；但每次提交仍有 Python、框架、运行时或驱动侧的开销。kernel 很短而数量很多时，这些开销尤其值得关注。

CUDA Graph 将定义和执行分开：先构建或捕获操作及依赖，实例化为可执行的图，之后反复启动这份已经准备好的图，从而减少重复提交和准备的成本。CPU 不再需要每次通过原有调用路径逐个提交图内的操作。参见 [NVIDIA CUDA Graph 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)。

**可以把它看作复用执行安排，但它并没有缓存这次计算的答案。** 输入缓冲区换成新内容之后，重放仍会执行 kernel，读取新输入并产生新结果。

| 机制            | 复用什么                     | 下次省掉什么                                         |
| --------------- | ---------------------------- | ---------------------------------------------------- |
| KV cache        | 历史 token 已经算出的 K、V   | 重算历史 K、V；新的 query 仍需读取并使用历史 KV      |
| 编译缓存        | 已生成的代码和相关编译产物   | 重复编译的准备工作                                   |
| CUDA Graph 重放 | 已准备的操作、依赖及执行资源 | 逐操作重新提交和部分运行时准备工作；图内计算仍然执行 |

编译图侧重张量运算怎样转为高效执行，CUDA Graph 侧重实际 CUDA 操作怎样重复提交；两者可配合，也可分别使用。CUDA Graph 可以重放已有的库 kernel，不以先调用 `torch.compile` 为必要条件。

常见模型图捕获要求兼容的操作结构、形状和稳定的缓冲区地址；缓冲区里的数值可以改变。形状变化时，引擎可以使用多份图、按容量填充，或选择其他执行路径；CUDA 也有更新图参数等机制，具体支持范围取决于接口和框架。不能把它理解成任意动态 Python 程序都能原样录制并重放。

对生成而言，引擎可以反复执行适用的 decode 计算图，每轮写入新的 token 和相关元数据。下一个 token 仍依赖前面的采样结果，CUDA Graph 不会提前知道整条答案，也不会自动消除逐 token 生成或项目中的工具交互。实际捕获范围由 vLLM 后端和配置决定。

## 二、把引擎放回项目：对象、进程和调用关系

### 10. SFT 更新为什么不需要 vLLM

SFT 已经有保存好的轨迹。程序把轨迹编码成输入和监督位置，算损失、反向、更新，无需先让模型自己生成答案。

```text
保存好的训练轨迹
       ↓ 项目编码与分组
PyTorch/Transformers Qwen 模型
       ↓ 项目计算 SFT loss
DeepSpeed 管理反向、分片、通信和优化器更新
       ↓
更新后的模型参数
```

训练前向需要保留可求导关系，当前代码使用训练模型计算；vLLM 用在需要生成轨迹的地方。因此“项目使用 vLLM”不代表每次模型前向都经过 vLLM。

#### 10.1 双卡 SFT 的进程和训练状态

双卡 SFT 由 `torchrun` 启动两个同级训练进程，各自运行 `train_sft.main()`，共同加入 `world_size=2` 的训练通信组。每个进程都有自己的 `DeepSpeedTrainer` 和 `DeepSpeedEngine`，两张卡都执行前向、反向和参数更新。

```text
torchrun --nproc_per_node=2 -m minitoolrl.train.train_sft ...
                           ↓ 启动两个训练进程

训练侧 Python 进程（rank 0）                            训练侧 Python 进程（rank 1）
┌────────────────────────────────┐                      ┌────────────────────────────────┐
│train_sft.main()                │                      │train_sft.main()                │
│ └─ DeepSpeedTrainer            │                      │ └─ DeepSpeedTrainer            │
│     └─ DeepSpeedineEngine         │                      │     └─ DeepSpeedEngine         │
│         ├─ 完整模型参数        │                      │         ├─ 完整模型参数        │
│         ├─ 梯度分片 A          │                      │         ├─ 梯度分片 B          │
│         └─ 优化器分片 A        │                      │         └─ 优化器分片 B        │
│                                │                      │                                │
│microbatch 0、2、4、6           │                      │microbatch 1、3、5、7           │
│本地前向、反向                  │←── NCCL 梯度规约 ──→ │本地前向、反向                  │
│更新负责的参数分片              │                      │更新负责的参数分片              │
│完整模型参数刷新                │←── NCCL 参数收集 ──→ │完整模型参数刷新                │
└────────────────────────────────┘                      └────────────────────────────────┘
             训练 GPU 0                                              训练 GPU 1
```

图中用一个完整的基础 SFT 更新组举例：16 条记录分成 8 个 microbatch，每个包含 2 条记录；rank 0 处理编号 0、2、4、6，rank 1 处理 1、3、5、7。两边的损失都除以这 16 条记录的全局监督 token 总数，然后由 DeepSpeed 合并梯度贡献。每个 rank 调用 4 次 `backward(); step()`，在最后一次累积边界共同完成一次全局梯度裁剪和一次优化器更新。

A、B 表示各 rank 负责的训练状态分片，包括规约并累积的梯度、FP32 主权重和 AdamW 状态；实际按参数和通信桶切分。每张卡始终保留完整的前向模型参数，处理本地数据时会为各参数产生梯度贡献。各 rank 更新自己负责的参数分片后，DeepSpeed 通过通信刷新两边的完整模型副本，再进入下一个更新组。图中的“梯度规约”和“参数收集”表示逻辑作用，具体通信由引擎组织。

训练轨迹里的工具返回已经保存在数据中，本次 SFT 更新直接读取这些内容。两张卡上都没有 `VLLMRollout` 或常驻 vLLM 采样服务。三个监督续训分支使用同样的双 rank 结构；其中 `U_mixed` 每更新有 9 个 microbatch，较少数据的一侧需要补零损失微步，见第 16 节。

### 11. GRPO 为什么同时有两份模型

GRPO 先让当前策略做题，再用轨迹和奖励更新模型。项目将这两个工作放到两张物理 GPU：

```text
训练侧 Python 进程                         采样侧 Python 子进程
┌─────────────────────────┐             ┌────────────────────────┐
│ GRPOTrainer             │             │ _serve()               │
│  ├─ DeepSpeedTrainer    │             │  └─ vllm.LLM           │
│  │   └─ DeepSpeedEngine │             │      └─ 推理模型和缓存 │
│  │       └─ 训练模型    │             │                        │
│  ├─ Episode 工具环境    │             │                        │
│  └─ VLLMRollout 客户端  │──生成请求──→│                        │
│                         │←─token结果─│                        │
│    更新后的训练参数     │══NCCL传参═→│      更新推理模型      │
└─────────────────────────┘             └────────────────────────┘
         训练 GPU 0                              采样 GPU 1
```

图只画项目显式管理的进程边界；vLLM 自己还可能建立内部引擎进程或 worker，不能由图推断操作系统进程总数。

两份模型最初从同一个 SFT 检查点加载，却是不同执行系统持有的独立参数存储。训练侧更新自己的参数，不会自动修改采样侧参数，所以每次有效更新后必须传参。

训练侧需要梯度、AdamW 状态和反向激活；采样侧需要生成调度与 KV cache。分开部署让两种执行系统各自管理资源，也让采样器持续驻留，同时增加第二份权重、通信和第二张卡的成本。当前流程依次完成采样、更新、同步，并不保证两卡始终同时繁忙。

GRPO 的 DeepSpeed **训练通信组只有 1 个 rank**，此阶段没有双训练 rank 分片的显存收益。采用同一后端保持更新语义一致；双 rank ZeRO-2 用于 SFT 与续训。权重传输另有包含发送方与推理 worker 的通信组，两者用途不同。

#### 11.1 三张、四张卡可以怎样分工

GRPO 算法不要求 GPU 数量为偶数。“一张训练、一张采样”是本项目的部署约定。更大的系统要分别决定训练侧用几张卡、采样侧用几张卡，以及每一侧怎样并行。

采样侧有两种不同的多卡用法：独立副本各持有完整模型，处理不同请求；或者用张量并行、流水线并行等方式让多卡共同执行一份模型。独立副本常用于扩大请求吞吐，模型分片常用于模型装不下单卡的情况，也可能改变吞吐和延迟。它们都不是把 GPU 数量填大就自动得到。

| 总卡数 | 训练侧 | 采样侧 | 一种可能的安排                                   |
| ------ | ------ | ------ | ------------------------------------------------ |
| 2      | 1      | 1      | 当前项目的部署                                   |
| 3      | 1      | 2      | 两个采样副本分担轨迹，或两卡共同执行一个推理模型 |
| 3      | 2      | 1      | 两个训练 rank 分担更新，采样模型在单卡上运行     |
| 4      | 1      | 3      | 三个采样副本分担轨迹，训练在单卡完成             |
| 4      | 2      | 2      | 两个训练 rank；采样侧用两个副本或一个两卡模型    |
| 4      | 3      | 1      | 三个训练 rank，采样在单卡完成                    |

数据并行的三个 rank 不需要按“卡数必须是 2 的幂”来约束。张量并行则可能要求模型维度、注意力头数等与并行度匹配；例如某种实现要求 16 个头平均分配，就不能直接分给 3 张卡。因此“三卡可以部署”不等于“任意模型都能直接设置 TP=3”。

多个采样副本处理同一任务的不同轨迹时，仍需按任务聚合结果再计算组内优势。所有参与当前批次的副本必须使用约定的策略版本；有效更新后，所需副本完成传参、清缓存和版本确认，才能接收下一批。若训练权重或推理权重被分片，传输还要匹配双方布局，不能简单沿用单发送方、单接收方的假设。

怎样分卡应先测瓶颈。对于按阶段等待的同步流程，一轮时间大致为：

```text
采样与工具交互时间 + 奖励处理时间 + 训练更新时间 + 权重同步时间
```

若采样占主要时间，增加采样吞吐可能更有价值；若训练状态装不下或更新时间很长，则需要考虑训练侧资源。请求太少时，增加采样副本可能用不满每张卡；多卡通信和权重广播也会增加成本。因此要测墙钟秒数和 GPU 秒数，不能仅按卡数推算加速。

**当前代码只实现表中的 1+1 GRPO。** `train_grpo.py` 显式要求训练通信组 `world_size == 1`；`VLLMRollout` 管理一个采样服务，权重传输组按一个发送方和一个推理 worker 建立。三卡、四卡的表格解释通用设计，不是本项目现成的配置入口。扩展需要修改训练分工或采样路由、权重同步和正确性检查；当前实现与运行预算保持原有约定。

### 12. 为什么会出现 `LocalVLLM`，它与评测有什么关系

固定检查点评测要加载一个训练好的模型，让它实际执行退款、补偿等任务，再统计成功率。这里需要生成和工具执行，模型参数在评测期间保持不变。下面按真实代码说明对象怎样创建、谁调用谁，以及结果怎样返回。

#### 12.1 `vllm.LLM` 是什么：包名、类名和对象

`vllm` 是安装的 Python 包，`LLM` 是这个包提供的一个类。`vllm.LLM(...)` 表示调用这个类的构造入口，创建一个管理模型加载、请求调度和生成的推理对象。它内部使用 Qwen 等模型；`LLM` 不是本项目给 Qwen 网络起的新名字。

两种导入写法表达同一件事：

```python
# 写法一：先导入包，通过包名访问类。
import vllm
llm = vllm.LLM(model=model_path)

# 写法二：直接导入类。这是项目实际采用的写法。
from vllm import LLM
llm = LLM(model=model_path)
```

所以上面文档用 `vllm.LLM` 指明类来自哪个包，源码中却可以只出现 `LLM(...)`。这两段是替代写法，实际使用时选其中一种。

`LocalVLLM` 则是项目在 [vllm_backend.py](../train/vllm_backend.py) 中定义的另一个类。它的构造方法实际写着：

```python
def __init__(self, model_path: str, *, max_model_len: int = 4096,
             gpu_memory_utilization: float = 0.85, dtype: str = 'bfloat16'):
    from vllm import LLM
    self.llm = LLM(model=model_path, dtype=dtype, max_model_len=max_model_len,
                   gpu_memory_utilization=gpu_memory_utilization,
                   enforce_eager=False)
    self.version = 0
```

这里的 `self` 是当前这个 `LocalVLLM` 对象，`self.llm` 是它保存的一个属性，属性值为刚创建的 vLLM 推理对象。因此有两层对象：外层是项目的 `LocalVLLM`，内层是框架的 `LLM`。

#### 12.2 `LocalVLLM.generate()` 是什么：定义的方法和实际调用

Python 在类里面用 `def generate(self, ...)` 定义一个方法。文档写 `LocalVLLM.generate()`，是在指“`LocalVLLM` 类定义的 `generate` 方法”，不是说源码中一定有这样一行原样调用。

先看 [eval/evaluate.py](../eval/evaluate.py) 的实际构造函数：

```python
def make_sampler(model_path, max_model_len=4096,
                 gpu_memory_utilization=0.85):
    from minitoolrl.train.vllm_backend import LocalVLLM
    return LocalVLLM(model_path, max_model_len=max_model_len,
                     gpu_memory_utilization=gpu_memory_utilization)
```

评测入口把返回对象存到变量 `sampler`。例如 [eval/rollout_eval.py](../eval/rollout_eval.py) 的 `main()` 中：

```python
sampler = make_sampler(args.model,
                       gpu_memory_utilization=args.gpu_memory_utilization)
```

所以此时 `sampler` 指向一个 `LocalVLLM` 对象。它经过 `run_greedy_pool()` 被传入 `rollout_pool()`。后者在 [rollout.py](../train/rollout.py) 中实际调用：

```python
generations = sampler.generate(requests)
```

这就是在调用 `LocalVLLM` 定义的 `generate` 方法，Python 自动把 `sampler` 作为 `self` 传进去。**`sampler.generate(...)` 和“进入 `LocalVLLM.generate`”描述的是同一次调用，不能把它们理解为两个先后执行的生成步骤。**

这个方法的实际代码如下：

```python
def generate(self, requests: Sequence[GenRequest]) -> List[Generation]:
    from vllm import TokensPrompt
    prompts = [TokensPrompt(prompt_token_ids=list(r.prompt_token_ids)) for r in requests]
    params = [_sampling_params(r) for r in requests]
    outputs = self.llm.generate(prompts, params)
    return outputs_to_generations(requests, outputs, self.version)
```

`requests: Sequence[GenRequest]` 和 `-> List[Generation]` 是类型标注，说明输入是一组项目请求、返回是一组项目结果；转换工作由方法里的代码执行。逐行看这段代码：

1. 把项目请求里的输入 token IDs 放进 vLLM 的 `TokensPrompt`。
2. `_sampling_params()` 把温度、生成长度等配置放进 vLLM 的 `SamplingParams`。
3. `self.llm.generate(...)` 调用内层 vLLM 对象的方法，执行这一批生成。
4. `outputs_to_generations(...)` 把 vLLM 返回的对象整理成项目自己的 `Generation` 对象，再返回给外层调度器。

两个方法都叫 `generate`，但属于不同对象：

```text
sampler                     项目的 LocalVLLM 对象
  └─ sampler.llm            框架的 LLM 对象

rollout_pool()
  └─ sampler.generate(requests, ...)
       进入 LocalVLLM 定义的 generate 方法
       ├─ 转换输入：GenRequest → TokensPrompt、SamplingParams
       ├─ self.llm.generate(prompts, params) → vLLM 输出对象
       └─ outputs_to_generations(...) → 项目 Generation 列表
            返回给 rollout_pool() 的 generations 变量
```

#### 12.3 输出具体在哪转换：`outputs_to_generations()`

转换函数就在 [vllm_backend.py](../train/vllm_backend.py) 中，位于 `LocalVLLM` 类定义的前面。下面是函数内实际提取结果的循环：

```python
for req, out in zip(requests, outputs):
    completion = out.outputs[0]
    token_ids = [int(t) for t in completion.token_ids]
    generations.append(Generation(
        request_id=req.request_id,
        token_ids=token_ids,
        finish_reason=str(completion.finish_reason),
        policy_version=policy_version,
    ))
```

`outputs` 是 vLLM 为这一批请求返回的结果列表。循环中的 `out` 对应一条请求，它的 `outputs` 属性是这条请求的候选输出列表；本项目取第一个候选 `out.outputs[0]`。两个名字相似，但外层是一批请求，内层是一条请求的候选。

| 项目结果字段       | 从哪里取得、怎样整理                                                               |
| ------------------ | ---------------------------------------------------------------------------------- |
| `request_id`     | 沿用项目输入请求的 ID，让调度器找回对应的任务                                      |
| `token_ids`      | 从候选结果提取生成的 token ID，整理成 Python 整数列表                              |
| `finish_reason`  | 提取 vLLM 返回的生成结束原因                                                       |
| `policy_version` | 加上项目记录的参数版本；这个固定模型适配器使用版本 0                               |

所以这里说的“输出转换”是**整理结果对象及字段**。返回给调度器的仍是 token IDs；文字解码发生在 `rollout_pool()` 的后续代码：

```python
text = tokenizer.decode(gen_ids, skip_special_tokens=True)
info = eps[i].step(text, n_tokens=len(gen_ids))
```

例如生成的 token 解码后包含 `create_refund` 工具调用，`Episode.step()` 才会解析它、执行退款工具并取得返回。`LocalVLLM.generate()` 本身不负责创建退款单或判断业务成功。

#### 12.4 评测也叫 rollout 吗

**评测中让模型实际执行任务、产生轨迹的过程，也叫 rollout。** 在这个项目中，一条 rollout 包括：模型选择工具 → 执行工具 → 把返回追加进上下文 → 模型继续选择下一步，直到任务结束或达到预算。

一次 `generate()` 通常只产生一轮 assistant 输出；`rollout_pool()` 会反复调用它，让这一批任务逐步完成。例如退款回库任务需要依次查询订单、创建退款单、回补商品和更新状态，这些交互共同组成任务轨迹。

| 场景                       | 产生轨迹后做什么                                         |
| -------------------------- | -------------------------------------------------------- |
| GRPO 训练中的 rollout      | 计算奖励和组内优势，有信号时更新模型                     |
| 固定检查点评测中的 rollout | 记录成功、错误、调用数等，汇总评测指标，模型参数保持不变 |

因此，“rollout”描述执行过程，“评测”描述这批执行的用途。评测还包含评分、汇总和报告；也有给定正确历史计算验证损失的评测，那种检查无需自主 rollout。即使采用 `temperature=0` 的确定性选择，实际执行整条任务的过程仍然叫 rollout。

在 [eval/rollout_eval.py](../eval/rollout_eval.py) 的 `evaluate_pool()` 中，可以看到先调用 `run_greedy_pool(...)` 收集任务轨迹，再调用 `M.summarize(...)` 汇总指标。整个评测结束后，`main()` 的 `finally` 调用 `close_sampler(sampler)`，再进入 `sampler.close()` 释放持有的 LLM 对象；并非每调用一次 `generate()` 就重新加载、释放模型。

这里“适配层”的具体含义，就是把项目的请求和结果字段连接到 vLLM 的接口。这样 `rollout_pool()` 可以使用相同的 `generate(requests, ...)` 调用，接收 `LocalVLLM` 或 `VLLMRollout` 返回的同一种 `Generation`。`Local` 表示调用方直接持有 LLM 对象，不经过项目的 ZMQ 客户端；vLLM 内部仍可能创建引擎进程或 worker。

### 13. `VLLMRollout` 会启动 `LocalVLLM` 吗

**不会。它们是两个并列的项目封装。** 当前源码中，`VLLMRollout` 启动子进程入口：

```text
python -m minitoolrl.train.vllm_backend --serve ...
```

子进程进入 `_serve()`，直接执行 `llm = LLM(...)`，没有创建 `LocalVLLM`。

```text
直接调用路径：
  LocalVLLM → vllm.LLM

GRPO 跨进程调用路径：
  VLLMRollout → ZMQ 请求 → _serve() → vllm.LLM
```

它们最终使用同一种 vLLM 引擎，并都向 `rollout_pool()` 提供 `generate(requests)`。这样多轮工具调度无需处理两种部署的通信差异。

**这两个类的划分是项目的代码组织选择，vLLM 没有要求一定写这两个类。** 可以把直接调用的转换代码放进别的函数，也可以设计统一接口；现有划分把“固定模型直接生成”和“常驻服务加权重同步”两种生命周期分开表达。

GRPO 训练中的采样模型由常驻采样进程持有，训练侧通过 `VLLMRollout` 发起采样与权重同步；SFT 开发集检查和正式评测由 `LocalVLLM` 在评测进程内加载固定检查点。

| 当前工作               | 训练引擎             | 生成对象               | 谁持有推理模型                   |
| ---------------------- | -------------------- | ---------------------- | -------------------------------- |
| SFT 或监督续训的更新   | `DeepSpeedTrainer` | 更新本身不需要生成对象 | 此更新过程没有 vLLM 推理模型     |
| GRPO 训练             | `DeepSpeedTrainer` | 训练侧 `VLLMRollout` | `_serve()` 创建的 `vllm.LLM` |
| 固定检查点自主任务评测 | 不做梯度更新         | `LocalVLLM`          | 由该适配器创建的 `vllm.LLM`    |

SFT 脚本给已有验证序列计算 loss/token accuracy，也使用训练模型前向。它与“模型自己选择工具、执行完整任务”的自主评测不同，后者才需要生成循环。

## 三、现在读初始化和一次实际调用

### 14. `DeepSpeedTrainer(...)` 建立了什么

`DeepSpeedTrainer(...)` 接收已经加载的模型，准备通信、优化器和训练引擎；构造完成后，还没有用训练数据执行更新。实际前向、反向传播和参数更新发生在后续的 `trainer.update(...)` 中。

先看这些训练进程从哪里来。[run_session.py](../train/run_session.py) 使用 `python -m torch.distributed.run`（即 `torchrun`）启动训练。单机双卡 SFT 设置 `--nproc_per_node 2`，启动两个 Python 进程，分别运行同一份训练代码、创建自己的模型和 `DeepSpeedTrainer`。启动器为各进程设置环境变量：

| 环境变量                         | 训练进程 A       | 训练进程 B       | 用途                                         |
| -------------------------------- | ---------------- | ---------------- | -------------------------------------------- |
| `RANK`                         | 0                | 1                | 当前进程在整个训练组中的编号                 |
| `LOCAL_RANK`                   | 0                | 1                | 当前进程在本机这组训练进程中的序号，用于选卡 |
| `WORLD_SIZE`                   | 2                | 2                | 训练组的总进程数                             |
| `MASTER_ADDR`、`MASTER_PORT` | 相同的地址和端口 | 相同的地址和端口 | 通信初始化时的共同会合地址                   |

`LOCAL_RANK` 不是操作系统的进程 ID。项目读取它并选择对应的 `cuda:local_rank`；设备编号相对于当前进程可见的 GPU 列表。例如 `CUDA_VISIBLE_DEVICES=2,5` 时，`LOCAL_RANK=0` 使用物理 GPU 2，`LOCAL_RANK=1` 使用物理 GPU 5。当前项目是单机训练，`RANK` 与 `LOCAL_RANK` 恰好相同；多机时本机序号会在每台机器上从 0 开始。

`torchrun` 负责启动进程和提供环境变量，不替训练进程初始化 PyTorch 默认进程组。每个训练进程创建 `DeepSpeedTrainer` 时，[deepspeed_backend.py](../train/deepspeed_backend.py) 统一根据 `LOCAL_RANK` 绑定 GPU，再调用 `deepspeed.init_distributed()` 建立训练进程组。单卡 SFT、双卡 SFT 和 GRPO 都走这一个初始化入口；进程组已经建立时直接复用，不重复初始化。

多进程初始化时，各进程根据启动器提供的身份、总人数和共同地址，通过会合服务交换连接信息，建立同一训练组的通信关系。GPU 训练使用 NCCL 后端，后续张量通信由它组织，不要求梯度都经过会合服务转发。

进程组建立后，各进程仍持有各自的 Python 对象。它们按同一份训练代码创建模型和引擎，并按匹配的顺序参加通信；通信初始化不会自动检查另一个进程是否已经创建优化器。训练时，DeepSpeed 安排梯度归约和参数同步，NCCL 组织 GPU 间的数据传输与归约计算，无需额外的中央 Python 进程收齐梯度再计算。

本项目 GRPO 虽然使用两张 GPU，但只有一个 DeepSpeed 训练进程，因此训练组的 `WORLD_SIZE=1`；另一张卡上的 vLLM 是独立采样进程。训练与采样之间的权重传输组另行建立，见第 20 节。

接着确定累积边界。`gas_global` 是一次更新的全局 microbatch 数；轮转分配到各 rank 后，`gas_local = ceil(gas_global / world_size)`。引擎据此判断本地处理多少微步才完成一次更新。

然后项目创建 AdamW，给出学习率和权重衰减，把模型、优化器与执行配置一起交给 `deepspeed.initialize()`。`init_distributed()` 准备进程通信，`initialize()` 创建训练引擎；即使通信已经初始化，仍需创建引擎。下面摘出核心配置：

```python
config = {
    'train_micro_batch_size_per_gpu': 1,
    'gradient_accumulation_steps': self.gas_local,
    'bf16': {'enabled': True, 'check_grad_overflow': True},
    'zero_optimization': {'stage': 2},
    'gradient_clipping': float(grad_clip),
}
self.engine, _, _, _ = deepspeed.initialize(
    model=model, optimizer=optimizer, config=config)
```

`stage` 决定状态分片方式，bf16 配置决定低精度执行路径，`gradient_clipping` 把裁剪放进真实更新边界。项目使用 bf16 模型计算与 fp32 优化器主权重/状态，不做 CPU/NVMe offload。

`train_micro_batch_size_per_gpu=1` 是提供给引擎的配置值；项目手动构造的真实张量批次另有大小：SFT 完整 microbatch 为 2 条，GRPO 为 1 条。它不会把实际传入的两条序列改成一条，真实曝光量按上层数据统计。

初始化不是替项目选择训练内容。数据顺序、损失公式、学习率日程和完成预算仍由阶段脚本决定。

### 15. 为什么项目还直接访问 `engine.module`

项目要只在监督位置计算词表投影，所以损失函数直接访问底层 Qwen 模型和输出头。`engine.module` 就是引擎持有的模型，并非另复制一套模型。

```text
项目的 loss_fn(engine.module, microbatch)
  → 模型前向，生成可求导的 loss
  → engine.backward(loss, ...)
  → engine.step()
```

网络计算仍由 PyTorch/Transformers 执行，DeepSpeed 管理训练状态和更新。正因为业务损失由项目提供，同一个训练后端才可以同时支持 SFT 与 GRPO。

### 16. 当前代码怎样做到“拆开算，还是同一次更新”

当前 SFT 中，每个训练进程都在 CPU 内存中准备相同的完整更新组并统计全局监督 token 数 `n_total`（无需为此通信），随后按 rank 分配组内的完整 microbatch，各 GPU 只对分到的 microbatch 计算损失和梯度。

SFT 先计算整个更新组的监督 token 数 `n_total`，每个 microbatch 返回：

```python
loss = F.cross_entropy(logits, target, reduction='sum') / n_total
```

若两个 microbatch 分别有 10、90 个监督 token，分母都用 100。每个 token 因而等权；如果先各自平均再平均，前 10 个 token 整体就会占一半权重。

每个 loss 已经是全局目标的一部分。当前后端用 `ds_loss_scale(gas_global, world_size)` 计算 `gas_local × world_size`，在真实 microbatch 的反向调用前补偿 DeepSpeed 的累积缩放与跨 rank 平均：

```python
self.scale = ds_loss_scale(self.gas_global, self.world)
self.engine.backward(loss * self.scale)
```

`gas_local` 补偿按累积步数的缩放，`world_size` 补偿跨 rank 平均。损失函数仍决定全局平均口径，后端负责这两项缩放补偿；零损失微步直接对零损失反向传播。

实际的 `U_mixed` 每更新使用 18 条记录，每 microbatch 2 条，共 9 个：

```text
rank 0：0、2、4、6、8
rank 1：1、3、5、7、零损失微步
每个 rank 的 gas_local = 5
```

rank 1 最后补带计算图的零损失，让双方经过相同的累积边界和通信次数。复制真实样本会改变数据权重，提前退出则可能让通信和计数错位。

每微步调用一次 backward 和 step，最后一步才真正裁剪、更新。`update()` 随后检查微步计数、真实更新次数、损失和梯度范数。这是项目为自己的更新约定增加的验证，不能仅凭“用了 DeepSpeed”推定集成正确。

GRPO 使用同一执行机制，但平均口径是“每条保留轨迹先按自己的策略 token 平均，再对轨迹平均”。工具观察进入上下文，不直接计入策略 token 损失。

### 17. `VLLMRollout(...)` 怎样建立生成服务

训练侧先创建训练模型和 `DeepSpeedTrainer`，随后构造 `VLLMRollout(model_path, gpu=...)`。后者准备子进程、命令通道和版本状态，不在训练侧创建 `self.llm`。

它给子进程单独设置 `CUDA_VISIBLE_DEVICES`，删除继承的训练 rank/master 环境变量，再启动 `--serve`。这样采样器使用指定物理 GPU，不会按继承的 rank 配置误加入训练通信组。

例如训练用物理卡 0，子进程只看见物理卡 1，物理卡 1 在子进程内部也可以叫 `cuda:0`。局部设备编号要结合可见设备列表理解。

子进程 `_serve()` 创建：

```python
llm = LLM(
    model=args.model,
    dtype='bfloat16',
    max_model_len=args.max_model_len,
    gpu_memory_utilization=args.gpu_memory_utilization,
    weight_transfer_config=WeightTransferConfig(backend='nccl'),
)
```

比固定模型直接调用多出的权重接收配置，是因为这个实例要在训练过程中不断换参数。加载完成后服务才开始响应命令；客户端用 health 检查等待它就绪，避免刚启动进程便向尚未可用的引擎请求生成。

这里有三个时刻：创建客户端、启动并加载采样引擎、首次同步时建立 NCCL 权重传输引擎。它们不是同一行代码同时完成的。

### 18. 一次 generate 调用怎样穿过这些对象

`rollout_pool()` 收集尚未结束的 Episode，将当前上下文 token 包装成请求。请求包含唯一 ID、token IDs、本轮输出预算和采样设置。

```text
rollout_pool
  → VLLMRollout.generate
  → ZMQ 发送 JSON 请求
  → _serve.handle('generate')
  → 转成 TokensPrompt 与 SamplingParams
  → llm.generate
  → vLLM 内部调度、模型执行、采样
  → 返回 token、结束原因与参数版本
  → 项目加上 policy_version，回传 JSON
  → VLLMRollout 检查版本，交还 rollout_pool
```

`TokensPrompt` 让引擎使用项目已经编码的输入；`SamplingParams` 告诉引擎怎样从概率分布选择输出、何时停止；`detokenize=False` 让项目取得原始 token，再自行解码给工具解析器。

随后 `Episode.step()` 解析调用并执行工具。例如全额退款回库任务中，`create_refund` 生成退款单，库存仍需用 `restock_item` 回补。工具返回追加到上下文，再产生下一轮生成请求。

所以一次 `llm.generate()` 通常只完成一轮 assistant 输出，完整业务轨迹由项目循环串起来。vLLM 的调度器不会自行知道“现在应执行退款工具并继续下一轮”。

固定模型直接调用的路径更短：`LocalVLLM.generate()` 转换请求后直接调用 `self.llm.generate()`，无需项目 ZMQ 往返。两条路径共享请求/结果格式，上层工具执行逻辑因此可以复用。

## 四、训练参数改变之后，两边怎样保持一致

### 19. 一轮 GRPO 的时间顺序

从相同 SFT 权重加载后，双方初始版本均为 0。假设这一轮存在非零学习信号：

| 时刻 | 训练侧                                            | 采样侧                               |
| ---- | ------------------------------------------------- | ------------------------------------ |
| ①   | 请求生成，每轮输出后执行工具                      | 用版本 0 生成 token                  |
| ②   | 轨迹全部结束后，计算成功奖励和组内优势            | 等待命令                             |
| ③   | 带梯度前向计算 logprob 和损失，DeepSpeed 更新一次 | 仍持有版本 0，尚不开始下一批         |
| ④   | 发送更新后的参数并等待应答                        | 接收、装载参数，清旧缓存，确认版本 1 |
| ⑤   | 确认后请求下一批                                  | 用版本 1 生成                        |

全对或全错的任务组没有组内优势；若整轮没信号，直接跳过更新和同步，版本不变。240 次采样迭代不一定对应 240 次有效更新。

每批轨迹只做一次优化器更新，所有 microbatch 的前向都发生在更新之前。因此损失前向得到 `lp` 后，直接用 `old_lp = lp.detach()` 固定基准，不另做一次无梯度前向。概率比 `exp(lp - old_lp)` 的值为 1，但只有分子参与求导，仍产生策略梯度。

### 20. 为什么同步又要初始化一套对象

生成客户端已经能发请求，却还不能仅凭 JSON 命令把大量参数张量高效送到另一张卡。项目通过 ZMQ 传控制消息、生成请求和 token 结果，通过官方 NCCL 权重传输发送张量。

首次同步时创建：

```python
self._engine = WeightTransferTrainerFactory.trainer_init(
    init_info=NCCLTrainerInitInfo(
        master_address=self.host,
        master_port=self.nccl_port,
        world_size=2,
        rank=0,
        packed=True,
    ),
    client=_WeightSyncClient(self.transport),
    source=ModuleSource(model),
)
```

`ModuleSource(model)` 提供当前模型的命名参数；`NCCLTrainerInitInfo` 给出参与方和连接设置；`_WeightSyncClient` 把官方控制接口转成发往采样服务的 ZMQ 命令；工厂将这些组合为可反复发送参数的传输引擎。

这里 `world_size=2` 是一个训练发送方和一个推理 worker，属于权重传输组；DeepSpeed 的 GRPO 训练组仍只有一个 rank。

之后调用 `send_weights()`，官方引擎负责开始、更新、结束调用，以及发送/接收端所需配合。项目保留引擎，后续同步继续复用，不每次重新建立。生命周期与 API 见 [vLLM 权重传输文档](https://docs.vllm.ai/en/v0.31.0/training/weight_transfer/)。

### 21. 参数传完为什么还不能立刻生成

KV 是模型用某套权重对前缀算出的中间状态。前缀 token 没变，权重变了，旧 KV 也不能继续当作新模型的结果。

服务端因此有三个状态：

```text
ready
  → start_weight_update
receiving
  → update_weights → finish_weight_update
loaded
  → confirm_version：清前缀缓存 → 提交版本 → 读回核对
ready，新版本可用
```

只有 `ready` 接受生成。`finish_weight_update` 只表示装载完成；项目还要清缓存并确认版本。客户端收到匹配应答才更新自己的版本号。

采样、训练更新和权重同步按顺序执行，一批多轮轨迹采样期间不更新权重。`VLLMRollout.generate()` 核对每次返回结果的版本是否与采样客户端记录一致；`rollout_pool()` 将采样版本随批次返回，GRPO 主循环核对它是否等于训练端当前策略版本，以发现训练已经更新但遗漏同步的情况。同步失败会终止阶段。版本检查核对的是版本标签，不校验权重张量内容。

官方传输解决张量怎样送达，项目的状态机解决何时允许继续业务执行。每次有效更新遵循“传参→清缓存→确认版本→接受下一批”。

## 五、回到“能手写，为什么还要引入”

### 22. 比较需要实现的能力，而不仅是能否运行

一个简单循环可以逐步扩展成有相似能力的系统。应比较需要达到的执行能力，而不仅看两边能否输出文本或让 loss 下降。

| 目标       | 简单实现已经可以做的事                 | 为达到当前引擎能力还要实现的事                       |
| ---------- | -------------------------------------- | ---------------------------------------------------- |
| 单卡学习   | 前向、反向、AdamW 更新                 | 多卡状态分片、通信、累积边界、精度状态与参数刷新     |
| 单请求生成 | 逐 token 前向和选择，可能已有 KV cache | 多请求调度、分块缓存管理、执行后端与内存规划         |
| 在线 GRPO  | 生成后训练同一个模型                   | 分离部署下的独立模型副本、传输、缓存失效、版本一致性 |

框架也有初始化、显存、通信、依赖和调试成本。小模型、小批次、单卡任务下，简单实现完全可能足够，不能凭组件名称断言更快。项目使用这套实现，并通过独立工程短测记录实际性能。

双卡 SFT 测试固定全局工作量下的速度与显存；GRPO 测试两卡部署中采样、训练、同步各耗多少时间。GPU 秒数按墙钟乘占用卡数计量，不能只看墙钟缩短而忽略额外用卡成本。

### 23. 工程测量记录什么

工程阶段测量单/双 rank SFT 各 40 更新、训推分离 GRPO 20 迭代的性能。

当前没有完整 GPU 运行结果。实现说明和数学推导不能写成已经取得的吞吐或任务成绩。操作入口见 [运行指南](运行指南.md)，完整实验定义见 [训练与评测](训练与评测.md)。

## 源码阅读顺序

理解关系后，每次只追一个完整动作：

1. [vllm_backend.py](../train/vllm_backend.py)：对照 `LocalVLLM.__init__/generate` 与 `_serve()` 中的 `LLM/generate`，确认两处如何持有同一种引擎。
2. 同一文件的 `VLLMRollout.__init__/generate`：从子进程启动追到一次请求和结果。
3. [rollout.py](../train/rollout.py)：看输出怎样变成工具调用，工具返回怎样成为下一轮输入。
4. [deepspeed_backend.py](../train/deepspeed_backend.py)：从构造函数追到 `update()`，看配置怎样约束执行。
5. [train_sft.py](../train/train_sft.py)：看 `run_update()` 怎样提供全局分母和更新组。
6. [train_grpo.py](../train/train_grpo.py)：顺着 `run()` 看采样、优势、更新、同步，再回到传参状态机。
7. [eval/evaluate.py](../eval/evaluate.py)：确认固定检查点评测为何直接创建 `LocalVLLM`，对照 GRPO 训练如何通过 `VLLMRollout` 使用常驻采样器。

读每段代码时问三个问题：这个对象持有什么状态？谁调用它？调用结束后，哪个模型、哪份缓存或哪个计数发生了变化？这样函数名才能连接成实际运行的系统。
