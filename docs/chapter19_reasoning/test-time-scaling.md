# 16.3 Test-time Compute Scaling

16.2 节说明了 R1-Zero 怎样用结果奖励强化长推理。模型能够生成更长的推理链以后，还要决定一次任务究竟投入多少推理计算。[Snell et al.](https://arxiv.org/abs/2408.03314) 系统比较了训练计算与推理计算，并把问题表述为 **Test-time Compute Scaling**：在模型参数不变时，通过增加候选、修订或搜索提高当前任务的成功率。

## 1. 在训练与推理之间分配算力

传统 LLM 的算力分配是高度倾斜的：

```text
预训练算力：~10^23 FLOPs（GPT-4 级别）
后训练算力：~10^21 FLOPs
推理算力（每次调用）：~10^15 FLOPs
```

也就是说，**预训练比推理多了 8 个数量级的算力**。这个分配在"模型一次前向给出答案"的范式下是合理的——推理时算少了，因为不需要思考。

但推理模型打破了这个假设。o1 在做一道 AIME 题时，可能生成 10K-100K token 的 CoT——这比传统 LLM 的 200-500 token 答案多了两个数量级。**o1 把推理算力从 ~10^15 提升到了 ~10^17**。

Snell et al. 的关键问题是：**如果固定总预算（训练 + 推理），应该花在哪里？**

### 1.1 Snell 等人的预算实验

[Snell et al. 2024](https://arxiv.org/abs/2408.03314)（"Scaling LLM Test-Time Compute Optimally")是 Test-time Compute Scaling 的奠基性论文。它的实验设计很巧妙：

**实验设置**：固定一个 base model（Llama-3-8B-Instruct），在不同难度的数学题上，比较两种提升方式：

- **方式 A**：用更多推理算力——让模型生成 N 个候选解，用 verifier 选最好的（best-of-N）
- **方式 B**：用更多训练算力——把 base model 升级为更大的模型（参数量增加）

**核心发现**：

1. **在简单题上**，增加推理算力的收益**超过**增加训练算力。一个 8B 模型 + 充分推理，可以打败一个 70B 模型不推理。
2. **在难题上**，增加推理算力的收益**递减**——base model 的能力上限决定了推理的上限。
3. **最佳的推理策略**取决于题目难度：简单题用 best-of-N，难题用 sequential revision（修订）。

这个发现的工程含义巨大：

- **推理算力是"可调的"**——可以根据任务难度动态决定花多少算力
- **训练算力是"固定的"**——一旦训练完，参数就定了

所以推理模型的核心优势不是"参数更多"，而是**"算力分配更灵活"**。

## 2. 三种增加推理计算的方法

Snell et al. 把 test-time compute 的使用方式归纳为两类：

### 2.1 并行采样

让模型独立生成 N 个候选解，然后用一个 verifier 选最好的。这是 best-of-N 的思路。

```python
# 并行采样示意
candidates = [model.generate(prompt) for _ in range(N)]
scores = [verifier.score(prompt, c) for c in candidates]
best = candidates[argmax(scores)]
```

**优点**：

- 天然并行，速度快
- 简单题效果好（N 越大，命中正确解的概率越高）

**缺点**：

- 难题效果差——如果 base model 的单次解题概率 < 1/N，N 个采样也大概率全错
- 需要 verifier（这是 [第 17 章 PRM](../chapter20_prm_search/outcome-vs-process) 的核心话题）

### 2.2 顺序修订

让模型生成一个初始解，然后基于这个解生成修订版本，反复迭代。

```python
# 顺序修订示意
solution = model.generate(prompt)
for _ in range(K):
    feedback = model.critique(prompt, solution)
    solution = model.revise(prompt, solution, feedback)
```

**优点**：

- 适合难题——每次修订都能纠错
- 不需要外部 verifier

**缺点**：

- 串行，速度慢
- 修订可能越改越错（feedback 本身可能错）

### 2.3 树搜索

更复杂的方式是树搜索——把推理过程展开成一棵树，每个节点是一个中间推理步骤，用搜索算法（MCTS、beam search）找最优路径。这是 [第 17 章 PRM 与推理时搜索](../chapter20_prm_search/inference-time-search) 的核心内容，这里先不展开。

## 3. 用 Deep Think 理解并行推理

2025 年 10 月，Google 发布了 [Gemini 3 Pro Deep Think](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/)——把 test-time compute scaling 推到了一个新的极端。Deep Think 的核心思想是：**在 MoE 模型上叠加一层"并行推理层"**。

传统推理模型（o1、R1）是**串行思考**——生成 token 1 → token 2 → token 3，每个 token 依赖前一个。这种串行结构让推理速度受限于自回归生成的速度。

Deep Think 引入了**并行推理**：

- 同时生成多条独立的推理路径
- 在路径之间做信息聚合（类似 ensemble）
- 用一个"协调器"决定何时停止、如何合并

这种结构让 Deep Think 可以在固定时间内**生成比串行模型多 N 倍的推理 token**（N 是并行路径数）。

### 3.1 Deep Think 的评测表现

Deep Think 在发布时的几个关键数字：

- **IMO 2025**：金牌（Gold），证明数学推理能力达到 IMO 顶尖选手水平
- **HLE（Humanity's Last Exam）**：48.4%，远超同期 GPT-5（约 30%）、Claude Opus 4.5（约 35%）
- **ARC-AGI-2**：84.6%，比 o3 的 75% 进一步突破
- **Codeforces rating**：超过 3000（人类 top 0.01%）

### 3.2 动态调整并行路径

2026 年 2 月，Google 发布了 Gemini 3.1 Pro Deep Think。主要改进：

- **动态并行路径数**：根据题目难度自动调整并行度（简单题 4 路，难题 32 路）
- **跨推理路径的注意力**：让不同推理路径之间可以"看到"彼此的中间结果，形成弱协调
- **更长的上下文**：从 1M token 扩展到 10M token，支持超长 CoT

  3.1 Deep Think 在 ARC-AGI-2 上达到 91.2%，HLE 上达到 52.7%——再次刷新了 test-time scaling 的上限。

## 4. 在质量、延迟与成本之间取舍

test-time compute scaling 不是免费的。每多花一倍推理算力，意味着：

- **延迟翻倍**：用户等待时间变长
- **API 费用翻倍**：按 token 计费的模型，思考 token 也算钱
- **能耗翻倍**：大规模部署的能源成本上升

这引出一个工程问题：**什么时候该开推理，什么时候不该开？**

| 任务类型                               | 推荐策略                   |
| -------------------------------------- | -------------------------- |
| 简单问答（"今天天气"）                 | 关闭推理，直接给答案       |
| 中等难度（"写个排序算法"）             | 轻量推理，几十到几百 token |
| 数学竞赛 / 代码生成                    | 充分推理，几千到几万 token |
| 科研推理（OpenAI o1-pro / Deep Think） | 极致推理，十万级 token     |

这也构成了 [16.4 Hybrid Thinking 与思考预算](./hybrid-thinking) 的工程动机：根据任务难度选择推理模式和预算。

### 4.1 Test-Time Scaling 何时饱和

Snell et al. 的实验发现，test-time compute 的收益在难题上递减。后续研究（[DeepSeek R1 论文](https://arxiv.org/abs/2501.12948)、[Qwen3 技术报告](https://arxiv.org/abs/2505.09388)）在更大规模上确认了这个现象：

- **简单题**：少量推理通常已经足够，继续增加计算的边际收益很低
- **中等题**：test-time compute 在某个点之后开始收益递减
- **难题**：test-time compute 很快饱和——base model 的能力不足是硬约束

这个发现的深层含义是：**test-time compute scaling 不能无限替代 training compute scaling**。两者是互补的：

- training compute 决定**能力上限**
- test-time compute 决定**接近上限的程度**

如果基座模型没有掌握解题所需的知识或操作，增加采样和修订次数也难以产生正确路径。因此，推理计算依赖基座模型已经具备的能力。

## 小结

Test-time Compute Scaling 提供了三种可以直接调节的资源：并行候选数量、单条推理链的修订次数和搜索树的规模。任务越难，增加这些资源越可能带来收益；基座能力不足或任务已经很简单时，收益会很快降低。

下一节把这条规律用于部署：同一个模型怎样在直接回答与深度思考之间切换，并用思考预算限制延迟和成本。
