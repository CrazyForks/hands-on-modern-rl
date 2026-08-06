# 18.6 并行协调推理总结

至此我们讨论了 PRM 的三条训练路线（判别式、生成式、形式化）和四种推理时搜索方法（Beam Search、ToT、MCTS、AlphaCodium）。这些方法都隐含一个假设：**推理是串行的**——模型一步一步往下走。

但 2025 年下半年出现了一个新方向：**并行协调推理（Parallel Coordinated Reasoning, PaCoRe）**——不串行，而是**并行展开多条独立推理，然后聚合**。这个思路与 [Gemini Deep Think](../chapter19_reasoning/test-time-scaling) 的"并行推理层"在哲学上一致。

这一节讨论 PaCoRe 路线，以及 PRM 的一个相关概念——**GenRM（Generative Reward Model）**。

## 11.6.1 深度 vs 广度 与 推理算力的两种花法

[第 8 章 Test-time Compute Scaling](../chapter19_reasoning/test-time-scaling) 我们讨论过两种推理算力的花法：

- **串行深度（Sequential Depth）**：模型生成一条很长的 CoT
- **并行广度（Parallel Breadth）**：模型生成多条独立的 CoT，用某种方式聚合

| 方式     | 代表                        | 算力分配           | 优势           | 劣势             |
| -------- | --------------------------- | ------------------ | -------------- | ---------------- |
| 串行深度 | o1、R1、Qwen3 thinking      | 全部算力花在一条链 | 适合难题       | 速度慢、错误累积 |
| 并行广度 | Best-of-N、Self-Consistency | 算力分散到多条     | 速度快、多样性 | 需要聚合机制     |

**PaCoRe** 是并行广度的极致版本——把 16-32 条独立推理聚合，但聚合方式不是简单的 majority vote，而是**用 LLM 协调**。

## 11.6.2 PaCoRe 的设计

[PaCoRe](https://github.com/stepfun-ai/PaCoRe)（StepFun，ACL 2026 论文）的设计核心：

### 核心流程

```text
┌─────────────────────────────────────────────────────┐
│ Step 1: 并行生成 N 条推理                            │
│   - 同一个 prompt                                   │
│   - 每条推理独立生成（不同温度、不同 seed）          │
│   - 不需要 PRM 引导                                 │
├─────────────────────────────────────────────────────┤
│ Step 2: 协调器（Coordinator）聚合                    │
│   - LLM 读取所有 N 条推理                           │
│   - 综合判断：哪些推理是正确的？                    │
│   - 输出最终答案                                    │
├─────────────────────────────────────────────────────┤
│ Step 3: （训练时）用 outcome reward 强化            │
└─────────────────────────────────────────────────────┘
```

### 协调器 vs 投票

PaCoRe 与 Best-of-N + Majority Vote 的关键区别是**协调器**：

- **Majority Vote**：选择出现次数最多的答案（简单统计）
- **PaCoRe Coordinator**：让 LLM 阅读所有推理，"判断"哪个推理最可信（语义聚合）

协调器的优势：

- **能处理多解问题**：如果 N 条推理得到不同的"正确"答案（比如一道题有多个等价解），majority vote 会随机选一个，PaCoRe 能识别它们都对
- **能识别推理质量**：即使两条推理得到相同答案，PaCoRe 能判断哪条推理过程更严谨

### 训练 PaCoRe

PaCoRe 的训练用 **outcome-based RL**——只奖励最终答案的正确性，不奖励中间推理。这与 R1-Zero 一致，简单且不需要 PRM 标注。

```python
def pacore_reward(prompt, target_answer):
    # 1. 并行生成 N 条推理
    reasonings = [model.generate(prompt, temperature=t) for t in temperatures]

    # 2. 协调器聚合
    final_answer = coordinator.aggregate(prompt, reasonings)

    # 3. 用 outcome reward 训练
    reward = 1.0 if final_answer == target_answer else 0.0

    # 把 reward 反向传播到所有 reasonings 和 coordinator
    return reward
```

这种训练方式让**整个 PaCoRe 系统作为一个整体被 RL 优化**，而不是单独优化每条推理。

## 11.6.3 PaCoRe 的实验结果

PaCoRe 在 [AIME 2025](https://github.com/stepfun-ai/PaCoRe) 上的结果：

| 方法                          | AIME 2025 | 推理 token 数  |
| ----------------------------- | --------- | -------------- |
| 单次 thinking（baseline）     | 60-70%    | ~10K           |
| Best-of-32 + Majority Vote    | 80%       | 320K（32×10K） |
| **PaCoRe（16 路并行）**       | **94.4%** | 160K（16×10K） |
| Gemini 3.1 Deep Think（参考） | 90%+      | 数百万 token   |

PaCoRe 用**比 Best-of-N 更少的算力**达到了更高的准确率——这说明协调器（LLM 聚合）比简单投票更有效。

## 11.6.4 PaCoRe vs Deep Think vs MCTS

三种推理范式的对比：

| 维度     | PaCoRe             | Deep Think              | MCTS         |
| -------- | ------------------ | ----------------------- | ------------ |
| 推理结构 | N 条独立 + 协调    | N 条并行 + 跨路径注意力 | 树形展开     |
| 算力开销 | N × 单条           | N × 单条                | 指数级       |
| 训练要求 | outcome RL         | 模型架构改动            | PRM + value  |
| 适合任务 | 中等难度，多样性高 | 难题，需要协调          | 形式化、精确 |
| 实现难度 | 中                 | 高（模型改动）          | 高           |

### 何时用 PaCoRe？

PaCoRe 的优势在于：

- **简单**：不需要 PRM、不需要修改模型架构
- **高效**：比 Best-of-N 准确率更高
- **可扩展**：N 路并行可以线性扩展

适合的场景：

- 多解问题（一道题有多种正确解法）
- 中等难度推理（不需要深度树搜索）
- 算力充足但模型改动困难

## 11.6.5 GenRM 与 生成式奖励模型

讨论 PaCoRe 时我们提到"协调器是一个 LLM"。这引出一个更广泛的概念——**GenRM（Generative Reward Model）**。

GenRM 的核心思想：**把 reward 计算变成生成任务**。传统 RM 是一个回归模型——输入 prompt + response，输出标量分数。GenRM 是一个 LLM——输入 prompt + response，输出自然语言评价 + 最终判断。

### GenRM 的形式

```text
输入：prompt + response + "请评价这个回答"
输出：自然语言评价 + [GOOD/BAD]
```

GenRM 可以用 [verbal confidence](https://arxiv.org/abs/2305.14992)——让 LLM 输出概率作为 reward：

$$\text{GenRM}(q, o) = P(\text{"good"} | q, o, \text{prompt})$$

即给定 prompt，模型输出"good"这个 token 的概率。

### GenRM vs 判别式 RM

| 维度     | 判别式 RM        | GenRM              |
| -------- | ---------------- | ------------------ |
| 架构     | Encoder + 回归头 | 标准 LLM           |
| 输出     | 标量分数         | 自然语言 + 概率    |
| 训练     | 回归损失         | 语言模型损失       |
| 解释性   | 弱（只有分数）   | 强（自然语言理由） |
| 推理速度 | 快               | 慢                 |

### GenRM 与 PRM 的关系

GenRM 是一个更宽泛的概念——它可以做 ORM（输出整条回答的评价）或 PRM（输出每步推理的评价）。

[11.3 节的 ThinkPRM](./generative-prm) 是 GenRM 做 PRM 的代表。其他 GenRM 工作：

- **Generative Verifiers**（[Zhang et al.](https://arxiv.org/abs/2408.15240)）：用 Chain-of-Thought 评价
- **LLM-as-Judge**（[Zheng et al.](https://arxiv.org/abs/2306.05685)）：用 GPT-4 评价其他模型的输出

## 11.6.6 LLM-as-Judge 与 Self-Rewarding

LLM-as-Judge 是 GenRM 的一种工业实践——用一个强 LLM（GPT-4、Claude）评价其他模型的输出。

### LLM-as-Judge 的应用

- **benchmark 评测**：用 GPT-4 作为 MT-Bench、AlpacaEval 的裁判
- **训练数据筛选**：用 LLM 过滤高质量训练数据
- **RLHF 替代**：用 LLM 替代人类标注偏好数据（RLAIF）

### Self-Rewarding Language Models

[Self-Rewarding LM](https://arxiv.org/abs/2401.10020)（Meta 2024）把 LLM-as-Judge 推到极致——**让模型评价自己的输出**：

```python
def self_reward_training(prompt, model):
    # 1. 生成多个回答
    responses = [model.generate(prompt) for _ in range(N)]

    # 2. 让模型自己评价
    rewards = [model.judge(prompt, r) for r in responses]

    # 3. 用自评价做 RL（DPO 或 PPO）
    model = rl_update(model, prompt, responses, rewards)
```

这种方法**完全摆脱了外部 RM**——模型自己既是 policy 又是 reward。优点是不需要 RM 训练，缺点是**自我评价可能强化已有的偏见**（模型认为自己好的，会被强化；模型不擅长的，会被弱化）。

## 11.6.7 PRM 与 Verifier 的未来

到 2026 年中，PRM 和 verifier 研究的几个趋势：

### 从判别到生成

判别式 PRM → 生成式 PRM → 自我评价——趋势是用 LLM 的内在推理能力替代外部 verifier。

### 从深度到广度

Tree of Thoughts → MCTS → PaCoRe——趋势是从深度串行搜索到广度并行协调。

### 从静态到动态

固定 PRM → 动态 verifier → 自适应搜索——趋势是让 verifier 在推理过程中动态调整。

### 从单一到混合

ORM-only → PRM-only → PRM + ORM + 形式化 + LLM-as-Judge 混合——趋势是用多种 verifier 互补。

## 本章总结

这一章我们把 PRM 和推理时搜索的全貌梳理了一遍：

- **11.1 节**：Outcome vs Process 奖励——稀疏奖励问题与信用分配
- **11.2 节**：判别式 PRM——OpenAI Let's Verify 与 PRM800K
- **11.3 节**：生成式 PRM——ThinkPRM 用 1% 标签达到 SOTA
- **11.4 节**：形式化 PRM——AlphaProof、Lean4、DeepSeek-Prover-V2
- **11.5 节**：推理时搜索——Beam Search、ToT、MCTS、AlphaCodium
- **11.6 节**：并行协调推理 PaCoRe 与 GenRM、LLM-as-Judge

**核心收获**：

1. **PRM 是长 CoT 任务的关键技术**——把稀疏奖励变成密集奖励
2. **三条路线各有优势**——判别式精确、生成式高效、形式化零误判
3. **推理时搜索可以进一步提升性能**——但算力开销大，工业上 Best-of-N 仍是主流
4. **并行协调（PaCoRe）是新方向**——把深度搜索转为广度并行，平衡算力和质量
5. **GenRM 和 LLM-as-Judge 是 verifier 的趋势**——用 LLM 的内在能力替代外部 verifier

**接下来的章节**：

- [第 10 章 Agentic RL](../chapter22_agentic/intro)——多步 trajectory 中的 PRM
- [第 12 章奖励黑客](../chapter15_rlhf/evaluation)——PRM 的 reward hacking 问题
- [第 13 章工业实战](../chapter17_dpo/industrial-post-training)——PRM 在工业训练中的使用
