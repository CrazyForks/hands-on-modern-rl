# 18.1 Outcome vs Process 奖励

上一章我们看到推理模型在硬任务上取得了突破——o1、DeepSeek-R1、Claude Opus 4.6 都能做复杂数学、代码、科研推理。但这些模型都依赖一个关键假设：**最终答案的对错可以作为 RL 的奖励信号**。

这个假设在简单任务上成立——一道数学题答对就是答对，答错就是答错。但随着任务变复杂，这个假设开始失效：

- **长 CoT 任务**：一个 10000 token 的推理链，可能前 8000 token 都对，最后 2000 token 错。模型只看到"答错了"，不知道哪里错了
- **代码生成**：一个程序编译失败，是哪一行错了？哪个逻辑错了？
- **多步 agent 任务**：10 步 trajectory 失败了，是第几步失败的？

这就是 **稀疏奖励问题（sparse reward problem）**——奖励信号只在序列结尾出现，中间步骤得不到反馈。**过程奖励模型（Process Reward Model, PRM）** 就是为这个问题而生：它给推理过程的每一步打分，把稀疏奖励变成密集奖励。

## 本章要回答的问题

- **Outcome 奖励 vs Process 奖励**的本质区别是什么？
- **判别式 PRM**（OpenAI 的经典路线）怎么工作？标注成本为什么是瓶颈？
- **生成式 PRM**（ThinkPRM）为什么用更少标签就能超越判别式？
- **形式化 PRM**（AlphaProof、Lean4）怎么实现"零误判"验证？
- **推理时搜索**（MCTS、Tree of Thoughts、Beam Search）怎么用 PRM 引导？
- **并行协调推理**（PaCoRe）如何替代传统的深度优先推理？

## 章节地图

```text
Outcome 奖励 vs Process 奖励
     ├── 稀疏奖励问题
     ├── 信用分配的本质
     └── 为什么 PRM 在长 CoT 任务里不可替代
判别式 PRM（经典路线）
     ├── OpenAI "Let's Verify Step by Step"
     ├── PRM800K 数据集
     ├── PRM 作为 Re-ranking 模型
     └── 局限：标注成本高、泛化弱
生成式 PRM（新路线）
     ├── ThinkPRM：生成式 verifier
     ├── 标签少 100 倍的关键
     ├── Verifier Compute Scaling
     └── 生成式 vs 判别式对比
形式化 PRM（终极 Verifier）
     ├── Lean4 / Coq：零误判验证
     ├── AlphaProof：IMO 银牌
     ├── AlphaGeometry 2：几何专用
     └── DeepSeek-Prover-V2：MiniF2F 88.9%
推理时搜索
     ├── Beam Search over Thoughts
     ├── MCTS over Thoughts
     ├── Tree of Thoughts
     └── AlphaCodium / rStar
并行协调推理（PaCoRe）
     ├── 16 路并行 rollout
     ├── outcome-based RL 训练
     ├── AIME 2025: 94.4
     └── 深度 vs 广度的权衡
GenRM 与 Verifier 模型
     ├── Generative Reward Model
     ├── LLM-as-Judge
     └── Self-Rewarding Language Models
```

## 与其他章的关系

这一章假定你已经读过：

- [第 13 章 RLHF 奖励模型](../chapter15_rlhf/reward-function-design)——Outcome Reward Model 的基础
- [第 16 章 GRPO 改进家族](../chapter18_grpo/grpo-family)——信用分配问题在 GRPO 中的体现
- [第 17 章推理模型](../chapter19_reasoning/intro)——为什么推理模型需要 PRM

本章后续会指向：

- [第 20 章 Agentic RL](../chapter22_agentic/overview)——多步 trajectory 的过程奖励
- [第 28 章奖励黑客](../chapter30_alignment_failures/classical-failures)——PRM 的 reward hacking 问题

## 一个直觉性的开场

在进入正式内容前，先建立两个关键直觉：

**直觉一：PRM 是把"考试评分"变成"作业批改"**。传统 outcome reward 就像考试评分——只看最终答案对不对，对了 100 分，错了 0 分。PRM 就像老师批改作业——每一步推理都打分，对的步骤给正分，错的步骤给负分，半对的步骤给部分分。批改虽然更费时，但反馈更细致，学生（模型）能学到更多。

**直觉二：PRM 是 verifier，不是 policy**。一个常见的混淆是把 PRM 当作"另一种 reward model"。严格来说，PRM 是一个 **verifier**——它的工作不是"生成好的推理"，而是"判断推理好不好"。Verifier 和 policy 的训练目标不同：policy 要学会"做什么"，verifier 要学会"评价什么"。这一区分对理解后续的 GenRM、LLM-as-Judge 很重要。

下面从稀疏奖励开始，说明最终结果为什么不足以指导长推理链。

这一节我们从最基础的问题开始：**为什么 outcome reward 在长 CoT 任务上不够？为什么需要 process reward？**

## 稀疏奖励问题

考虑一个具体的例子：让模型证明"√2 是无理数"。

模型生成的 CoT 长这样（简化版）：

```text
Step 1: 假设 √2 = p/q，其中 p, q 互质
Step 2: 那么 2 = p²/q²，即 p² = 2q²
Step 3: 所以 p² 是偶数
Step 4: 所以 p 是偶数（这一步用了"偶数的平方是偶数"的逆否命题）
Step 5: 设 p = 2k
Step 6: 代入：4k² = 2q²，即 2k² = q²
Step 7: 所以 q² 是偶数，q 也是偶数
Step 8: 这与 p, q 互质矛盾
Step 9: 所以 √2 是无理数  ✓
```

假设这个证明在 Step 6 出错了——比如写成"4k² = 2q²，即 4k = q²"（漏了平方）。最终结论"√2 是无理数"还是对的（结论正确），但推理过程有错。

**Outcome reward** 给这个回答打分：

- 如果用最终答案（√2 是无理数）作为对错标准 → 正确 → reward = 1
- 但实际上推理过程错了，模型应该学到"Step 6 是错的"

**Outcome reward 的问题**：

1. **信号稀疏**：10000 token 的推理链，只得到 1 个 reward 信号
2. **错误归因**：模型不知道是哪一步错了，无法精准修正
3. **奖励误标**：推理错了但答案对了（运气好）→ 正反馈，强化错误推理
4. **学习低效**：模型只能从整体 reward 反推哪些步骤重要，效率极低

这就是**稀疏奖励问题（sparse reward problem）**——奖励信号在时间维度上分布太稀疏，无法提供有效的学习信号。

## 信用分配问题

稀疏奖励问题在 RL 里有一个更正式的名字：**信用分配问题（credit assignment problem）**。

具体定义：给定一个序列决策任务，最终 reward 是 $r_T$，怎么把这个 reward 分配回序列中的每一步 $a_1, a_2, \ldots, a_T$？哪些步骤应该被强化，哪些应该被抑制？

经典 RL 用几个方法解决这个问题：

### 折扣回报（Discounted Return）

把未来 reward 折扣到现在：

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \ldots + \gamma^{T-t} r_T$$

这是 [第 3 章 MDP](../chapter03_mdp/value-bellman) 讨论的经典方法。它隐含一个假设：**离当前时刻越远的 reward，对当前决策的影响越小**。这个假设在物理控制任务里成立（推小车时，10 步后的 reward 对当前推力的影响确实小），但在 LLM 推理里不成立——一个数学证明的第 1 步和第 10 步同等重要。

### GAE（Generalized Advantage Estimation）

[第 8 章 PPO 的 GAE](../chapter10_ppo/gae-reward-model) 通过引入 $\lambda$ 参数，在 bias 和 variance 之间做权衡。GAE 是 PPO 的标准做法，但它的根本局限是——**它需要 value function**，而 GRPO 故意省略了 value function。

### Token-level loss

[DAPO](../chapter18_grpo/deepseek-dapo) 的 Token 级损失是一种近似 PRM 的方法——不是给"整条推理链"打分，而是给"每个 token"打分。但 token 级 loss 仍然依赖 outcome reward 反向传播——它没有独立的 verifier 来评估"这个 token 好不好"。

### PRM（Process Reward Model）

这是本章的主角——**训练一个独立的 verifier，对每一步推理打分**。PRM 的输出是密集的——每一步都有一个分数，模型可以精确地知道"哪一步好，哪一步坏"。

## Outcome Reward vs Process Reward：形式化对比

让我们用数学形式化两者的区别。

### Outcome Reward Model (ORM)

ORM 接受一个 prompt $q$ 和一个完整回答 $o$，输出一个标量分数：

$$\text{ORM}(q, o) \in \mathbb{R}$$

这个分数代表"回答整体有多好"。在数学任务里，它通常是 0 或 1（答错或答对）。

ORM 训练数据形式：

```text
(prompt, response, final_correctness)
```

例：("证明 √2 是无理数", "<完整证明>", 1)

### Process Reward Model (PRM)

PRM 接受 prompt $q$、回答 $o$、和回答中的某个步骤位置 $i$，输出该步骤的分数：

$$\text{PRM}(q, o, i) \in \mathbb{R}$$

这个分数代表"第 $i$ 步推理好不好"。在数学任务里，它可以是：

- 二元：1（正确）/ 0（错误）/ -1（无关）
- 连续：[0, 1] 之间的概率

PRM 训练数据形式：

```text
(prompt, response, step_index, step_correctness)
```

例：("证明 √2 是无理数", "<完整证明>", 4, 1) # 第 4 步是正确的

### 在 RL 训练中的使用

ORM 用于 RL 训练时：

$$r_{\text{ORM}} = \text{ORM}(q, o)$$

整个序列共享一个 reward。

PRM 用于 RL 训练时（一种常见做法）：

$$r_t = \text{PRM}(q, o, \text{step}(t))$$

每个 token $t$ 根据"它属于哪个推理步骤"获得对应的 PRM 分数。同一推理步骤内的 token 共享该步骤的分数。

这种做法把稀疏 reward 变成了密集 reward，每个 token 都有清晰的训练信号。

## 为什么 PRM 在长 CoT 任务里不可替代

PRM 的价值在长 CoT 任务里最明显。考虑三个场景：

### 短回答任务（function calling、简单问答）

- CoT 长度：100-500 token
- ORM 信号密度：每 100-500 token 一个 reward
- PRM 价值：**有限**——序列短，ORM 信号已经够密

### 中等推理任务（GSM8K、MATH）

- CoT 长度：500-2000 token
- ORM 信号密度：每 500-2000 token 一个 reward
- PRM 价值：**显著**——可以精确定位错误步骤

### 长 CoT 推理任务（AIME、IMO、科研）

- CoT 长度：5000-50000 token
- ORM 信号密度：每 5000+ token 一个 reward
- PRM 价值：**不可替代**——ORM 信号几乎无法提供有效学习信号

[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 训练时报告了一个现象：训练初期，模型的 CoT 长度迅速从几百 token 增长到几千 token，但 AIME 准确率提升缓慢。直到训练后期，模型学会"在关键步骤做检查"（self-verification），AIME 才有显著突破。这说明长 CoT 任务**需要过程级信号才能高效学习**。

## PRM 的三条工业路线

PRM 的工业实现有两条主要路线，对应不同的训练方法：

### 判别式 PRM（Discriminative PRM）

把 PRM 当作一个**分类器**——输入 prompt + step，输出"这步对/错"的概率。

代表工作：OpenAI 的 [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)（Lightman et al. 2023）。

训练数据：人工标注的步骤正确性（PRM800K 数据集）。

模型：BERT-style encoder 或 decoder-only LLM 加分类头。

### 生成式 PRM（Generative PRM）

把 PRM 当作一个**生成器**——让 LLM 用自然语言"评价"每个步骤。

代表工作：[ThinkPRM](https://arxiv.org/abs/2504.16828)（2025.04）。

训练数据：少量种子示例 + LLM 生成的评价。

模型：任何 LLM（LLaMA、Qwen、DeepSeek），用 prompting + 少量 fine-tune。

### 形式化 PRM（Formal PRM）

把 PRM 当作一个**形式化验证器**——用 Lean4 / Coq 这种定理证明器自动验证。

代表工作：DeepMind 的 [AlphaProof](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/)（2024.07）、DeepSeek 的 [DeepSeek-Prover-V2](https://arxiv.org/abs/2504.21801)（2025.04）。

训练数据：形式化的数学定理（Lean4 格式）。

模型：LLM + Lean4 verifier。

这三条路线是接下来三节的主题：[18.2 判别式 PRM](./discriminative-prm)、[18.3 生成式 PRM](./generative-prm)、[18.4 形式化 PRM](./formal-prm)。

## 小结

Outcome reward 在简单任务上足够，但在长 CoT 任务上信号太稀疏，无法提供有效的学习信号。Process reward 通过给每一步推理打分，把稀疏奖励变成密集奖励，是长 CoT 任务的关键技术。

PRM 有三条工业路线：

- **判别式**：分类器思路，准确但标注成本高
- **生成式**：LLM 评价思路，标注少但精度依赖 prompt engineering
- **形式化**：Lean4 验证，零误判但只适用于形式化任务

下面三节分别详细讨论这三条路线，最后两节讨论 PRM 在推理时搜索（MCTS、ToT）和并行协调推理（PaCoRe）中的应用。
