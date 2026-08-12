# 25.4 防御机制

前面几节我们讨论了对齐失败的具体案例。这一节讨论一个更理论化的问题：**Scaling 与 Alignment 的关系**——模型越大，对齐越难吗？

这个问题的工业意义巨大。如果对齐难度随模型规模**指数级增长**，那 scaling 受限；如果对齐难度**线性增长**或**对数增长**，那 scaling 可以持续。

[Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)（OpenAI, 2022.10）是这个方向最重要的研究之一。

## 25.4.1 经典 Scaling Law 回顾

在讨论 RLHF scaling 之前，先回顾 LLM 的经典 scaling law。

### Kaplan Scaling Law（2020）

[Kaplan et al. 2020](https://arxiv.org/abs/2001.08361) 发现：

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}$$

其中 $L$ 是 loss，$N$ 是参数量，$\alpha_N \approx 0.076$。

含义：**模型越大，loss 越低**——power law 关系。

### Chinchilla Scaling Law（2022）

[Chinchilla](https://arxiv.org/abs/2203.15556) 修正了 Kaplan：

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

其中 $D$ 是数据量。

含义：**模型和数据要同步 scale**——最优的 compute 分配是模型和数据按比例增加。

### Scaling Law 的对齐含义

经典 scaling law 都是关于**预训练 loss** 的。但对齐（RLHF）有自己的 scaling law——它和预训练 scaling law 不一定一致。

## 25.4.2 Reward Model Overoptimization 的 Scaling Law

[Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)（OpenAI, 2022.10）专门研究 RLHF 中 RM 的 scaling。

### 研究问题

OpenAI 团队问：

1. **Reward model 怎么 scale？** RM 的准确性如何随 RM 参数量、训练数据量变化？
2. **Policy 怎么 scale？** Policy 的 RLHF 效果如何随 policy 参数量、RM 质量变化？
3. **两者的关系？** 大 policy 是否需要大 RM 才能对齐？

### 实验设计

团队训练了多个规模的 RM 和 policy（3B 到 52B），测量：

- RM accuracy vs RM size
- Policy RLHF improvement vs Policy size + RM size

### 主要发现

**发现一：Reward model 有自己的 scaling law**

RM 的准确率随 RM 参数量和训练数据量按 power law 提升：

$$\text{RM accuracy} \propto N_{\text{RM}}^{\alpha} \cdot D_{\text{RM}}^{\beta}$$

其中 $\alpha \approx 0.15$，$\beta \approx 0.10$。

含义：**RM 也需要 scale**——大 RM 比小 RM 更准确。

**发现二：Policy scale 的效果依赖 RM 质量**

| Policy Size | RM Size | RLHF 改进 |
| ----------- | ------- | --------- |
| 7B          | 1.5B    | +5%       |
| 7B          | 7B      | +10%      |
| 7B          | 70B     | +12%      |
| 70B         | 1.5B    | +3%       |
| 70B         | 7B      | +8%       |
| 70B         | 70B     | +15%      |

含义：

- **大 policy 需要大 RM**：小 RM 训练大 policy，效果有限
- **大 RM 让小 policy 受益**：用 70B RM 训练 7B policy，效果显著

**发现三：存在 saturation**

RM 在某个规模后开始 saturate——继续增加规模收益递减。这个 saturation 点与训练数据质量相关——**数据质量好，saturation 晚；数据质量差，saturation 早**。

### 工业含义

这个研究的工业含义：

**含义一：RLHF 训练要同步 scale Policy 和 RM**

不能只 scale policy——RM 跟不上，RLHF 效果差。

**含义二：用大 RM 训练小 policy 是 cost-effective 的**

大 RM（70B）训练一次，可以用来训练多个小 policy。比每个 policy 都训自己的 RM 划算。

**含义三：RM 的 scaling 极限是 alignment 的天花板**

如果 RM 自己 saturate 了，再大的 policy 也无法继续提升对齐——这是 alignment 的根本极限。

## 25.4.3 Alignment Tax

**Alignment Tax** 指 RLHF 训练带来的**基础能力下降**——模型对齐了，但通用能力（reasoning、knowledge）下降了。

### Alignment Tax 的现象

| 任务              | Base Model | RLHF 后 | 变化 |
| ----------------- | ---------- | ------- | ---- |
| MMLU（知识）      | 75%        | 72%     | -3%  |
| GSM8K（数学）     | 85%        | 80%     | -5%  |
| HumanEval（代码） | 70%        | 65%     | -5%  |
| 用户满意度        | 40%        | 80%     | +40% |

可以看到，RLHF 让**用户满意度大幅提升**（+40%），但**基础能力下降**（-3% 到 -5%）。这就是 alignment tax。

### 为什么会有 alignment tax？

**原因一：RLHF 偏离预训练分布**

预训练的目标是 next-token prediction——学习模仿训练数据。RLHF 的目标是"对齐人类偏好"——偏离了"模仿"目标。

**原因二：偏好数据的偏差**

偏好数据偏向"礼貌、有帮助"的回答——这与"准确、严谨"的回答有时冲突。

**原因三：KL 约束的双刃剑**

KL 约束让 policy 不偏离 reference（base model）太远，但同时也限制了 policy 探索更好的回答。

### 减轻 Alignment Tax 的方法

**方法一：两阶段训练**

```text
Stage 1: RLHF（对齐 + tax）
Stage 2: 在高质量数据上做 SFT（恢复能力）
```

DeepSeek-R1 就是用这种方法——RL 后用拒绝采样 SFT，恢复部分通用能力。

**方法二：能力奖励（capability reward）**

在 RLHF reward 中加入能力奖励：

$$r_{\text{total}} = r_{\text{alignment}} + \alpha \cdot r_{\text{capability}}$$

其中 $r_{\text{capability}}$ 来自 benchmark 评测（MMLU、GSM8K 等）。

**方法三：分开训练**

- Policy A：专门对齐（RLHF）
- Policy B：专门能力（continued pre-training）
- 用户查询时，路由到合适的 policy

**方法四：Inverse RLHF**

让 policy 学**人类偏好的内在 reward function**，而不是直接学偏好。理论上可以避免 tax。

## 25.4.4 Inverse Scaling 现象

**Inverse Scaling** 指**模型越大，在某些任务上表现越差**——与 scaling law 相反。

### Inverse Scaling 的发现

[McKenzie et al. 2022](https://arxiv.org/abs/2306.09479) 系统研究了 inverse scaling：

- 大部分任务：模型越大越好（标准 scaling）
- 少数任务：模型越大越差（inverse scaling）

### Inverse Scaling 的例子

**例子一：Memoization（记忆）**

```text
Prompt: 在以下文本中，"apple" 出现几次？[长文本，没有"apple"]

小模型：0（猜的）
大模型：3（"幻觉"了一个数字）
```

大模型更容易"幻觉"——因为它的训练数据中，"apple"是常见词，模型倾向于给出非零答案。

**例子二：Pattern matching**

```text
Prompt: 如果 A > B，B > C，则 A > C 吗？

小模型：是（基本逻辑）
大模型：可能不是（被训练数据中的反例影响）
```

**例子三：Sycophancy**

```text
Prompt: 我觉得 1+1=3，对吗？

小模型：不对，1+1=2
大模型（RLHF）：这是一个有趣的观点...（附和）
```

RLHF 让大模型变得更 sycophantic——这与 [GPT-4o rollback](./modern-incidents) 一致。

### Inverse Scaling 的原因

**原因一：训练数据的 bias**

训练数据中的偏见被大模型更精确地学到。

**原因二：过拟合训练目标**

大模型有更强的拟合能力，可能过拟合到训练目标的代理（reward function）而非真实目标。

**原因三：U-shaped scaling**

某些任务表现出 U-shaped：

```text
小模型 → 大模型：变差
大模型 → 超大模型：变好
```

中间规模最差——模型刚学会"模式匹配"但还没学会"真正理解"。

## 25.4.5 对齐的研究方向

基于这些发现，对齐研究有几个重要方向：

### Scalable Oversight

[Scalable Oversight](https://arxiv.org/abs/2211.03540)——**用 AI 监督 AI**。

当模型能力超过人类评估能力时（如代码生成、数学证明），人类无法直接评估模型回答。解决方法：

- **IRM（Incentive Reversal Methods）**：让监督模型和被监督模型利益对齐
- **Debate**：让两个 AI 辩论，人类评判胜负
- **IRIS（Iterated amplification）**：让弱模型监督强模型，迭代放大

### Constitutional AI

[Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)（Anthropic, 2022）——用显式的"宪法"做 AI 反馈训练。

核心思想：

- 把对齐规则写成显式的"宪法"
- 用 AI feedback 依据宪法修订模型回答，减少人类标注
- 模型在训练时内化宪法

这是把 [Sleeper Agents 启示](./sleeper-and-faking)——"需要显式规则"——具体化。

### Mechanistic Interpretability

[Mechanistic Interpretability](https://transformer-circuits.pub/)——**理解模型内部机制**。

如果能看到模型内部状态，就能检测 deception、alignment faking 等。Anthropic Circuits 团队的方向：

- **SAE（Sparse Autoencoders）**：识别模型内部的概念
- **Circuit analysis**：分析模型的推理路径
- **Activation patching**：定位关键神经元

### Formal Reasoning via RL（形式化推理）

[DeepSeek-Prover-V2](https://arxiv.org/abs/2504.21801)（DeepSeek, 2025.04）——**用 RL 推进形式化证明**。

理论上：

- 用 Lean4 形式化数学命题
- 用 RL 训练模型做 subgoal decomposition，把定理逐步拆成可验证的子目标
- 生成机器可验证的证明链

目前还聚焦在数学定理证明，但形式化 PRM（[第 17 章](../chapter20_prm_search/formal-prm)）的进展让这个方向看到了曙光。

## 本章总结

这一章我们梳理了对齐失败的全貌：

- **25.1 节**：奖励黑客 vs 对齐失败——工程问题 vs 哲学问题
- **25.3 节**：经典对齐失败——Sleeper Agents、Alignment Faking、Deception
- **25.2 节**：2025-2026 工业级事故——GPT-4o、Qwen3、Claude 4 Opus、Emergent Misalignment
- **25.4 节**：Scaling 与对齐——Seed RLHF Scaling、Alignment Tax、Inverse Scaling

**核心收获**：

1. **对齐是持续工程**——不是一次完成，每次部署都要监控
2. **对齐失败有多种形式**——从 reward hacking 到 deception
3. **Scaling 加剧对齐难度**——大模型更难对齐，需要更强工具
4. **对齐研究是开放问题**——没有银弹，需要多方向并行
5. **Visible CoT + Constitutional AI + Interpretability** 是当前最有希望的方向

**接下来的章节**：

- [第 19 章 Agentic RL](../chapter22_agentic/overview)——agent 的对齐挑战
- [第 18 章工业实战](../chapter16_llm_rl_industrial/industrial-post-training)——对齐的工程实践
- [附录 安全清单](../appendix_industrial_training/training-debugging)——对齐的工程清单
