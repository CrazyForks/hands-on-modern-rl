# 18.4 形式化 PRM Verifier

前面两节的 PRM 路线——判别式和生成式——都有一个共同的根本问题：**它们都是基于 LLM 的，所以会继承 LLM 的错误**。

- 判别式 PRM 错标：标注员判断错，PRM 学到错的标签
- 生成式 PRM 错判：LLM 自己对推理理解错，输出错的评价

如果有一种 verifier，**对正确的推理永远说对，对错误的推理永远说错，零误判**，那 PRM 的所有问题都消失了。

这种 verifier 存在——它就是**形式化证明器**（formal theorem prover），如 Lean4、Coq、Isabelle。这一节我们讨论形式化 PRM 路线，这是 PRM 研究的"终极 verifier"方向。

## 11.4.1 为什么形式化是终极 verifier

### 形式化语言 vs 自然语言

数学证明可以用两种语言表达：

**自然语言**（informal）：

```text
证明 √2 是无理数：
假设 √2 = p/q，p 和 q 互质。
那么 p² = 2q²，所以 p 是偶数。
设 p = 2k，代入得 2k² = q²，所以 q 也是偶数。
这与 p, q 互质矛盾。所以 √2 是无理数。
```

人类可读，但**可能有歧义、可能有跳跃、可能有错误**。

**形式化语言**（Lean4）：

```lean
theorem sqrt_two_irrational : Irrational (√2) := by
  intro h
  rcases h with ⟨p, q, h1, h2⟩
  -- 假设 √2 = p/q
  have h3 : p^2 = 2 * q^2 := by
    have : (√2)^2 = (p/q)^2 := by rw [h1]
    simp at this
    rw [div_pow] at this
    field_simp at this
    linarith
  -- ...
  sorry  -- (省略完整证明)
```

完全无歧义，**每一步推理都必须用 Lean4 的规则严格证明**。如果证明通过 Lean4 编译器，**它就是数学上正确的**——没有讨论余地。

### Lean4 verifier 的特点

Lean4 的特性让它成为理想的 PRM：

- **零误判**：通过编译 = 证明正确。这是数学定理保证的
- **自动化**：编译过程是自动的，不需要人工判断
- **可扩展**：可以定义新的数学结构、新的定理
- **社区支持**：[Mathlib](https://github.com/leanprover-community/mathlib4) 已经形式化了大学水平的数学

### 形式化的代价

但形式化也有代价：

- **领域受限**：Lean4 主要用于数学。其他领域（自然语言推理、代码逻辑）没有成熟的形式化系统
- **数据稀缺**：Lean4 代码相对少，LLM 在 Lean4 上的预训练数据不足
- **门槛高**：写 Lean4 代码需要专门训练，大部分数学家不熟悉

## 11.4.2 AlphaProof 与 DeepMind 的 IMO 银牌

2024 年 7 月，DeepMind 宣布 [AlphaProof](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/) 在国际数学奥林匹克（IMO）2024 上达到**银牌水平**——解决了 6 道题中的 4 道。这是形式化 PRM 路线的里程碑。

### AlphaProof 的架构

AlphaProof 的核心思路是**AlphaZero 风格的自我博弈 + Lean4 verifier**：

```text
┌────────────────────────────────────────────────────┐
│ 1. 问题形式化：把数学题翻译成 Lean4                 │
│                                                    │
│ 2. AlphaZero 风格搜索：                            │
│    - 策略网络：推荐下一步 Lean4 tactic            │
│    - 价值网络：评估当前证明状态                    │
│    - MCTS：搜索证明树                              │
│                                                    │
│ 3. Lean4 verifier：每个 tactic 自动验证           │
│                                                    │
│ 4. 自我博弈训练：用搜索结果训练策略和价值网络      │
└────────────────────────────────────────────────────┘
```

这个架构与 [AlphaGo Zero](https://www.nature.com/articles/nature24270) 几乎完全一样——区别只是动作空间从"围棋棋盘"变成了"Lean4 tactic 序列"。

### AlphaProof 的关键设计

**设计一：问题形式化**

IMO 题目是自然语言写的，AlphaProof 需要先翻译成 Lean4。DeepMind 用了一个**形式化器**（formalizer）——一个专门训练的 LLM，把自然语言数学题翻译成 Lean4。

这一步本身有挑战——自然语言的歧义性、数学符号的多样性、定理表述的不唯一性。DeepMind 报告这一步的准确率约 50%——一半的题目能正确翻译。

**设计二：大规模 Lean4 训练**

AlphaProof 在约 **100 万个** Lean4 问题上做了训练。这些问题包括：

- Mathlib 中已有的定理
- 自动生成的问题（用 LLM 生成 Lean4 命题）
- 历年 IMO、Putnam 题目的 Lean4 版本

**设计三：Lean4 MCTS**

AlphaProof 用 MCTS 搜索证明——每个节点是一个证明状态，每个动作是一个 Lean4 tactic，reward 是"是否完成证明"。Lean4 编译器作为 MCTS 的环境——告诉搜索"这个 tactic 是否有效"。

### AlphaProof 的成绩

IMO 2024 的 6 道题，AlphaProof 解出 4 道：

- **Algebra 1**：✓（4 分满分）
- **Algebra 2**：✓
- **Combinatorics**：✓
- **Number Theory**：✓
- **Geometry 1**：✗（Lean4 翻译失败）
- **Geometry 2**：✗

总分约 25 分（满分 42），相当于银牌水平（金牌门槛约 29 分）。

Geometry 失败的原因不是推理能力不足，而是 **Lean4 翻译失败**——几何题的形式化比代数难得多。

## 11.4.3 AlphaGeometry 2 与 几何专用

针对 AlphaProof 在几何上的弱点，DeepMind 发布了 [AlphaGeometry 2](https://www.nature.com/articles/s41586-024-07819-5)——专门解决几何题的形式化系统。

AlphaGeometry 2 的关键创新：

- **合成数据**：自动生成 5 亿个几何题及其证明
- **辅助构造**：让模型学会"添加辅助线"这一关键技巧
- **符号 + 神经**：符号推理（DD = Deductive Database）+ 神经网络（LM）混合

  2024.07 的报告中，AlphaGeometry 2 在 IMO 几何题上达到了金牌水平。

## 11.4.4 DeepSeek-Prover-V2 与 开源的形式化 PRM

[DeepSeek-Prover-V2](https://arxiv.org/abs/2504.21801)（2025.04）是 DeepSeek 开源的形式化 PRM 工作。它的目标是：

- 用 Lean4 + RL 训练一个能解数学竞赛题的开源模型
- 推进形式化 PRM 的工业可用性

### Prover-V2 的方法

DeepSeek 的方法与 AlphaProof 类似，但有几个改进：

**改进一：递归证明搜索**

Prover-V2 用**递归定理证明**——把一个难定理分解为若干子目标，每个子目标再分解，直到子目标可以独立证明。

```text
主目标：证明 A
  ├── 子目标 1：证明 B（如果 B 成立，则 A 成立）
  │     ├── 子子目标 1.1：证明 C
  │     └── 子子目标 1.2：证明 D
  └── 子目标 2：证明 E
```

这种分解让证明搜索更结构化，避免在一个大的证明空间里乱找。

**改进二：binary reward**

Prover-V2 用最简单的 binary reward——证明通过 = 1，证明失败 = 0。Lean4 verifier 提供这个 reward，零噪声。

**改进三：大规模数据合成**

DeepSeek 自动生成了大量 Lean4 定理和证明，用于训练。生成的过程包括：

- 用 LLM 从自然语言数学题生成 Lean4 命题
- 用 Monte Carlo Tree Search 找证明
- 把找到的证明作为训练数据

### Prover-V2 的成绩

Prover-V2 在 MiniF2F（形式化数学 benchmark）上的成绩：

| 模型                           | MiniF2F（验证集） |
| ------------------------------ | ----------------- |
| AlphaProof（公开版）           | ~70%              |
| **DeepSeek-Prover-V2**         | **88.9%**         |
| GPT-5（natural language 推理） | ~50%              |

这是开源模型在 MiniF2F 上的 SOTA。88.9% 意味着 Prover-V2 在大学水平的形式化数学上接近完美。

## 11.4.5 形式化 PRM 的代价

虽然形式化 PRM 在数学上"零误判"，但它有几个代价：

### 领域受限

Lean4 主要用于数学。其他领域：

- **代码逻辑**：可以用形式化方法（如 Dafny、F\*），但工业可用性低
- **自然语言推理**：没有成熟的形式化系统
- **多模态推理**：完全没有形式化方法

所以形式化 PRM 目前只适用于**数学和形式化数学相关任务**。

### 形式化数据稀缺

Lean4 代码相对少：

- Mathlib 有约 100 万行 Lean4 代码
- 但与自然语言的预训练数据（万亿 token）相比，少 6 个数量级

LLM 在 Lean4 上的能力远弱于自然语言。这是形式化 PRM 的根本瓶颈。

### 翻译成本

形式化 PRM 要求把任务翻译成 Lean4。这个翻译本身是 LLM 的工作，会引入错误。AlphaProof 报告 50% 的翻译成功率——一半的题目被翻译错误。

### 训练成本

Lean4 MCTS 训练的计算成本极高——每次 tactic 调用都要触发 Lean4 编译（每次编译几百毫秒到几秒）。AlphaProof 训练了几个月，消耗的算力与 GPT-4 相当。

## 11.4.6 形式化 PRM 的未来

尽管有这些代价，形式化 PRM 的研究方向仍然非常重要。几个未来方向：

### 自动形式化

让 LLM 学会把自然语言自动翻译成 Lean4。这是 [AlphaProof 的形式化器](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/) 和 [Autoformalization with Large Language Models](https://arxiv.org/abs/2205.12615) 等研究的方向。

### Lean4 + LLM 混合

不要求 LLM 完全生成 Lean4，而是让 LLM 生成自然语言推理 + Lean4 验证。这样既有 LLM 的灵活性，又有形式化的严谨性。

### 扩展到其他领域

把形式化方法扩展到代码（用 Dafny、F\*）、物理（用 Lean4 物理库）、生物（用 Lean4 化学库）。这是 Mathlib 之外的扩展方向。

### 神经符号集成

把 LLM 的"灵感"和形式化的"严谨"结合——LLM 提出证明思路，形式化系统验证。这是 DeepMind AlphaProof 和 AlphaGeometry 2 共同的方向。

## 小结

形式化 PRM 是 PRM 研究的终极方向——通过 Lean4 等形式化系统实现零误判验证。AlphaProof 在 IMO 上达到银牌水平，DeepSeek-Prover-V2 在 MiniF2F 上达到 88.9%，证明了这条路的技术可行性。

但形式化 PRM 受限于领域（主要是数学）、数据稀缺（Lean4 训练数据少）、翻译成本（自然语言→Lean4 不完美）。它的工业应用目前集中在数学推理领域，难以推广到通用 LLM 任务。

至此，我们看完了 PRM 的三条路线：

- **判别式 PRM**（11.2）：标注成本高，但分类精确
- **生成式 PRM**（11.3）：标注少，泛化好，但依赖 LLM 自己判断
- **形式化 PRM**（11.4）：零误判，但领域受限

接下来两节讨论 PRM 在推理时搜索（MCTS、ToT、PaCoRe）中的应用——PRM 不只是训练时的 reward，还是推理时的搜索引导。
