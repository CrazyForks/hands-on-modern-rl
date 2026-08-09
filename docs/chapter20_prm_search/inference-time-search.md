# 18.5 推理时搜索

前面三节我们讨论了 PRM 的训练——判别式、生成式、形式化。这一节我们换一个角度：**PRM 在推理时怎么用**。

最直接的应用是 [Best-of-N + Re-ranking](./discriminative-prm)（11.2 节讨论过）——生成 N 个候选，用 PRM 选最好的。但 Best-of-N 是**无记忆的并行采样**，没有利用推理的中间状态。

**推理时搜索（Inference-time Search）** 是更结构化的方法——把推理过程展开成一棵**思考树（Thought Tree）**，用 PRM 评估每个节点，用搜索算法（DFS、BFS、Beam Search、MCTS）找最优路径。

这一节讨论几种主要的推理时搜索方法。

## 11.5.1 为什么需要搜索？

考虑一个数学题：

```text
求 x² + 5x + 6 = 0 的解
```

模型可以生成多种推理路径：

```text
路径 A：用求根公式
  → x = (-5 ± √(25-24)) / 2 = (-5 ± 1) / 2
  → x = -2 或 x = -3

路径 B：用因式分解
  → x² + 5x + 6 = (x+2)(x+3) = 0
  → x = -2 或 x = -3

路径 C：尝试配方法
  → x² + 5x = -6
  → x² + 5x + 25/4 = 25/4 - 6 = 1/4
  → (x + 5/2)² = 1/4
  → x + 5/2 = ±1/2
  → x = -2 或 x = -3
```

三条路径都得到正确答案。但如果模型在某条路径上出错（比如路径 A 算错根号），单次采样的结果就是错的。

Best-of-N 解决这个问题——生成多条独立路径，用 PRM 选最好的。但 Best-of-N 有局限：

- **没有利用路径间的相似性**：如果两条路径前半段相同，Best-of-N 会重复生成
- **无法在中间回退**：如果一条路径走到一半发现走错了，Best-of-N 只能从头再来
- **搜索效率低**：N 条独立采样相当于暴力枚举

**推理时搜索**通过结构化的搜索树解决这些问题：

- **共享前缀**：相同的前缀推理只算一次
- **中间评估**：用 PRM 评估中间状态，决定继续走还是回退
- **资源分配**：把搜索算力用在最有希望的方向

## 11.5.2 Beam Search over Thoughts

**Beam Search** 是最简单的搜索方法——维护 K 个最优的"部分推理"（beam），每步扩展所有 beam，用 PRM 评估，保留 K 个最好的。

### 算法描述

```python
def beam_search_thoughts(prompt, model, prm, K=4, max_steps=10):
    # 初始 beam：只有一个空状态
    beams = [{"thought": "", "score": 1.0}]

    for step in range(max_steps):
        # 扩展每个 beam：让模型生成下一步推理
        candidates = []
        for beam in beams:
            for _ in range(N_expansions):
                next_thought = model.generate_next(prompt, beam["thought"])
                score = prm.score(prompt, beam["thought"] + next_thought)
                candidates.append({
                    "thought": beam["thought"] + next_thought,
                    "score": score
                })

        # 选 top-K 作为新的 beams
        beams = sorted(candidates, key=lambda x: x["score"], reverse=True)[:K]

        # 如果找到完整答案，停止
        if any(is_complete(b["thought"]) for b in beams):
            break

    return beams[0]["thought"]  # 返回最优 beam
```

### Beam Search 的特点

**优点**：

- 实现简单
- 速度快（K 个 beam 并行）
- 适合宽度广的搜索空间

**缺点**：

- K 是固定的——简单题浪费算力，难题不够
- 没有"回退"机制——一旦 beam 被淘汰，不再考虑
- 容易陷入局部最优

### 适用场景

Beam Search 适合：

- 推理空间广（多种解法）
- 单步推理容易评估
- 简单到中等难度的任务

## 11.5.3 Tree of Thoughts（ToT）

[Tree of Thoughts](https://arxiv.org/abs/2305.10601)（Yao et al. 2023）是 Beam Search 的扩展——支持**分支、回退、DFS/BFS 混合**。

### ToT 的核心结构

```text
                Root
              /      \
            A1        A2
           /  \      /  \
         B1   B2   B3   B4
        / \    |    |   / \
       C1  C2  C3   C4 C5  C6

       搜索算法：BFS（广度优先）或 DFS（深度优先）
       评估：每步用 PRM 打分
       回退：低分节点被剪枝
```

### ToT 的算法

```python
def tree_of_thoughts(prompt, model, prm, max_depth=10, breadth=4):
    # 从根开始 DFS
    def dfs(thought, depth):
        if depth >= max_depth:
            return [{"thought": thought, "score": prm.score(prompt, thought)}]

        # 生成 N 个候选下一步
        candidates = []
        for _ in range(breadth):
            next_thought = model.generate_next(prompt, thought)
            full_thought = thought + next_thought
            score = prm.score(prompt, full_thought)
            candidates.append({"thought": full_thought, "score": score})

        # 按分数排序，剪枝低分
        candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates = candidates[:breadth // 2]  # 剪枝一半

        # 对保留的 candidate 递归
        results = []
        for c in candidates:
            results.extend(dfs(c["thought"], depth + 1))

        return results

    return dfs("", 0)
```

### ToT 的特点

**优点**：

- 支持回退和剪枝
- 可以处理深度推理任务
- 比 Best-of-N 更高效

**缺点**：

- 速度慢（指数级展开）
- PRM 评估的次数多
- 不适合超长 CoT 任务

### ToT 的实验结果

在 [24 Game](https://arxiv.org/abs/2305.10601)（24 点游戏）任务上：

| 方法                              | 成功率    |
| --------------------------------- | --------- |
| Greedy decoding                   | 7.3%      |
| CoT prompting                     | 4.0%      |
| Self-consistency（多采样 + 投票） | 9.0%      |
| **Tree of Thoughts**              | **74.0%** |

这是一个巨大的提升——同样的 GPT-4 base，仅仅通过更好的推理时搜索，成功率从 7% 提升到 74%。

## 11.5.4 MCTS over Thoughts

**Monte Carlo Tree Search（MCTS）** 是 AlphaGo 用的算法。在 LLM 推理上，MCTS 的核心思想是：

- 用 PRM 作为 value function（评估节点价值）
- 用模型作为 policy（推荐下一步）
- 用 UCB 公式平衡探索与利用

### MCTS 的四个步骤

每次迭代执行：

1. **Selection（选择）**：从根开始，用 UCB 公式选择最优子节点，直到到达叶子
2. **Expansion（扩展）**：在叶子节点生成 N 个子节点
3. **Simulation（模拟）**：对子节点做 rollout（快速生成完整推理）
4. **Backpropagation（回传）**：把 rollout 的 reward 回传到所有祖先节点

### UCB 公式

UCB（Upper Confidence Bound）平衡探索与利用：

$$\text{UCB}(n) = Q(n) + c \cdot \sqrt{\frac{\ln N(p)}{N(n)}}$$

其中：

- $Q(n)$：节点 $n$ 的平均 reward（来自 PRM）
- $N(n)$：节点 $n$ 被访问的次数
- $N(p)$：父节点被访问的次数
- $c$：探索常数

直觉：第一项是"已知价值"（exploitation），第二项是"未探索的潜力"（exploration）。

### MCTS 的特点

**优点**：

- 自适应——多探索有希望的方向
- 理论保证（收敛到最优）
- 可以处理深度搜索

**缺点**：

- 实现复杂
- 计算成本高（每步多次 rollout）
- 对 PRM 质量敏感

### 代表工作

- **rStar**（[arXiv:2408.06195](https://arxiv.org/abs/2408.06195)）：MCTS + 自我对弈，用于数学推理
- **AlphaProof**（[DeepMind 2024](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/)）：MCTS + Lean4 verifier
- **RAP**（[Reasoning via Planning](https://arxiv.org/abs/2305.14992)）：MCTS + LLM 作为 world model

## 11.5.5 AlphaCodium 与 代码生成搜索

[AlphaCodium](https://arxiv.org/abs/2401.08500)（2024.01）是专门为代码生成设计的搜索方法。它的核心思想：

- 代码任务的"正确性"可以用**单元测试**自动验证（类似 Lean4）
- 用迭代式搜索：生成 → 测试 → 修复 → 再测试

### AlphaCodium 的流程

```text
1. 问题理解：让 LLM 提取关键信息、生成测试用例
2. 初步解：生成一个候选解
3. 迭代修复：
   a. 运行测试用例
   b. 如果失败，分析错误
   c. 让 LLM 修复错误
   d. 重复直到所有测试通过
4. 输出最终解
```

### AlphaCodium 的特点

- 不需要 PRM——单元测试就是 verifier
- 迭代式（不是树搜索）——简单高效
- 在 Codeforces 上比单次生成提升 30%+

## 11.5.6 推理时搜索的算力开销

不同搜索方法的算力开销（以"模型 forward 次数"为度量）：

| 方法                     | forward 次数（典型） |
| ------------------------ | -------------------- |
| Greedy decoding          | 1                    |
| Best-of-N                | N（通常 4-64）       |
| Beam Search（K, D）      | K × D                |
| Tree of Thoughts（B, D） | O(B^D)（指数级）     |
| MCTS（N_iter, N_expand） | N_iter × N_expand    |

可以看到，**Tree of Thoughts 和 MCTS 的算力开销是指数级的**。这是它们在工业部署上不如 Best-of-N 普及的原因。

但在科学计算、形式化证明、竞赛编程等"对就是对的"的任务上，搜索的算力开销是值得的——因为这些任务对正确性的要求极高。

## 11.5.7 训练时 vs 推理时搜索

一个深刻的问题是：**搜索应该在训练时做，还是推理时做？**

**训练时搜索**（如 AlphaProof 的自博弈）：

- 把搜索的结果作为训练数据
- 训练完的模型"内化"了搜索的能力
- 推理时不需要搜索

**推理时搜索**（如 ToT、MCTS）：

- 训练时不搜索，正常 RL
- 推理时用搜索提升性能
- 算力开销大但灵活

**工业实践**通常是混合：

- 训练时轻度搜索（加速收敛）
- 推理时根据任务难度决定是否搜索

这与 [第 17 章 Test-time Compute Scaling](../chapter19_reasoning/test-time-scaling) 的思想一致——把算力花在哪里，是一个工程权衡。

## 小结

推理时搜索是 PRM 的另一面——不仅训练时给密集 reward，推理时也用 PRM 引导搜索。

主要方法：

- **Beam Search**：简单并行，适合中等任务
- **Tree of Thoughts**：支持回退和剪枝，适合复杂任务
- **MCTS**：自适应探索，理论保证
- **AlphaCodium**：代码专用，用单元测试作为 verifier

搜索的算力开销是指数级的，所以**工业上 Best-of-N 仍然是主流**，搜索只在科研、形式化、竞赛等高价值任务上使用。

下一节讨论 PaCoRe——一种新的推理范式，把"深度搜索"转为"广度并行"，平衡算力和质量。
