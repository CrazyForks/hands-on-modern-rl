# 17.5 推理时搜索

17.2—17.4 节已经能够判断一个中间步骤是否值得继续。现在把评价器放回生成过程：模型写到一半时，如果当前等式已经错误，就不必继续花几千 token 完成整条答案；如果两个候选共享正确前缀，也不必从题目重新生成两次。

推理时搜索保存中间状态，并在“扩展哪条路径、保留多少分支、何时回退”之间分配计算。本节沿着同一道二次方程观察四种方案：独立采样怎样浪费前缀，Beam Search 怎样固定保留若干路径，Tree of Thoughts 怎样允许回退，MCTS 又怎样用访问次数平衡已知高分路径与尚未充分尝试的路径。最后再用代码测试说明何时额外搜索值得付出成本。

## 1. 为什么需要复用中间推理

假设模型已经正确写出 $(x+2)(x+3)=0$，接下来只需要分别令两个因子为零。另一条路径却把判别式 $25-24$ 算成 $-1$。独立采样会把两条路径都生成到结尾；有了中间评价，系统可以继续扩展第一条，并提前停止第二条。

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

推理时搜索把这些重复工作改成一棵显式的状态树：

- **共享前缀**：相同的前缀推理只算一次
- **中间评估**：用 PRM 评估中间状态，决定继续走还是回退
- **资源分配**：把搜索算力用在最有希望的方向

接下来的算法差别主要在两个问题：每轮保留多少节点，以及被暂时淘汰的路径以后还有没有机会回来。

## 2. Beam Search 与 ToT 如何扩展推理树

**Beam Search** 先采用最直接的规则：任何时刻只维护 $K$ 个得分最高的部分推理。每轮扩展这些节点，用 PRM 重新评分，再留下新的前 $K$ 个。

### 2.1 Beam Search 的算法

```python
def beam_search_thoughts(prompt, model, prm, K=4, expansions=2, max_steps=10):
    # 初始 beam：只有一个空状态
    beams = [{"thought": "", "score": 1.0}]

    for step in range(max_steps):
        # 扩展每个 beam：让模型生成下一步推理
        candidates = []
        for beam in beams:
            for _ in range(expansions):
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

### 2.2 Beam Search 的特点

固定宽度使 Beam Search 容易实现，$K$ 条路径也可以并行扩展。代价同样来自固定的 $K$：简单题可能维护了多余路径，难题又可能过早删掉后来才显示价值的分支。被淘汰节点不会重新进入 beam，因此早期 PRM 误判会持续影响结果。

### 2.3 Beam Search 的适用场景

当步骤边界清楚、单步分数可靠、每层只需保留少量候选时，Beam Search 是合适的起点。若错误只有到很后面才暴露，当前分数很难决定早期剪枝，固定 beam 的风险就会增大。

### 2.4 ToT 如何保留分支与回退机会

[Tree of Thoughts](https://arxiv.org/abs/2305.10601)（Yao et al. 2023）是 Beam Search 的扩展——支持**分支、回退、DFS/BFS 混合**。

#### ToT 的核心结构

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

#### ToT 的算法

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

#### ToT 的特点

ToT 允许系统先把推理切成较粗的 thought，再用 BFS、DFS 或受限 beam 扩展。它能回到尚未删除的中间节点尝试新后续，但不会天然比 Best-of-N 更省计算；收益取决于共享前缀和中间评分是否真的有用。若每层完整保留 $B$ 个分支，节点数会随深度指数增长，因此实现时必须设置宽度、深度或总节点预算。

#### ToT 的实验结果

在 [24 Game](https://arxiv.org/abs/2305.10601)（24 点游戏）任务上：

| 方法                              | 成功率    |
| --------------------------------- | --------- |
| Greedy decoding                   | 7.3%      |
| CoT prompting                     | 4.0%      |
| Self-consistency（多采样 + 投票） | 9.0%      |
| **Tree of Thoughts**              | **74.0%** |

在这个任务和提示设置中，GPT-4 配合 ToT 从个位数成功率提高到 74%。24 点游戏的中间状态容易判断，特别适合搜索；这个幅度不能直接外推到没有明确状态和验证规则的开放任务。

## 3. MCTS 如何利用验证反馈选择路径

Beam Search 每一层都只看当前分数，早期被低估的路径一旦淘汰就不会回来。MCTS 增加访问次数，让系统既利用当前高分节点，也给尚未充分探索的节点保留机会。

**Monte Carlo Tree Search（MCTS）** 通过重复访问树来分配预算。在 LLM 推理中，可以让模型提出下一步，再用 rollout 结果、PRM 或外部检查器更新节点价值：

- 用结果奖励、PRM 或外部 verifier 评估节点
- 用模型作为 policy（推荐下一步）
- 用 UCB 公式平衡探索与利用

### 3.1 MCTS 的四个步骤

每次迭代执行：

1. **Selection（选择）**：从根开始，用 UCB 公式选择最优子节点，直到到达叶子
2. **Expansion（扩展）**：在叶子节点生成 N 个子节点
3. **Simulation（模拟）**：对子节点做 rollout（快速生成完整推理）
4. **Backpropagation（回传）**：把 rollout 的 reward 回传到所有祖先节点

### 3.2 UCB 公式

选择节点时需要兼顾两件事：当前平均分高的节点值得继续利用，访问次数少的节点也应该得到尝试机会。UCB 把这两项相加：

$$\text{UCB}(n) = Q(n) + c \cdot \sqrt{\frac{\ln N(p)}{N(n)}}$$

其中：

- $Q(n)$：节点 $n$ 的平均 reward（来自 PRM）
- $N(n)$：节点 $n$ 被访问的次数
- $N(p)$：父节点被访问的次数
- $c$：探索常数

第一项 $Q(n)$ 是已观察到的平均价值。第二项随父节点访问次数 $N(p)$ 增长而增大，却随当前节点访问次数 $N(n)$ 增长而减小，因此会优先补充探索访问较少的子节点。$c$ 越大，搜索越愿意尝试新分支；$c$ 越小，搜索越集中在当前高分分支。实现时通常先访问尚未探索的子节点，或在分母加入很小的平滑项，避免 $N(n)=0$ 时除零。

例如两个子节点的平均分同为 0.6，其中一个访问了 20 次，另一个只访问 2 次。第二项会让访问 2 次的节点获得更高 UCB，直到它积累足够多的证据。这里的 $Q$ 可以来自最终结果、PRM 或二者组合。

### 3.3 MCTS 的特点

访问次数让 MCTS 能把更多预算给高价值分支，同时继续试探访问较少的分支。渐近性质依赖有限动作空间、充分探索和可靠回报等假设；在开放文本生成中，动作候选由模型截断，评价器也会出错，因此不能把经典保证直接视为答案正确性保证。它的主要代价是多次 rollout、状态缓存和价值更新。

### 3.4 代表工作

- **rStar**（[arXiv:2408.06195](https://arxiv.org/abs/2408.06195)）：MCTS + 自我对弈，用于数学推理
- **AlphaProof**（[DeepMind 2024](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/)）：AlphaZero 风格强化学习、证明搜索与 Lean verifier
- **RAP**（[Reasoning via Planning](https://arxiv.org/abs/2305.14992)）：MCTS + LLM 作为 world model

### 3.5 AlphaCodium 的代码生成搜索

[AlphaCodium](https://arxiv.org/abs/2401.08500)（2024.01）把代码生成组织成“理解问题—生成测试—写初稿—执行—修复”的迭代流程：

- 代码任务可以用**已有测试**自动检查；模型生成的新测试只能补充覆盖，不能保证完整验证
- 用迭代式搜索：生成 → 测试 → 修复 → 再测试

#### AlphaCodium 的流程

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

#### AlphaCodium 的特点

- 已有测试可以直接充当结果 verifier；测试覆盖不足时，仍可能漏掉错误实现
- 迭代式（不是树搜索）——简单高效
- 论文在 CodeContests 等基准上报告了相对直接生成的提升，具体幅度随模型和评测设置变化

## 4. 搜索何时值得计算开销

搜索不会免费提高正确率。每扩展一个节点都要调用生成模型，很多方法还要调用 PRM、执行测试或运行证明检查器。是否采用搜索，取决于中间反馈的可靠程度和一次失败的代价。

不同方法可以先用“生成分支或扩展次数”估算开销：

| 方法                      | 主要预算项                                    |
| ------------------------- | --------------------------------------------- |
| Greedy decoding           | 一条完整生成                                  |
| Best-of-N                 | $N$ 条完整生成与 $N$ 次结果评分               |
| Beam Search（$K,D$）      | 约 $K\times D$ 组节点扩展与逐层评分           |
| Tree of Thoughts（$B,D$） | 完整展开为 $O(B^D)$，实际由剪枝和节点上限控制 |
| MCTS                      | 迭代次数、每次扩展数、rollout 长度与评分成本  |

若 Tree of Thoughts 在每一层保留全部 $B$ 个分支，深度达到 $D$ 时节点数会按 $B^D$ 增长，所以实际系统必须剪枝或设置节点预算。MCTS 不展开整棵树，它的计算量主要由迭代次数和每次扩展数决定。两者都比独立采样多了状态维护和逐步评分，因此是否使用搜索要看中间反馈能否抵消这些开销。

科学计算、形式化证明和竞赛编程通常有可执行检查器。搜索每扩展一条路径都能得到较可靠反馈，此时额外计算更容易转化为更高成功率。没有可靠 verifier 时，搜索也可能沿着错误评分反复扩展。

### 4.1 训练时搜索与推理时搜索

还需要决定搜索发生在训练阶段还是推理阶段。

**训练时使用搜索结果**（如 AlphaProof 的强化学习循环）：

- 把搜索的结果作为训练数据
- 让模型提高高价值步骤的概率
- 部署时仍可按任务需要继续搜索

**推理时搜索**（如 ToT、MCTS）：

- 推理时用搜索提升性能
- 无需重新训练即可改变搜索预算

两种方式也可以组合：

- 训练时轻度搜索（加速收敛）
- 推理时根据任务难度决定是否搜索

这与 [第 16 章 Test-time Compute Scaling](../chapter19_reasoning/test-time-scaling) 的思想一致——把算力花在哪里，是一个工程权衡。

## 小结

PRM 在训练时可以提供过程奖励，在推理时也可以为部分路径评分。搜索算法利用这些分数决定保留、扩展或放弃哪些中间步骤。

主要方法：

- **Beam Search**：简单并行，适合中等任务
- **Tree of Thoughts**：支持回退和剪枝，适合复杂任务
- **MCTS**：按访问次数在探索与利用之间分配预算
- **AlphaCodium**：代码专用，用单元测试作为 verifier

完整展开搜索树的节点数会随深度迅速增长；Beam Search 和 MCTS 通过限制保留宽度或采样次数控制开销。因此，实际系统要根据任务价值和验证器成本选择 Best-of-N、受限搜索或直接生成。

[17.6 并行推理与答案汇总](./parallel-reasoning-and-summary) 继续讨论另一种分配方式：并行生成多条推理，再让模型或 verifier 交换信息并聚合结果。
