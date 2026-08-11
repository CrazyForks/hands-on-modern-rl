# 16.4 Hybrid Thinking 如何控制思考预算

16.3 节展示了增加推理计算的三种方法，但简单翻译、事实问答和短代码通常不需要长推理链。若所有请求都使用同一套高预算，服务延迟和成本会随输出长度一起增加。

**Hybrid Thinking** 让同一个模型同时支持直接回答与深度思考。本节先介绍两种模式怎样进入同一模型，再比较长思考与并行短回答，随后说明 long2short 压缩，最后讨论部署时的模式切换和预算控制。

## 1. 同一模型如何支持两种模式

[DeepSeek V3.1](https://api-docs.deepseek.com/news/news250821)（2025.08）是 Hybrid Thinking 的早期工业实现。V3.1 的设计思路是：

- **同一个模型**：不维护两个独立的"思考模型"和"非思考模型"
- **模式切换**：通过 prompt 中的特殊标记（如 `<think>` / `</think>`）控制是否进入思考模式
- **思考内容可选**：思考部分可以保留在输出里（用户可见），也可以剥离（用户只看到最终答案）

V3.1 的训练流程包括两个阶段：

1. **推理 RL**（reasoning RL）：在数学 / 代码任务上做 RL，让模型学会深度推理
2. **通用 RL**（general RL）：在对话 / 工具调用任务上做 RL，让模型学会在不需要推理时直接给答案

第二阶段的关键是**保持模型已有的推理能力，同时学会"什么时候该推理、什么时候不该"**。DeepSeek 的具体做法是混合训练数据——一部分 prompt 显式触发推理（数学题），一部分 prompt 不触发（闲聊），让模型自己学到模式切换的判断力。

### 1.1 Qwen3 怎样融合两种模式

[Qwen3 技术报告](https://arxiv.org/abs/2505.09388)（2025.05）提出了更系统的 Hybrid Thinking 方案。Qwen3 全系模型（从 0.6B 到 235B）都支持两种模式：

#### Thinking Mode Fusion

Qwen3 不是简单地"训练一个推理模型 + 训练一个非推理模型"，而是**在单一模型中融合两种模式**。具体做法：

- 训练数据中混合了 thinking 和 non-thinking 两种样本
- thinking 样本：完整的长 CoT + 答案
- non-thinking 样本：直接的短答案
- 通过 `<think>` token 触发模式切换

这个训练方式让模型**自动学会根据 prompt 选择模式**——遇到数学题就开 think，遇到闲聊就不开。

#### Thinking Budget 与可控的思考深度

Qwen3 引入了一个工程参数 **thinking budget**——用户可以在 API 调用中指定 `thinking_budget=N`，限制模型最多花 N 个 token 在思考上。

```python
# Qwen3 thinking budget 示例
response = client.chat.completions.create(
    model="qwen3-235b-a22b",
    messages=[{"role": "user", "content": "证明根号2是无理数"}],
    extra_body={"thinking_budget": 2000}  # 最多 2000 token 思考
)
```

这个参数的工程价值是：

- **延迟可控**：高频服务可以设小 budget，保证响应时间
- **成本可控**：按 token 计费时，可以预算控制
- **质量权衡**：难题设大 budget，简单题设小 budget

但 thinking budget 也带来一个**算法挑战**：怎么让模型"在 budget 用完时优雅地停下来"？Qwen3 的做法是在 RL 训练中加入**长度惩罚**——超过 budget 的回答会被惩罚。这与 [DAPO 的 Overlong Reward Shaping](../chapter18_grpo/deepseek-dapo) 思路一致。

## 2. 长思考和并行短回答如何选择

2025 年 4 月，Ma et al. 发表了一篇有趣的研究——[Reasoning Models Can Be Effective Without Thinking](https://arxiv.org/abs/2504.09858)（常称 NoThinking）。这篇论文提出了一个反直觉的主张：**在很多任务上，"不思考 + Best-of-N" 比 "思考" 效果更好**。

### 2.1 NoThinking 的核心实验

实验设置：

- base model：Qwen3-32B
- 任务：AIME、GPQA、MMLU-Pro
- 对比两种推理方式：
  - **Thinking**：标准 thinking mode，单次生成（带 CoT）
  - **NoThinking + BoN**：禁用 thinking，但生成 N 个候选解，用 majority vote 选最好

实验结果：

| 任务         | Thinking（单次） | NoThinking + BoN（N=32） |
| ------------ | ---------------- | ------------------------ |
| AIME 2024    | 60.2             | **65.1**                 |
| GPQA Diamond | 55.3             | **58.7**                 |
| MMLU-Pro     | 72.1             | 71.8（持平）             |

也就是说，**把 N 次 thinking 的算力，改成 N 次 nothinking + 投票，效果反而更好**。

### 2.2 NoThinking 为什么在部分任务上有效

Ma et al. 给出了几个解释：

1. **Thinking 的 CoT 不是免费的**——错误的推理步骤会累积，导致最终答案错。NoThinking 跳过推理直接给答案，避免了推理错误。
2. **Best-of-N 提供 diversity**——N 个独立采样比一条长链的多样性更高，命中正确解的概率更大。
3. **CoT 在某些任务上是 over-thinking**——简单任务不需要长推理，思考反而引入噪声。

但 NoThinking 也有边界：

- **难题上效果不如 Thinking**——AIME 上的难题，NoThinking + BoN 仍然打不过 Thinking
- **依赖 verifier**——BoN 需要一个判断"哪个答案更好"的机制，多数投票只对有明确答案的任务有效

这个研究的意义不是否定 Thinking，而是揭示了**test-time compute 有多种花法，没有一种是最优的**——任务特征决定最优策略。

## 3. Long2Short 如何压缩推理链

推理模型训练后期会出现一个普遍问题：**CoT 越来越长**。R1-Zero、o1、Qwen3 都报告了类似现象——训练步数越多，模型的回答越长，最终长度可以膨胀到 50K+ token。这有几个原因：

- **奖励信号鼓励"答对"**，更长 CoT 意味着更多检查机会，答对概率更高
- **反思和验证行为被强化**，模型学会"再检查一遍"
- **没有显式的长度约束**，模型没有动机压缩 CoT

但部署时，50K token 的 CoT 是不可接受的——延迟太高、成本太高。怎么把训练后的长 CoT 压缩到部署可接受的长度，同时保持推理质量？

[Kimi k1.5](https://arxiv.org/abs/2501.12599)（2025.01）提出了 **long2short RL**——一个分两阶段的训练方案：

### 3.1 先训练长推理链

先用标准 RL（GRPO + 数学奖励）训练一个长 CoT 推理模型。这个阶段不限制长度，让模型充分学习推理能力。训练后，模型的典型 CoT 长度可能在 5K-20K token。

### 3.2 再蒸馏并加入长度奖励

第二阶段是把长 CoT 的能力迁移到短 CoT 上：

1. **数据生成**：用阶段一的模型生成长 CoT 答案（高质量）
2. **蒸馏压缩**：用一个小模型（或另一个 LLM）把长 CoT 压缩成短 CoT，保留关键推理步骤
3. **SFT**：用压缩后的短 CoT 数据微调原模型
4. **length-penalty RL**：再做一轮 RL，这次在奖励中加入长度惩罚

length penalty 的具体形式：

$$r_{\text{total}} = r_{\text{correct}} - \lambda \cdot \max(0, |o| - L_{\text{target}})$$

其中 $|o|$ 是回答长度，$L_{\text{target}}$ 是目标长度，$\lambda$ 是惩罚强度。这个形式与 [DAPO 的 Overlong Reward Shaping](../chapter18_grpo/deepseek-dapo) 类似，但 Kimi 把它当作"主动压缩"的训练信号，不只是"防止过长"的工程约束。

### 3.3 比较压缩前后的效果

Kimi k1.5 论文报告：

- **长 CoT 模型**（baseline）：AIME 60.1 分，平均 CoT 长度 12K token
- **long2short 模型**：AIME 58.7 分（损失 1.4 分），平均 CoT 长度 3.2K token（压缩 73%）

也就是说，**用 25% 的 token 数，保留了 97.7% 的能力**。这个 trade-off 在工业部署上是非常划算的。

### 3.4 训练压缩与推理预算怎样配合

long2short 和 thinking budget 是互补的：

- **long2short** 是**训练阶段**的压缩——改变模型本身，让它倾向于生成短 CoT
- **thinking budget** 是**推理阶段**的控制——不改变模型，而是在生成时强制截断

工业部署通常两者结合：

```text
训练阶段：long2short RL 让模型学会短 CoT
推理阶段：thinking budget 设一个安全上限
```

## 4. 部署时如何控制模式与预算

Hybrid Thinking 不是"加一个开关"这么简单。它带来几个新的算法问题：

### 4.1 判断是否需要思考

模型怎么决定"这道题该 think 还是不 think"？

- **用户显式指定**：最简单，用户在 prompt 里加 `<think>` 或不加
- **模型自动判断**：训练时让模型自己学，根据 prompt 内容决定

Qwen3 用的是混合策略——默认让模型自动判断，但用户可以显式覆盖。模型自动判断的训练数据需要包含两类样本：thinking 适合的（数学 / 代码）和不适合的（闲聊 / 翻译）。

### 4.2 保持两种模式的答案一致

thinking 模式和 nothinking 模式应该给出**一致的答案**——不能 think 说"A"，nothinking 说"B"。但训练时这两种模式是分开优化的，容易出现不一致。

DeepSeek 的做法是在 RL 阶段加入**一致性约束**——同一 prompt 的两种模式答案应该一致，不一致就降低奖励。

### 4.3 在预算用完时结束推理

当 thinking budget 用完时，模型需要"中断当前思考，直接给答案"。但 RL 训练时模型没见过这种情况，强行截断会生成质量很差的答案。

Qwen3 的做法是在训练时**模拟 budget 用完的场景**——在 rollout 阶段随机截断 CoT，让模型学会"被截断时也能给个合理的答案"。

## 小结

Hybrid Thinking 用模式切换避免简单任务产生不必要的长推理，思考预算则为复杂任务设置计算上限。long2short 从训练阶段缩短推理链，二者可以配合使用。

DeepSeek V3.1、Qwen3、Kimi K2 都在这个方向上做了重要探索：

- **V3.1**：模式融合 + 可选思考内容
- **Qwen3**：Thinking Mode Fusion + thinking_budget 参数
- **Kimi k1.5**：long2short RL 主动压缩 CoT
- **NoThinking 研究**：揭示了"不思考 + 投票"在某些任务上反超思考

固定的模式开关仍然需要用户或路由器预先判断任务难度。[16.5 模型如何自适应思考](./adaptive-thinking) 进一步让模型根据输入动态决定推理深度。
