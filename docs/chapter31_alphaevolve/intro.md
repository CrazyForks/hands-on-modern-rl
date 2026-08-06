# 第 29 章 · 进化式 LLM 搜索与生成式世界模型

[前面几节](./intro) 我们讨论了 RL 的传统前沿——embodied intelligence、self-play、multi-agent。这一节我们看一个全新的方向：**LLM 驱动的科学发现**。

这个方向的特点是：

- **LLM 不只是 actor**——它是 idea generator、code writer、experiment designer
- **RL 不只是 policy optimization**——它是搜索、是进化、是自我改进
- **目标不是"打游戏"或"对话"**——是发现新算法、新数学、新科学

2024-2026 年的几个代表工作：

- **AlphaEvolve**（DeepMind, 2024.05）：LLM + 进化算法发现新数学
- **Genie 3**（DeepMind, 2025.08）：生成式 world model
- **Titans**（Google, 2024.12）：长期记忆架构
- **Multi-Agent Deep Research**（字节 Seed, 2025.11）：用 M-GRPO 训练多智能体搜索系统

这些工作代表了 RL 与 LLM 结合的"下一代范式"——从"训练一个 policy"到"训练一个研究系统"。

## 12.6.1 AlphaEvolve 与 LLM + 进化的数学发现

[AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)（DeepMind, 2024.05 公开，2025 完整版）是 LLM 驱动科学发现的旗舰案例。

### AlphaEvolve 的核心思想

把数学发现建模为**进化搜索 + LLM 代码生成**：

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. 种群初始化：已有的算法/证明                              │
│    （例如：矩阵乘法的 Strassen 算法）                       │
├─────────────────────────────────────────────────────────────┤
│ 2. LLM 变异：让 Gemini 生成分支                              │
│    "请改进这个算法" / "尝试不同的方法"                       │
│    输出：新代码                                              │
├─────────────────────────────────────────────────────────────┤
│ 3. 自动评估：运行代码，测量效果                              │
│    （例如：矩阵乘法的乘法次数）                              │
├─────────────────────────────────────────────────────────────┤
│ 4. 选择：保留效果好的，淘汰效果差的                          │
├─────────────────────────────────────────────────────────────┤
│ 5. 迭代：返回步骤 2                                          │
└─────────────────────────────────────────────────────────────┘
```

这个流程与 [经典遗传算法（GA）](../chapter03_mdp/dp-mc-td) 几乎完全一样——区别只是变异操作从"随机改动"变成"LLM 智能生成"。

### AlphaEvolve 的关键创新

**创新一：LLM 作为智能变异算子**

传统 GA 的变异是随机改动——成功率低。LLM 变异是"理解当前代码 + 提出有意义的改进"——成功率高。

**创新二：代码作为基因**

不是用 bit string 作为基因，而是用**可执行的代码**。这让 fitness 可以**自动测量**——运行代码就知道效果。

**创新三：Gemini 作为 LLM 后端**

AlphaEvolve 用 Gemini Pro/Ultra 作为 LLM 后端——强 LLM 让变异质量大幅提升。

### AlphaEvolve 的发现

AlphaEvolve 在多个领域做出了**真正的新发现**：

**发现 1：矩阵乘法新算法**

1969 年 Strassen 发现 $4 \times 4$ 矩阵乘法可以用 49 次乘法（之前认为是 64 次）。AlphaEvolve 发现了一个新算法，用 **48 次**——**超越了人类 50 多年的研究**。

**发现 2：组合数学新界**

在 [tensor decomposition](https://en.wikipedia.org/wiki/Tensor_decomposition)、[sorting networks](https://en.wikipedia.org/wiki/Sorting_network) 等组合问题上，AlphaEvolve 发现了多个新界限，超越了已知最优。

**发现 3：谷歌基础设施优化**

DeepMind 内部用 AlphaEvolve 优化了：

- 数据中心调度算法（节省 0.7% 全球计算资源）
- TPU 矩阵乘法硬件设计
- 机器学习 kernel 优化

### AlphaEvolve 的意义

AlphaEvolve 证明了：

1. **LLM 可以做真正的科学研究**——不只是"回答问题"，是"发现新知识"
2. **进化 + LLM 是强大的组合**——LLM 提供智能，进化提供探索
3. **自动评估是关键**——只有可自动评估的领域才适合这种范式

## 12.6.2 Genie 3 与 生成式 World Model

[Genie 3](https://deepmind.google/models/genie/)（DeepMind, 2025.08）是生成式 world model 的代表作。

### 什么是 World Model？

World model 是一个能**预测环境动态**的模型：

```text
输入：当前状态 s_t + 动作 a_t
输出：下一状态 s_{t+1}
```

在 RL 中，world model 可以**替代真实环境**——policy 在 world model 上训练，避免昂贵的真实环境交互。

### Genie 系列的演化

**Genie 1**（2024.02）：从视频学习世界模型

- 输入：互联网视频
- 输出：可以生成可控的"游戏"环境
- 关键：没有显式动作标签，模型自己学到"什么是动作"

**Genie 2**（2024.12）：3D world model

- 输入：一张图片
- 输出：可以生成可交互的 3D 环境
- 关键：环境可以维持几分钟的一致性

**Genie 3**（2025.08）：大规模、可控、长 horizon

- 输入：自然语言描述
- 输出：完全可控的、长 horizon 的 3D 环境
- 关键：可用于训练 embodied agent

### Genie 3 的训练

```text
┌─────────────────────────────────────────────────────────┐
│ Phase 1: 视频预训练                                     │
│   - 大量无标签视频                                      │
│   - 学习"世界如何运作"                                 │
├─────────────────────────────────────────────────────────┤
│ Phase 2: 动作标注                                       │
│   - 让 LLM 给视频中的动作打标签                         │
│   - 学会"什么动作导致什么变化"                          │
├─────────────────────────────────────────────────────────┤
│ Phase 3: World Model 训练                               │
│   - (s_t, a_t, s_{t+1}) 三元组                          │
│   - 训练一个能预测 s_{t+1} 的模型                       │
├─────────────────────────────────────────────────────────┤
│ Phase 4: RL 训练                                        │
│   - Policy 在 world model 上训练                        │
│   - 避免昂贵的真实环境交互                              │
└─────────────────────────────────────────────────────────┘
```

### Genie 3 的应用

**应用 1：Embodied agent 训练**

机器人在 world model 中学习走路、抓取、操作——避免在真实机器人上试错（昂贵且危险）。

**应用 2：游戏生成**

用 Genie 3 自动生成可玩游戏——玩家描述想要的游戏，Genie 3 生成完整环境。

**应用 3：模拟训练**

自动驾驶、工业控制、医疗手术等高风险场景——在 world model 中训练，部署到真实环境。

### Genie 3 的局限

- **准确性**：world model 不是 100% 准确——长期预测会漂移
- **泛化**：训练分布外的环境难以模拟
- **计算成本**：高质量 world model 推理成本高

## 12.6.3 Titans 与 长期记忆架构

[Titans](https://arxiv.org/abs/2501.00663)（Google, 2024.12 公开，2025 修改版）是 LLM 架构的新方向——**长期记忆**。

### Titans 的动机

Transformer 有一个根本限制：**context window**。即使扩展到 1M token，也无法处理"无限长"的输入。Titans 试图解决这个问题。

### Titans 的设计

Titans 引入**神经长期记忆**：

```text
┌──────────────────────────────────────────────────────────┐
│ 短期记忆：attention（标准 Transformer）                  │
│   - 处理最近的 token                                     │
│   - 容量有限（context window）                          │
├──────────────────────────────────────────────────────────┤
│ 长期记忆：神经记忆模块（新）                             │
│   - 持续学习 + 存储                                      │
│   - 容量无限                                             │
├──────────────────────────────────────────────────────────┤
│ 持久记忆：任务相关知识（系统 prompt、知识库）            │
│   - 不变                                                 │
└──────────────────────────────────────────────────────────┘
```

三层记忆让 Titans 能处理**无限长输入**——长期记忆持续存储历史信息。

### Titans 与 RL 的关系

Titans 的核心是**学习如何记忆**——这本身就是一个 RL 问题：

- 状态：当前输入 + 当前记忆
- 动作：怎么更新记忆（write / forget / update）
- 奖励：未来任务表现（如果记忆了有用的信息，任务表现好）

Titans 用**惊喜度（surprise）**作为内部 reward——当输入"令人惊讶"时，加强记忆；当输入"重复"时，减弱记忆。这是一种**自监督 RL**——模型自己生成 reward。

### Titans 的实验结果

在长 horizon 任务上，Titans 显著优于 Transformer：

| 任务                             | Transformer | Titans              |
| -------------------------------- | ----------- | ------------------- |
| Language modeling（10M context） | OOM         | 67% perplexity 改进 |
| 长文档 QA                        | 55%         | 78%                 |
| 时序预测                         | 65%         | 82%                 |

Titans 证明了**长期记忆是 scaling 的下一个方向**——不只是"更宽、更深"，还要"更记得"。

## 12.6.4 M-GRPO 与多智能体搜索训练

[Multi-Agent Deep Research: Training Multi-Agent Systems with M-GRPO](https://arxiv.org/abs/2511.13288)（字节 Seed, 2025.11）用 M-GRPO——Group Relative Policy Optimization 的多智能体扩展——训练多智能体搜索系统。

### M-GRPO 的系统设计

多智能体系统由主 agent 和多个子 agent 组成：

```text
┌─────────────────────────────────────────────────────────┐
│ 主 Agent（planner）：总规划                              │
│   - 接收任务                                            │
│   - 拆解为子任务                                        │
│   - 调度子 agents                                       │
├─────────────────────────────────────────────────────────┤
│ 子 Agents（tool executors）：工具执行                    │
│   - 多轮调用搜索、代码等工具                            │
│   - 各自频率不同、调用次数可变                          │
├─────────────────────────────────────────────────────────┤
│ 层级 Credit Assignment                                  │
│   - 主 agent 和子 agent 各自计算 group-relative         │
│   - 通过共享 store 交换最小统计量                       │
└─────────────────────────────────────────────────────────┘
```

### M-GRPO 的训练方法

M-GRPO 解决多智能体 RL 训练的三个难点：

- **层级 credit assignment**：主 agent 和子 agent 分别计算 group-relative advantage，避免"贡献混淆"
- **轨迹对齐**：子 agent 调用次数可变，用轨迹对齐方案生成固定 size 的 batch
- **解耦训练**：agent 分布在独立服务器上，通过共享 store 交换统计量，不需要跨服务器反向传播

在 GAIA、XBench-DeepSearch、WebWalkerQA 等基准上，M-GRPO 一致优于单 agent GRPO 和"冻结子 agent"的多智能体 GRPO。

### M-GRPO 与 AlphaEvolve 的关系

两者都用 LLM + RL/搜索，但角度不同：

- **AlphaEvolve**：进化搜索（无梯度，population-based），面向算法发现
- **M-GRPO**：多智能体 RL（基于 GRPO），面向工具增强的深度研究

它们是 LLM-driven discovery 的两个互补范式。

## 12.6.5 递归自我改进

**递归自我改进（Recursive Self-Improvement, RSI）** 是 LLM-driven discovery 的终极形态——**模型自己改进自己**。

### RSI 的核心循环

```text
┌─────────────────────────────────────────────────────┐
│ 1. 当前模型 M_t 评估自己的能力                      │
│    - 在哪些任务上表现好？在哪些任务上差？           │
├─────────────────────────────────────────────────────┤
│ 2. 生成改进方案                                    │
│    - 设计新的训练数据                              │
│    - 调整训练超参                                  │
│    - 改进算法                                     │
├─────────────────────────────────────────────────────┤
│ 3. 执行改进                                        │
│    - 用方案训练新模型 M_{t+1}                     │
├─────────────────────────────────────────────────────┤
│ 4. 评估新模型                                      │
│    - M_{t+1} 比 M_t 好吗？                         │
│    - 如果好，保留 M_{t+1}；如果差，回退            │
├─────────────────────────────────────────────────────┤
│ 5. 返回步骤 1                                      │
└─────────────────────────────────────────────────────┘
```

### RSI 的当前状态

到 2026 年中，RSI 仍然是**研究概念**，没有工业级实现。原因：

**挑战 1：自我评估的不准确性**

模型很难准确评估自己的能力——容易高估（Dunning-Kruger 效应）。

**挑战 2：改进方案的搜索空间爆炸**

可能的训练数据、超参、算法组合是天文数字。

**挑战 3：安全风险**

如果模型可以无限改进自己，可能突破人类控制——这是 [AI safety](../chapter30_alignment_failures/intro) 的核心担忧。

### RSI 的部分实现

虽然没有完整 RSI，但有几个**部分实现**：

- **AutoGPT**（2023）：早期尝试，效果有限
- **SRPO**（[arXiv:2406.01660](https://arxiv.org/abs/2406.01660)）：用自改进过程训练偏好模型（Cohere, 2024）
- **Voyager**（[arXiv:2305.16291](https://arxiv.org/abs/2305.16291)）：Minecraft agent 自主学习新技能
- **DeepMind 的自我博弈系统**：模型与自己对弈提升

这些是 RSI 的早期形态——证明了部分可行性，但离真正的"递归自我改进"还很远。

## 12.6.6 LLM-driven Discovery 的共同模式

这些工作（AlphaEvolve、Genie 3、Titans、MIRAS、RSI）的共同模式：

### LLM 作为搜索的智能指导

传统搜索（MCTS、Beam Search）需要人为设计 heuristic。LLM 可以**自动生成 heuristic**——让搜索更智能。

### 自动评估是关键

AlphaEvolve 之所以能发现新算法，是因为**算法的效果可以自动测量**（运行代码就知道）。这是 LLM-driven discovery 的前提——**只有可自动评估的领域才适合**。

### 组合 > 单一

- AlphaEvolve = LLM + 进化
- Genie 3 = LLM + world model
- Titans = LLM + 长期记忆
- Multi-Agent Deep Research = LLM + multi-agent + RL

**组合多种方法**比单一方法更强——这是 RL 在 LLM 时代的新形态。

### 从"训 policy"到"训系统"

传统 RL 训一个 policy。LLM-driven discovery 训一个**完整的研究系统**——多个 agent + memory + search + tools。

## 12.6.7 未来方向

### 科学发现

把 AlphaEvolve 思路扩展到：

- **生物学**：蛋白质设计、药物发现
- **化学**：新分子合成路径
- **物理**：新实验设计、新理论验证

### 教育

用 LLM-driven discovery 个性化教育——发现每个学生最适合的学习路径。

### AGI

递归自我改进是 AGI 的潜在路径之一——如果模型能持续改进自己，可能在某个时间点超越人类能力。

但这伴随**严重的安全风险**——这也是 [Alignment 研究](../chapter30_alignment_failures/scaling-and-defenses) 的核心议题。

## 小结

LLM-driven discovery 是 RL 在 2025-2026 年的新前沿：

- **AlphaEvolve**：LLM + 进化，发现新数学
- **Genie 3**：生成式 world model，用于 embodied agent
- **Titans**：长期记忆架构，扩展 context
- **M-GRPO**：多智能体 RL 训练
- **RSI**：递归自我改进（部分实现）

这些工作共同指向一个新范式——**从训练 policy 到训练研究系统**。这是 RL 与 LLM 深度结合的下一代方向，也是 AGI 研究的最前沿。
