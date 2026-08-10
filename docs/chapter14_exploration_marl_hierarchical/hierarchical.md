# 12.3 分层 RL 与生成式世界模型引子

> [12.2](./marl) 处理了多 agent 场景。本节处理第三种被刻意回避的情形——**任务 horizon 极长**（如机器人完成整个房屋清洁，需要 1000+ 步动作）。单层策略在这种长程任务上几乎学不到信号。**分层 RL** 把长程决策分解为 option 序列：高层策略选子目标，低层策略执行子目标。

## 分层 RL 与 Options、FeUdal Networks 与 HIRO

长 horizon 任务是 RL 的另一类硬骨头。考虑 _Atari Montezuma's Revenge_：智能体需要先拿钥匙、再开门、最后进下一关。直接用 PPO 训练，几千步后梯度信号已经被淹没。**分层 RL** 把决策分解成两层（或多层）：

- **高层策略**：偶尔决策，输出"子目标"或"option"
- **底层策略**：在高层给出的子目标下，执行原子动作直到子目标完成

这样高层只关心稀疏的子目标序列，把长 horizon 切成短 horizon，梯度信号在每个子层内可传播。

### Options 框架

Sutton, Precup & Singh 1999 的 **options** 是分层 RL 的形式化基础。一个 option $\omega = (\mathcal{I}_\omega, \pi_\omega, \beta_\omega)$ 由三部分组成：

- ** initiation set** $\mathcal{I}_\omega$：可启动该 option 的状态集合
- ** intra-option policy** $\pi_\omega$：option 执行期间遵循的策略
- ** termination function** $\beta_\omega(s)$：到达 $s$ 后终止该 option 的概率

半马尔可夫决策过程（SMDP）的 Bellman 方程扩展为对 option 求期望：

$$Q^\mu(s, \omega) = \mathbb{E}\left[\sum_{t=0}^{T-1}\gamma^t r_t + \gamma^T \max_{\omega'} Q^\mu(s_T, \omega')\right]$$

其中 $T$ 是 option 终止时步。Options 框架的优美之处在于：高层可当成普通 MDP 学（用 SMDP-Q 学习），底层可以独立训（任何 model-free 算法）。

### FeUdal Networks 与 manager 输出方向，worker 执行

FeUdal Networks（Vezhnevets et al. 2017）把 options 做成端到端可学习。两个网络：

- **Manager** $M_\theta$：每 $c$ 步输出一个隐藏空间方向向量 $g_t \in \mathbb{R}^k$（不直接是子目标）
- **Worker** $W_\phi$：在 $c$ 步窗口内，每个原子步输出动作 $\pi_\phi(a \mid s; g_t)$，目标分布方向由 $g_t$ 调制

Manager 的训练目标很巧妙：让 $g_t$ 预测**未来 $c$ 步的隐藏状态变化方向**：

$$\mathcal{L}_M = -\langle g_t,\ \hat{z}_{t+c} - \hat{z}_t\rangle$$

其中 $\hat{z}$ 是共享编码器的输出。这是一个**自监督**目标——Manager 不需要任何外部奖励就能学会"指向有信息增量的方向"。Worker 仍然用环境奖励训练，但条件在 $g_t$ 上。

FeUdal 在 _Montezuma's Revenge_ 上首次让端到端深度 RL 拿到正分数，但训练不稳定、对超参敏感，工程复现困难。

### 异策略分层 RL

Data-Efficient Hierarchical Reinforcement Learning (HIRO, Nachum et al. 2018) 是 FeUdal 的现代化改进，关键创新是**off-policy 训练 + 目标转移**：

- 高层输出连续子目标 $g_t \in \mathbb{R}^d$（直接是状态空间内的位移），每 $c$ 步切换
- 底层奖励是内在的：$r^l_t = -\|s_{t+1} - (s_t + g_t)\|$，鼓励底层达到高层指定的位移
- 高层用 off-policy 算法（如 TD3）训练

最大的技术难点是**off-policy 偏差**：高层从 replay buffer 取出的旧子目标 $g$，对应底层当时执行的策略，但现在底层策略已经变了。HIRO 用**目标转移**（goal transition）解决：把旧子目标 $g$ 重新映射成"如果用当前底层策略执行，能达到的新子目标 $g'$"，使高层训练数据保持一致。

```python
# HIRO 主循环骨架
for step in range(total_steps):
    if step % c == 0:
        # 高层每 c 步采样子目标
        goal = high_level_policy(state)
    # 底层条件策略
    a = low_level_policy(state, goal)
    s_next, r_ext, done = env.step(a)
    # 底层内在奖励
    r_int = -np.linalg.norm(s_next - (state + goal))
    low_buffer.add(state, a, r_int, s_next)
    if step % c == 0:
        # 高层奖励是 c 步累积外部奖励
        high_buffer.add(state, goal, ext_reward_sum, s_next_c)
    update(low_level_policy, low_buffer)
    update(high_level_policy, high_buffer, goal_transition=transition_fn)
```

### 分层 RL 算法对比

| 算法    | 高层输出     | 底层目标    | 训练方式          | 主要问题      |
| ------- | ------------ | ----------- | ----------------- | ------------- |
| Options | option id    | 固定子策略  | SMDP-Q            | 需预设 option |
| FeUdal  | 隐藏方向 $g$ | worker 内在 | on-policy，端到端 | 训练不稳定    |
| HIRO    | 状态位移     | 状态匹配    | off-policy        | 目标转移设计  |

::: warning 分层 RL 的实际困境
分层 RL 听起来优雅，但工业落地少。原因：(1) 层次结构本身是强归纳偏置，错配会反向伤害性能；(2) 高层与底层耦合训练易陷入"互相欺骗"局部解——Manager 给无意义方向，Worker 学着忽略它；(3) LLM 时代的"分层"已经从神经网络架构转移到 prompt 层（plan-then-act、ReAct），更易调试。但思想仍深刻影响 agentic RL（[第 19 章](../chapter22_agentic/tool-use-and-trajectory)）和 [第 26 章 多智能体](../chapter32_selfplay/llm-multi-agent-rl/)。
:::

## 生成式世界模型作为 RL 环境

前三节的方法都在"如何让智能体更高效地探索既有环境"上下功夫。最后一节换视角：**当环境本身也是学习的产物**，探索范式会如何变化？

### 从 Dreamer 到 Genie

[第 9 章 Dreamer V3](../chapter11_continuous_control/intro#_12-7-dreamer-v3-世界模型的新世代) 已经展示了"在世界模型里训练 actor-critic"的可行性：先用真实数据训一个 RSSM 世界模型，再在想象轨迹里优化策略。Dreamer 的世界模型仍是任务相关的（在某个 Atari 游戏或 MuJoCo 环境上训）。

Genie（Bruce et al. 2024）把世界模型推到**生成式、跨任务**的新阶段。给定一段视频或一张图片，Genie 能学出一个可交互的"游戏引擎"——你可以输入一个动作，模型生成下一帧。这意味着：

- **环境数据来自互联网视频**，不再依赖游戏引擎或物理仿真
- **同一模型可生成多个环境**，跨任务泛化
- **RL 训练可在生成环境中进行**，无需真实物理引擎

Genie 3 进一步引入**潜在动作**（latent action）学习：模型自动发现视频中"导致下一帧变化"的潜在控制变量，无需任何动作标签。形式上：

$$z_t = \text{LatentAction}(x_t, x_{t+1}),\quad x_{t+1} = \text{Decoder}(x_t, z_t)$$

学到的 $z_t$ 可作为 RL 的动作空间，使得在 Genie 生成的环境中训练的 agent 能迁移到真实控制任务。这是 model-based RL（[第 9 章](../chapter11_continuous_control/intro#_12-5-model-based-rl-学习环境模型)）+ 视频生成模型 + 探索-利用理论的交汇点。

### 探索、多智能体、分层在新范式下的角色

把世界模型当作可生成环境后，本章三主题重新组合：

1. **探索**：内在奖励可以作用在生成环境的隐藏空间上，而不是像素空间——ICM 的"前向预测误差"本质就是世界模型的训练 loss
2. **多智能体**：Genie 类模型可生成包含 NPC 的环境，多智能体可在生成环境中做 self-play（[第 26 章 self-play](../chapter32_selfplay/self-play-outlook/)）
3. **分层**：高层策略可以直接输出"潜在子目标"，由世界模型解码出环境状态变化，相当于 option 的隐式学习

工业影响：DeepMind 的 SIMA（Scalable Instructable Multi-World Agent）已经在 Genie 生成的多游戏环境中训练通用 agent；Tongyi DeepResearch 等 LLM agent 也开始用 LLM 自生成的"code world model"作为训练环境（[第 26 章 LLM 驱动的科学发现](../chapter32_selfplay/alphaevolve/)）。世界模型从"训练辅助工具"升级为"训练环境本身"，是 2024-2026 年 RL 最深刻的变化之一。

## 本章总结

本章覆盖了经典深度 RL 假设被破坏后的三条救火路线：

1. **奖励稀疏 → 内在奖励**：ICM 用前向预测误差，RND 用随机网络蒸馏；NGU 把短期 episodic 与长期 RND 结合，Agent57 自适应切换探索-利用，是 Atari 57 上首个全游戏超人类的算法
2. **多智能体非平稳 → CTDE**：MADDPG 给每个智能体配集中 critic，MAPPO 共享 critic + on-policy clip，是 SMAC、Hanabi 的事实标准
3. **长 horizon → 分层**：Options 框架、FeUdal Networks 的端到端 manager-worker、HIRO 的 off-policy 目标转移，让高层只关心子目标序列

这三条路线在 LLM 时代再次汇合：agentic RL 的工具调用本质是 option、多 agent 协作本质是 CTDE、LLM 内嵌的世界知识本质是 Genie 式生成环境。下一章 [第 13 章 RLHF 训练流水线](../chapter15_rlhf/base-model-to-assistant) 进入大模型对齐主线，那里的"环境"是 LLM 自身，但本章的探索、多智能体、分层思想将贯穿后续所有 LLM RL 章节。

## 延伸阅读

- [Pathak et al. 2017 "Curiosity-driven Exploration by Self-Supervised Prediction" (ICM)](https://arxiv.org/abs/1705.05363)
- [Burda et al. 2018 "Exploration by Random Network Distillation" (RND)](https://arxiv.org/abs/1810.12894)
- [Badia et al. 2020 "Agent57: Outperforming the Atari Human Benchmark"](https://arxiv.org/abs/2003.13350)
- [Lowe et al. 2017 "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" (MADDPG)](https://arxiv.org/abs/1706.02275)
- [Yu et al. 2022 "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (MAPPO)](https://arxiv.org/abs/2103.01955)
- [Vezhnevets et al. 2017 "FeUdal Networks for Hierarchical Reinforcement Learning"](https://arxiv.org/abs/1703.01161)
- [Nachum et al. 2018 "Data-Efficient Hierarchical Reinforcement Learning" (HIRO)](https://arxiv.org/abs/1805.08296)
- [Bruce et al. 2024 "Genie: Generative Interactive Environments"](https://arxiv.org/abs/2402.15391)
