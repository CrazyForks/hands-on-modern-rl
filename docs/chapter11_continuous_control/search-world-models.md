# 9.4 AlphaZero、MuZero 与 Dreamer V3

> [11.3](./model-based) 讲了 model-based RL 的"数据增强"路线——Dyna/PETS/MBPO 用模型生成数据加速 model-free 训练。本节讲 model-based 的另一条旗舰路线：**显式搜索 + 神经网络估值**。从 AlphaGo（2016）到 AlphaZero（2017）到 MuZero（2019）再到 Dreamer V3（2023），这条线代表了 model-based RL 的理论天花板，也直接启发了 LLM 时代的 Process Reward Model 搜索。

## AlphaZero 与 搜索 + 学习的极致

AlphaGo（2016）→ AlphaGo Zero（2017）→ AlphaZero（2017）→ MuZero（2019）这条线代表了 model-based RL 的另一种哲学：**显式搜索 + 神经网络估值**。

### AlphaZero 的核心循环

```python
def alphazero_search(state, neural_net, n_simulations=800):
    root = MCTSNode(state)
    for _ in range(n_simulations):
        # 1. Selection: 按 PUCT 选最优子节点
        node = root
        while not node.is_leaf():
            node = node.select_child()

        # 2. Expansion: 神经网络评估叶子
        policy, value = neural_net(node.state)
        node.expand(policy)

        # 3. Backup: 把 value 反向传播到根
        node.backup(value)

    # 返回根节点的访问次数分布作为动作概率
    return root.compute_action_distribution()
```

AlphaZero 把 MCTS（蒙特卡洛树搜索）和神经网络结合：

- **策略网络** $p_\theta(a \mid s)$：缩小搜索宽度（只搜有希望的动作）
- **价值网络** $v_\theta(s)$：缩短搜索深度（叶子直接估值，不必搜到底）

### PUCT 公式

AlphaZero 用 PUCT（Predictor + UCB）选择子节点：

$$\text{PUCT}(a) = Q(s, a) + c_{\text{prior}} \cdot p_\theta(a \mid s) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$

- $Q(s, a)$：动作 $a$ 的当前价值估计
- $p_\theta(a \mid s)$：策略网络先验
- $\sqrt{N(s)} / (1 + N(s, a))$：探索奖金（UCB 思想）

第一项利用当前知识，第二项用先验缩小搜索范围，第三项确保每个动作都被尝试过。

### 自我对弈训练

这两个网络**自我对弈**训练：

1. 用当前网络 + MCTS 与自己对弈 1 局
2. 搜索结果是"更好的策略标签"——MCTS 给出的动作分布是改进版策略
3. 胜负是"更好的价值标签"——赢了 +1，输了 -1
4. 用这些标签监督训练网络

```python
def self_play_training(network, n_games=10000):
    for game in range(n_games):
        # 1. 自我对弈
        trajectory = []
        state = initial_state()
        while not state.is_terminal():
            policy = alphazero_search(state, network)
            action = sample_from(policy)
            trajectory.append((state, policy, action))
            state = state.next(action)

        # 2. 标注胜负
        winner = state.winner()
        for s, p, a in trajectory:
            value = +1 if winner == s.current_player else -1
            train_network(s, p, value)
```

**无需人类棋谱**——AlphaZero 从零开始 4 小时打败 Stockfish，72 小时超越所有人类围棋程序。

## MuZero 与 隐式模型学习

AlphaZero 需要知道游戏规则（状态转移、合法动作）。MuZero（Schrittwieser et al. 2019）的关键创新：**学一个隐式模型**，把状态 $s$ 映射到隐藏表示 $h(s)$，在隐藏空间做规划和价值估计。

$$s \xrightarrow{h} x_0 \xrightarrow{g} x_1 \xrightarrow{g} x_2 \to \ldots$$

### MuZero 的三大网络

- **表示网络** $h(s) \to x$：把真实状态编码到隐藏空间
- **动力学网络** $g(x, a) \to x', r$：在隐藏空间预测下一状态和奖励
- **预测网络** $f(x) \to p, v$：从隐藏状态预测策略和价值

```python
class MuZero:
    def plan(self, state, n_simulations):
        # 1. 编码真实状态到隐藏空间
        root_hidden = self.representation(state)
        root_policy, root_value = self.prediction(root_hidden)

        # 2. MCTS 在隐藏空间搜索
        for _ in range(n_simulations):
            self._mcts_iteration(root_hidden)

        # 3. 返回根的动作分布
        return root.action_distribution()

    def _mcts_iteration(self, root):
        # 在隐藏空间选择、扩张、回溯
        path = self._select_path(root)
        next_hidden, reward = self.dynamics(path[-1].hidden, path[-1].action)
        policy, value = self.prediction(next_hidden)
        path[-1].expand(policy, reward)
        for node in path:
            node.update(value, reward)
```

### MuZero 的意义

MuZero 不知道游戏规则也能学——它**自己学规则**。这让它能推广到：

- **Atari**（不需要模拟器，直接从像素学）
- **棋盘游戏**（围棋、象棋、将棋）
- **扑克**（部分可观察）
- **任何 MDP**

MuZero 是 model-based RL 的"统一架构"——同一个算法、同一套网络结构，跨越视觉输入、矢量输入、离散动作、连续动作。

## Dreamer V3 与 世界模型的新世代

Dreamer 系列（Hafner et al. 2020-2023）是 model-based RL 的现代旗舰。核心思想：**学一个循环隐变量世界模型**，在这个模型里"做梦"训练 actor-critic。

### 循环状态空间模型

Dreamer 用 **Recurrent State-Space Model**（RSSM）同时建模：

- **确定性轨迹**（RNN hidden state $h_t$）
- **随机后验**（编码器从观察推断 $z_t$）
- **随机先验**（从 $h_t$ 预测 $\hat{z}_t$）

训练时让 $\hat{z}_t$ 匹配 $z_t$，模型就能"想象"出符合真实环境的轨迹。

```python
class RSSM:
    def forward(self, obs_seq, action_seq):
        h = zeros(batch, hidden_dim)
        posterior_zs = []
        prior_zs = []

        for t in range(T):
            # 先验：从 h_t 预测 z_t
            prior_mean, prior_std = self.prior(h)
            prior_zs.append((prior_mean, prior_std))

            # 后验：从 h_t 和 obs_t 推断 z_t
            posterior_mean, posterior_std = self.posterior(h, encoder(obs_seq[t]))
            z = reparameterize(posterior_mean, posterior_std)
            posterior_zs.append((posterior_mean, posterior_std))

            # 更新 RNN hidden state
            h = self.rnn(h, z, action_seq[t])

        return prior_zs, posterior_zs
```

### Actor-Critic in Imagination

训练 actor 不用真实数据，而是在世界模型里 rollout：

```python
# 在世界模型里"做梦"
h = world_model.encode(real_observation_sequence)
for t in range(H):  # H = 15 想象 horizon
    a = actor(h)
    h, r = world_model.predict(h, a)
    imagined_trajectory.append((h, a, r))

# 在想象轨迹上训练 actor-critic
for (h, a, r) in imagined_trajectory:
    critic_loss = ...
    actor_loss = ...
```

### Dreamer V3 的统一性

Dreamer V3（Hafner et al. 2023）的关键贡献：**单一超参数设置**跨越 150+ 任务，包括：

- Atari（离散动作 + 视觉输入）
- MuJoCo（连续动作 + 矢量输入）
- Crafter（开放世界生存）
- DMLab（第一人称 3D 导航）
- BSuite（认知任务）

无需调参，Dreamer V3 在大多数基准上**超越 model-free SOTA**。这是 model-based RL 第一次在通用性上击败 SAC、PPO 等。

### 三个关键工程创新

1. **离散化 latent**：把 $z$ 从高斯分布改成 categorical 分布，训练更稳定
2. **symlog 损失**：用 $\text{symlog}(x) = \text{sign}(x) \log(|x| + 1)$ 压缩值函数范围，适应不同任务的奖励尺度
3. **不使用 KL annealing**：直接最大化 ELBO，让后验匹配先验

这三招让 Dreamer V3 能在 150+ 任务上"开箱即用"。

## Model-Based vs Model-Free 与 何时用哪个

| 维度         | Model-Free       | Model-Based                |
| ------------ | ---------------- | -------------------------- |
| **样本效率** | 低（百万步）     | 高（万步）                 |
| **渐进性能** | 高               | 受模型误差限制             |
| **计算成本** | 低（直接用数据） | 高（训练模型 + 搜索/规划） |
| **可解释性** | 黑盒             | 模型可分析                 |
| **迁移能力** | 弱               | 模型可迁移到下游任务       |
| **超参敏感** | 中               | 高（模型质量决定一切）     |

**何时选 Model-Free：**

- 仿真器便宜（Atari、MuJoCo、StarCraft）
- 关注最终性能（不受样本数限制）
- 部署时无模型推理开销

**何时选 Model-Based：**

- 真实环境采样贵（机器人、自动驾驶、化学反应）
- 需要快速适应（meta-RL、在线学习）
- 需要可解释性（安全关键场景）

## 与 LLM RL 的连接

LLM 训练中：

- **Model-Free**：RLHF/GRPO 直接用 RM 奖励训练（model-free）
- **Model-Based**：Process Reward Model、Verifier 模型就是某种"环境模型"，PRM 引导的搜索（[第 18 章 PRM 与搜索](../chapter20_prm_search/inference-time-search)）类比 AlphaZero
- **World Model**：Code World Model（[第 21 章 SWE-Agent](../chapter23_rl_based_swe/world-model-and-deep-swe)）预测代码执行结果，是 LLM 时代的 MuZero

理解了 model-based 与 model-free 的权衡，你就能理解为什么 Tongyi DeepResearch 用 PRM 引导搜索、为什么 SWE-Agent 用 Code World Model 提升样本效率。

## 本章总结

连续控制和 model-based RL 是经典深度 RL 的两大进阶方向：

1. **DDPG → TD3 → SAC** 是确定性策略梯度的演进：从外加噪声探索，到双 Q + 延迟更新稳定，再到最大熵自动探索
2. **Dyna → PETS → MBPO** 是 model-based 数据增强的演进：模型作为数据生成器
3. **AlphaZero → MuZero → Dreamer V3** 是显式搜索 + 学习模型的旗舰路线，代表了 model-based RL 的天花板

下一章 [第 10 章 离线强化学习](../chapter12_offline_rl/intro) 转向另一个角度——**当智能体不能交互，只能用历史数据时怎么办**？这是 LLM 后训练、推荐系统等真实场景的核心问题。

## 延伸阅读

- [Silver et al. 2018 "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play" (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404)
- [Schrittwieser et al. 2020 "Mastering Atari, Go, chess and shogi by planning with a learned model" (MuZero)](https://arxiv.org/abs/1911.08265)
- [Hafner et al. 2023 "Mastering Diverse Domains through World Models" (Dreamer V3)](https://arxiv.org/abs/2301.04104)
- [Janner et al. 2019 "When to Trust Your Model: Model-Based Policy Optimization" (MBPO)](https://arxiv.org/abs/1906.08253)
