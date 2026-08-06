# 11.2 逆向 RL 与 GAIL

> [13.1](./bc-dagger) 讲了行为克隆——直接模仿专家动作。但 BC 有个根本限制：它学不到"专家为什么这么做"。本节走另一条路——**逆向 RL**：从专家轨迹反推一个奖励函数，再用标准 RL 算法在这个奖励上训练。这绕过了 BC 的分布漂移，也学到了更可迁移的奖励信号。

## 最大熵反向 RL

反向 RL（Inverse RL）假设专家之所以优秀，是因为他在**优化某个隐藏的奖励函数**。与其直接模仿动作，不如**先反推出这个奖励函数**，再用普通 RL 求解。

### 反向 RL 的基本设定

给定专家轨迹 $\mathcal{D}_{\text{expert}} = \{\tau_1, \ldots, \tau_M\}$，每条 $\tau = (s_0, a_0, \ldots, s_T)$。目标是学一个奖励函数 $r_\psi(s, a)$ 满足：

$$\text{专家策略在 } r_\psi \text{ 下是最优的}$$

但这个条件**严重不唯一**——所有常数奖励 $r_\psi \equiv c$ 都满足。需要额外的**正则化**或**最大熵原理**来唯一确定 $r_\psi$。

### MaxEnt IRL 的目标

Ziebart et al. 2008 提出最大熵反向 RL。假设专家策略服从**最大熵**分布（既匹配特征期望，又尽可能随机）：

$$\pi(a \mid s) \propto \exp\left(Q^{\text{soft}}_{r_\psi}(s, a)\right)$$

那么专家轨迹的似然为：

$$p(\tau \mid r_\psi) = \frac{1}{Z(r_\psi)} \exp\left(\sum_t r_\psi(s_t, a_t)\right)$$

最大化专家数据的对数似然：

$$\max_\psi \; \mathcal{L}(\psi) = \sum_{\tau \in \mathcal{D}_{\text{expert}}} \left[\sum_t r_\psi(s_t, a_t)\right] - |\mathcal{D}_{\text{expert}}| \log Z(r_\psi)$$

第一项是专家轨迹的累积奖励，第二项 $\log Z$ 是配分函数（所有可能轨迹上的指数和的对数）。梯度为：

$$\nabla_\psi \mathcal{L} = \mathbb{E}_{\tau \sim \text{expert}}\left[\sum_t \nabla_\psi r_\psi(s_t, a_t)\right] - \mathbb{E}_{\tau \sim p(\cdot \mid r_\psi)}\left[\sum_t \nabla_\psi r_\psi(s_t, a_t)\right]$$

直白的解读：**让专家访问的 $(s, a)$ 的奖励升高，让当前策略（按 $r_\psi$ 滚动的策略）访问的 $(s, a)$ 的奖励降低**。当两者特征期望一致时，梯度为零。

### MaxEnt IRL 的难点

$\log Z(r_\psi)$ 在连续状态-动作空间下**不可解析**。三种主流近似：

1. **基于模型**：用学到的环境模型做 forward rollout 估计 $Z$
2. **基于采样的 soft Q iteration**：用软 Bellman 备份近似（Guided Cost Learning, Finn et al. 2016）
3. **对抗式（GAIL）**：用判别器隐式表达 $r_\psi$（下一节）

```python
def maxent_irl_step(reward_net, expert_states_actions, env_sampler, soft_q_planner):
    # 1. 当前奖励下做 soft Q planning，得到采样分布
    current_rewards = reward_net(states_actions_tensor)
    sampled_trajectories = soft_q_planner.rollout(reward_net)

    # 2. 计算特征期望差
    expert_feat = feature_expectation(expert_states_actions, reward_net)
    sampled_feat = feature_expectation(sampled_trajectories, reward_net)

    # 3. 梯度上升更新奖励
    grad = expert_feat - sampled_feat
    reward_net.update(grad)
```

MaxEnt IRL 的代价高昂：每次外层更新需要内层求解一个完整的 soft Q 问题。这使它难以扩展到高维问题（如视觉输入）。**GAIL** 用对抗训练避开显式 $Z$ 计算。

## 生成对抗模仿学习

Generative Adversarial Imitation Learning（Ho & Ermon 2016）借用 GAN 的思想：把反向 RL 写成**判别器 $D_\phi$ vs 生成器 $\pi_\theta$** 的博弈。

### GAIL 的目标

判别器区分"专家数据"和"策略数据"：

$$\max_\phi \; \mathbb{E}_{(s,a) \sim \mathcal{D}_{\text{expert}}}\left[\log D_\phi(s, a)\right] + \mathbb{E}_{(s,a) \sim \pi_\theta}\left[\log (1 - D_\phi(s, a))\right]$$

策略通过"骗过判别器"学习：

$$\min_\theta \; \mathbb{E}_{(s,a) \sim \pi_\theta}\left[\log D_\phi(s, a)\right] - \lambda \mathcal{H}(\pi_\theta)$$

第二项是熵正则化，避免策略过早坍缩。这里 $-\log D_\phi(s, a)$ 充当**隐式奖励**——等价于 MaxEnt IRL 中 $r_\psi(s, a) = \log D_\phi(s, a) - \log(1 - D_\phi(s, a))$ 的对抗推导。

```python
class GAIL:
    def __init__(self, expert_data, policy, discriminator):
        self.expert_buffer = expert_data   # 专家 (s, a) 对
        self.policy = policy               # 任意 RL 算法（PPO/TRPO/SAC）
        self.disc = discriminator          # 二分类网络

    def update(self, n_policy_steps=5, n_disc_steps=1):
        # === 1. 训练判别器 ===
        for _ in range(n_disc_steps):
            # 采样策略数据
            policy_states, policy_actions = self.policy.sample_rollout()
            # 二分类交叉熵
            expert_logits = self.disc(self.expert_buffer.sample())
            policy_logits = self.disc(policy_states, policy_actions)
            d_loss = (
                F.binary_cross_entropy_with_logits(expert_logits, ones) +
                F.binary_cross_entropy_with_logits(policy_logits, zeros)
            )
            self.disc_optim.zero_grad(); d_loss.backward(); self.disc_optim.step()

        # === 2. 训练策略：用 -log D 作为奖励 ===
        for _ in range(n_policy_steps):
            states, actions, next_states, _ = self.policy.rollout()
            with torch.no_grad():
                rewards = -F.logsigmoid(self.disc(states, actions))  # r = -log(1 - D)
            # 喂给任意 RL 算法（这里假设 PPO）
            self.policy.ppo_update(states, actions, rewards, next_states)
```

### GAIL 与 MaxEnt IRL 的等价性

形式上，GAIL 是 MaxEnt IRL 在**奖励函数无约束**（任意神经网络）下的对偶问题。判别器 $D_\phi^*$ 的最优解为：

$$D_\phi^*(s, a) = \frac{p_{\text{expert}}(s, a)}{p_{\text{expert}}(s, a) + p_{\pi_\theta}(s, a)}$$

代入后，最优奖励正是 $r^*(s, a) = \log D^* - \log(1 - D^*) = \log \frac{p_{\text{expert}}}{p_{\pi_\theta}}$——即**对数似然比**。这与 MaxEnt IRL 推出的奖励一致，但 GAIL 不需要显式计算配分函数 $Z$。

### 三大模仿学习路线对比

| 维度             | BC  | MaxEnt IRL        | GAIL                |
| ---------------- | --- | ----------------- | ------------------- |
| 是否解决分布偏移 | ❌  | ✅                | ✅                  |
| 需要环境模型     | ❌  | ✅（或软 Q 近似） | ❌                  |
| 显式奖励函数     | —   | ✅（可解释）      | ❌（隐式）          |
| 计算成本         | 低  | 高（内层 RL）     | 中（对抗训练）      |
| 扩展到高维       | 易  | 难                | 中                  |
| LLM 中的对应     | SFT | —                 | DPO 隐式（见 14.6） |

::: details GAIL 的训练不稳定性
GAN 的通病：判别器过强时生成器梯度消失，过弱时学不到信号。实践中常用 Tricks：

- 判别器梯度惩罚（Wasserstein GAIL）
- 判别器更新比策略慢（每 5 步策略更新 1 步判别器）
- 熵正则化系数 $\lambda$ 调到 0.1-1.0 防止策略坍缩
  :::

GAIL 在 MuJoCo 上接近专家水平，但需要数百万步环境交互——**样本效率仍是瓶颈**。这推动了对**离线模仿学习**的研究（如 DemoDICE、DWBC），把专家数据与次优数据结合，无需在线交互。

## 本节总结

逆向 RL（IRL）从专家行为反推奖励函数，最大熵 IRL 解决了 IRL 的不适定问题。GAIL 用 GAN 框架绕开显式 reward 推断，让模仿学习的可扩展性大幅提升。GAIL 启发了后来的对抗 RL 和 RLHF 中的 reward model 训练。

下一节 [13.3 元 RL：MAML、RL²、PEARL、In-Context RL](./meta-rl) 转向另一个问题——**当环境不断变化时，agent 如何快速适应新任务**？
