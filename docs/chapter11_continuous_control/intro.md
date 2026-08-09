# 第 9 章 · 连续控制与基于模型的深度 RL

> [第 8 章 PPO](../chapter10_ppo/intro) 解决了连续动作空间的策略学习问题——用高斯策略输出连续动作 + clip 保证稳定更新。但 PPO 是 on-policy 的：每次策略更新后必须重新采样数据，**样本效率极低**。本章解决两个问题：(1) 如何 off-policy 地处理连续动作（DDPG/TD3/SAC）；(2) 如何用环境模型进一步提升样本效率（Model-Based RL + AlphaZero/Dreamer）。

## 9.1 确定性策略梯度与 DDPG

CartPole、Atari 这类问题动作是离散的（左/右、上下左右），用 Q-Learning 或 softmax 策略直接处理。但机器人控制、自动驾驶、机械臂操作——动作是**连续的**：关节角度 $\theta \in \mathbb{R}^n$、油门开度 $[0, 1]$、方向盘转角 $[-\pi, \pi]$。

连续动作带来两个挑战：

1. **Q 函数无法 argmax**：离散情况 $a^* = \arg\max_a Q(s, a)$ 直接遍历；连续情况无法枚举
2. **策略输出维度变化**：不再是 softmax 概率，而是分布参数（均值、方差）

### 策略梯度定理的连续版本

[第 6 章策略梯度](../chapter08_policy_gradient/reinforce)给出：

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^\pi, a \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a\mid s) \cdot Q^\pi(s, a)\right]$$

这要求策略 $\pi_\theta(a \mid s)$ 是**随机的**（概率分布）。但 Silver et al. 2014 证明：如果策略是**确定性**的 $a = \mu_\theta(s)$，也存在类似的梯度定理：

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^\mu}\left[\nabla_\theta \mu_\theta(s) \cdot \nabla_a Q^\mu(s, a)\big|_{a=\mu_\theta(s)}\right]$$

这就是**确定性策略梯度（Deterministic Policy Gradient, DPG）**定理。它比随机版本更样本高效：

- **不需要对 $a$ 积分**：随机 PG 要对所有可能动作求期望，确定性 PG 只需对状态求期望
- **off-policy 友好**：可以用任何行为策略采集的数据训练确定性策略

但有个致命问题：**确定性策略不探索**。如果 $\mu_\theta(s)$ 总是返回同一个 $a$，智能体永远不会尝试别的动作。DDPG 的解决方案：**训练时给动作加噪声**。

### 深度确定性策略梯度

Deep Deterministic Policy Gradient（Lillicrap et al. 2015）把 DPG 和 DQN 的深度网络技巧结合：

- **Actor**：$\mu_\theta(s)$ 输出连续动作（神经网络直接回归）
- **Critic**：$Q_\phi(s, a)$ 评估动作价值
- **目标网络**：稳定训练（继承自 DQN）
- **经验回放**：off-policy 重用数据（继承自 DQN）

### 算法主循环

```python
class DDPG:
    def __init__(self, state_dim, action_dim, action_max):
        # 主网络
        self.actor = Actor(state_dim, action_dim, action_max)
        self.critic = Critic(state_dim, action_dim)
        # 目标网络（软更新）
        self.actor_target = copy(self.actor)
        self.critic_target = copy(self.critic)
        self.replay_buffer = ReplayBuffer(capacity=1_000_000)
        self.gamma = 0.99
        self.tau = 0.005  # 软更新系数

    def select_action(self, state, explore=True):
        with torch.no_grad():
            action = self.actor(state)
        if explore:
            # Ornstein-Uhlenbeck 或高斯噪声探索
            action += np.random.normal(0, 0.1, size=action.shape)
        return np.clip(action, -self.action_max, self.action_max)

    def update(self, batch_size=256):
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(batch_size)

        # === Critic 更新 ===
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * (1 - dones) * target_q
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optim.zero_grad(); critic_loss.backward()
        self.critic_optim.step()

        # === Actor 更新：最大化 Q(s, μ(s)) ===
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optim.zero_grad(); actor_loss.backward()
        self.actor_optim.step()

        # === 软更新目标网络 ===
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
```

DDPG 在 MuJoCo 物理环境（HalfCheetah、Hopper、Walker2d）上首次让深度 RL 打败了基于线性特征的 TRPO/CES 等方法。但 DDPG 有几个广受诟病的缺陷：

- **Q 值过估计**：target Q 用 max，易被噪声推高
- **超参数敏感**：学习率、噪声尺度、网络结构稍变就发散
- **训练不稳定**：critic 学坏了 actor 跟着坏，正反馈死循环

## 本节总结

确定性策略梯度（DPG）定理把策略梯度从随机策略扩展到确定性策略，让连续动作空间也能 off-policy 训练。DDPG 把 DPG 与 DQN 的深度网络技巧结合，在 MuJoCo 上首次让深度 RL 打败经典方法。

但 DDPG 有 Q 值过估计、超参敏感、训练不稳定三大缺陷。下一节 [9.2 TD3 与 SAC](./td3-sac) 给出两套互补的修补方案——TD3 用工程 trick 稳定 DDPG，SAC 用最大熵 RL 从根本上重构目标函数。
