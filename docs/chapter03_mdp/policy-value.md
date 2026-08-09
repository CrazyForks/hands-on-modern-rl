# 2.3 策略、价值与回报

> [2.2](./mdp) 定义了 MDP 的五元组 $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$。但 MDP 本身只是"环境"——智能体如何在这个环境里决策？本节引入三个核心概念：**策略**（agent 怎么选动作）、**回报**（怎么衡量一条轨迹的好坏）、**价值函数**（怎么评估一个状态或动作的长期收益）。这三个概念是后续所有 RL 算法的基础。

## 策略与决策规则

**策略**（policy）是 agent 从状态到动作的映射，分两类：

- **确定性策略**：$\pi: \mathcal{S} \to \mathcal{A}$，给定状态直接输出动作 $a = \pi(s)$
- **随机策略**：$\pi: \mathcal{S} \to \Delta(\mathcal{A})$，给定状态输出动作上的分布 $a \sim \pi(\cdot \mid s)$

随机策略更一般，确定性策略是它的特例（分布坍缩到单点）。RL 一般讨论随机策略。

```python
# CartPole 的一个简单随机策略示例
import torch
import torch.nn as nn

class CartPolePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32), nn.Tanh(),
            nn.Linear(32, 2)  # 2 个动作的 logits
        )

    def forward(self, state):
        logits = self.net(state)
        return torch.distributions.Categorical(logits=logits)

    def act(self, state):
        dist = self.forward(state)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
```

### 最优策略

RL 的目标是找到**最优策略** $\pi^*$，使长期累积奖励最大化：

$$\pi^* = \arg\max_\pi \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t)\right]$$

后续所有算法（DQN、PPO、SAC）都是在近似求解这个优化问题。

## 回报与轨迹衡量

智能体在一个 episode 中经历一条轨迹 $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$。**回报**（return）是从 $t$ 时刻起的累积奖励：

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

### 折扣因子 γ 的作用

$\gamma \in [0, 1]$ 是**折扣因子**，让远期奖励权重递减。它有三个作用：

1. **数学上保证收敛**：无限和 $\sum \gamma^k R$ 在 $|\gamma| < 1$ 时收敛
2. **反映不确定性**：未来奖励本来就难预测，应该给低权重
3. **稳定训练**：避免"延迟奖励"导致的高方差

| γ 值  | 含义                | 应用                 |
| ----- | ------------------- | -------------------- |
| 0     | 只看眼前（贪心）    | 很少使用             |
| 0.9   | 较短视野（10 步内） | 棋盘游戏、推荐系统   |
| 0.99  | 中等视野（100 步）  | Atari、CartPole      |
| 0.999 | 长视野（1000 步）   | 长程任务、机器人导航 |
| 1.0   | 无折扣              | 有限 horizon 任务    |

### CartPole 中的回报

CartPole 每步 reward = 1（杆子还立着）。episode 终止时杆子倒了或 cart 出界。回报：

$$G_0 = 1 + \gamma + \gamma^2 + \cdots + \gamma^{T-1} = \frac{1 - \gamma^T}{1 - \gamma}$$

其中 $T$ 是 episode 长度。当 $\gamma = 0.99, T = 500$ 时，$G_0 \approx 99$。

## 价值函数与长期收益

**价值函数**衡量"在某个状态/动作下，按当前策略 $\pi$ 行动，期望能拿到多少回报"。分两种：

### 状态价值 V(s)

$$V^\pi(s) = \mathbb{E}_\pi\left[G_t \mid s_t = s\right] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid s_t = s\right]$$

含义：从状态 $s$ 出发，按策略 $\pi$ 走，期望累积回报。

### 动作价值 Q(s, a)

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[G_t \mid s_t = s, a_t = a\right]$$

含义：从状态 $s$ 出发，**先采取动作 $a$，之后按 $\pi$**，期望累积回报。

### V 与 Q 的关系

$$V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s, a)$$

即 $V^\pi(s)$ 是 $Q^\pi(s, a)$ 在策略 $\pi$ 下的期望。

### GridWorld 数值示例

考虑一个 4×4 GridWorld，目标在右下角（reward = +1 终止），其他位置每步 reward = 0：

```
┌───┬───┬───┬───┐
│0.0│0.5│0.8│0.9│   ← V(s) values
├───┼───┼───┼───┤
│0.5│0.7│0.9│1.0│ ★ (goal)
├───┼───┼───┼───┤
│0.7│0.9│0.95│  │
├───┼───┼───┼───┤
│0.8│0.95│ │  │   ← 没显示的位置 V 较低
└───┴───┴───┴───┘
```

距离 goal 越近，V 越高——因为未来能拿到的奖励更近（折扣更少）。

## 优势函数与动作评价

**优势函数**（advantage function）衡量动作 $a$ 相对于平均水平好多少：

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

- $A > 0$：动作 $a$ 比平均好
- $A < 0$：动作 $a$ 比平均差
- $A = 0$：动作 $a$ 是平均水平

优势函数在策略梯度（[第 6 章](../chapter08_policy_gradient/policy-gradient)）和 Actor-Critic（[第 7 章](../chapter09_actor_critic/actor-critic)）中是核心概念。

## 贝尔曼方程的预告

价值函数满足**贝尔曼方程**——一个递归关系，把 $V(s)$ 写成 $V(s')$ 的函数：

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma V^\pi(s')\right]$$

这个方程是所有 RL 算法的核心。下一节 [2.4 折扣、轨迹与 POMDP](./panorama) 会进一步讨论轨迹的细节，然后 [第 3 章 价值函数与贝尔曼方程](./value-bellman) 会深入贝尔曼方程。

## 本节总结

策略、回报、价值函数是 MDP 的三个核心概念：

1. **策略 $\pi$**：agent 的决策规则，随机策略 $a \sim \pi(\cdot \mid s)$ 是最一般形式
2. **回报 $G_t$**：从 $t$ 起的折扣累积奖励 $\sum \gamma^k r$
3. **价值函数**：$V^\pi(s)$ 状态价值，$Q^\pi(s, a)$ 动作价值；优势 $A = Q - V$ 衡量相对好坏

下一节 [2.4 折扣、轨迹与 POMDP](./panorama) 讨论轨迹的形式化、POMDP（部分可观察 MDP）的扩展。
