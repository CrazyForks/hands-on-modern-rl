# 1.3 动手：PPO 训练可视化

> **本节目标**：在 `CartPole-v1` 上完成一次 PPO 训练，用奖励曲线、策略回放和训练指标判断小车是否学会保持平衡。

> **学习路径**：[1.1 CartPole 控制原理](./principles) → [1.2 奖励与训练指标](./metrics) → **1.3 PPO 训练可视化**

> **本节代码**：[Stable-Baselines3 版本](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter01_cartpole/1-ppo_cartpole.py) · [纯 PyTorch 版本](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter01_cartpole/2-pytorch_ppo.py) · [曲线绘制](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter01_cartpole/plot_curves.py) · [依赖](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter01_cartpole/requirements.txt)

前两节已经说明了 CartPole 的状态、动作和奖励，也解释了训练曲线中的回报与回合长度。现在把这些概念放进一次完整训练：策略先用随机权重控制小车，再通过反复采样和更新，逐渐把杆子保持在竖直位置。

## 训练流程概览

PPO 训练 CartPole 的完整流程：

```
┌──────────────────────────────────────────────┐
│ 1. 初始化：策略网络 π_θ(a|s) 随机权重       │
├──────────────────────────────────────────────┤
│ 2. Rollout：用当前 π_θ 跑 N 条轨迹          │
│    收集 (s_t, a_t, r_t, s_{t+1}, done)       │
├──────────────────────────────────────────────┤
│ 3. 计算 advantage Â_t（用 GAE）              │
├──────────────────────────────────────────────┤
│ 4. PPO 更新：maximize clipped objective     │
│    L = E[min(r_t Â_t, clip(r_t, 1±ε) Â_t)] │
├──────────────────────────────────────────────┤
│ 5. 重复 2-4 直到收敛（reward 达到 500）     │
└──────────────────────────────────────────────┘
```

## 完整训练代码

下面是一个最小化的 PPO + CartPole 实现。完整可运行版本见 `code/chapter01_cartpole/2-pytorch_ppo.py`。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque

# === 策略网络：状态 → 动作概率 ===
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, s):
        logits = self.net(s)
        return torch.distributions.Categorical(logits=logits)

# === 价值网络：状态 → V(s) ===
class ValueNetwork(nn.Module):
    def __init__(self, state_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, s):
        return self.net(s).squeeze(-1)

# === GAE：广义优势估计 ===
def compute_gae(rewards, values, next_values, episode_ends,
                gamma=0.99, lam=0.95):
    """计算 GAE，并在每次环境 reset 处切断递推。"""
    raw_advantages = []
    gae = 0
    for r, v, v_next, episode_end in zip(
        reversed(rewards), reversed(values), reversed(next_values),
        reversed(episode_ends)
    ):
        # 真正终止时 v_next=0；时间截断时仍从 V(s') bootstrap。
        delta = r + gamma * v_next - v
        gae = delta + gamma * lam * (1.0 - float(episode_end)) * gae
        raw_advantages.insert(0, gae)
    return torch.tensor(raw_advantages, dtype=torch.float32)

# === 主训练循环 ===
def train_ppo(env_name='CartPole-v1', n_iters=200, n_steps=2048,
              gamma=0.99, lam=0.95, clip_eps=0.2, lr=3e-4,
              n_epochs=10, batch_size=64):
    env = gym.make(env_name)
    policy = PolicyNetwork()
    value_fn = ValueNetwork()
    optimizer = optim.Adam(list(policy.parameters()) + list(value_fn.parameters()), lr=lr)

    reward_history = deque(maxlen=20)

    for iter in range(n_iters):
        # === 1. Rollout ===
        states, actions, rewards = [], [], []
        values, next_values, episode_ends, log_probs_old = [], [], [], []
        s, _ = env.reset()
        ep_reward = 0

        for step in range(n_steps):
            s_tensor = torch.FloatTensor(s)
            dist = policy(s_tensor)
            v = value_fn(s_tensor)
            a = dist.sample()

            s_next, r, terminated, truncated, _ = env.step(a.item())
            done = terminated or truncated
            with torch.no_grad():
                v_next = 0.0 if terminated else value_fn(torch.FloatTensor(s_next)).item()

            states.append(s); actions.append(a.item()); rewards.append(r)
            values.append(v.item()); next_values.append(v_next); episode_ends.append(done)
            log_probs_old.append(dist.log_prob(a).item())
            ep_reward += r

            if done:
                reward_history.append(ep_reward)
                ep_reward = 0
                s, _ = env.reset()
            else:
                s = s_next

        # === 2. 计算 advantage ===
        raw_advantages = compute_gae(
            rewards, values, next_values, episode_ends, gamma, lam
        )
        # Critic 学习未归一化的回报；归一化只用于策略更新。
        returns = raw_advantages + torch.FloatTensor(values)
        advantages = (raw_advantages - raw_advantages.mean()) / (
            raw_advantages.std(unbiased=False) + 1e-8
        )

        # === 3. PPO 更新（多个 epoch） ===
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions)
        log_probs_old_t = torch.FloatTensor(log_probs_old)

        for epoch in range(n_epochs):
            idx = torch.randperm(len(states_t))
            for start in range(0, len(states_t), batch_size):
                end = start + batch_size
                mb_idx = idx[start:end]

                mb_states = states_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_lp = log_probs_old_t[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ret = returns[mb_idx]

                dist = policy(mb_states)
                new_lp = dist.log_prob(mb_actions)
                ratio = (new_lp - mb_old_lp).exp()

                # PPO Clip
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                v_pred = value_fn(mb_states)
                value_loss = ((v_pred - mb_ret) ** 2).mean()

                loss = policy_loss + 0.5 * value_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(policy.parameters()) + list(value_fn.parameters()), 0.5
                )
                optimizer.step()

        if iter % 10 == 0:
            avg_r = np.mean(reward_history) if reward_history else 0
            print(f"Iter {iter}: avg_reward = {avg_r:.1f} / 500")

    return policy

if __name__ == '__main__':
    policy = train_ppo()
```

## 训练曲线与可视化

下图是预期趋势示意。实际曲线受随机种子和软件版本影响，不会逐轮单调上升。

```
reward
 500 │                              ╭───── converged
 400 │                          ╭───╯
 300 │                      ╭───╯
 200 │                  ╭───╯
 100 │              ╭───╯
     0 │───────────╯
       └─────────────────────────────────
        0    50   100   150   200 iterations
```

观察 4 个关键阶段：

| 阶段   | iteration | 平均 reward | 现象                                              |
| ------ | --------- | ----------- | ------------------------------------------------- |
| 探索期 | 0-20      | 10-30       | 智能体随机倒杆，几乎学不到信号                    |
| 学习期 | 20-100    | 30-200      | reward 快速上升，策略开始稳定                     |
| 收敛期 | 100-150   | 200-450     | 进步变慢，但仍稳定提升                            |
| 已解决 | 150+      | 475+        | Gymnasium 定义"解决"为最近 100 episode 平均 ≥ 475 |

## 超参数的影响

PPO 在 CartPole 上对超参数不算敏感，但仍需理解每个参数的作用：

| 超参数      | 默认值 | 影响范围                                         |
| ----------- | ------ | ------------------------------------------------ |
| `lr`        | 3e-4   | 学习率太高（>1e-3）训练崩溃；太低（<1e-5）训练慢 |
| `clip_eps`  | 0.2    | 越大越激进（更接近原 PG）；越小越保守            |
| `gamma`     | 0.99   | 折扣因子；CartPole 设 0.99 几乎等同于无折扣      |
| `lam` (GAE) | 0.95   | 越大越接近 Monte Carlo；越小越接近 TD            |
| `n_epochs`  | 10     | 每次 rollout 数据复用次数；太大 overfit          |
| `n_steps`   | 2048   | rollout 长度；CartPole 短任务可减到 512          |

### 失败模式排查

如果训练不收敛，按以下顺序排查：

1. **策略 entropy 是否归零**：如果动作概率过早 collapse 到 1.0/0.0，加 entropy bonus `loss += -0.01 * dist.entropy().mean()`
2. **advantage 是否正确归一化**：忘归一化会导致梯度尺度不稳定
3. **value loss 是否爆炸**：检查 returns 是否包含异常值
4. **学习率是否过大**：用 `lr=1e-4` 重试

## 用 TensorBoard 可视化

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/cartpole_ppo')

for iter in range(n_iters):
    # ... 训练代码 ...

    writer.add_scalar('train/reward_mean', avg_r, iter)
    writer.add_scalar('train/policy_loss', policy_loss.item(), iter)
    writer.add_scalar('train/value_loss', value_loss.item(), iter)
    writer.add_scalar('train/entropy', dist.entropy().mean().item(), iter)
    writer.add_scalar('train/clip_frac', clip_fraction, iter)
```

启动 TensorBoard：

```bash
tensorboard --logdir=runs
```

监控 5 个关键指标：

- **reward_mean**：训练核心指标，其移动平均应整体上升，允许短期回落
- **policy_loss**：PPO clip 后的 loss，振荡正常
- **value_loss**：应平稳下降
- **entropy**：策略熵，应从 ~0.69（初始 ln 2）缓慢下降到 ~0.1
- **clip_frac**：被 clip 的比例，应稳定在 0.1-0.3；> 0.5 说明策略变化太快

## 用绘制工具对比实验

```python
import matplotlib.pyplot as plt

def plot_experiments(results):
    """results: dict of name -> list of rewards"""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, rewards in results.items():
        # 滑动平均
        smoothed = np.convolve(rewards, np.ones(20)/20, mode='valid')
        ax.plot(smoothed, label=name)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Reward (20-episode mean)')
    ax.set_title('PPO on CartPole: Hyperparameter Sweep')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=475, color='r', linestyle='--', label='Solved threshold')
    plt.savefig('cartpole_ppo_sweep.png', dpi=120, bbox_inches='tight')

# 对比不同超参数
results = {
    'lr=3e-4 (default)': run_experiment(lr=3e-4),
    'lr=1e-4 (slow)': run_experiment(lr=1e-4),
    'lr=1e-3 (fast)': run_experiment(lr=1e-3),
    'clip=0.1 (conservative)': run_experiment(clip_eps=0.1),
    'clip=0.3 (aggressive)': run_experiment(clip_eps=0.3),
}
plot_experiments(results)
```

以下是待验证的实验假设，不是固定的收敛保证：

- 较小的学习率可能更稳定，但需要更多更新。
- 较大的学习率或裁剪范围可能加快早期学习，也可能增大波动。
- 用多个随机种子比较达到目标回报所需的环境步数，并报告均值和离散程度。

## 本节小结

PPO 训练 CartPole 是 RL 入门最经典的"hello world"。本节给出了完整可运行的 PPO 实现，覆盖了 rollout 收集、GAE 优势估计、PPO Clip 更新、超参数调优、TensorBoard 可视化的全流程。

关键收获：

1. **PPO = Rollout + GAE + Clipped Update** 的三步循环
2. **超参数不敏感** 是 PPO 流行的关键——默认值在 CartPole 上几乎一定 work
3. **可视化训练** 是 debug RL 代码的核心工具——entropy、clip_frac 等辅助指标往往能提前发现训练问题

下一章 [强化学习过程的基本定义](../chapter03_mdp/bandit) 会先把视角拉回到 RL 的最简形式——无状态、即时奖励的多臂老虎机——专门研究探索和利用问题，然后再引入状态转移和长期回报。
