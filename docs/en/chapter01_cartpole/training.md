# 1.3 Hands-On: Visualizing PPO Training

> **Section objective**: Complete a PPO training run on `CartPole-v1`, then use reward curves, policy replays, and training metrics to determine whether the cart has learned to balance the pole.

> **Learning path**: [1.1 How CartPole Control Works](./principles) → [1.2 Rewards and Training Metrics](./metrics) → **1.3 Visualizing PPO Training**

> **Code for this section**: [Stable-Baselines3 version](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter01_cartpole/1-ppo_cartpole.py) · [Pure PyTorch version](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter01_cartpole/2-pytorch_ppo.py) · [Curve plotting](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter01_cartpole/plot_curves.py) · [Dependencies](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter01_cartpole/requirements.txt)

The preceding two sections explained CartPole's states, actions, and rewards, as well as the return and episode length shown in training curves. We will now place these concepts into a complete training run. The policy first controls the cart with random weights. Through repeated sampling and updating, it gradually learns to keep the pole upright.

## Overview of the Training Process

The complete process for training CartPole with PPO is as follows:

```
┌─────────────────────────────────────────────────────┐
│ 1. Initialize: random weights for policy π_θ(a|s)   │
├─────────────────────────────────────────────────────┤
│ 2. Rollout: run N trajectories with current π_θ     │
│    Collect (s_t, a_t, r_t, s_{t+1}, done)           │
├─────────────────────────────────────────────────────┤
│ 3. Compute advantage Â_t (using GAE)                 │
├─────────────────────────────────────────────────────┤
│ 4. PPO update: maximize clipped objective            │
│    L = E[min(r_t Â_t, clip(r_t, 1±ε) Â_t)]          │
├─────────────────────────────────────────────────────┤
│ 5. Repeat steps 2–4 until convergence                │
│    (reward reaches 500)                              │
└─────────────────────────────────────────────────────┘
```

## Complete Training Code

The following is a minimal PPO implementation for CartPole. See `code/chapter01_cartpole/2-pytorch_ppo.py` for the complete runnable version.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque

# === Policy network: state → action probabilities ===
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

# === Value network: state → V(s) ===
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

# === GAE: Generalized Advantage Estimation ===
def compute_gae(rewards, values, next_values, episode_ends,
                gamma=0.99, lam=0.95):
    """Compute GAE and stop the recursion at every environment reset."""
    raw_advantages = []
    gae = 0
    for r, v, v_next, episode_end in zip(
        reversed(rewards), reversed(values), reversed(next_values),
        reversed(episode_ends)
    ):
        # v_next is zero after true termination; truncation still bootstraps V(s').
        delta = r + gamma * v_next - v
        gae = delta + gamma * lam * (1.0 - float(episode_end)) * gae
        raw_advantages.insert(0, gae)
    return torch.tensor(raw_advantages, dtype=torch.float32)

# === Main training loop ===
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

        # === 2. Compute advantages ===
        raw_advantages = compute_gae(
            rewards, values, next_values, episode_ends, gamma, lam
        )
        # The critic uses raw return targets; normalization is only for the policy loss.
        returns = raw_advantages + torch.FloatTensor(values)
        advantages = (raw_advantages - raw_advantages.mean()) / (
            raw_advantages.std(unbiased=False) + 1e-8
        )

        # === 3. PPO update (multiple epochs) ===
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

                # PPO clipping
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

## Training Curves and Visualization

The following diagram shows the expected trend. The actual curve depends on the random seed and software versions and need not rise monotonically at every iteration.

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

The curve has four key stages:

| Stage       | Iteration | Average reward | Behavior                                                     |
| ----------- | --------- | -------------- | ------------------------------------------------------------ |
| Exploration | 0–20      | 10–30          | The agent drops the pole randomly and receives little signal |
| Learning    | 20–100    | 30–200         | Reward rises rapidly and the policy begins to stabilize      |
| Convergence | 100–150   | 200–450        | Improvement slows but remains steady                         |
| Solved      | 150+      | 475+           | Gymnasium defines "solved" as a mean ≥ 475 over 100 episodes |

## The Effect of Hyperparameters

PPO is not particularly sensitive to hyperparameters on CartPole, but the role of each parameter still matters:

| Hyperparameter | Default | Effect                                                                                         |
| -------------- | ------- | ---------------------------------------------------------------------------------------------- |
| `lr`           | 3e-4    | Too high (>1e-3) causes collapse; too low (<1e-5) slows training                               |
| `clip_eps`     | 0.2     | Larger values are more aggressive (closer to vanilla PG); smaller values are more conservative |
| `gamma`        | 0.99    | Discount factor; on CartPole, 0.99 is almost equivalent to no discounting                      |
| `lam` (GAE)    | 0.95    | Larger values approach Monte Carlo; smaller values approach TD                                 |
| `n_epochs`     | 10      | Number of times each rollout is reused; values that are too large cause overfitting            |
| `n_steps`      | 2048    | Rollout length; short tasks such as CartPole can reduce it to 512                              |

### Diagnosing Failure Modes

If training does not converge, check the following items in order:

1. **Has policy entropy fallen to zero?** If action probabilities collapse prematurely to 1.0/0.0, add an entropy bonus: `loss += -0.01 * dist.entropy().mean()`
2. **Are advantages normalized correctly?** Omitting normalization makes the gradient scale unstable
3. **Has the value loss exploded?** Check whether the returns contain anomalous values
4. **Is the learning rate too high?** Retry with `lr=1e-4`

## Visualization with TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/cartpole_ppo')

for iter in range(n_iters):
    # ... training code ...

    writer.add_scalar('train/reward_mean', avg_r, iter)
    writer.add_scalar('train/policy_loss', policy_loss.item(), iter)
    writer.add_scalar('train/value_loss', value_loss.item(), iter)
    writer.add_scalar('train/entropy', dist.entropy().mean().item(), iter)
    writer.add_scalar('train/clip_frac', clip_fraction, iter)
```

Start TensorBoard:

```bash
tensorboard --logdir=runs
```

Monitor five key metrics:

- **reward_mean**: the primary training metric; its moving average should trend upward, although short-term drops are normal
- **policy_loss**: the loss after PPO clipping; oscillation is normal
- **value_loss**: should decrease steadily
- **entropy**: the policy entropy; it should slowly decrease from about 0.69 (the initial ln 2) to about 0.1
- **clip_frac**: the clipped fraction; it should remain around 0.1–0.3, while a value above 0.5 indicates that the policy is changing too quickly

## Comparing Experiments with a Plotting Tool

```python
import matplotlib.pyplot as plt

def plot_experiments(results):
    """results: dict of name -> list of rewards"""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, rewards in results.items():
        # Moving average
        smoothed = np.convolve(rewards, np.ones(20)/20, mode='valid')
        ax.plot(smoothed, label=name)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Reward (20-episode mean)')
    ax.set_title('PPO on CartPole: Hyperparameter Sweep')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=475, color='r', linestyle='--', label='Solved threshold')
    plt.savefig('cartpole_ppo_sweep.png', dpi=120, bbox_inches='tight')

# Compare different hyperparameters
results = {
    'lr=3e-4 (default)': run_experiment(lr=3e-4),
    'lr=1e-4 (slow)': run_experiment(lr=1e-4),
    'lr=1e-3 (fast)': run_experiment(lr=1e-3),
    'clip=0.1 (conservative)': run_experiment(clip_eps=0.1),
    'clip=0.3 (aggressive)': run_experiment(clip_eps=0.3),
}
plot_experiments(results)
```

Treat the following as hypotheses to test, not fixed convergence guarantees:

- A smaller learning rate may be more stable but require more updates.
- A larger learning rate or clipping range may speed up early learning but can also increase variability.
- Compare the environment steps needed to reach the target return across multiple random seeds, reporting both the mean and dispersion.

## Section Summary

Training CartPole with PPO is the classic "hello world" of introductory RL. This section presented a complete, runnable PPO implementation covering rollout collection, GAE advantage estimation, PPO Clip updates, hyperparameter tuning, and TensorBoard visualization.

The key lessons are:

1. **PPO = Rollout + GAE + Clipped Update**, repeated in a three-step loop
2. **Low hyperparameter sensitivity** is central to PPO's popularity—the defaults almost always work on CartPole
3. **Visualizing training** is essential for debugging RL code—auxiliary metrics such as entropy and `clip_frac` often reveal training problems early

The next chapter, [Basic Definitions of the Reinforcement Learning Process](../chapter03_mdp/bandit), returns to the simplest form of RL—the stateless, immediate-reward multi-armed bandit—to study exploration and exploitation before introducing state transitions and long-term returns.
