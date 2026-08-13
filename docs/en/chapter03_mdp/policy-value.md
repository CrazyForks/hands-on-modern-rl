# 2.3 Policies, Values, and Returns

> [2.2](./mdp) defined an MDP as the tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$. An MDP itself, however, describes only the "environment"—how does an agent make decisions within that environment? This section introduces three core concepts: the **policy** (how the agent selects actions), the **return** (how the quality of a trajectory is measured), and the **value function** (how the long-term benefit of a state or action is evaluated). These concepts form the foundation of every RL algorithm discussed later.

## Policies and Decision Rules

A **policy** is a mapping from states to actions. Policies fall into two categories:

- **Deterministic policy**: $\pi: \mathcal{S} \to \mathcal{A}$, which directly outputs an action $a = \pi(s)$ for a given state
- **Stochastic policy**: $\pi: \mathcal{S} \to \Delta(\mathcal{A})$, which outputs a distribution over actions for a given state, with $a \sim \pi(\cdot \mid s)$

Stochastic policies are more general. A deterministic policy is the special case in which the distribution collapses to a single point. RL generally considers stochastic policies.

```python
# A simple stochastic policy for CartPole
import torch
import torch.nn as nn

class CartPolePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32), nn.Tanh(),
            nn.Linear(32, 2)  # Logits for the two actions
        )

    def forward(self, state):
        logits = self.net(state)
        return torch.distributions.Categorical(logits=logits)

    def act(self, state):
        dist = self.forward(state)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
```

### The Optimal Policy

The goal of RL is to find an **optimal policy** $\pi^*$ that maximizes the long-term cumulative reward:

$$\pi^* = \arg\max_\pi \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t)\right]$$

Every algorithm discussed later, including DQN, PPO, and SAC, approximately solves this optimization problem.

## Returns and Trajectory Evaluation

During an episode, an agent experiences a trajectory $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$. The **return** is the cumulative reward from time step $t$ onward:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

### The Role of the Discount Factor γ

$\gamma \in [0, 1]$ is the **discount factor**, which gives progressively less weight to rewards farther in the future. It serves three purposes:

1. **Guaranteeing convergence mathematically**: the infinite sum $\sum \gamma^k R$ converges when $|\gamma| < 1$
2. **Representing uncertainty**: future rewards are harder to predict and therefore receive less weight
3. **Stabilizing training**: discounting reduces the high variance caused by delayed rewards

| γ value | Meaning                          | Applications                      |
| ------- | -------------------------------- | --------------------------------- |
| 0       | Immediate rewards only (greedy)  | Rarely used                       |
| 0.9     | Short horizon (about 10 steps)   | Board games, recommender systems  |
| 0.99    | Medium horizon (about 100 steps) | Atari, CartPole                   |
| 0.999   | Long horizon (about 1,000 steps) | Long-term tasks, robot navigation |
| 1.0     | No discounting                   | Finite-horizon tasks              |

### Returns in CartPole

CartPole gives a reward of 1 at every step while the pole remains upright. An episode ends when the pole falls or the cart leaves the permitted range. The return is

$$G_0 = 1 + \gamma + \gamma^2 + \cdots + \gamma^{T-1} = \frac{1 - \gamma^T}{1 - \gamma}$$

where $T$ is the episode length. When $\gamma = 0.99, T = 500$, $G_0 \approx 99$.

## Value Functions and Long-Term Benefit

A **value function** measures "the expected return obtained by following the current policy $\pi$ from a particular state or after taking a particular action." There are two forms.

### State Value V(s)

$$V^\pi(s) = \mathbb{E}_\pi\left[G_t \mid s_t = s\right] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid s_t = s\right]$$

Meaning: the expected cumulative return obtained by starting from state $s$ and following policy $\pi$.

### Action Value Q(s, a)

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[G_t \mid s_t = s, a_t = a\right]$$

Meaning: the expected cumulative return obtained by starting from state $s$, **first taking action $a$, and then following $\pi$**.

### The Relationship Between V and Q

$$V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s, a)$$

Thus, $V^\pi(s)$ is the expectation of $Q^\pi(s, a)$ under policy $\pi$.

### A Numerical GridWorld Example

Consider a 4×4 GridWorld with a goal in the lower-right corner (reward = +1 and termination) and a reward of 0 at every other position:

```
┌───┬───┬───┬───┐
│0.0│0.5│0.8│0.9│   ← V(s) values
├───┼───┼───┼───┤
│0.5│0.7│0.9│1.0│ ★ (goal)
├───┼───┼───┼───┤
│0.7│0.9│0.95│  │
├───┼───┼───┼───┤
│0.8│0.95│ │  │   ← Unshown positions have lower V values
└───┴───┴───┴───┘
```

The closer a state is to the goal, the higher its value, because the future reward arrives sooner and is discounted less.

## The Advantage Function and Action Evaluation

The **advantage function** measures how much better action $a$ is than the average action:

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

- $A > 0$: action $a$ is better than average
- $A < 0$: action $a$ is worse than average
- $A = 0$: action $a$ is average

The advantage function is central to policy-gradient methods ([Chapter 6](../chapter08_policy_gradient/policy-gradient)) and Actor-Critic methods ([Chapter 7](../chapter09_actor_critic/actor-critic)).

## A Preview of the Bellman Equation

Value functions satisfy the **Bellman equation**, a recursive relationship that expresses $V(s)$ in terms of $V(s')$:

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma V^\pi(s')\right]$$

This equation lies at the heart of every RL algorithm. The next section, [2.4 Discounting, Trajectories, and POMDPs](./panorama), examines trajectories in greater detail. [Chapter 3: Value Functions and Bellman Equations](./value-bellman) then develops the Bellman equation fully.

## Section Summary

Policies, returns, and value functions are three core concepts in an MDP:

1. **Policy $\pi$**: the agent's decision rule; the stochastic policy $a \sim \pi(\cdot \mid s)$ is the most general form
2. **Return $G_t$**: the discounted cumulative reward from time step $t$ onward, $\sum \gamma^k r$
3. **Value functions**: $V^\pi(s)$ is the state value and $Q^\pi(s, a)$ is the action value; the advantage $A = Q - V$ measures relative quality

The next section, [2.4 Discounting, Trajectories, and POMDPs](./panorama), formalizes trajectories and extends the framework to POMDPs (partially observable MDPs).
