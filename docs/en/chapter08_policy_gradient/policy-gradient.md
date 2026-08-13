---
title: 6.1 Policy Gradient Theorem
---

# 6.1 Policy Gradient Theorem

Chapter 3 followed one route: learn $Q(s,a)$, use it to score each action, and choose the action with the highest score. See [Q-Values and the Greedy Policy](../chapter03_mdp/value-q) for a review. This route works well for CartPole and Atari because their action sets are finite.

CartPole has only two actions, so DQN can produce two $Q$ values and compare them. A robotic arm is different. Its shoulder, elbow, and wrist can each apply a continuous torque, giving us infinitely many possible action combinations. A language model also makes a sequence of decisions while preserving the ability to sample rather than always selecting the single most likely token.

These tasks suggest a second route: represent the policy distribution itself and optimize it directly.

## Learning the Policy Directly

We skip the intermediate $Q$ values and learn a **policy** $\pi_\theta(a\mid s)$ directly. The policy answers a concrete question: given state $s$, how likely should each action be?

This is the idea behind the [policy objective $J(\theta)$](../chapter03_mdp/policy-value) from Chapter 3. We define a score for the policy as a whole, then adjust its parameters $\theta$ to increase that score. This chapter develops that route from the policy gradient theorem to REINFORCE and variance-reducing baselines.

::: tip Prerequisites
Three earlier ideas will be used throughout the chapter:

- [Policy $\pi_\theta$ and objective $J(\theta)$](../chapter03_mdp/policy-value): how a parameterized policy is represented and evaluated
- [Monte Carlo methods](../chapter03_mdp/dp-mc-td): finish an episode, then compute its return
- [State value $V(s)$](../chapter03_mdp/value-bellman): the expected return from a state and the source of a useful baseline
  :::

## The Route Through This Chapter

The theory begins with the policy gradient theorem, turns it into REINFORCE, and then introduces baselines and advantage functions to reduce variance. The experiments follow the same order: run vanilla REINFORCE on CartPole, observe its noisy updates, and then add a value baseline.

| Section                                              | Question                                                                          |
| ---------------------------------------------------- | --------------------------------------------------------------------------------- |
| [Why Policy Gradients?](./pg-necessity)              | Where does DQN's $\arg\max$ become inconvenient, and why learn a policy directly? |
| [REINFORCE and Value Baselines](./reinforce)         | How does the policy gradient theorem become a trainable loss?                     |
| [Hands-on: CartPole](./cartpole)                     | What does high variance look like in a control task?                              |
| [Policy Gradient Improvements](./pg-improvements)    | Why does a baseline reduce variance?                                              |
| [Hands-on: Baseline Comparison](./cartpole-baseline) | How much does the value baseline change reward and variance?                      |

We first establish the full argument on this page. The next section repeats the central calculation with more numerical detail.

The bandit experiment gave us one compact line of code:

```python
loss = -log_prob * reward
```

The policy gradually favored the better arm, but the code alone did not explain why multiplying a log-probability by a reward produces that behavior. The policy gradient theorem supplies the missing argument.

## Value-Based and Policy-Based Methods

Before deriving the theorem, let us state exactly what changes when we move from DQN to a policy gradient.

### What Each Method Learns

A value-based method learns $Q(s,a)$, a score for each action. Its policy is obtained indirectly through $\arg\max_a Q(s,a)$.

A policy-based method learns $\pi_\theta(a\mid s)$, a probability distribution over actions. It can sample an action directly without first constructing a score table.

|                        | Value-based method                       | Policy-based method                   |
| ---------------------- | ---------------------------------------- | ------------------------------------- |
| Learned object         | $Q(s,a)$                                 | $\pi_\theta(a\mid s)$                 |
| Action selection       | $\arg\max_a Q(s,a)$                      | sample from $\pi_\theta(\cdot\mid s)$ |
| Policy form            | usually deterministic at evaluation time | explicitly stochastic                 |
| Main mathematical tool | Bellman targets and TD learning          | policy gradients                      |

### Action Spaces

Standard DQN produces one value for every candidate action. This is straightforward when CartPole offers two actions. For a six-joint arm with torque vector $a\in[-10,10]^6$, however, the candidate set is continuous. Maximizing a learned $Q(s,a)$ then becomes a separate continuous optimization problem.

A policy network represents the action distribution directly. A Softmax distribution works for discrete actions; a Gaussian distribution can represent continuous actions. The learning rule remains the same while the output distribution changes.

### Exploration

A greedy DQN policy always chooses $\arg\max_a Q(s,a)$, so training usually adds exploration through a rule such as $\varepsilon$-greedy. The value of $\varepsilon$ must be scheduled separately.

A policy gradient already produces a distribution. If an action receives probability $0.3$, sampling tries it about 30 percent of the time. Exploration is therefore part of the policy representation.

### Data Reuse

DQN is off-policy. A replay buffer can store older transitions and reuse them across many updates.

Vanilla REINFORCE is on-policy. Its gradient is an expectation under the current policy $\pi_\theta$, so data collected by an older policy cannot be reused in the same direct way after the policy changes. This is one reason basic policy-gradient methods need more interaction data.

### Summary

| Property                | Value-based methods                     | Policy-based methods                   |
| ----------------------- | --------------------------------------- | -------------------------------------- |
| Action space            | finite discrete actions in standard DQN | discrete or continuous                 |
| Exploration             | added by a separate rule                | represented by the policy distribution |
| Data reuse              | off-policy replay is common             | vanilla REINFORCE is on-policy         |
| Typical source of noise | bootstrapped TD targets                 | sampled Monte Carlo returns            |
| Representative method   | DQN                                     | REINFORCE                              |

The two routes can be combined. Actor-Critic methods use a policy network to choose actions and a value network to reduce the variance of the policy update. We first need the policy-gradient calculation that makes the actor trainable.

## The Policy Objective

We need a scalar quantity that says how well the policy performs. Let $J(\theta)$ be the expected discounted return obtained by policy $\pi_\theta$:

$$
J(\theta)=\mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{\infty}\gamma^t r_t\right].
$$

Here $\theta$ denotes the policy-network parameters, $r_t$ is the reward at time $t$, and $\gamma\in[0,1]$ discounts rewards farther in the future. The expectation averages over trajectories generated by the policy.

Our objective is to find parameters that maximize $J(\theta)$.

## Gradient Ascent

To increase the objective, we update the parameters in the direction of its gradient:

$$
\theta \leftarrow \theta + \alpha\nabla_\theta J(\theta),
$$

where $\alpha$ is the learning rate. The plus sign indicates gradient ascent because the objective is being maximized.

The difficulty lies in computing $\nabla_\theta J(\theta)$. The expectation in $J(\theta)$ ranges over every trajectory the policy could generate. Enumerating all of them is impossible, so the gradient must be rewritten into a form that can be estimated from sampled episodes.

## The Policy Gradient Theorem

The policy gradient theorem gives that sample-based form:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\pi_\theta}
\left[
\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\,G_t
\right],
$$

where $G_t$ is the return from time $t$ onward. Each sampled action contributes two quantities:

- $\nabla_\theta\log\pi_\theta(a_t\mid s_t)$ tells us how a parameter change would alter the probability of that action;
- $G_t$ tells us how much return followed that action in the sampled episode.

An action followed by a large return receives a stronger positive update. An action followed by a small or negative return receives a weaker or negative update. Averaging these contributions over sampled trajectories estimates the true policy gradient.

### The Log-Derivative Trick

The logarithm appears through the identity

$$
\nabla_\theta\log\pi_\theta(a\mid s)
=
\frac{\nabla_\theta\pi_\theta(a\mid s)}{\pi_\theta(a\mid s)}.
$$

To see why this helps, write the objective as a sum over trajectories:

$$
J(\theta)=\sum_\tau P(\tau;\theta)R(\tau).
$$

Its gradient is

$$
\nabla_\theta J(\theta)
=
\sum_\tau \nabla_\theta P(\tau;\theta)R(\tau).
$$

Using $\nabla P=P\nabla\log P$ turns this sum back into an expectation:

$$
\nabla_\theta J(\theta)
=
\sum_\tau P(\tau;\theta)\nabla_\theta\log P(\tau;\theta)R(\tau).
$$

The probability of a trajectory factors into policy choices and environment transitions:

$$
P(\tau;\theta)
=p(s_0)\prod_t \pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t).
$$

Only the policy depends on $\theta$. After taking the log and differentiating, the environment-transition terms disappear:

$$
\nabla_\theta\log P(\tau;\theta)
=
\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t).
$$

This is the important practical result: policy gradients do not require a differentiable model of the environment.

## The REINFORCE Algorithm

REINFORCE is the direct Monte Carlo implementation of the theorem:

1. Run a complete episode using the current policy.
2. For each step, compute
   $G_t=\sum_{k=t}^{T}\gamma^{k-t}r_k$.
3. Estimate the gradient with
   $\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)G_t$.
4. Update the policy parameters in the ascent direction.

PyTorch optimizers minimize a loss, so the ascent update is written with a minus sign:

```python
loss = -log_prob * G_t
```

For a complete episode:

```python
loss = 0.0
for t in range(len(rewards)):
    G_t = sum(
        gamma ** k * rewards[t + k]
        for k in range(len(rewards) - t)
    )
    loss += -log_probs[t] * G_t

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

The one-step bandit update `-log_prob * reward` is the special case $G_t=r_t$.

## Why REINFORCE Has High Variance

The return $G_t$ contains every random event that occurs after action $a_t$. The same action can therefore receive very different learning signals in different episodes.

| Episode                | What happens after the action    | $G_t$ |
| ---------------------- | -------------------------------- | ----- |
| fortunate trajectory   | later steps receive high rewards | large |
| unfortunate trajectory | later steps receive low rewards  | small |

REINFORCE uses $G_t$ to judge the sampled action, even though part of $G_t$ may be caused by later randomness. A useful action can receive a weak update after an unlucky continuation, while a poor action can receive a strong update after a lucky continuation. The expected gradient remains correct, but individual updates are noisy.

A baseline can remove part of this shared fluctuation without changing the expected gradient. That observation leads to value baselines and Actor-Critic methods.

## Discrete and Continuous Action Spaces

The theorem does not depend on whether actions are discrete or continuous. Only the policy distribution changes:

|               | Discrete actions                     | Continuous actions                   |
| ------------- | ------------------------------------ | ------------------------------------ |
| Example       | CartPole left/right, token selection | joint torque, steering angle         |
| Policy output | Softmax probabilities                | Gaussian mean and standard deviation |
| Sampling      | categorical sample                   | $a\sim\mathcal{N}(\mu,\sigma^2)$     |
| $\log\pi$     | categorical log-probability          | Gaussian log-density                 |

This is why the same policy-gradient framework can train a CartPole controller, a continuous-control policy, or a language model. The next section works through the objective and the update numerically: [REINFORCE and Value Baselines](./reinforce).

---

[^1]: Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. _Machine Learning_, 8(3-4), 229–256. [DOI](https://doi.org/10.1007/BF00992696)

[^2]: Sutton, R. S., et al. (2000). Policy gradient methods for reinforcement learning with function approximation. _Advances in Neural Information Processing Systems_, 12.
