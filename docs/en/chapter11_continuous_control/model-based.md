# 9.3 Model-Based RL

> [9.2](./td3-sac) brought model-free continuous-control algorithms to a stable, practical level: SAC and TD3 can learn good policies on MuJoCo after one million training steps. One million steps are prohibitively expensive for a physical robot, however, because motor wear, battery life, and safety constraints make real-environment sampling costly. The central idea of **model-based RL** is to **learn an environment model**, $\hat{P}(s' \mid s, a), \hat{R}(s, a)$, and train the policy within that model, reducing the required samples from millions of steps to tens of thousands.

## The Fundamental Difference Between Model-Based and Model-Free RL

All preceding algorithms—DDPG, TD3, and SAC—are **model-free**: the agent does not attempt to understand the environment and learns a policy only from the rewards supplied by the environment. **Model-based RL** takes the opposite approach. It first learns an environment model $\hat{P}(s' \mid s, a), \hat{R}(s, a)$ and then uses the model to plan or generate data.

### Why Use a Model?

The main reason is **sample efficiency**. MuJoCo's physics simulations are inexpensive, but **every sample from a physical robot is costly** because the robot may be damaged and its battery and mechanical components wear down. Model-free methods need millions of steps to learn a good policy, which is impractical for a physical robot. Model-based methods need only tens of thousands of steps because, after learning the model, they can sample from it without limit.

### Overview of Three Major Paradigms

| Paradigm | Central idea                                    | Representative algorithm | Applicable setting                                     |
| -------- | ----------------------------------------------- | ------------------------ | ------------------------------------------------------ |
| **Dyna** | Use the model for data augmentation             | Dyna-Q                   | Discrete actions and rapid training                    |
| **PETS** | Probabilistic ensembles and trajectory sampling | PETS                     | High-precision control where model uncertainty matters |
| **MBPO** | Short-horizon rollouts                          | MBPO                     | General continuous control                             |

We now examine each paradigm in turn.

## The Model as Data Augmentation

Dyna is Sutton's classic 1990 approach. It divides each real interaction into four steps: the third trains the model, and the fourth uses the model to generate "synthetic" data that accelerates model-free training.

```python
for step in range(total_steps):
    # 1. Interact with the real environment
    a = policy.select(s)
    s_prime, r = env.step(a)
    replay_buffer.add(s, a, r, s_prime)

    # 2. Update a model-free algorithm such as Q-Learning with real data
    q_learning_update(replay_buffer.sample())

    # 3. Train the environment model with real data
    model.train(s, a, r, s_prime)

    # 4. Generate synthetic data with the model and perform N more Q-Learning updates
    for _ in range(N):  # N = 10–100
        s_sim, a_sim = replay_buffer.sample_state_action()
        s_sim_next, r_sim = model.predict(s_sim, a_sim)
        q_learning_update(s_sim, a_sim, r_sim, s_sim_next)
```

Dyna treats the model as an additional data generator. After every real interaction, it performs $N$ simulated updates, improving **sample efficiency by approximately a factor of $N$**.

### A Key Limitation of Dyna

Dyna assumes a deterministic model that uses $(s, a)$ to predict $s'$ directly. This works in discrete environments such as GridWorld, but model errors accumulate in continuous physics environments such as MuJoCo:

$$\|s_T^{\text{predicted}} - s_T^{\text{true}}\| \sim \mathcal{O}(\epsilon^T)$$

where $\epsilon$ is the one-step prediction error. When $\epsilon = 0.1, T = 10$, the prediction error reaches $10^{10}$ and becomes unusable. Later methods such as PETS and MBPO therefore address the question of how to quantify model error.

## Probabilistic Ensembles with Trajectory Sampling

The key observation behind Probabilistic Ensembles with Trajectory Sampling (Chua et al., 2018) is that the model itself has **two kinds of uncertainty**:

- **Epistemic uncertainty**: uncertainty in the model caused by limited training data, represented by an **ensemble** $M_1, \ldots, M_K$
- **Aleatoric uncertainty**: randomness inherent in the environment, such as a die roll, represented by a **probabilistic output** $p(s' \mid s, a)$

### Model Architecture

PETS uses an ensemble of $K$ probabilistic neural networks:

```python
class PEtsModel:
    def __init__(self, n_models=5):
        self.models = [ProbabilisticNN() for _ in range(n_models)]

    def predict(self, s, a):
        # Each model outputs (mean, variance)
        means, vars = [], []
        for m in self.models:
            mu, sigma = m(s, a)
            means.append(mu); vars.append(sigma)
        return means, vars  # Ensemble disagreement = epistemic uncertainty
```

Planning uses samples from the ensemble rather than a single model, making the policy robust to the possibility that the model is inaccurate.

### Trajectory Sampling Strategy

PETS plans with the **Cross-Entropy Method (CEM)**. At every step, it samples and selects among candidate action sequences $\{a_1, \ldots, a_H\}$:

```python
def cem_planning(model, s, horizon=10, n_samples=500, n_iters=5):
    # Initialize the action distribution
    action_mean = zeros(horizon, action_dim)
    action_var = ones(horizon, action_dim)

    for it in range(n_iters):
        # 1. Sample N action sequences
        action_seqs = sample_normal(action_mean, action_var, n_samples)

        # 2. Roll out each sequence with a randomly selected ensemble model
        rewards = []
        for seq in action_seqs:
            model_id = random_int(0, K)
            s_pred = s
            total_r = 0
            for a in seq:
                s_pred, r = model[model_id].predict(s_pred, a)
                total_r += r
            rewards.append(total_r)

        # 3. Select the top 20% of sequences and update the distribution
        elite = top_k_indices(rewards, k=0.2 * n_samples)
        action_mean = action_seqs[elite].mean(0)
        action_var = action_seqs[elite].var(0)

    return action_mean[0]  # Execute only the first action, following MPC
```

### Experimental Results for PETS

PETS was the first model-based method to match model-free performance on MuJoCo while using **10–50 times fewer samples**. Its cost is expensive planning: every environment step requires 500 model rollouts.

## Model-Based Policy Optimization

The central innovation of Model-Based Policy Optimization (Janner et al., 2019) is to **generate finite-length rollouts with the model**, such as five steps, before returning to the real environment. This prevents model error from growing without bound as rollout length increases.

### Short-Horizon Rollouts

The key MBPO parameter is the rollout length $k$. The paper proves that when the one-step model error is $\epsilon$, the accumulated error of a $k$-step rollout is $\leq k \epsilon$, which remains controllable.

```python
# Short-horizon rollouts keep model error under control
for rollout_step in range(K_short):  # K_short = 5
    a = policy(s_sim)
    s_sim, r = model.predict(s_sim, a)
    replay_buffer.add(s_sim, a, r, s_sim)
    # Reset to a real state every five steps
    if rollout_step % K_short == 0:
        s_sim = real_env.state
```

### MBPO Training Process

```
┌──────────────────────────────────────────────────────┐
│ 1. Train model M with real data                      │
│    M.predict(s, a) → s', r                           │
├──────────────────────────────────────────────────────┤
│ 2. Generate short rollouts (5 steps) with M          │
│    Start: a state s from the real data               │
│    Each step: a = policy(s), s' = M(s, a)            │
│    Add the five (s, a, r, s') tuples to replay       │
├──────────────────────────────────────────────────────┤
│ 3. Update SAC on a replay buffer mixing real and     │
│    synthetic data                                    │
└──────────────────────────────────────────────────────┘
```

MBPO matches the performance of model-free SAC on MuJoCo while using **10–100 times fewer samples**.

### Comparing Three Model-Based RL Algorithms

| Algorithm | Model type             | Planning method         | Sample efficiency | Computational cost |
| --------- | ---------------------- | ----------------------- | ----------------- | ------------------ |
| Dyna      | Deterministic          | One-step synthetic data | ~10×              | Low                |
| PETS      | Probabilistic ensemble | CEM MPC                 | ~50×              | High               |
| MBPO      | Deterministic          | Short rollouts          | ~100×             | Moderate           |

Practical choices:

- **Rapid experiments**: Dyna, which is simple and stable
- **High-precision control**: PETS, for robotic manipulation and precision manufacturing
- **General continuous control**: MBPO, across the full MuJoCo suite

## Section Summary

Model-based RL improves sample efficiency by **learning an environment model**:

1. **Dyna** uses the model for data augmentation and performs N simulated updates after every real interaction
2. **PETS** represents model uncertainty with probabilistic ensembles and maintains robustness through CEM planning
3. **MBPO** uses short-horizon rollouts to limit error accumulation, matching SAC's performance with 100 times fewer samples

The next section, [9.4 Search and World Models](./search-world-models), turns to another branch of model-based methods: explicit search with neural-network evaluation, tracing its development from AlphaGo to Dreamer V3.
