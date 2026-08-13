# 20.2 Code World Model and DeepSWE

In the previous section, we observed the core bottleneck of Meta SWE-RL — **unstable training with long horizons**. Trajectories longer than 16 steps are difficult for RL to learn credit assignment.

A deeper issue is: **each rollout requires running real tests, which is slow and costly**. A trajectory involves multiple `pytest` calls, each taking several seconds to several minutes. If an RL training requires one million rollouts, the total time may be several weeks.

In the second half of 2025, two breakthrough directions emerged:

- **Code World Model (CWM)**: Train a model to "simulate" code execution, avoiding real testing
- **DeepSWE**: Use world model + long sequence RL to train deep agents

This section will discuss these two directions in detail.

## 20.2.1 Code World Model (CWM)

[Code World Model](https://arxiv.org/abs/2510.02387) (CWM, 2025.09) has the core idea: **model code execution as an MDP and train a world model to predict the state changes of code**.

### CWM's MDP Definition

Modeling SWE tasks as an MDP:

| MDP Elements                        | SWE Correspondence                                              |
| ----------------------------------- | --------------------------------------------------------------- |
| State $s_t$                         | Repository code + current modification history + test results   |
| Action $a_t$                        | The model's next step (read file, modify code, run test)        |
| Transition $T(s_{t+1} \| s_t, a_t)$ | Code execution — how the state changes after file modifications |
| Reward $r_t$                        | Step feedback (intermediate state) + final reward (test pass)   |

### Training the World Model

CWM trains an independent **world model** $\hat{T}$:

$$\hat{T}(s_{t+1} | s_t, a_t) \approx T(s_{t+1} | s_t, a_t)$$

This world model is a large language model (LLM), which takes as input $(s_t, a_t)$ and outputs $s_{t+1}$.

Training data:

- Trajectories collected from real SWE tasks
- $(s_t,a_t,s_{t+1})$ triplets as training samples
- Let the world model learn to "predict the next state given the current code state and action"

### Training Process of CWM

```text
┌────────────────────────────────────────────────────────────┐
│ Phase 1: World Model Pre-training                           │
│   - Collect trajectories from real SWE tasks               │
│   - Train the world model to predict code state transitions │
├────────────────────────────────────────────────────────────┤
│ Phase 2: RL with World Model                               │
│   - Policy interacts with the world model                  │
│   - The world model quickly simulates "code execution"    │
│   - No need for real testing; 100x faster                  │
├────────────────────────────────────────────────────────────┤
│ Phase 3: Real Testing Fine-tuning                          │
│   - Fine-tune the policy trained with the world model in  │
│   - the real environment for the final RL                  │
│   - Correct the deviation between the world model and the │
│   - real environment                                        │
└────────────────────────────────────────────────────────────┘
```

### Advantages of CWM

**Advantage 1: Speed**

A world model is an LLM forward—taking a few milliseconds. Real-world testing takes several seconds to several minutes. **CWM accelerates training by 100–1000 times**.

**Advantage 2: Ability to Simulate Failures**

A world model can simulate "what happens if we make this change"—a policy can explore many failure modes within the world model, learning to avoid them.

**Advantage 3: High Data Efficiency**

A world model learns the "rules" of code execution—these rules can generalize to new tasks.

### Limitations of CWM

**Limitation 1: Accuracy of the World Model**

A world model is an LLM, which can make mistakes. If it predicts the wrong "code execution result," the policy will learn an incorrect strategy.

Practical mitigation in industry: **Regularly correct the world model with real-world testing**—every N steps of rollout, use real-world testing ground truth to correct the model.

**Limitation 2: Complex Dependencies**

Code execution involves complex dependencies (library versions, environment variables, external services). A world model struggles to fully simulate these.

**Limitation 3: Training Cost**

Training a world model itself requires a large amount of trajectory data and computational power—more complex than directly training a policy.

### Relationship Between CWM and Model-Based RL

CWM is an application of model-based RL in the domain of Software Engineering (SWE). Classical model-based RL (e.g., MuZero, Dreamer) has already demonstrated value in games and control tasks. CWM brings this idea to the LLM + SWE domain.

Reference: [Chapter 8: Model Planning in Long-Horizon Tasks](../chapter10_ppo/rl-long-horizon-planning) and [24.3 VLA and Embodied World Models](../chapter28_vla/embodied-intelligence/model-based-rl/).

## 20.2.2 DeepSWE and Long Horizon Agents in RL

[DeepSWE-Preview](https://www.together.ai/blog/deepswe) (Agentica × Together AI, 2025.07) is another breakthrough in SWE-RL. Its core contribution is: **training long horizon agents (trajectories longer than 32 steps) with verifiable reward**.

### Core Idea of DeepSWE

The key insight of DeepSWE: **the fundamental reason for instability in long horizon RL is the difficulty of credit assignment**. A 32-step trajectory only has the final test reward. How can this reward be backpropagated to all 32 steps?

DeepSWE solves this with three techniques:

**Technique One: Step-level Reward Shaping**

Instead of only the final reward, DeepSWE gives a shaping reward to each step:

```python
def deep_swe_reward(trajectory, final_test_result):
    # Base reward: final test result
    base_reward = 1.0 if final_test_result else 0.0

    # Shaping reward: "contribution" of each step
    step_rewards = []
    for step in trajectory:
        # Use LLM judge to evaluate if this step is "meaningful"
        step_quality = llm_judge(step)
        step_rewards.append(step_quality)

    # Total reward = base + sum(step rewards)
    return base_reward + sum(step_rewards) * 0.1
```

This shaping allows the model to receive feedback at each step, avoiding the difficulty of credit assignment.

**Technique Two: Value Model**

DeepSWE reintroduces the value model (consistent with the VAPO approach) — [see Chapter 15 on VAPO](../chapter18_grpo/grpo-family).

The value model $V_\phi(s_t)$ estimates the "expected future reward" of the current state. This allows RL to use GAE for credit assignment:

$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \ldots$$

where $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$.

**Technique Three: Hierarchical RL**

Breaking long trajectories into layers:

- **High-level policy**: Decides "which file to fix next" (coarse-grained)
- **Low-level policy**: Decides "how to modify this file" (fine-grained)

The high-level policy uses sparse reward (final test), while the low-level policy uses dense reward (shaping per step).

### Training Process of DeepSWE

```text
┌──────────────────────────────────────────────────────────┐
│ Phase 1: Data Collection                                │
│   - Rollout with the SFT model on SWE-bench             │
│   - Collect trajectories of 32–64 steps                │
├──────────────────────────────────────────────────────────┤
│ Phase 2: World Model Training (similar to CWM)          │
│   - Accelerate subsequent RL                           │
├──────────────────────────────────────────────────────────┤
│ Phase 3: Value Model Training                           │
│   - Train $ V_\phi $ using collected trajectories      │
├──────────────────────────────────────────────────────────┤
│ Phase 4: Hierarchical RL                                │
│   - High-level policy: PPO + sparse reward             │
│   - Low-level policy: GRPO + dense reward              │
├──────────────────────────────────────────────────────────┤
│ Phase 5: Test-time search                              │
│   - Use MCTS or Beam Search during inference           │
│   - Leverage the value model to evaluate intermediate states │
└──────────────────────────────────────────────────────────┘
```

### Achievements of DeepSWE

Performance of DeepSWE on SWE-bench Verified:

| Model                   | SWE-bench Verified |
| ----------------------- | ------------------ |
| Meta SWE-RL             | 41.0%              |
| **DeepSWE (ByteDance)** | **50.0%**          |
| SWE-Lancer (OpenAI)     | 45.0%              |
| Claude Opus 4.5 + Tools | 60%+               |

DeepSWE achieves 50% on open-source models — demonstrating the feasibility of long-horizon reinforcement learning training.

### Relationship Between DeepSWE and VAPO

The design of DeepSWE is highly similar to [ByteDance's VAPO](../chapter18_grpo/grpo-family) — both replace the "no critic" approach of GRPO with a value model. This reflects the consensus within ByteDance Seed that **"long-horizon tasks require critics."**

This also validates the conclusion from [Chapter 15 on GRPO improvements](../chapter18_grpo/grpo-family) — **critic-free is an engineering compromise, not an algorithmic necessity.** On long-horizon tasks (long CoT reasoning, long SWE trajectories), value models have once again proven their value.

## 20.2.3 Test-time Search Integration

CWM and DeepSWE both integrate **test-time search**—using MCTS or Beam Search during inference to improve performance.

### MCTS on CWM

CWM's world model makes MCTS efficient:

```python
def cwm_mcts(issue, model, world_model, depth=10):
    # Perform MCTS on the world model
    root_state = initialize_state(issue)

    for _ in range(N_iter):
        # Selection: Select the most promising child node using UCB
        node = select(root_state)

        # Expansion: Generate action using policy, simulate next state using world model
        action = model.policy(node.state)
        next_state = world_model.predict(node.state, action)

        # Simulation: Quick rollout to termination
        rollout_reward = quick_rollout(next_state, world_model)

        # Backprop: Update node statistics
        backpropagate(node, rollout_reward)

    # Return the best action from the root state
    return best_action(root_state)
```

The entire MCTS is performed on the world model—**no real-world testing is needed**, making it extremely fast.

### Beam Search on DeepSWE

DeepSWE uses Beam Search during inference:

```python
def deep_swe_beam_search(issue, model, value_model, K=4):
    beams = [{"state": init_state(issue), "score": 0}]

    for step in range(MAX_STEPS):
        candidates = []
        for beam in beams:
            # Generate K candidate actions
            actions = model.generate_actions(beam["state"], n=K)

            for action in actions:
                next_state = apply_action(beam["state"], action)
                # Use value model to evaluate
                value = value_model.estimate(next_state)
                candidates.append({
                    "state": next_state,
                    "score": beam["score"] + value
                })

        # Select top-K
        beams = sorted(candidates, key=lambda x: x["score"], reverse=True)[:K]

    return beams[0]["state"]
```

Beam Search allows DeepSWE to trade computational resources for higher accuracy during inference — consistent with [Chapter 19 on Test-time Compute Scaling](../chapter19_reasoning/test-time-scaling).

## 20.2.4 Comparison with Industrial Practices

By mid-2026, mainstream SWE-RL industrial solutions include:

| Solution                    | Representative   | Features               | SWE-bench Verified |
| --------------------------- | ---------------- | ---------------------- | ------------------ |
| Simple GRPO                 | Meta SWE-RL      | Open-source, simple    | 41.0%              |
| + World Model               | Code World Model | Fast training          | ~45%               |
| + Value + Search            | DeepSWE          | Long horizon           | 50.0%              |
| + Multi-agent Collaboration | Claude Opus 4.7  | Closed-source, complex | 65%+               |

As can be seen, **algorithm complexity is positively correlated with performance** — from simple GRPO to multi-agent collaboration, each improvement brings a few percentage points of performance gain.

## Summary

Code World Model and DeepSWE represent two significant breakthroughs in SWE-RL:

- **CWM**: Uses a world model to accelerate training, avoiding the high cost of real testing
- **DeepSWE**: Uses a value model + hierarchical RL + test-time search to handle long horizons

Both approaches reflect a common characteristic of SWE-RL: **long-horizon tasks require more refined algorithms**. Simple GRPO is suitable for short tasks (< 8 steps), but SWE tasks with 16–64-step trajectories require stronger tools.

In the next section, we will examine Self-play SWE-RL — **letting models generate their own training data** — further reducing the reliance on human-generated data.
