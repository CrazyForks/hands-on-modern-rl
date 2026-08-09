# 9.3 Model-Based RL 与 Dyna、PETS、MBPO

> [9.2](./td3-sac) 让连续控制的 model-free 算法达到了稳定可用——SAC、TD3 在 MuJoCo 上训练百万步能给出好策略。但百万步对真实机器人是不可承受的：机械臂的电机磨损、电池续航、安全约束让真实环境采样极其昂贵。**Model-Based RL** 的核心思想是：**学一个环境模型** $\hat{P}(s' \mid s, a), \hat{R}(s, a)$，让策略在模型上训练，把样本效率从百万步降到万步。

## Model-Based 与 Model-Free 的根本区别

前面所有算法（DDPG/TD3/SAC）都是 **model-free**——智能体不试图理解环境，只用环境给的 reward 学习策略。**Model-Based RL** 反其道：先学一个环境模型 $\hat{P}(s' \mid s, a), \hat{R}(s, a)$，再用模型做规划或生成数据。

### 为什么用模型？

**样本效率**。MuJoCo 的物理仿真虽然便宜，但真实机器人实验**每次采样都很贵**（机械臂可能损坏、电池续航、磨损）。Model-free 要几百万步才能训出好策略，真实机器人不可承受。Model-Based 只需几万步——因为模型学完后可以"在模型里"无限采样。

### 三大范式概览

| 范式     | 核心思想            | 代表算法 | 适用场景                     |
| -------- | ------------------- | -------- | ---------------------------- |
| **Dyna** | 模型作为数据增强    | Dyna-Q   | 离散动作，快速训练           |
| **PETS** | 概率集成 + 轨迹采样 | PETS     | 高精度控制，模型不确定性重要 |
| **MBPO** | 短 horizon rollout  | MBPO     | 通用连续控制                 |

下面逐一拆解。

## 模型作为数据增强

Sutton 1990 的经典思路。Dyna 把每次真实交互分成四步，其中第三步训练模型，第四步用模型生成"假"数据加速 model-free 训练：

```python
for step in range(total_steps):
    # 1. 真实交互
    a = policy.select(s)
    s_prime, r = env.step(a)
    replay_buffer.add(s, a, r, s_prime)

    # 2. 用真实数据更新 model-free 算法（如 Q-Learning）
    q_learning_update(replay_buffer.sample())

    # 3. 用真实数据训练环境模型
    model.train(s, a, r, s_prime)

    # 4. 用模型生成"假"数据，再做 N 次 Q-Learning 更新
    for _ in range(N):  # N = 10-100
        s_sim, a_sim = replay_buffer.sample_state_action()
        s_sim_next, r_sim = model.predict(s_sim, a_sim)
        q_learning_update(s_sim, a_sim, r_sim, s_sim_next)
```

Dyna 把模型当成"额外数据生成器"，每次真实交互后做 $N$ 次模拟更新，**样本效率提升约 $N$ 倍**。

### Dyna 的关键限制

Dyna 假设模型是确定性的——用 $(s, a)$ 直接预测 $s'$。这在离散环境（GridWorld）里有效，但在连续物理环境（MuJoCo）里模型误差累积：

$$\|s_T^{\text{predicted}} - s_T^{\text{true}}\| \sim \mathcal{O}(\epsilon^T)$$

其中 $\epsilon$ 是单步预测误差。当 $\epsilon = 0.1, T = 10$ 时，预测误差达到 $10^{10}$——完全不可用。这就是为什么后续工作（PETS、MBPO）都在解决"模型误差如何量化"。

## 概率集成轨迹采样

Probabilistic Ensembles with Trajectory Sampling（Chua et al. 2018）的关键观察：模型本身有**两种不确定性**：

- **认知不确定**（epistemic uncertainty）：因为训练数据有限，模型本身不确定 → 用**集成** $M_1, \ldots, M_K$ 表达
- **偶然不确定**（aleatoric uncertainty）：环境本身有随机性（如骰子）→ 用**概率输出** $p(s' \mid s, a)$ 表达

### 模型架构

PETS 的模型是 $K$ 个概率神经网络的集成：

```python
class PEtsModel:
    def __init__(self, n_models=5):
        self.models = [ProbabilisticNN() for _ in range(n_models)]

    def predict(self, s, a):
        # 每个模型输出 (mean, var)
        means, vars = [], []
        for m in self.models:
            mu, sigma = m(s, a)
            means.append(mu); vars.append(sigma)
        return means, vars  # 集成散度 = epistemic uncertainty
```

规划时不是用单个模型，而是从集成中随机采样，让策略对"模型可能错"保持鲁棒。

### Trajectory Sampling 策略

PETS 用 **CEM**（Cross-Entropy Method）做规划：在每一步，对候选动作序列 $\{a_1, \ldots, a_H\}$ 做采样 + 选择：

```python
def cem_planning(model, s, horizon=10, n_samples=500, n_iters=5):
    # 初始化动作分布
    action_mean = zeros(horizon, action_dim)
    action_var = ones(horizon, action_dim)

    for it in range(n_iters):
        # 1. 采样 N 条动作序列
        action_seqs = sample_normal(action_mean, action_var, n_samples)

        # 2. 用模型 rollout，每条序列用随机一个集成模型
        rewards = []
        for seq in action_seqs:
            model_id = random_int(0, K)
            s_pred = s
            total_r = 0
            for a in seq:
                s_pred, r = model[model_id].predict(s_pred, a)
                total_r += r
            rewards.append(total_r)

        # 3. 选 top 20% 的序列，更新分布
        elite = top_k_indices(rewards, k=0.2 * n_samples)
        action_mean = action_seqs[elite].mean(0)
        action_var = action_seqs[elite].var(0)

    return action_mean[0]  # 只执行第一步（MPC 思想）
```

### PETS 的实验结果

PETS 在 MuJoCo 上首次让 model-based 达到 model-free 同等性能，且采样步数减少 **10-50 倍**。但 PETS 的代价是**规划时计算昂贵**——每步要做 500 次模型 rollout。

## 模型策略迭代

Model-Based Policy Optimization（Janner et al. 2019）的核心创新：**用模型生成长度有限的 rollout**（如 5 步），然后切换回真实环境。这避免了模型误差随 rollout 长度累积爆炸。

### 短 horizon rollout

MBPO 的关键参数是 rollout 长度 $k$。论文证明：当单步模型误差为 $\epsilon$ 时，$k$ 步 rollout 的累积误差 $\leq k \epsilon$，可控。

```python
# 短 horizon rollout 与 模型误差可控
for rollout_step in range(K_short):  # K_short = 5
    a = policy(s_sim)
    s_sim, r = model.predict(s_sim, a)
    replay_buffer.add(s_sim, a, r, s_sim)
    # 关键：每 5 步必须"重置"到真实状态
    if rollout_step % K_short == 0:
        s_sim = real_env.state
```

### MBPO 训练流程

```
┌────────────────────────────────────────────┐
│ 1. 用真实数据训练模型 M                     │
│    M.predict(s, a) → s', r                  │
├────────────────────────────────────────────┤
│ 2. 用 M 生成短 rollout（5 步）              │
│    起点：真实数据中的某个 s                  │
│    每步：a = policy(s), s' = M(s, a)        │
│    结果：(s, a, r, s') × 5 加入 replay      │
├────────────────────────────────────────────┤
│ 3. 在 replay buffer（混合真假）上 SAC 更新  │
└────────────────────────────────────────────┘
```

MBPO 在 MuJoCo 上达到 model-free SAC 同等性能，但**采样步数减少 10-100 倍**。

### Model-Based RL 三大算法对比

| 算法 | 模型类型 | 规划方式   | 样本效率 | 计算成本 |
| ---- | -------- | ---------- | -------- | -------- |
| Dyna | 确定性   | 1 步假数据 | ~10×     | 低       |
| PETS | 概率集成 | CEM MPC    | ~50×     | 高       |
| MBPO | 确定性   | 短 rollout | ~100×    | 中       |

实战选择：

- **快速实验**：Dyna（简单稳定）
- **高精度控制**：PETS（机器人操作、精密加工）
- **通用连续控制**：MBPO（MuJoCo 全套环境）

## 本节总结

Model-Based RL 通过**学习环境模型** 提升样本效率：

1. **Dyna** 把模型作为数据增强，每次真实交互后做 N 次模拟更新
2. **PETS** 用概率集成表达模型不确定性，CEM 规划保持鲁棒
3. **MBPO** 用短 horizon rollout 避免误差累积，达到 SAC 性能但采样少 100×

下一节 [9.4 AlphaZero、MuZero 与 Dreamer V3](./search-world-models) 转向 model-based 的另一旗舰路线——显式搜索 + 神经网络估值，从 AlphaGo 到 Dreamer V3 的演进。
