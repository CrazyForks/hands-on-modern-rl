# 11.2 逆强化学习与 GAIL

[11.1](./bc-dagger)直接模仿专家在每个状态采取的动作。遇到专家数据没有覆盖的新状态时，策略仍然缺少判断依据。逆向 RL 改为从整条专家轨迹推断奖励，再用这个奖励训练策略。

本节先说明奖励为什么无法由示范唯一确定，再用最大熵原则选择一个可学习的奖励，随后介绍 GAIL 怎样用判别器绕开配分函数，最后比较三条模仿学习路线的代价与适用条件。

## 1. 为什么要从示范推断奖励

逆向 RL（Inverse RL）假设专家行为来自某个尚未观测到的奖励函数。训练先从轨迹推断这个奖励，再用普通 RL 求解对应策略。

### 1.1 逆向 RL 的基本设定

给定专家轨迹 $\mathcal{D}_{\text{expert}}=\{\tau_1,\ldots,\tau_M\}$，每条 $\tau=(s_0,a_0,\ldots,s_T)$。我们希望找到奖励函数 $r_\psi(s,a)$，使专家轨迹在这个奖励下比其他轨迹更有可能出现：

$$\text{专家策略在 } r_\psi \text{ 下是最优的}$$

只要求“专家最优”无法唯一确定奖励。例如给所有状态同一个常数奖励时，所有策略的回报都相同，专家依然可以被称为最优。最大熵 IRL 通过对轨迹分布增加熵约束，在满足专家特征的解中选择不过度集中的一个。

## 2. 用最大熵原则确定奖励

Ziebart et al. 2008 提出最大熵逆向 RL。它要求轨迹既匹配专家的特征期望，又在满足约束的轨迹之间保留尽可能高的熵：

$$\pi(a \mid s) \propto \exp\left(Q^{\text{soft}}_{r_\psi}(s, a)\right)$$

在最大熵模型中，一条轨迹的概率与它的累计奖励指数成正比：

$$p(\tau \mid r_\psi) = \frac{1}{Z(r_\psi)} \exp\left(\sum_t r_\psi(s_t, a_t)\right)$$

其中 $Z(r_\psi)$ 把所有轨迹的未归一化分数加起来，使概率总和等于 1。对 $M=|\mathcal D_{\text{expert}}|$ 条专家轨迹取对数后，训练目标为：

$$\max_\psi \; \mathcal{L}(\psi) = \sum_{\tau \in \mathcal{D}_{\text{expert}}} \left[\sum_t r_\psi(s_t, a_t)\right] - |\mathcal{D}_{\text{expert}}| \log Z(r_\psi)$$

第一项提高专家轨迹的累计奖励，第二项防止所有轨迹的分数一起无界增大。对参数 $\psi$ 求梯度得到：

$$\nabla_\psi \mathcal{L} = \mathbb{E}_{\tau \sim \text{expert}}\left[\sum_t \nabla_\psi r_\psi(s_t, a_t)\right] - \mathbb{E}_{\tau \sim p(\cdot \mid r_\psi)}\left[\sum_t \nabla_\psi r_\psi(s_t, a_t)\right]$$

第一项来自专家轨迹，第二项来自当前奖励模型诱导的轨迹分布。如果某类状态—动作在专家中更常见，梯度会提高它的奖励；如果它只在当前策略中频繁出现，梯度会降低它的奖励。当两边的特征统计接近时，更新趋于停止。

### 2.1 配分函数为什么难以计算

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

## 3. 用 GAIL 直接匹配访问分布

Generative Adversarial Imitation Learning（Ho & Ermon 2016）借用 GAN 的思想，把逆向 RL 写成判别器 $D_\phi$ 与策略 $\pi_\theta$ 之间的博弈。

### 3.1 判别器与策略怎样交替训练

判别器区分"专家数据"和"策略数据"：

$$\max_\phi \; \mathbb{E}_{(s,a) \sim \mathcal{D}_{\text{expert}}}\left[\log D_\phi(s, a)\right] + \mathbb{E}_{(s,a) \sim \pi_\theta}\left[\log (1 - D_\phi(s, a))\right]$$

策略需要让自己的状态—动作分布更接近专家。若约定 $D_\phi(s,a)$ 表示“样本来自专家”的概率，一种常用的策略目标是最小化：

$$\min_\theta \; \mathbb{E}_{(s,a) \sim \pi_\theta}\left[\log (1-D_\phi(s, a))\right] - \lambda \mathcal{H}(\pi_\theta)$$

第二项是熵正则化，避免策略过早只输出少数动作。实现时常把 $-\log(1-D_\phi(s,a))$ 或 $\log D_\phi(s,a)$ 的等价变体作为隐式奖励；具体符号取决于判别器把专家标成 1 还是 0，代码与公式必须使用同一约定。

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
                # D 表示“来自专家”的概率，所以奖励取 -log(1-D)
                rewards = -F.logsigmoid(-self.disc(states, actions))
            # 喂给任意 RL 算法（这里假设 PPO）
            self.policy.ppo_update(states, actions, rewards, next_states)
```

### 3.2 GAIL 与最大熵 IRL 的联系

固定策略后，二分类判别器的最优解可以写成两个访问分布的比例：

$$D_\phi^*(s, a) = \frac{p_{\text{expert}}(s, a)}{p_{\text{expert}}(s, a) + p_{\pi_\theta}(s, a)}$$

把这个 $D^*$ 代入对数比，可得 $\log D^* - \log(1-D^*)=\log\frac{p_{\text{expert}}}{p_{\pi_\theta}}$。当策略访问分布接近专家时，这个比值接近 1、对数接近 0。GAIL 通过判别器估计这种分布差异，因此不需要显式枚举所有轨迹来计算 $Z$。

## 4. 比较三条模仿学习路线

| 维度             | BC  | MaxEnt IRL        | GAIL                |
| ---------------- | --- | ----------------- | ------------------- |
| 是否解决分布偏移 | ❌  | ✅                | ✅                  |
| 需要环境模型     | ❌  | ✅（或软 Q 近似） | ❌                  |
| 显式奖励函数     | —   | ✅（可解释）      | ❌（隐式）          |
| 计算成本         | 低  | 高（内层 RL）     | 中（对抗训练）      |
| 扩展到高维       | 易  | 难                | 中                  |
| LLM 中的对应     | SFT | —                 | DPO 隐式（见 14.6） |

### 4.1 GAIL 的训练稳定性

GAN 的通病：判别器过强时生成器梯度消失，过弱时学不到信号。实践中常用 Tricks：

- 判别器梯度惩罚（Wasserstein GAIL）
- 判别器更新比策略慢（每 5 步策略更新 1 步判别器）
- 熵正则化系数 $\lambda$ 调到 0.1-1.0 防止策略坍缩
  GAIL 在 MuJoCo 上接近专家水平，但需要数百万步环境交互——**样本效率仍是瓶颈**。这推动了对**离线模仿学习**的研究（如 DemoDICE、DWBC），把专家数据与次优数据结合，无需在线交互。

## 本节总结

逆向 RL（IRL）从专家行为反推奖励函数，最大熵 IRL 解决了 IRL 的不适定问题。GAIL 用 GAN 框架绕开显式 reward 推断，让模仿学习的可扩展性大幅提升。GAIL 启发了后来的对抗 RL 和 RLHF 中的 reward model 训练。

下一节 [11.3 元 RL：MAML、RL²、PEARL、In-Context RL](./meta-rl) 转向另一个问题——**当环境不断变化时，agent 如何快速适应新任务**？
