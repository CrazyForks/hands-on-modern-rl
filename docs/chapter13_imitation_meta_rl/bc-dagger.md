# 第 11 章 · 模仿学习、反向 RL 与元 RL

> [第 10 章 离线强化学习](../chapter12_offline_rl/intro)处理"只有历史数据、不能交互"的场景，但仍假设数据带有显式奖励信号。本章处理两种更极端的情形：(1) **完全没有奖励函数**——只有专家示范轨迹，怎么办？(2) **环境本身在不断变化**——智能体必须学会"快速适应新任务"。前者引出**模仿学习（Imitation Learning, IL）**与**反向 RL（Inverse RL）**，后者引出**元 RL（Meta-RL）**。两者最终在 LLM 时代合流：SFT 本质是行为克隆，InstructGPT 三阶段可重写为 BC + RL + RL，而 In-Context RL 揭示了"RL 算法本身可被蒸馏进 transformer"。

## 13.1 行为克隆与 DAgger

[第 6 章策略梯度](../chapter08_policy_gradient/reinforce)假设环境提供 reward。但很多真实任务中我们只有**专家示范**——人类驾驶员的轨迹、熟练工人的操作记录、高质量问答对。**模仿学习**直接从示范学策略，跳过奖励函数的设计。

### 监督学习的视角

最朴素的方案是把模仿学习当成监督学习：把 $(s_t, a_t)$ 当作 (输入, 标签)，最小化负对数似然：

$$\mathcal{L}_{BC}(\theta) = -\mathbb{E}_{(s, a) \sim \mathcal{D}_{\text{expert}}}\left[\log \pi_\theta(a \mid s)\right]$$

其中 $\mathcal{D}_{\text{expert}} = \{(s_i, a_i)\}_{i=1}^N$ 是专家示范数据集。这就是 LLM 监督微调（SFT）的标准损失。

```python
def behavior_cloning_step(policy_net, expert_batch):
    states, actions = expert_batch
    log_probs = policy_net.log_prob(states, actions)
    loss = -log_probs.mean()  # 负对数似然
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

### BC 的致命缺陷

BC 训练时专家状态分布是 $d_{\text{expert}}(s)$，但部署时智能体访问的是 $d_{\pi_\theta}(s)$。一旦策略在某一步犯了小错，状态就偏离专家轨迹，进入**从未见过的区域**，下一步错误概率更高——误差**指数级累积**。

形式化地，假设每步错误概率为 $\epsilon$，$T$ 步后仍在专家分布附近的概率约为 $(1-\epsilon)^T \to 0$。DAgger 论文（Ross et al. 2011）证明了 BC 的期望误差上界：

$$\mathbb{E}\left[\sum_{t=0}^T \mathbb{1}[\pi_\theta(s_t) \neq \pi^*(s_t)]\right] \leq O(T^2 \epsilon)$$

误差随 horizon **平方级放大**。这就是为什么纯行为克隆的自动驾驶在长程任务上几乎不可用。

### DAgger 与 迭代收集"失败状态"

Dataset Aggregation 的核心洞察：与其让智能体在专家没见过的状态下挣扎，不如**主动收集这些失败状态，请专家标注**。

```python
def dagger(env, expert, policy_net, n_iterations=20, n_traj_per_iter=50):
    dataset = []
    for it in range(n_iterations):
        # 1. 用当前策略 rollout（注意：不是用专家！）
        trajectories = []
        for _ in range(n_traj_per_iter):
            s = env.reset()
            traj = []
            done = False
            while not done:
                # β 混合：早期多用专家保证安全，后期多用策略
                beta = max(0.0, 1.0 - it / 10)
                if np.random.rand() < beta:
                    a = expert(s)
                else:
                    a = policy_net.act(s)
                s_next, r, done, _ = env.step(a)
                traj.append((s, a))
                s = s_next
            trajectories.append(traj)

        # 2. 关键：对策略访问到的状态（包括失败状态）请专家重新标注
        for traj in trajectories:
            for s, _ in traj:
                a_expert = expert(s)
                dataset.append((s, a_expert))

        # 3. 用扩展后的数据集重训策略
        train_bc(policy_net, dataset)
```

DAgger 让训练分布从 $d_{\text{expert}}$ 逐渐逼近 $d_{\pi_\theta}$——**解决分布偏移的根本病因**。理论上 DAgger 的误差上界降为 $O(T \epsilon)$，线性增长，远好于 BC 的 $O(T^2 \epsilon)$。

| 方法   | 训练数据来源          | 是否解决分布偏移 | 需要专家在线标注      |
| ------ | --------------------- | ---------------- | --------------------- |
| BC     | 仅离线专家数据        | ❌               | ❌                    |
| DAgger | 专家 + 策略访问的状态 | ✅               | ✅（关键限制）        |
| GAIL   | 专家 + 策略 rollout   | ✅（隐式）       | ❌（只需状态-动作对） |

DAgger 的工程瓶颈是**需要专家在线交互**。人类驾驶员很难实时为 AI 生成的"奇怪状态"标注正确动作。这推动了下一节"从示范反推奖励"的反向 RL 路线。

## 本节总结

行为克隆（BC）是最朴素的模仿学习——把专家轨迹当作监督数据训练策略。但它有**分布漂移**问题：训练时只在专家状态分布上学习，部署时一旦偏离就再也回不来。DAgger 通过让专家纠正 agent 的实际轨迹解决这个问题。

下一节 [13.2 逆向 RL 与 GAIL](./irl-gail) 不再直接模仿动作，而是**从专家行为反推奖励函数**——这就是逆向强化学习（IRL）。
