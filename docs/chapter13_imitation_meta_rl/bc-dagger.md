# 11.1 行为克隆与交互式模仿学习

[第 10 章离线强化学习](../chapter12_offline_rl/intro)使用固定历史数据改进策略，但数据中仍然保留奖励。模仿学习面对的条件更少：数据只告诉我们专家在某个状态采取了什么动作，没有现成的奖励函数解释这个动作为什么好。

第 11 章沿着三步展开：先用行为克隆与 DAgger 直接学习专家动作，再用逆强化学习与 GAIL 从示范推断奖励，最后进入 MAML、RL²、PEARL 与上下文强化学习，研究策略如何快速适应新任务。本节先回答四个问题：行为克隆怎样训练，误差为什么会累积，DAgger 怎样收集错误状态，以及几种路线各自需要什么数据。

## 1. 把专家示范写成监督学习

[第 6 章策略梯度](../chapter08_policy_gradient/reinforce)假设环境提供 reward。但很多真实任务中我们只有**专家示范**——人类驾驶员的轨迹、熟练工人的操作记录、高质量问答对。**模仿学习**直接从示范学策略，跳过奖励函数的设计。

### 1.1 行为克隆的训练目标

最直接的方法是把专家数据写成监督学习样本：状态 $s$ 是输入，专家动作 $a$ 是标签。策略给专家动作的概率越高，损失越小：

$$\mathcal{L}_{BC}(\theta) = -\mathbb{E}_{(s, a) \sim \mathcal{D}_{\text{expert}}}\left[\log \pi_\theta(a \mid s)\right]$$

其中 $\mathcal{D}_{\text{expert}}=\{(s_i,a_i)\}_{i=1}^N$ 是专家示范数据集，$\pi_\theta(a\mid s)$ 是策略在状态 $s$ 选择专家动作 $a$ 的概率。负号把“提高专家动作概率”变成最小化问题。离散动作通常使用交叉熵，连续动作可以改用均方误差或概率分布的负对数似然。LLM 的监督微调也使用同样的条件似然目标，只是动作变成了下一个 token。

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

## 2. 为什么行为克隆会累积错误

BC 训练时看到的是专家访问的状态分布 $d_{\text{expert}}(s)$，部署时访问的却是当前策略产生的 $d_{\pi_\theta}(s)$。一次小错误可能把智能体带到训练集中没有出现的状态，后续动作就更容易继续出错。

先看一个简单数字。若每一步在专家状态上的错误率是 $\epsilon=0.01$，连续执行 $T=100$ 步都不出错的概率约为 $(1-0.01)^{100}\approx0.366$。这还没有计入偏离专家轨迹后错误率继续升高的影响。DAgger 论文（Ross et al. 2011）把累计任务代价写成 $O(T^2\epsilon)$ 量级：

$$\mathbb{E}\left[\sum_{t=0}^T \mathbb{1}[\pi_\theta(s_t) \neq \pi^*(s_t)]\right] \leq O(T^2 \epsilon)$$

这里的 $T$ 是任务长度，$\epsilon$ 是监督学习误差。$T^2$ 表示早期错误不仅影响当前一步，还会改变后面许多步看到的状态。任务越长，只在专家状态上训练的代价越明显。

## 3. 用 DAgger 收集策略真正访问的状态

Dataset Aggregation 直接补充策略会访问、而专家数据没有覆盖的状态。当前策略先执行任务，专家再为这些状态提供正确动作。

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

DAgger 把当前策略实际访问的状态加入数据集，再请专家给这些状态标注动作。训练数据因此逐渐覆盖 $d_{\pi_\theta}$。在无遗憾在线学习等条件下，累计代价可以从 BC 的 $O(T^2\epsilon)$ 改善到 $O(T\epsilon)$ 量级；代价是训练期间必须反复调用专家。

## 4. 比较 BC、DAgger 与 GAIL

| 方法   | 训练数据来源          | 是否解决分布偏移 | 需要专家在线标注      |
| ------ | --------------------- | ---------------- | --------------------- |
| BC     | 仅离线专家数据        | ❌               | ❌                    |
| DAgger | 专家 + 策略访问的状态 | ✅               | ✅（关键限制）        |
| GAIL   | 专家 + 策略 rollout   | ✅（隐式）       | ❌（只需状态-动作对） |

DAgger 的工程瓶颈是**需要专家在线交互**。人类驾驶员很难实时为策略访问到的异常状态标注正确动作。这推动了下一节从示范推断奖励的逆向 RL 路线。

## 本节总结

行为克隆（BC）是最朴素的模仿学习——把专家轨迹当作监督数据训练策略。但它有**分布漂移**问题：训练时只在专家状态分布上学习，部署时一旦偏离就再也回不来。DAgger 通过让专家纠正 agent 的实际轨迹解决这个问题。

下一节 [11.2 逆强化学习与 GAIL](./irl-gail) 不再直接模仿动作，而是**从专家行为反推奖励函数**——这就是逆强化学习（IRL）。
