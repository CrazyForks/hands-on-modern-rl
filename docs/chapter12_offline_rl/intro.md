# 10.1 离线 RL 的挑战与经典方法

Part II 依靠智能体持续与环境交互来收集经验。许多真实系统只能使用已有日志，新的试错可能昂贵、缓慢或存在安全风险。Part III 从离线强化学习开始，随后把固定数据学习推进到模仿学习、逆向强化学习、元强化学习、探索、多智能体与分层决策。

> [第 9 章](../chapter11_continuous_control/intro) 解决了连续动作与样本效率问题——DDPG/TD3/SAC 通过 replay buffer 复用历史数据，model-based RL 通过环境模型减少真实交互。但所有这些算法仍然允许智能体**继续与环境交互**：replay buffer 里的数据是旧策略采的，新策略采到的新数据会持续加入。本章处理一个更严苛的设定——**当智能体完全不能交互，只能从一个固定的历史数据集学习时，如何训出可靠策略？** 这就是 **Offline RL（离线强化学习）**，也称作 batch RL。它是 LLM 后训练、推荐系统、医疗决策、工业机器人等真实场景的核心范式，并且通过 **Decision Transformer** 这一分支，与现代序列建模（GPT）建立了直接联系。

## 离线 RL 的核心挑战：分布偏移

[第 5 章 DQN](../chapter07_dqn/from-q-to-dqn) 和 [第 9 章 SAC](../chapter11_continuous_control/intro) 都依赖同一个机制：Bellman 备份。无论 on-policy 还是 off-policy，价值函数的更新都写成：

$$y = r + \gamma \cdot \mathbb{E}_{s' \sim P(\cdot \mid s, a)}\left[V(s')\right]$$

在线 RL 中，target 里那个 $V(s')$ 来自未来的探索——即使新策略走到一个没见过的状态，智能体会继续与环境交互、采到新数据，从而修正估值。**离线 RL 没有这个保险。** 数据集 $\mathcal{D} = \{(s, a, r, s')\}$ 由某个行为策略 $\pi_\beta$ 采得，训练时**完全冻结**：

$$\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^{N}, \quad (s, a) \sim d^{\pi_\beta}(s) \pi_\beta(a \mid s)$$

新策略 $\pi_\theta$ 训练完成后部署，但它选择动作的分布 $\pi_\theta(a \mid s)$ 与 $\pi_\beta(a \mid s)$ 不同。**分布偏移（distribution shift）** 由此产生。

### Extrapolation Error 的形式化定义

Fujimoto et al. 2019 在 BCQ 论文中精确刻画了离线 RL 失败的根源。设数据集支撑集为 $\mathcal{D}_\mathcal{A}(s) = \{a : (s, a) \in \text{support}(\pi_\beta(\cdot \mid s))\}$。Bellman 算子在 $a' \notin \mathcal{D}_\mathcal{A}(s')$ 上的取值没有任何监督信号——神经网络在这些 OOD（out-of-distribution）点上 **外推**，结果是任意的。

把估值误差分解为三类来源：

$$\underbrace{Q_\phi(s, a) - Q^\pi(s, a)}_{\text{总误差}} = \underbrace{\epsilon_{\text{stat}}}_{\substack{\text{统计误差}\\\text{(样本有限)}}} + \underbrace{\epsilon_{\text{approx}}}_{\substack{\text{函数逼近误差}\\\text{(网络容量)}}} + \underbrace{\max_{a'} Q_\phi(s', a') - Q^\pi(s', \pi(s'))}_{\text{外推误差 (Extrapolation Error)}}$$

第三项是关键。Q-Learning 的 target 用 $\max_{a'} Q(s', a')$，在 OOD 动作上 $Q$ 可能因为外推给出**虚高的值**，于是策略被引向这些"幻想"动作。

外推误差的累积过程可以递归展开。设 $Q_0$ 是初始估值，Bellman 迭代 $T$ 次后误差满足：

$$\|Q_T - Q^\pi\|_\infty \leq \gamma^T \|Q_0 - Q^\pi\|_\infty + \sum_{k=0}^{T-1} \gamma^k \|\mathcal{T} Q_k - \mathcal{T}^\pi Q_k\|_\infty$$

其中 $\mathcal{T}$ 是数据约束下的 Bellman 算子（含 max），$\mathcal{T}^\pi$ 是真策略算子。当 max 算子在 OOD 上每次产生误差 $\epsilon_{\text{ood}}$，单步误差就以 $\sum \gamma^k \approx 1/(1-\gamma) \approx 100$（$\gamma = 0.99$）的系数累积。在线 RL 中，下一次交互会立即揭露这个错误（实际 reward 很低），Q 被拉回；离线 RL 中没有这种纠错机会，误差在 Bellman 迭代中**指数级累积**。

::: warning 为什么加更多数据救不了
直觉上，扩大数据集覆盖度可以缓解 OOD 问题。但实际上，连续动作空间里无论采多少数据，$\mathcal{D}_\mathcal{A}(s)$ 都是 $|\mathcal{A}|$ 维空间里的稀疏支撑。$a'$ 距离最近数据点的欧氏距离可能很小，但 $Q$ 函数在这个方向上的梯度可以任意大。**外推误差不是数据量的问题，而是 Q-Learning 的 max 算子与函数逼近器组合的结构性缺陷**。
:::

### 离线 RL 的目标函数

有了上面的诊断，离线 RL 的目标可形式化为：在数据集支撑下学一个策略 $\pi_\theta$，使其期望回报尽可能大，但 $\pi_\theta$ **不能偏离 $\pi_\beta$ 太远**——否则就会进入 OOD 区域。所有现代离线 RL 算法都是在这两个目标间求平衡：

$$\max_\theta \; \mathbb{E}_{s \sim \mathcal{D}}\left[Q^\pi(s, \pi_\theta(s))\right] \quad \text{subject to} \quad D(\pi_\theta \| \pi_\beta) \leq \epsilon$$

接下来三节按"如何实现这个约束"分三条路线展开。

## 悲观主义路线：CQL、IQL 与 BCQ

最直接的思路：**让 Q 函数对 OOD 动作悲观**。如果 $Q(s, a)$ 在没见过的 $a$ 上给低值，$\max_a Q(s, a)$ 自然不会选到幻想动作。三大经典算法——BCQ、CQL、IQL——从不同角度实现这一原则。

### 动作空间约束

Batch-Constrained Q-Learning（Fujimoto et al. 2019）是第一个被证明能在连续动作离线数据上稳定的深度算法。核心约束：**target 动作 $a'$ 必须落在 $\pi_\beta$ 的支撑集内**。

BCQ 训一个条件 VAE $\pi_\beta(a \mid s)$ 近似行为策略，采样候选动作 $\{a_i\} \sim \pi_\beta$，再在这些候选上做 max：

$$a' = \arg\max_{a \in \{a_i + \xi \Phi(s, a_i)\}} Q_\phi(s', a)$$

其中 $\Phi(s, a)$ 是一个扰动网络，对采样动作做小幅修正以逼近局部最优。$\xi$ 是扰动幅度。这把"连续动作 argmax"约束在行为策略的高密度区域内。

### 值函数层面的悲观

Conservative Q-Learning（Kumar et al. 2020）从另一个角度切入——不约束动作，而是**直接惩罚 Q 在 OOD 上的值**。在标准 Bellman 误差之外加一个正则项：

$$\mathcal{L}_{\text{CQL}}(Q) = \alpha \left(\mathbb{E}_{s \sim \mathcal{D}}\left[\log \sum_a \exp(Q(s, a))\right] - \mathbb{E}_{(s, a) \sim \mathcal{D}}[Q(s, a)]\right) + \mathcal{L}_{\text{Bellman}}(Q)$$

第一项 $\log \sum_a \exp(Q(s, a))$ 是 logsumexp，对 **所有动作**（包括 OOD）的 Q 做软最大值；让它变小的唯一办法是把所有动作的 Q 都压低。第二项把数据集里实际见过的 $(s, a)$ 的 Q 拉回正常范围。两者的差形成一个"惩罚 gap"——OOD 动作的 Q 被系统性低估。

CQL 的理论保证：学到的 $\hat{Q}$ 是真实 $Q^\pi$ 的**下界**，即 $\hat{Q}(s, a) \leq Q^\pi(s, a)$ 对所有 $(s, a)$ 成立；进一步可以证明 $\hat{Q}$ 在 OOD 动作上的值比 in-distribution 动作低一个 $\mathcal{O}(\alpha)$ 的 gap。因此由 $\hat{Q}$ 推出的策略不会高估任何动作的回报。在实践中 $\alpha$ 用 Lagrangian 自动调节，让保守性恰到好处：

$$\mathcal{L}(\alpha) = -\alpha \cdot \left(\mathbb{E}_s\left[\log\sum_a \exp(\hat{Q}(s, a))\right] - \mathbb{E}_{(s, a) \sim \mathcal{D}}[\hat{Q}(s, a)] - \xi\right)$$

其中 $\xi$ 是目标 gap（如 5.0）。当实际 gap 低于 $\xi$ 时增大 $\alpha$，反之减小，使 gap 自动稳定在目标附近。

```python
class CQL(SAC):
    def critic_loss(self, batch):
        s, a, r, s_next, done = batch
        # 标准 Bellman 误差（继承自 SAC）
        with torch.no_grad():
            a_next = self.actor(s_next)
            q_target = torch.min(self.critic_target1(s_next, a_next),
                                  self.critic_target2(s_next, a_next))
            y = r + self.gamma * (1 - done) * q_target
        bellman_loss = F.mse_loss(self.critic1(s, a), y) + \
                       F.mse_loss(self.critic2(s, a), y)

        # CQL 保守正则
        # 第一项：对随机动作（OOD）做 logsumexp
        rand_a = torch.rand_like(a) * 2 - 1
        q_rand1 = self.critic1(s, rand_a).flatten()
        q_curr1 = self.critic1(s, a).flatten()  # in-dist
        q_next1 = self.critic1(s, a_next).flatten()
        cat_q1 = torch.cat([q_rand1, q_curr1, q_next1], dim=1)
        logsumexp_q1 = torch.logsumexp(cat_q1, dim=1).mean()

        conservative_loss = \
            self.alpha * (logsumexp_q1 - q_curr1.mean()) \
            + self.alpha * (logsumexp_q2 - q_curr2.mean())

        return bellman_loss + conservative_loss
```

### 避免显式 OOD 评估

Implicit Q-Learning（Kostrikov et al. 2022）的洞察更深一层：**根本不需要评估任何 OOD 动作的 Q**。它用一个分位数回归（quantile regression）学 $V(s)$，让 $V$ 偏向数据中较好的动作：

$$\mathcal{L}_V = \mathbb{E}_{(s, a) \sim \mathcal{D}}\left[L_2^\tau(Q_{\bar{\theta}}(s, a) - V_\psi(s))\right]$$

其中 $L_2^\tau(x) = |\tau - \mathbb{1}(x < 0)| \cdot x^2$ 是期望分位数为 $\tau$（通常 $\tau = 0.7$）的分位数损失。这把 $V$ 学成 "数据中较好动作的价值"，而**不需要 max 任何东西**。然后用 advantage $A(s, a) = Q_{\bar{\theta}}(s, a) - V_\psi(s)$ 做 advantage-weighted regression 训练策略：

$$\mathcal{L}_\pi = -\mathbb{E}_{(s, a) \sim \mathcal{D}}\left[\exp(\beta \cdot A(s, a)) \cdot \log \pi_\theta(a \mid s)\right]$$

$\exp(\beta A)$ 给数据中表现好的动作更大权重，让 $\pi_\theta$ 向它们靠拢。$\beta$ 是温度。IQL 完全绕开了 Q-Learning 的 max 算子，因此**不会产生外推误差**——这是它和 CQL 的本质区别。

### 三大算法对比

| 维度               | BCQ             | CQL                 | IQL                  |
| ------------------ | --------------- | ------------------- | -------------------- |
| 约束位置           | 动作空间        | 值函数              | 隐式（分位数 + AWR） |
| 是否评估 OOD 动作  | 否（采样约束）  | 是（logsumexp）     | 否（完全规避）       |
| 额外网络           | VAE $\pi_\beta$ | 无                  | $V$ 网络             |
| 超参敏感           | 高（扰动幅度）  | 中（$\alpha$ 自动） | 低（$\tau, \beta$）  |
| 对中等数据集表现   | 中              | 强                  | 强                   |
| 对稀疏数据集稳定性 | 中              | 偶发不稳定          | 强                   |
| 实现复杂度         | 高              | 中                  | 低                   |

**实战建议**：从 IQL 开始（最稳定、最少调参）；若 baseline 偏低再换 CQL（更激进）；BCQ 已较少作为新 baseline。

## AWAC 与 TD3+BC：保守约束和行为克隆正则化

另一条路线更工程化——**保留 on-policy / off-policy actor-critic 主循环，在策略损失里直接加行为克隆（BC）正则**。这类方法的优势是与第 8 至 9 章的 PPO/SAC 框架兼容，工程改造量极小。

### TD3+BC：BC 正则化的最简形式

Fujimoto & Gu 2021 提出的 TD3+BC 把思想推到极致：在 TD3 的 actor loss 上加一个 BC 项，权重 $\lambda$ 自适应调节：

$$\mathcal{L}_{\text{actor}} = -\mathbb{E}_{s \sim \mathcal{D}}\left[Q(s, \mu_\theta(s))\right] + \lambda \cdot \mathbb{E}_{(s, a) \sim \mathcal{D}}\left[(\mu_\theta(s) - a)^2\right]$$

其中 $\lambda = \frac{\alpha}{\frac{1}{N}\sum_i |Q(s_i, \mu_{\theta_{\text{old}}}(s_i))|}$。分母是当前 Q 值的尺度——这让 $\lambda$ 自动适应不同环境的 reward scale，无需调参。论文里 $\alpha = 2.5$ 在所有 D4RL MuJoCo 任务上都是同一设置。

TD3+BC 的简洁性使它成为离线 RL 的强基线。其表现提示一个反直觉的事实：**很多离线 RL benchmark 上，最朴素的 BC 正则化就能达到接近 CQL/IQL 的性能**。

### 优势加权的 BC

Advantage-Weighted Actor-Critic（Nair et al. 2020）和 IQL 的策略损失有相同的来源——advantage-weighted regression——但 AWAC 用显式 Q 而不是分位数 V：

$$\mathcal{L}_\pi^{\text{AWAC}} = -\mathbb{E}_{(s, a) \sim \mathcal{D}}\left[\underbrace{\exp\left(\frac{A(s, a)}{\beta}\right)}_{\text{advantage 权重}} \cdot \log \pi_\theta(a \mid s)\right]$$

其中 $A(s, a) = Q(s, a) - V(s)$，$\beta$ 是温度。直观地：数据中表现优于平均的动作被放大权重，劣于平均的被压低。AWAC 把 BC 推广为"加权 BC"——只模仿好的部分。

AWAC 的工程亮点是**支持离线到在线的平滑过渡**：先纯离线预训练，再少量在线交互微调。这一点对真实机器人、推荐系统等场景非常实用。

### AWAC 与 IQL：同源的策略损失

仔细比较两个公式：

$$\mathcal{L}_\pi^{\text{AWAC}} = -\mathbb{E}\left[\exp\left(\frac{A(s, a)}{\beta}\right) \log \pi(a \mid s)\right], \quad \mathcal{L}_\pi^{\text{IQL}} = -\mathbb{E}\left[\exp\left(\beta \cdot A(s, a)\right) \log \pi(a \mid s)\right]$$

形式上几乎一致（$\beta$ 的位置不同，但都可以看作温度）。差异在 $A(s, a)$ 的估计：

- **AWAC**：$A = Q_\phi(s, a) - V_\psi(s)$，其中 $Q$ 仍走标准 Bellman 备份（target 里仍有 max $\pi$）
- **IQL**：$A = Q_\phi(s, a) - V_\psi(s)$，但 $Q$ 通过 $V$ 备份（target 用 $V(s')$ 而非 $\max_a Q(s', a)$），$V$ 用分位数回归偏向数据中较好的动作

IQL 通过把 Bellman target 改成 $V(s')$（不再 max），从根源上消除了外推误差的产生路径。AWAC 保留了标准 Bellman target，靠加权 BC 来约束策略——这种约束比 IQL 的隐式约束弱，因此 AWAC 在数据集 Q 值噪声大时更容易踩到 OOD 雷区。

### AWAC vs TD3+BC vs IQL

| 方法   | 策略损失形式                            | 是否需要 $V$ | 在线微调友好 |
| ------ | --------------------------------------- | ------------ | ------------ |
| TD3+BC | $-\!Q + \lambda \|\mu - a\|^2$          | 否           | 中           |
| AWAC   | $-\!w(A) \log \pi$，$w = \exp(A/\beta)$ | 是           | 强           |
| IQL    | $-\!\exp(\beta A) \log \pi$（AWR）      | 是           | 中           |

注意 AWAC 和 IQL 的策略损失结构高度相似，区别在 $A$ 的来源——AWAC 用显式 Q-V 差，IQL 用分位数回归隐式估计。这种细微差别在稀疏数据上对稳定性影响很大。

## 本节总结

本节梳理了离线 RL 的核心挑战（分布偏移与外推误差）与三大保守路线：BCQ 约束动作空间、CQL 惩罚 OOD Q 值、IQL 完全规避 max 算子。这些算法都在 Bellman 框架内做文章。

下一节 [10.2 Decision Transformer、Trajectory Transformer 与 Diffuser](./sequence-modeling) 走另一条路——彻底抛弃 Bellman，把 RL 写成条件序列生成。
