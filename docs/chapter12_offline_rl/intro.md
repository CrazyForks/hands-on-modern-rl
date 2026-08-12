# 10.1 离线数据与分布偏移

Part II 依靠智能体持续与环境交互来收集经验。许多真实系统只能使用已有日志，新的试错可能昂贵、缓慢或存在安全风险。Part III 从离线强化学习开始，随后把固定数据学习推进到模仿学习、逆向强化学习、元强化学习、探索、多智能体与分层决策。

[第 9 章](../chapter11_continuous_control/intro)中的 DDPG、TD3 和 SAC 可以用 replay buffer 复用历史数据，基于模型的强化学习也能借助环境模型减少真实交互。不过，这些方法仍允许新策略继续采样并修正旧经验。离线强化学习取消了这条反馈通道：训练期间只能使用一份固定数据集。

本节沿着三个问题展开：固定数据为什么会导致分布偏移，BCQ、CQL 与 IQL 怎样限制错误估值，AWAC 与 TD3+BC 又怎样把行为克隆加入策略更新。理解这条线以后，[10.2](./sequence-modeling) 才会转向 Decision Transformer 的序列建模路线。

## 1. 固定数据为什么会产生分布偏移

[第 5 章 DQN](../chapter07_dqn/from-q-to-dqn) 和 [第 9 章 SAC](../chapter11_continuous_control/intro) 都会用下一状态的估值更新当前状态。先写出这个一步目标：

$$y = r + \gamma \cdot \mathbb{E}_{s' \sim P(\cdot \mid s, a)}\left[V(s')\right]$$

这行公式可以从左向右读：$y$ 是本次要拟合的目标，$r$ 是当前动作已经得到的奖励，$V(s')$ 是下一状态之后的长期价值，$\gamma$ 控制未来价值在目标中占多大比重。在线训练即使暂时高估了某个新状态，策略以后仍有机会访问它，并用真实奖励修正估值。

在线 RL 中，target 里那个 $V(s')$ 来自未来的探索——即使新策略走到一个没见过的状态，智能体会继续与环境交互、采到新数据，从而修正估值。**离线 RL 没有这个保险。** 数据集 $\mathcal{D} = \{(s, a, r, s')\}$ 由某个行为策略 $\pi_\beta$ 采得，训练时**完全冻结**：

$$\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^{N}, \quad (s, a) \sim d^{\pi_\beta}(s) \pi_\beta(a \mid s)$$

新策略 $\pi_\theta$ 训练完成后部署，但它选择动作的分布 $\pi_\theta(a \mid s)$ 与 $\pi_\beta(a \mid s)$ 不同。**分布偏移（distribution shift）** 由此产生。

### 1.1 外推误差从哪里产生

Fujimoto et al. 2019 在 BCQ 论文中精确刻画了离线 RL 失败的根源。设数据集支撑集为 $\mathcal{D}_\mathcal{A}(s) = \{a : (s, a) \in \text{support}(\pi_\beta(\cdot \mid s))\}$。Bellman 算子在 $a' \notin \mathcal{D}_\mathcal{A}(s')$ 上的取值没有任何监督信号——神经网络在这些 OOD（out-of-distribution）点上 **外推**，结果是任意的。

为了看清问题来自哪里，可以把估值误差按来源写成一个示意分解：

$$\underbrace{Q_\phi(s, a) - Q^\pi(s, a)}_{\text{总误差}} = \underbrace{\epsilon_{\text{stat}}}_{\substack{\text{统计误差}\\\text{(样本有限)}}} + \underbrace{\epsilon_{\text{approx}}}_{\substack{\text{函数逼近误差}\\\text{(网络容量)}}} + \underbrace{\max_{a'} Q_\phi(s', a') - Q^\pi(s', \pi(s'))}_{\text{外推误差 (Extrapolation Error)}}$$

前两项在在线和离线训练中都会出现。第三项只在最大化操作选中了缺少数据支持的动作时出现：网络可能碰巧给这个动作很高的 $Q$ 值，$\max$ 又会优先选中它，于是这个未经验证的估值进入下一轮目标。

外推误差的累积过程可以递归展开。设 $Q_0$ 是初始估值，Bellman 迭代 $T$ 次后误差满足：

$$\|Q_T - Q^\pi\|_\infty \leq \gamma^T \|Q_0 - Q^\pi\|_\infty + \sum_{k=0}^{T-1} \gamma^k \|\mathcal{T} Q_k - \mathcal{T}^\pi Q_k\|_\infty$$

其中 $\mathcal{T}$ 表示带动作最大化的 Bellman 更新，$\mathcal{T}^\pi$ 表示按真实策略计算的更新。右边第一项是初始误差，乘上 $\gamma^T$ 后会逐渐衰减；求和项是每一轮新引入的误差。若每轮都产生大小接近 $\epsilon_{\text{ood}}$ 的误差，它们的总影响会被几何级数放大到约 $\epsilon_{\text{ood}}/(1-\gamma)$。例如 $\gamma=0.99$ 时，放大系数接近 100。这里是**误差反复累加**，并非误差值本身指数增长。

::: warning 为什么加更多数据救不了
扩大数据覆盖可以减少 OOD 动作，却很难在连续动作空间中覆盖每个可能的 $a$。只要更新仍会在缺少数据的区域取最大值，外推误差就可能出现。因此，数据覆盖和保守更新需要同时处理。
:::

### 1.2 离线 RL 要同时优化什么

有了上面的诊断，离线 RL 的目标可形式化为：在数据集支撑下学一个策略 $\pi_\theta$，使其期望回报尽可能大，但 $\pi_\theta$ **不能偏离 $\pi_\beta$ 太远**——否则就会进入 OOD 区域。所有现代离线 RL 算法都是在这两个目标间求平衡：

$$\max_\theta \; \mathbb{E}_{s \sim \mathcal{D}}\left[Q^\pi(s, \pi_\theta(s))\right] \quad \text{subject to} \quad D(\pi_\theta \| \pi_\beta) \leq \epsilon$$

下面先看怎样在动作空间或价值函数中实现这个约束，再看怎样直接把行为克隆加入策略损失。

## 2. 用保守估值限制数据集外动作

最直接的思路：**让 Q 函数对 OOD 动作悲观**。如果 $Q(s, a)$ 在没见过的 $a$ 上给低值，$\max_a Q(s, a)$ 自然不会选到幻想动作。三大经典算法——BCQ、CQL、IQL——从不同角度实现这一原则。

### 2.1 BCQ：把动作限制在数据分布附近

Batch-Constrained Q-Learning（Fujimoto et al. 2019）是第一个被证明能在连续动作离线数据上稳定的深度算法。核心约束：**target 动作 $a'$ 必须落在 $\pi_\beta$ 的支撑集内**。

BCQ 训一个条件 VAE $\pi_\beta(a \mid s)$ 近似行为策略，采样候选动作 $\{a_i\} \sim \pi_\beta$，再在这些候选上做 max：

$$a' = \arg\max_{a \in \{a_i + \xi \Phi(s, a_i)\}} Q_\phi(s', a)$$

其中 $\Phi(s, a)$ 是一个扰动网络，对采样动作做小幅修正以逼近局部最优。$\xi$ 是扰动幅度。这把"连续动作 argmax"约束在行为策略的高密度区域内。

### 2.2 CQL：压低数据集外动作的价值

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

### 2.3 IQL：避免显式评估数据集外动作

Implicit Q-Learning（Kostrikov et al. 2022）避开了对数据集外动作取最大值。它用 expectile regression（期望分位回归）学习 $V(s)$，让 $V$ 偏向数据中价值较高的动作：

$$\mathcal{L}_V = \mathbb{E}_{(s, a) \sim \mathcal{D}}\left[L_2^\tau(Q_{\bar{\theta}}(s, a) - V_\psi(s))\right]$$

这里先计算残差 $x=Q_{\bar{\theta}}(s,a)-V_\psi(s)$，再用

$$L_2^\tau(x) = |\tau - \mathbb{1}(x < 0)| \cdot x^2$$

对正残差和负残差赋予不同权重。这叫 **expectile loss（期望分位损失）**。当 $\tau=0.7$ 时，$V(s)$ 会更靠近数据中较高的 $Q(s,a)$，但训练过程仍只使用数据集已经出现的动作。得到 $V$ 后，再定义优势 $A(s,a)=Q_{\bar\theta}(s,a)-V_\psi(s)$，并训练策略：

$$\mathcal{L}_\pi = -\mathbb{E}_{(s, a) \sim \mathcal{D}}\left[\exp(\beta \cdot A(s, a)) \cdot \log \pi_\theta(a \mid s)\right]$$

若 $A(s,a)>0$，说明这个动作在数据中优于当前状态的基准价值，指数权重就大于 1；若 $A(s,a)<0$，它的模仿权重就会降低。$\beta$ 控制这种差别被放大多少。IQL 不在数据集外动作上执行 $\max$，因此避开了这条外推误差路径。CQL 会主动压低数据集外动作的价值，IQL 则只从数据中的动作学习 $Q$、$V$ 和策略。

### 2.4 比较 BCQ、CQL 与 IQL

| 维度               | BCQ             | CQL                 | IQL                  |
| ------------------ | --------------- | ------------------- | -------------------- |
| 约束位置           | 动作空间        | 值函数              | 隐式（分位数 + AWR） |
| 是否评估 OOD 动作  | 否（采样约束）  | 是（logsumexp）     | 否（避免显式查询）   |
| 额外网络           | VAE $\pi_\beta$ | 无                  | $V$ 网络             |
| 超参敏感           | 高（扰动幅度）  | 中（$\alpha$ 自动） | 低（$\tau, \beta$）  |
| 对中等数据集表现   | 中              | 强                  | 强                   |
| 对稀疏数据集稳定性 | 中              | 偶发不稳定          | 强                   |
| 实现复杂度         | 高              | 中                  | 低                   |

第一次实现可以先用 IQL 建立基线，因为它的更新只依赖数据集内动作；需要显式控制保守程度时再比较 CQL。BCQ 适合帮助理解“限制候选动作”这条路线。

## 3. 用行为克隆约束策略更新

另一条路线更工程化——**保留 on-policy / off-policy actor-critic 主循环，在策略损失里直接加行为克隆（BC）正则**。这类方法的优势是与第 8 至 9 章的 PPO/SAC 框架兼容，工程改造量极小。

### 3.1 TD3+BC：在策略损失中加入行为克隆

Fujimoto & Gu 2021 提出的 TD3+BC 采用了直接的实现：在 TD3 的 Actor 损失上增加一个行为克隆项，并自适应调节权重 $\lambda$：

$$\mathcal{L}_{\text{actor}} = -\mathbb{E}_{s \sim \mathcal{D}}\left[Q(s, \mu_\theta(s))\right] + \lambda \cdot \mathbb{E}_{(s, a) \sim \mathcal{D}}\left[(\mu_\theta(s) - a)^2\right]$$

其中 $\lambda = \frac{\alpha}{\frac{1}{N}\sum_i |Q(s_i, \mu_{\theta_{\text{old}}}(s_i))|}$。分母是当前 Q 值的尺度——这让 $\lambda$ 自动适应不同环境的 reward scale，无需调参。论文里 $\alpha = 2.5$ 在所有 D4RL MuJoCo 任务上都是同一设置。

TD3+BC 的简洁性使它成为离线 RL 的强基线。其表现提示一个反直觉的事实：**很多离线 RL benchmark 上，最朴素的 BC 正则化就能达到接近 CQL/IQL 的性能**。

### 3.2 AWAC：提高优质动作的模仿权重

Advantage-Weighted Actor-Critic（Nair et al. 2020）和 IQL 的策略损失有相同的来源——advantage-weighted regression——但 AWAC 用显式 Q 而不是分位数 V：

$$\mathcal{L}_\pi^{\text{AWAC}} = -\mathbb{E}_{(s, a) \sim \mathcal{D}}\left[\underbrace{\exp\left(\frac{A(s, a)}{\beta}\right)}_{\text{advantage 权重}} \cdot \log \pi_\theta(a \mid s)\right]$$

其中 $A(s, a) = Q(s, a) - V(s)$，$\beta$ 是温度。直观地：数据中表现优于平均的动作被放大权重，劣于平均的被压低。AWAC 把 BC 推广为"加权 BC"——只模仿好的部分。

AWAC 的工程亮点是**支持离线到在线的平滑过渡**：先纯离线预训练，再少量在线交互微调。这一点对真实机器人、推荐系统等场景非常实用。

### 3.3 AWAC 与 IQL 的差别在哪里

仔细比较两个公式：

$$\mathcal{L}_\pi^{\text{AWAC}} = -\mathbb{E}\left[\exp\left(\frac{A(s, a)}{\beta}\right) \log \pi(a \mid s)\right], \quad \mathcal{L}_\pi^{\text{IQL}} = -\mathbb{E}\left[\exp\left(\beta \cdot A(s, a)\right) \log \pi(a \mid s)\right]$$

形式上几乎一致（$\beta$ 的位置不同，但都可以看作温度）。差异在 $A(s, a)$ 的估计：

- **AWAC**：$A = Q_\phi(s, a) - V_\psi(s)$，其中 $Q$ 仍走标准 Bellman 备份（target 里仍有 max $\pi$）
- **IQL**：$A = Q_\phi(s, a) - V_\psi(s)$，但 $Q$ 通过 $V$ 备份（target 用 $V(s')$ 而非 $\max_a Q(s', a)$），$V$ 用分位数回归偏向数据中较好的动作

IQL 通过把 Bellman target 改成 $V(s')$（不再 max），从根源上消除了外推误差的产生路径。AWAC 保留了标准 Bellman target，靠加权 BC 来约束策略——这种约束比 IQL 的隐式约束弱，因此 AWAC 在数据集 Q 值噪声大时更容易踩到 OOD 雷区。

### 3.4 比较 AWAC、TD3+BC 与 IQL

| 方法   | 策略损失形式                            | 是否需要 $V$ | 在线微调友好 |
| ------ | --------------------------------------- | ------------ | ------------ |
| TD3+BC | $-\!Q + \lambda \|\mu - a\|^2$          | 否           | 中           |
| AWAC   | $-\!w(A) \log \pi$，$w = \exp(A/\beta)$ | 是           | 强           |
| IQL    | $-\!\exp(\beta A) \log \pi$（AWR）      | 是           | 中           |

注意 AWAC 和 IQL 的策略损失结构高度相似，区别在 $A$ 的来源——AWAC 用显式 Q-V 差，IQL 用分位数回归隐式估计。这种细微差别在稀疏数据上对稳定性影响很大。

## 本节总结

本节从分布偏移和外推误差出发，比较了三种处理方式：BCQ 把候选动作限制在数据附近，CQL 压低数据外动作的估值，IQL 避免对数据外动作显式取最大值。三者仍然使用 Bellman 更新，差别在于怎样阻止不可靠的估值进入策略改进。

下一节 [10.2 基于序列建模的离线强化学习](./sequence-modeling) 走另一条路——彻底抛弃 Bellman，把 RL 写成条件序列生成。
