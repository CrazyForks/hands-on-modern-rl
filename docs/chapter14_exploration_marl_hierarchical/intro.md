# 12.1 内在动机探索：ICM、RND、NGU 与 Agent57

[第 9 章连续控制](../chapter11_continuous_control/intro)研究单个智能体怎样稳定学习，[第 10 章离线 RL](../chapter12_offline_rl/intro)研究无法继续交互时怎样利用固定数据。第 12 章继续放宽经典设定：奖励可能很久才出现，其他智能体也会同时改变环境，任务还可能长到单层策略难以获得训练信号。

本节先处理稀疏奖励。学习顺序是：先说明随机探索为什么失效，再用 ICM 的预测误差构造内在奖励，接着用 RND 估计状态新颖度，最后看 NGU 与 Agent57 怎样同时管理短期和长期探索。

## 1. 稀疏奖励为什么需要内在动机

[第 2 章的探索和利用问题](../chapter03_mdp/bandit) 已经在无状态设定下引入了探索-利用权衡：每个臂的期望回报未知，智能体必须在"拉目前最优臂"（利用）与"拉不确定臂"（探索）之间分配预算。那里介绍的 UCB 会使用上置信界 $U_t(a) = \hat{\mu}_t(a) + c\sqrt{\ln t / N_t(a)}$，把不确定性直接编码进动作价值。

深度 RL 把这个问题放大了。在 Atari 的 _Montezuma's Revenge_ 或 _Pitfall_ 中，智能体从初始状态到第一个奖励需要执行几十个有意义的动作（跳过陷阱、拿钥匙、开门），随机探索 $\epsilon$-greedy 撞到第一个奖励的概率约为 $10^{-18}$。DQN 在这些 **hard-exploration** 游戏上的得分长期为零。

困难来自奖励信号过于稀疏。设智能体从 $s_0$ 出发，到达奖励状态 $s^\star$ 至少需要 $H^\star$ 步。任何只依赖环境奖励 $r_t$ 的更新都要等到第一条成功轨迹出现，而成功轨迹的密度随 $H^\star$ 增大而快速下降。**内在奖励**（intrinsic reward）为到达外部奖励以前的探索提供辅助信号，鼓励智能体访问新颖或难以预测的状态。

形式上，总奖励变为

$$r^{\text{total}}_t = r^{\text{ext}}_t + \beta \cdot r^{\text{int}}_t$$

其中 $r^{\text{ext}}$ 是环境给的外部奖励，$r^{\text{int}}$ 是智能体自己计算的内在奖励，$\beta$ 是权衡系数。关键在于 $r^{\text{int}}$ 必须满足两点：

1. **可计算**：只依赖已观测数据，不需要外部监督
2. **可耗尽**：一个状态被访问够多次后，其内在奖励应衰减到零，避免智能体陷入局部"刷分"

下面两节给出两条主流路线：基于预测误差的 ICM 与基于随机网络蒸馏的 RND。

## 2. 用 ICM 的预测误差驱动探索

### 2.1 从状态预测误差构造内在奖励

Intrinsic Curiosity Module（Pathak et al. 2017）的核心想法：如果智能体无法预测自己的下一步状态，说明这个区域是"陌生"的，值得探索。预测误差越大，内在奖励越高。

直接在像素空间做预测是失败的——下一帧的像素细节太多，预测误差会被无关高频噪声主导。ICM 先用一个**逆向模型** $g_\phi$ 学一个特征空间 $\Phi(s)$：输入 $(s_t, s_{t+1})$，预测动作 $a_t$。这个特征只保留"动作能影响的部分"，过滤掉背景闪烁、镜头抖动等无关变化。

然后训练**前向模型** $f_\psi$：

$$\hat{\Phi}(s_{t+1}) = f_\psi(\Phi(s_t), a_t)$$

内在奖励定义为前向预测误差：

$$r^{\text{int}}_t = \tfrac{1}{2}\|\Phi(s_{t+1}) - \hat{\Phi}(s_{t+1})\|^2$$

整体损失：

$$\mathcal{L} = \mathcal{L}_{\text{policy}}(\theta) + \lambda_{\text{inv}}\,\mathcal{L}_{\text{inv}}(\phi) + \lambda_{\text{fwd}}\,\mathcal{L}_{\text{fwd}}(\psi)$$

```python
class ICM(nn.Module):
    def __init__(self, feat_dim=256, action_dim=6):
        self.encoder = CNNtoMLP(out=feat_dim)              # Φ(s)
        self.inverse = nn.Linear(feat_dim * 2, action_dim) # g_φ
        self.forward_net = MLP(feat_dim + action_dim, feat_dim)

    def intrinsic_reward(self, s, a, s_next):
        phi, phi_next = self.encoder(s), self.encoder(s_next)
        phi_pred = self.forward_net(torch.cat([phi, a], -1))
        return 0.5 * (phi_next - phi_pred).pow(2).sum(-1)

    def forward_loss(self, s, a, s_next):
        phi, phi_next = self.encoder(s), self.encoder(s_next)
        phi_pred = self.forward_net(torch.cat([phi, a], -1))
        return F.mse_loss(phi_pred, phi_next.detach()) + \
               F.cross_entropy(self.inverse(torch.cat([phi, phi_next], -1)), a)
```

ICM 在 _Super Mario Bros_ 等连续控制 + 视觉输入的任务上让智能体在没有外部奖励的情况下穿过整张地图。它的弱点是**噪声电视问题**：如果环境中存在不可预测的随机源（屏幕角落的电视随机播雪花），前向模型永远学不会，内在奖励永远居高，智能体会被钉在电视前不动。

## 3. 用 RND 识别没有访问过的状态

Random Network Distillation（Burda et al. 2018）不再预测下一帧，而是固定一个随机初始化、永不更新的目标网络 $\hat{f}(s)$，再训练预测网络 $f_\psi(s)$ 拟合它：

$$\mathcal{L}_{\text{RND}}(\psi) = \mathbb{E}_s\bigl[\|f_\psi(s) - \hat{f}(s)\|^2\bigr]$$

$$r^{\text{int}}(s) = \|f_\psi(s) - \hat{f}(s)\|^2$$

机制很简单：已访问过的状态被预测网络学过，预测误差小；新状态没见过，预测误差大。随机目标网络本身没有任何语义，它的作用只是提供一个**固定但不可穷尽**的学习信号。

RND 的优势：

- **不需要逆向模型**，省一半计算
- **不依赖动作**，可叠加到任何 model-free 算法上（PPO、A2C）
- **天然抗噪声电视**：随机目标的复杂度有限，预测误差有上界，不会被无限推高

```python
class RND(nn.Module):
    def __init__(self, obs_shape, feat_dim=512):
        # 目标网络：冻结，永不更新
        self.target = CNN(obs_shape, feat_dim)
        for p in self.target.parameters():
            p.requires_grad = False
        # 预测网络：训练
        self.predictor = CNN(obs_shape, feat_dim)

    def intrinsic_reward(self, s):
        with torch.no_grad():
            target = self.target(s)
        pred = self.predictor(s)
        return (pred - target).pow(2).sum(-1)  # 每个状态一个标量
```

Burda et al. 在大规模实验中发现：仅用 RND 内在奖励（无任何外部奖励），PPO 智能体能在多个 Atari 游戏上探索出复杂行为；在外部奖励稀疏的 _Montezuma's Revenge_ 上首次突破零分。

### 3.1 比较 ICM 与 RND

| 维度             | ICM                      | RND                    |
| ---------------- | ------------------------ | ---------------------- |
| 是否依赖动作     | 是（前向模型需要 $a$）   | 否                     |
| 需要训练的子模块 | 编码器 + 逆向 + 前向     | 仅预测器               |
| 噪声电视鲁棒性   | 弱                       | 强                     |
| 计算开销         | 高                       | 中                     |
| 代表应用         | 视觉探索（Mario、DMLab） | Atari hard-exploration |

## 4. 同时管理短期与长期探索

ICM 和 RND 各自解决了部分问题，但仍有共同的盲区：**episodic 记忆缺失**。一个状态可能在单条 episode 内是新颖的（短期），但在跨 episode 看已经访问过千万次（长期）。仅凭神经网络拟合的预测误差无法区分这两种新颖度。Never Give Up（Badia et al. 2020）与后续的 Agent57（Badia et al. 2020）通过同时建模这两个时间尺度的探索，成为 Atari 全套 57 个游戏上**首个超越人类水平**的算法。

### 4.1 NGU 的双时间尺度内在奖励

NGU 的内在奖励由两部分拼接：

$$r^{\text{int}}_t(s) = r^{\text{episodic}}_t(s) \cdot r^{\text{life-long}}_t(s)$$

**短期（episodic）部分** $r^{\text{episodic}}$：维护一个固定容量的 controllable state 表，记录当前 episode 内访问过的状态特征。新状态如果与表中所有状态距离都远（kNN 距离大），则新颖度高；被频繁访问的状态新颖度衰减：

$$r^{\text{episodic}}_t = \frac{1}{\sqrt{k} + c \sum_{i=1}^{k} \frac{1}{\sqrt{N(s_i)}}}$$

其中 $N(s_i)$ 是状态 $s_i$ 的访问次数，$c$ 是衰减常数。这是 kNN 形式简化版。

**长期（life-long）部分** $r^{\text{life-long}}$ 使用 RND 跨 episode 记录新颖度，识别当前回合没有访问、但过去回合已经反复访问的状态。两者相乘后，只有在当前回合和长期训练中都较少出现的状态才能获得较高内在奖励。

### 4.2 用 Retrace 与分布式 Actor 传播奖励

NGU 用 R2D2 的分布式架构（多个 actor 并行采样 + LSTM 处理部分可观测性），用 Retrace($\lambda$) 估计 off-policy Q 值，把内在奖励信号稳定地传到长 horizon。整套系统训练成本极高（数十亿帧、数百 TPU），但它首次证明 Atari hard-exploration 游戏可以被端到端 RL 攻克。

### 4.3 Agent57 怎样自适应选择探索强度

NGU 仍有一个遗留问题：内在奖励权重 $\beta$ 是固定的。在简单游戏（_Pong_、_Space Invaders_）上 $\beta$ 太高会让智能体疯狂探索、不利用已知最优策略；在 hard-exploration 游戏上 $\beta$ 太低又探索不足。**Agent57** 引入一个**自适应策略调度器**：

- 维护一族策略 $\pi_i$，每个有不同的探索参数 $(\beta_i, \gamma_i, c_i)$，分布在"纯利用"到"纯探索"的区间上
- 用 meta-controller 在线估计每个策略的相对回报，优先采样表现好的策略
- 训练时各策略共享 replay buffer 和 Q 网络

这样便不必为每个游戏固定一组 $\beta$。Agent57 是第一个在 Atari 57 套件全部游戏上都超过人类基准分数的算法，说明同一系统可以同时覆盖强调探索和强调利用的任务。

## 本节总结

内在动机为稀疏奖励任务补充了可持续的训练信号。ICM 使用状态预测误差，RND 使用随机目标网络的预测误差，NGU 组合单回合与跨回合的新颖度，Agent57 再根据任务表现选择不同探索强度。这条方法线显著改善了 _Montezuma's Revenge_ 等 hard-exploration 游戏上的表现。

下一节 [12.2 多智能体 RL：CTDE、MADDPG、MAPPO](./marl) 转向另一个挑战——当环境里有多个 agent 同时学习时，非平稳性打破了 MDP 假设。
