# 10.2 Decision Transformer、Trajectory Transformer 与 Diffuser

[10.1](./intro)仍在 Bellman 框架中估计价值，只是限制策略或价值函数不要离开固定数据。Decision Transformer 提出了另一种做法：把状态、动作与回报排成序列，直接学习在给定目标回报时应该采取什么动作。

本节按四步展开：先用 return-to-go 建立 Decision Transformer，再说明 Trajectory Transformer 怎样搜索完整轨迹，随后介绍 Diffuser 的条件生成，最后比较三种方法各自适合的任务。

## 1. 用目标回报训练 Decision Transformer

[10.1](./intro)中的方法通过约束动作、压低数据集外的 Q 值或加入行为克隆正则来稳定 Bellman 更新。Decision Transformer（Chen et al. 2021）不再学习 Q 函数，而是把离线轨迹改写成条件序列生成问题。

### 1.1 把 Return-to-Go 作为条件

DT 的核心洞察是：在监督学习框架下，一条轨迹 $\tau = (s_1, a_1, r_1, s_2, a_2, r_2, \ldots, s_T, a_T, r_T)$ 里，每个动作 $a_t$ 都有一个**自然的目标**——从 $t$ 时刻起累计的回报：

$$\hat{R}_t = \sum_{t'=t}^{T} r_{t'}$$

称为 **return-to-go**。给定 $\hat{R}_t$ 和 $s_t$，预测 $a_t$ 就是普通的条件监督学习。

DT 把轨迹重组为三元组序列：

$$\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \ldots, \hat{R}_T, s_T, a_T$$

每个 timestep 包含 (RTG, state, action) 三个 token。然后用 GPT 风格的 causal transformer 自回归建模：

$$\pi_\theta(a_t \mid \hat{R}_t, s_t, a_{t-1}, \ldots) = \text{Transformer}(\hat{R}_{1:t}, s_{1:t}, a_{1:t-1})$$

```python
class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_dim, n_heads, n_layers,
                 max_ep_len=4096):
        super().__init__()
        # 三个嵌入层：RTG, state, action 各自映射到 hidden_dim
        self.embed_rtg  = nn.Linear(1, hidden_dim)
        self.embed_state = nn.Linear(state_dim, hidden_dim)
        self.embed_action = nn.Linear(act_dim, hidden_dim)
        self.embed_ln = nn.LayerNorm(hidden_dim)
        # 位置编码：timestep 嵌入
        self.pos_emb = nn.Embedding(max_ep_len, hidden_dim)
        # GPT 主体
        self.transformer = GPT(
            d_model=hidden_dim, n_heads=n_heads, n_layers=n_layers,
            # 关键：每个 timestep 占 3 个 token，attention mask 要对应
            attn_pdrop=0.1, resid_pdrop=0.1
        )
        # 动作预测头（回归，假设连续动作）
        self.action_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, rtg, states, actions, timesteps):
        B, T, _ = states.shape
        # 嵌入并交错：(R1, s1, a1, R2, s2, a2, ...)
        rtg_emb   = self.embed_rtg(rtg)
        state_emb = self.embed_state(states) + self.pos_emb(timesteps)
        action_emb = self.embed_action(actions)

        # 把三者堆叠成 (B, 3T, H)，顺序为 RTG, state, action
        stacked = torch.stack([rtg_emb, state_emb, action_emb], dim=1)
        stacked = stacked.permute(0, 2, 1, 3).reshape(B, 3 * T, -1)
        stacked = self.embed_ln(stacked)

        # causal attention：每个 token 只能看到过去
        h = self.transformer(stacked)
        # 取出 state 位置的输出预测对应 action
        h_states = h[:, 1::3, :]  # indices 1, 4, 7, ...
        return self.action_head(h_states)  # 回归到连续动作

    @torch.no_grad()
    def act(self, state, target_rtg, history, t):
        # 推理时把目标 RTG 作为"prompt"，自回归生成动作
        rtg_seq = torch.cat([history.rtg, target_rtg[None]], dim=0)[-self.K:]
        s_seq   = torch.cat([history.states, state[None]], dim=0)[-self.K:]
        a_seq   = history.actions[-self.K - 1:-1]  # 错位
        t_seq   = torch.arange(len(s_seq))
        pred_a = self.forward(rtg_seq, s_seq, a_seq, t_seq)
        return pred_a[-1]  # 取最后一个 timestep 的预测
```

### 1.2 用监督学习预测动作

DT 的训练损失就是连续动作回归的 MSE（或离散动作的交叉熵）：

$$\mathcal{L} = \mathbb{E}_{\tau \sim \mathcal{D}}\left[\sum_t \|\hat{a}_t - a_t\|^2\right]$$

**没有 Bellman，没有 Q-Learning，没有时序差分**。整个训练过程和训练 GPT 完全一样：扫描轨迹，做下一个 token 预测。这一性质让 DT 可以无缝接入 LLM 训练栈——数据加载、AdamW、cosine schedule、gradient checkpointing 全部沿用。

### 1.3 推理时怎样使用目标回报

DT 部署时不需要 argmax Q。你只要**指定一个目标 RTG**（比如该环境 expert 的分数），DT 自回归生成动作，使累计回报接近目标：

```python
target_return = 9000  # HalfCheetah expert-level
state = env.reset()
history = TrajectoryBuffer()
for t in range(max_steps):
    action = model.act(state, target_return, history, t)
    next_state, reward, done, _ = env.step(action)
    history.append(state, action, reward)
    state = next_state
    # 关键：RTG 每步扣掉实际 reward，作为"剩余要达成的目标"
    target_return -= reward
```

这个机制非常优雅——RTG 是一个**控制变量**，调高调低能生成不同性能水平的策略。实验上 DT 在 Atari、MuJoCo、Key-to-Door 上**达到或超越** CQL/IQL。

### 1.4 Decision Transformer 为什么能够工作

这是离线 RL 社区最有争议的问题之一。传统 RL 视角里，**没有 Bellman 备份就不可能学到长期回报的最优策略**——因为监督信号只能从采到的轨迹来。DT 的回答是：**当数据集足够丰富时，轨迹本身已经隐含了最优性信息**。

- 数据集里有 expert 轨迹（高 RTG）、medium 轨迹（中 RTG）、random 轨迹（低 RTG）
- 给定目标 RTG 高，transformer 学到的条件分布 $p(a \mid \hat{R}_{\text{high}}, s)$ 自然偏向高回报动作
- 这相当于一种**基于检索的策略学习**——本质是模仿"曾经达到过类似 RTG 的轨迹"

形式化地，DT 学到的策略可以写成：

$$\pi_\theta(a \mid s, \hat{R}) \propto \exp\left(-\frac{1}{2\sigma^2}\|a - f_\theta(s, \hat{R})\|^2\right)$$

其中 $f_\theta$ 是 transformer 的回归输出。当 $\sigma \to 0$ 时这退化为确定策略 $a = f_\theta(s, \hat{R})$。注意到这个分布与 $\pi_\beta$ 的关系：

$$\pi_\theta(a \mid s, \hat{R}) \approx \pi_\beta(a \mid s, \text{return} \approx \hat{R})$$

即 DT 学到的是行为策略在"指定回报条件下"的条件分布。这正是为什么 DT 不能超越数据集中最好的策略——它从未组合过两个次优轨迹的好的部分。

这一观察催生了后续大量工作：online RL 中的 RL via supervised learning、in-context RL（Algorithm Distillation）、Star-Vector、Eyre et al. 的 "language modeling is all you need for RL" 等。

### 1.5 Decision Transformer 的局限

1. **只能学到数据中存在的最优策略**——如果数据集里没有 expert 轨迹，再高的目标 RTG 也无法生成 expert 行为
2. **stitching 能力差**——传统离线 RL 可以把两条次优轨迹的好的部分"缝合"成更优策略（subtrajectory stitching），DT 因为是纯监督，做不到这种组合泛化
3. **RTG 选择敏感**——目标 RTG 设太高会生成不连贯动作，设太低则保守

## 2. 用 Trajectory Transformer 搜索轨迹

DT 之后，"RL 作为序列建模" 这条路线迅速衍生。其中两个代表性工作：Trajectory Transformer 把整个轨迹建模为 token 序列、用 beam search 推理；Diffuser 用扩散模型直接生成完整轨迹。

### 2.1 离散化轨迹并使用 Beam Search

Janner et al. 2021 把 RTG、state、action、reward 全部离散化成 token，然后训练一个标准 transformer 预测下一个 token：

$$p_\theta(\tau) = \prod_{t=1}^{T} p_\theta(s_t, a_t, r_t \mid s_{<t}, a_{<t}, r_{<t})$$

推理时用 beam search 最大化轨迹概率（可加 reward 约束）。TT 的特点：

- 把连续量离散化避免了回归问题，但 token 数量爆炸（state 每一维都要离散化）
- Beam search 推理慢（要展开多个候选轨迹）
- 优势：可以做 **planning**——在 search 时显式注入未来 reward 约束，相当于 implicit model-based RL

## 3. 用 Diffuser 生成完整轨迹

Janner et al. 2022 把扩散模型引入 RL。把一条轨迹 $\tau \in \mathbb{R}^{T \times (d_s + d_a)}$ 视为高维图像般的对象，训练一个扩散模型：

$$\min_\theta \; \mathbb{E}_{\tau, t, \epsilon}\left[\|\epsilon - \epsilon_\theta(\tau_t, t)\|^2\right]$$

其中 $\tau_t$ 是轨迹在 timestep $t$ 加噪后的版本，$\epsilon_\theta$ 是去噪网络（通常是 1D temporal U-Net 或 transformer）。推理时从纯噪声开始逐步去噪，得到完整轨迹。

Diffuser 使用 classifier-free guidance 控制生成方向。训练时随机丢弃状态或奖励条件，让模型同时学习条件分布和无条件分布：

$$\tilde{\epsilon}_\theta = (1 + w) \cdot \epsilon_\theta(\tau_t, t, c) - w \cdot \epsilon_\theta(\tau_t, t)$$

其中 $c$ 是条件（如未来奖励最大化），$w$ 控制条件强度。推理时，奖励条件改变去噪方向，使高奖励轨迹获得更高的生成概率。优化由此从显式选择最大价值动作，变成从受奖励引导的轨迹分布中采样。

## 4. 比较三种序列建模方法

| 维度                | Decision Transformer | Trajectory Transformer | Diffuser             |
| ------------------- | -------------------- | ---------------------- | -------------------- |
| 建模对象            | 给定 RTG 的条件策略  | 整条轨迹的联合分布     | 整条轨迹的扩散模型   |
| 离散化              | 否                   | 是（state 每维都离散） | 否                   |
| 推理方式            | 自回归采样           | Beam search            | 迭代去噪             |
| Planning 能力       | 弱（隐式）           | 强（显式）             | 强（条件生成）       |
| Stitching 能力      | 弱                   | 中                     | 强                   |
| 推理速度            | 快                   | 慢                     | 中（需要几十步去噪） |
| 与 LLM 训练栈兼容性 | 强（最像 GPT）       | 强                     | 弱（架构不同）       |

## 本节总结

Decision Transformer 在给定 return-to-go 时自回归生成动作，因此可以直接复用 Transformer 的监督训练方法。Trajectory Transformer 用 Beam Search 搜索完整轨迹，Diffuser 则通过迭代去噪生成轨迹。三者都依赖离线数据中已有的行为覆盖，只是表示轨迹和选择动作的方式不同。

下一节 [10.3 离线 RL 与 LLM 数据](./experiments)把固定数据的视角带到偏好优化，比较 DPO、偏好数据和序列建模之间的联系与差别。
