# 9.2 TD3 与 SAC

> [9.1](./intro) 讲清楚了确定性策略梯度（DPG）与 DDPG——把 DQN 的 off-policy 思想迁移到连续动作。但 DDPG 有三个广受诟病的缺陷：Q 值过估计、超参敏感、训练不稳定。本节给出两套互补的修补方案：**TD3** 用工程 trick 稳定 DDPG，**SAC** 用最大熵 RL 从根本上重构目标函数。

## DDPG 的稳定性补丁

Twin Delayed Deep Deterministic Policy Gradient（Fujimoto et al. 2018）针对 DDPG 的三个缺陷给出三招修补。

### 1. 双 Q 网络（Twin Q）

借鉴 Double DQN 的思想：训练**两个独立的 critic** $Q_{\phi_1}, Q_{\phi_2}$，取较小值作为 target：

$$y = r + \gamma \cdot \min(Q_{\phi_1'}, Q_{\phi_2'})(s', \mu_{\theta'}(s'))$$

这从结构上抑制了 Q 值过估计——两个网络同时过估计的概率比一个低得多。

```python
class TD3Critic:
    def __init__(self, state_dim, action_dim):
        self.Q1 = QNetwork(state_dim, action_dim)
        self.Q2 = QNetwork(state_dim, action_dim)  # 独立初始化

    def forward(self, s, a):
        return self.Q1(s, a), self.Q2(s, a)

    def target_min(self, s, a):
        return torch.min(self.Q1(s, a), self.Q2(s, a))
```

### 2. 延迟策略更新（Delayed Policy Updates）

Critic 比 Actor 难学得多——Critic 要拟合 $Q(s,a)$ 这个二元函数，Actor 只需学 $\mu(s)$ 一元函数。TD3 让 Actor 每 $d$ 步才更新一次（$d=2$），让 Critic 多收敛几轮再给 Actor 信号：

```python
for step in range(total_steps):
    # 每步都更新 critic
    update_critic()

    # 每 d=2 步才更新 actor + 目标网络
    if step_count % policy_delay == 0:
        update_actor()
        soft_update_targets()
```

直觉：critic 还没学好时，actor 收到的梯度是噪声。延迟更新避免 actor 被错误梯度带歪。

### 3. 目标策略平滑（Target Policy Smoothing）

DDPG 的 target 动作 $a' = \mu_{\theta'}(s')$ 是确定性的，但函数逼近器在 $s'$ 附近的值可能剧烈变化。TD3 给 target 动作加一点平滑噪声：

$$a' = \text{clip}(\mu_{\theta'}(s') + \epsilon, a_{\text{low}}, a_{\text{high}}), \quad \epsilon \sim \text{clip}(\mathcal{N}(0, \sigma), -c, c)$$

这相当于"在动作空间做局部平均"，让 Q 函数在动作维度更平滑，**降低 critic 对小扰动的敏感度**。$\sigma = 0.2, c = 0.5$ 是常用配置。

### 三招合起来的效果

TD3 在 MuJoCo 上**显著稳定 DDPG**，性能超越同期 SAC 的早期版本。直到今天，TD3 仍是连续控制的强基线算法。

```python
class TD3:
    def update(self, batch_size=256):
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(batch_size)

        # === Critic 更新（双 Q） ===
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            # 目标策略平滑
            noise = (torch.randn_like(next_actions) * 0.2).clamp(-0.5, 0.5)
            next_actions = (next_actions + noise).clamp(-self.action_max, self.action_max)
            # 双 Q 取 min
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * (1 - dones) * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optim.zero_grad(); critic_loss.backward()
        self.critic_optim.step()

        # === Actor 更新（延迟） ===
        if self.step_count % self.policy_delay == 0:
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
            self.actor_optim.zero_grad(); actor_loss.backward()
            self.actor_optim.step()

            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)
```

## 最大熵 RL

Soft Actor-Critic（Haarnoja et al. 2018）从完全不同的角度切入：**不要求策略最大化期望回报，而是最大化回报 + 熵**。

### 最大熵 RL 的目标

$$J(\pi) = \mathbb{E}_{(s_t, a_t) \sim \pi}\left[\sum_t \gamma^t \big(r_t + \alpha \mathcal{H}(\pi(\cdot \mid s_t))\big)\right]$$

其中 $\mathcal{H}(\pi) = -\mathbb{E}_{a \sim \pi}[\log \pi(a \mid s)]$ 是策略熵，$\alpha$ 是温度系数控制熵权重。

**为什么要加熵？**

- **鼓励探索**：高熵策略不会过早收敛到单一动作
- **鲁棒性**：多模式策略（多个好动作都给概率）对环境扰动更鲁棒
- **训练稳定**：熵正则化让 Q 函数更平滑，避免过估计

### Soft Bellman 方程

修改后的 Bellman 备份：

$$Q^\pi(s, a) = \mathbb{E}_{s'}\left[r + \gamma \cdot V^\pi(s')\right], \quad V^\pi(s) = \mathbb{E}_{a \sim \pi}[Q^\pi(s, a)] + \alpha \mathcal{H}(\pi(\cdot \mid s))$$

关键变化：$V$ 不再是 $\max_a Q$，而是 **soft max**（log-sum-exp 形式在连续动作下取期望）：

$$V^\pi(s) = \alpha \log \int \exp\left(\frac{Q^\pi(s, a)}{\alpha}\right) da$$

### 随机策略的 reparameterization

SAC 的策略 $\pi_\theta(a \mid s)$ 是高斯分布，用 reparameterization trick 计算 Actor 梯度：

$$a = \mu_\theta(s) + \sigma_\theta(s) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

这让 Actor 损失可微：

$$\mathcal{L}_{\text{actor}} = \mathbb{E}_{s \sim \mathcal{D}, \epsilon}\left[\alpha \log \pi_\theta(a \mid s) - Q_\phi(s, a)\right]$$

### 自动温度调节

最难的超参数是 $\alpha$。SAC 的工程创新是**自动调温**：

$$\alpha^* = \arg\max_\alpha \mathbb{E}\left[-\alpha \log \pi(a \mid s) - \alpha \mathcal{H}_0\right]$$

其中 $\mathcal{H}_0$ 是目标熵（通常设为 $-|\mathcal{A}|$）。这让 $\alpha$ 在训练中自动调整：熵太高就降低 $\alpha$，熵太低就提高 $\alpha$。

```python
# 自动调温的 alpha 优化
def update_alpha(self, states, actions):
    # alpha 作为可学习参数，目标是让策略熵接近 target_entropy
    log_pi = -self.actor.log_prob(states, actions)  # 当前策略的负对数似然
    alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
    self.alpha_optim.zero_grad()
    alpha_loss.backward()
    self.alpha_optim.step()
    self.alpha = self.log_alpha.exp()
```

### SAC 的优势

SAC 在 MuJoCo 上长期霸榜，原因：

1. **Off-policy 高样本效率**（继承自 DDPG）
2. **最大熵自动探索**（不用调噪声）
3. **训练稳定**（双 Q + soft target）
4. **超越人类水平**（在 HalfCheetah 上达到 15000+ 分）

### 三大算法对比

| 维度     | DDPG     | TD3          | SAC            |
| -------- | -------- | ------------ | -------------- |
| 策略类型 | 确定性   | 确定性       | 随机（高斯）   |
| Q 网络   | 1 个     | 2 个（Twin） | 2 个（Twin）   |
| 探索方式 | 外加噪声 | 外加噪声     | 熵奖励（内置） |
| 稳定性   | 差       | 中           | 强             |
| 超参敏感 | 高       | 中           | 低             |
| 推荐首选 | ❌       | ⚠️           | ✅             |

**实战建议**：连续控制首选 SAC；如果需要确定性策略（部署时无随机），选 TD3。

## HalfCheetah 上的训练曲线

在 MuJoCo HalfCheetah-v3 环境上训练 1M 步的对比：

```
回报
12000 │                    ╭─────── SAC (稳定收敛)
10000 │                  ╭─╯
 8000 │                ╭─╯  ╭─────── TD3 (稳定但稍慢)
 6000 │              ╭─╯   ╱
 4000 │            ╭─╯    ╱
 2000 │          ╭─╯     ╱  ╭───── DDPG (发散后偶尔恢复)
     0 │─────────╯──────╱──╯
       └───────────────────────────────
        0    200K  400K  600K  800K  1M steps
```

三个观察：

- **SAC** 收敛最快、最稳——最大熵自带的探索让前期学得快
- **TD3** 略慢于 SAC 但最终性能接近——稳定性补丁让 DDPG 可用
- **DDPG** 大部分时间在发散——只在某些种子下偶有训练成功

## 本节总结

DDPG → TD3 → SAC 是连续控制的三步演进：

1. **DDPG** 把 DQN 思想扩展到连续动作，但不稳定
2. **TD3** 用 Twin Q + 延迟更新 + 目标平滑三招稳定 DDPG
3. **SAC** 用最大熵 RL 从目标函数层面重构，内置探索，自动调温

实战中 SAC 是首选，TD3 是确定性策略场景的备选，DDPG 已不推荐。

下一节 [9.3 Model-Based RL](./model-based) 转向另一个方向——当真实环境采样昂贵时，学一个环境模型来生成"假"数据，把样本效率提升 10-100 倍。
