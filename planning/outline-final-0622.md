# 现代强化学习实战 — 完整 v5 大纲（MIT 级教材）

> 综合 v1→v2→v3→v4→v5 的所有修正，基于真实证据（OpenAI / Anthropic / DeepSeek / Qwen / Zhipu / StepFun JD + 2025-2026 论文）的最终方案。
>
> **本版（0622）核心修正**：统一三层标题层级，强制区分「Part / 单篇文章 / 文章内子大纲」，消除旧版"X.Y 编号歧义"（旧版 `1.1` 既可能是文章内小节、也可能是独立文章）。

---

## 标题层级规则（强制统一）

| 层级       | 用什么       | 含义                                                                                 |
| ---------- | ------------ | ------------------------------------------------------------------------------------ |
| **Part**   | `#`          | 大章（Part I、Part II、序章…）                                                       |
| **章节**   | `##`         | 一个教学单元，对应一个目录 `chapterNN_xxx/`，**必须标 `[单篇文章]` 或 `[多篇章节]`** |
| **文章**   | `###`        | **仅 `[多篇章节]` 下出现**，一个 `.md` 文件                                          |
| **子大纲** | `-` 缩进列表 | 该文章内部要展开的 H2/H3 要点（不是独立文件）                                        |

**强制规则：**

- 单篇章节下**禁止**用 `###`（避免误导读者以为是独立文件）
- 多篇章节下**必须**用 `###` 标每个文件
- 设计理由、备注一律放进 `>` 引用块，不混入正文
- `[v5 新增]`、`[v5.1 扩展]` 这类版本标签放进章节标题后的方括号，不污染层级
- 文件路径用 `→` 箭头标注；旧目录名历史遗留直接说明，不掩饰

---

## 阅读约定

| 标记         | 含义                                                         |
| ------------ | ------------------------------------------------------------ |
| `[单篇文章]` | 章节只有 1 个 `.md` 文件，下方"内部小节"是该文件内部的 H2/H3 |
| `[多篇章节]` | 章节有多个 `.md` 文件，下方每一项 `###` 是一个独立文章       |
| `📄`         | 文章（一个 `.md` 文件）                                      |
| `→ 路径`     | 文件实际位置                                                 |
| `子大纲:`    | 该文章内部要展开的 H2/H3 要点（不是独立文件）                |

---

## 设计哲学

### 为什么用这个结构

1. **理论 → 实战 → 前沿** 三段递进，符合 Stanford CS285 + Sutton & Barto + D2L 风格
2. **每个 Part 对应一个清晰的学习目标**，可独立教学
3. **Agentic 与实战不是附录，是核心 Part V**，反映 2025-2026 真实工业需求
4. **每章统一结构**：本章导读 → 理论 → 实现 → 实验 → 总结 → 延伸阅读
5. **真实论文出处**：每个关键论点都有 arXiv 编号或公司技术报告链接

### 与现有书籍的对比

| 维度       | Sutton & Barto | CS285 | Raschka (in progress) | **本书 v5**      |
| ---------- | -------------- | ----- | --------------------- | ---------------- |
| 经典 RL    | ✅ 完整        | ✅    | ❌                    | ✅               |
| 深度 RL    | ❌             | ✅    | ❌                    | ✅               |
| LLM RL     | ❌             | 部分  | ✅                    | ✅ 完整          |
| Agentic RL | ❌             | ❌    | ❌                    | **✅ 独立 5 章** |
| 多模态 RL  | ❌             | ❌    | ❌                    | **✅ 独立 4 章** |
| 安全/对齐  | ❌             | ❌    | ❌                    | **✅ 独立 3 章** |
| 工程系统   | ❌             | ❌    | 部分                  | **✅ 独立 2 章** |

---

# 序章 · 导论 `[多篇章节]`

> **设计理由**：本书承诺"先动手、后理论"。但旧版 `preface/intro.md` 开篇却是 Sutton 苦涩的教训哲学论述，直到第 1 章才让读者碰代码——导论自己违反了承诺。v5 修正：序章首节直接放可立刻玩的 CartPole 入口，30 秒内看到智能体从摇晃到站稳；**玩过之后再讲为什么**。
>
> **目录**：`docs/preface/`（共 3 个文件，覆盖 0.1-0.6 共 6 节）

### 写在开头（含 0.1 / 0.2 / 0.3 / 0.6）→ `docs/preface/intro.md`

**子大纲：**

- **0.1 先动手：30 秒玩转 CartPole** `[v5 新前置]`
  - 三层体验，海内外双源部署，覆盖所有读者类型：
  - **① 一键试玩（首选体验，零安装）— 双源部署**
    - 主源 ModelScope 创空间：`spaces.modelscope.cn/{namespace}/cartpole-playground`
      - 大陆访问稳定，本书主要受众（中文读者）默认走这里
      - gradio 应用，与 HF Space 代码几乎通用
    - 副源 HuggingFace Space：`hf.co/spaces/{namespace}/cartpole-playground`
      - 海外读者/镜像访问者走这里
    - 页内呈现：两个 iframe 标签切换 "🇨🇳 国内入口 / 🌍 海外入口"
      - 默认显示 ModelScope（主源）
      - 加载失败时自动 fallback 提示，引导切换
    - 体验：点击 "Train" 按钮 → 实时看 reward 曲线爬升 → 训练完成自动播放最终策略
  - **② 一行命令本地跑（深度选项）**
    - 醒目代码框：`pip install "gymnasium[classic-control]" stable-baselines3 && python 1-ppo_cartpole.py`
    - 30 秒 CPU 训练完毕，弹出 `--gui` 小车演示窗口
    - 衔接到 `code/chapter01_cartpole/` 完整代码
    - 大陆读者：`pip install` 走清华/阿里镜像源，文档里给出配置说明
  - **③ 视觉预览（离线兜底）**
    - 训练过程 GIF：reward 曲线从 20 爬到 500 的全过程动画（自托管在本书仓库 `docs/preface/images/`）
    - 最终策略演示视频：智能体从摇摇晃晃到稳稳站立
    - 双源都加载失败、不愿安装、离线读者的最低门槛体验
  - **承诺兑现点**：无论选哪层、身处哪个地区，此刻读者都已经"见过"一个智能体学会一件事——后续 0.2-0.6 都是回头解释刚才发生了什么
  - **经典 → 现代的呼应：未来剧透区** `[v5 新增]`
    - CartPole 是 RL 的过去（1990s 起经典任务），本书的真正主角是 LLM 时代的现代 RL
    - 剧透 1：DPO 让大模型学会"不盲从用户"（训练前 vs 训练后对话对比 GIF：用户要求写恶意代码 → 训练前照办 / 训练后婉拒）→ 衔接第 17 章 DPO 家族
    - 剧透 2：DeepSeek-R1 推理能力涌现（R1-Zero 纯 RL 训练 CoT 自发拉长的视频）→ 衔接第 18 章 GRPO 家族 + 第 19 章 Reasoning Models
    - 剧透 3：Computer Use 智能体操作浏览器（Claude Computer Use / OpenAI Operator 公开演示视频）→ 衔接第 24 章计算机使用与 GUI Agent
    - 剧透 4：SWE-Agent 自主修 Bug（SWE-bench 上智能体读代码 → 定位 → 修改 → 通过测试的完整流程）→ 衔接第 23 章代码智能体 RL
    - 呈现方式：全部自托管 GIF/视频在 `docs/preface/images/teasers/`，避免外链失效。读者看完即知本书终点，但不在序章强行要求能玩 LLM（门槛太高）

- **0.2 写在开头：为什么需要 RL**
  - 萨顿《苦涩的教训》与 70 年 AI 史的两条主线：搜索与学习
  - 为什么试错是学习最原始的形态：骑自行车类比
  - 从识别到决策：为什么监督学习无法覆盖连续决策
  - RL 提供了什么：不告诉怎么做，只告诉什么好什么不好
  - 衔接 0.1：你刚才在 CartPole 上看到的"试错 → 收敛"就是这一节的实例

- **0.3 什么是 RL：核心循环与关键术语**
  - 智能体-环境-状态-动作-奖励循环
  - 轨迹、回报、折扣因子 $\gamma$
  - 状态 vs 观测、离散 vs 连续动作空间
  - 衔接第 1 章：把 0.1 的 CartPole 用这些术语重新描述一遍

- **0.6 本书结构与读者路线图**
  - 全书 7 大 Part 的递进逻辑：基础 → 深度 → 高级 → LLM → Agentic → 多模态 → 安全与前沿
  - 三种背景读者的推荐路径：
    - ML 工程师：0.1 → Part IV–V（LLM + Agentic）
    - RL 背景读者：Part II–III + Part IV
    - 学生：从 Part I 循序渐进
  - 符号约定与记号表（详见附录 H）

### 0.4 强化学习简史 → `docs/preface/brief-history/index.md`

**子大纲：**

- 1950s-1980s：试错学习、Bellman 方程、TD 学习诞生
- 1992：TD-Gammon——第一个战胜人类冠军的 RL 系统
- 2013：DQN 玩 Atari——深度 RL 元年
- 2016：AlphaGo 击败李世石
- 2017-2019：AlphaGo Zero、MuZero、自我博弈
- 2017：PPO 发布，成为工业标准（你刚才在 0.1 用的就是 PPO）
- 2022：InstructGPT / RLHF 进入大模型训练
- 2023-2024：DPO、GRPO、Constitutional AI
- 2025：DeepSeek-R1、o1/o3、RLVR 范式确立
- 中国实验室崛起：Qwen3 GSPO、Step-Audio、DeepSeek 透明度

### 0.5 环境安装指南 → `docs/preface/env-setup.md`

**子大纲：**

- Python 环境：conda / venv 选择
- PyTorch 版本与 CUDA 配置
- Gymnasium 安装与验证
- veRL / OpenRLHF / TRL 工具链预告
- 训练硬件检查表：入门实验 / 核心实验 / 大型项目三档
- 仓库代码结构：`code/` 各章独立子目录

---

# Part I · 基础与经典强化学习（6 章）

## 第 1 章 CartPole：第一个强化学习实验 `[多篇章节]`

> **目录**：`chapter01_cartpole/`（目录名沿用旧编号 01，与章节号 1 不一致——历史遗留，暂不重命名）

### 1.1 CartPole 入门与原理 → `chapter01_cartpole/intro.md` + `principles.md`

**子大纲：**

- CartPole 问题与 Gym/Gymnasium 接口
- 状态、动作、奖励的工程化定义
- 随机策略基线与失败模式

### 1.2 训练指标设计 → `chapter01_cartpole/metrics.md`

**子大纲：**

- 回报曲线、成功率、稳定性
- 实验：从随机到收敛的完整流程

### 1.3 PPO 训练实战与可视化 → `chapter01_cartpole/training.md`

**子大纲：**

- stable-baselines3 PPO 训练入口
- 训练曲线可视化与超参含义
- 失败模式诊断（收敛震荡、reward 停滞）
- 衔接序章 0.1 试玩背后的训练流程

---

## 第 2 章 多臂老虎机与探索-利用理论 `[多篇章节]`

> **目录**：`chapter03_bandits/`

### 2.1 问题定义与 ε-贪心 → `chapter03_bandits/intro.md`

**子大纲：**

- 多臂老虎机问题与形式化定义
- 遗憾（Regret）作为衡量标准
- ε-贪心算法与衰减调度
- ε-贪心的遗憾界分析

### 2.2 UCB 与 Thompson 采样 → `chapter03_bandits/ucb-thompson.md`

**子大纲：**

- 上置信界（UCB）算法：乐观面对不确定性
- Thompson 采样：贝叶斯视角与概率匹配
- Bernoulli 奖励的 Beta 共轭先验
- UCB 与 Thompson 的对比与工业实践（Google AdWords 案例）

### 2.3 遗憾界、PAC 与上下文老虎机 → `chapter03_bandits/theory-contextual.md`

**子大纲：**

- Lai-Robbins 下界与渐近最优性 `[理论]`
- PAC 框架与样本复杂度
- 上下文老虎机（Contextual Bandits）：LinUCB / LinTS / NeuralUCB
- 与 RLHF 的连接：为什么 GRPO 采样多条 rollout、为什么 PPO 需要 IS

---

## 第 3 章 马尔可夫决策过程 `[多篇章节]`

> **历史遗留**：当前与第 4/5/6 章合并在 `chapter03_mdp/` 下。v5 建议拆分为独立目录 `chapter04_mdp/`，但因涉及大量交叉引用，迁移工作放到 Phase 2。

### 3.1 MDP 基础与马尔可夫性 → `chapter03_mdp/mdp.md`

**子大纲：**

- 从老虎机到序列决策
- 马尔可夫性的数学定义与直觉
- 状态空间、动作空间、转移函数、奖励函数

### 3.2 策略、价值与回报 → `chapter03_mdp/policy-value.md`

**子大纲：**

- 策略的定义：确定性 vs 随机性
- 回报与价值（V/Q 的初步引入，详见第 4 章）
- 策略评估与回合制 vs 连续任务

### 3.3 折扣、轨迹与 POMDP → `chapter03_mdp/panorama.md`

**子大纲：**

- 折扣因子与回报的数学意义
- 轨迹与回合
- 部分可观测 MDP（POMDP） `[LLM 多轮前置]`
- 真实世界的 POMDP 案例：机器人感知、对话历史

---

## 第 4 章 价值函数与贝尔曼方程 `[多篇章节]`

> **历史遗留**：与第 3/5/6 章合并。当前实际内容跨 `chapter03_mdp/value-bellman.md` 与 `value-q.md` 两个文件。

### 4.1 V/Q 函数与贝尔曼期望方程 → `chapter03_mdp/value-bellman.md`

**子大纲：**

- 状态价值函数 $V^\pi(s)$
- 动作价值函数 $Q^\pi(s,a)$
- 贝尔曼期望方程的推导
- V 与 Q 的相互关系

### 4.2 贝尔曼最优、压缩映射与最优策略 → `chapter03_mdp/value-q.md`

**子大纲：**

- 贝尔曼最优方程
- 贝尔曼算子的压缩映射性质 `[理论]`
- 最优策略的存在性与唯一性
- Banach 不动点定理应用

### 4.3 价值函数数值实验 → `chapter03_mdp/value-experiment.md`

**子大纲：**

- 网格世界（Gridworld）上的策略评估
- 价值迭代的收敛速度实证
- V/Q 可视化与策略改进
- 与第 5 章动态规划的衔接

---

## 第 5 章 动态规划、蒙特卡洛与时序差分 `[多篇章节]`

> **历史遗留**：与第 3/4/6 章合并。当前实际内容在 `chapter03_mdp/dp-mc-td.md`。

### 5.1 动态规划 → `chapter03_mdp/dp-mc-td.md`

**子大纲：**

- 策略评估与策略改进
- 策略迭代与价值迭代
- DP 的局限：需要完美模型

### 5.2 蒙特卡洛方法 → `chapter03_mdp/dp-mc-td.md`

**子大纲：**

- 首次访问与每次访问 MC
- MC 的 model-free 性质
- 方差问题与控制

### 5.3 时序差分、n-step 与资格迹 → `chapter03_mdp/dp-mc-td.md`

**子大纲：**

- TD(0) 学习
- n-step Bootstrap 与 TD(λ) 的偏差-方差权衡
- 资格迹（Eligibility Traces）
- DP / MC / TD 三类方法的对比与权衡

---

## 第 6 章 Q-Learning 与离策略控制 `[多篇章节]`

> **历史遗留**：与第 3/4/5 章合并。当前实际内容在 `chapter03_mdp/algorithm-taxonomy.md`。

### 6.1 on/off-policy、Q-Learning 与 SARSA → `chapter03_mdp/algorithm-taxonomy.md`

**子大纲：**

- 在策略（on-policy）与离策略（off-policy）的区别
- Q-Learning 算法与收敛性
- SARSA 算法
- Q-Learning vs SARSA： Cliff Walking 经典案例

### 6.2 重要性采样与 Deadly Triad → `chapter03_mdp/algorithm-taxonomy.md`

**子大纲：**

- 重要性采样（Importance Sampling） `[关键前置，PPO/GRPO 基础]`
- 函数逼近的挑战与 Deadly Triad（函数逼近 + bootstrapping + off-policy） `[理论]`
- 离策略梯度方法初步

### 6.3 奖励函数设计入门 → `chapter03_mdp/reward-design.md`

**子大纲：**

- 稀疏 vs 稠密奖励
- Reward shaping 与基于势能的整形（potential-based shaping）
- Reward hacking 的早期案例
- 衔接第 30 章奖励黑客专题

---

# Part II · 深度强化学习（5 章）

## 第 7 章 深度 Q 网络与 Distributional RL `[多篇章节]`

> **目录**：`chapter04_dqn/`（6 个文件，目录名沿用旧编号 04，与章节号 7 不一致——历史遗留）

### 7.1 从 Q-Learning 到 DQN → `chapter04_dqn/from-q-to-dqn.md`

**子大纲：**

- 从 Q-Learning 到 DQN 的动机
- 经验回放（Experience Replay）
- 目标网络（Target Network）

### 7.2 DQN 改进家族 → `chapter04_dqn/dqn-family.md`

**子大纲：**

- Double DQN：解决过估计
- Dueling DQN：状态-动作价值分解
- 优先经验回放（PER）
- Rainbow 与 NoisyNet

### 7.3 Distributional RL → `chapter04_dqn/dqn-components.md`

**子大纲：**

- C51、QR-DQN、IQN
- 分布式价值函数的数学基础

### 7.4 实验：LunarLander 与 Atari → `chapter04_dqn/lunar-lander.md` + `visual-game-projects.md`

**子大纲：**

- LunarLander 训练流程
- Atari 游戏基准
- 调参与可视化

---

## 第 8 章 策略梯度方法 `[多篇章节]`

> **目录**：`chapter05_policy_gradient/`（9 个文件，目录名沿用旧编号 05，与章节号 8 不一致——历史遗留）

### 8.1 策略梯度入门 → `chapter05_policy_gradient/intro.md` + `policy-gradient.md`

**子大纲：**

- 策略梯度方法的动机（连续动作、随机策略）
- 策略表示：Softmax、高斯、Categorical
- 策略梯度定理（完整推导） `[理论]`

### 8.2 REINFORCE 与基线 → `chapter05_policy_gradient/reinforce.md` + `baseline-experiment.md`

**子大纲：**

- REINFORCE 算法
- 方差问题与基线（Baseline）

### 8.3 策略梯度改进与实验 → `chapter05_policy_gradient/pg-improvements.md` + `cartpole.md` + `cartpole-baseline.md` + `dice-game.md` + `pg-necessity.md`

**子大纲：**

- Off-policy 策略梯度
- 实验：CartPole 与 Pendulum
- 策略梯度必要性证明（Dice 游戏）

---

## 第 9 章 Actor-Critic 架构 `[多篇章节]`

> **目录**：`chapter06_actor_critic/`（8 个文件，目录名沿用旧编号 06，与章节号 9 不一致——历史遗留）

### 9.1 优势函数与 Critic 训练 → `chapter06_actor_critic/advantage-function.md` + `critic-training.md`

**子大纲：**

- 优势函数 $A(s,a) = Q(s,a) - V(s)$
- Critic 网络的训练（价值函数拟合）

### 9.2 Actor-Critic 框架与同步更新 → `chapter06_actor_critic/actor-critic.md` + `ac-frontier.md`

**子大纲：**

- Actor-Critic 框架与同步更新
- 广义优势估计（GAE） `[PPO 前置]`
- A2C 与 A3C：异步并行

### 9.3 实验 → `chapter06_actor_critic/pendulum.md` + `bipedalwalker.md` + `alphago.md`

**子大纲：**

- 实验：Pendulum 与 BipedalWalker
- AlphaGo 作为 Actor-Critic 的应用案例

---

## 第 10 章 PPO 与信任域方法 `[多篇章节]`

> **目录**：`chapter07_ppo/`（7 个文件，目录名沿用旧编号 07，与章节号 10 不一致——历史遗留）

### 10.1 TRPO 与信任域 → `chapter07_ppo/trust-region-clipping.md` + `ppo-math.md`

**子大纲：**

- 策略更新的稳定性问题
- TRPO 与单调改进定理 `[理论]`
- PPO 数学推导

### 10.2 PPO-Clip 工程实现 → `chapter07_ppo/intro.md`

**子大纲：**

- PPO-Clip 算法
- PPO-Penalty 与自适应 KL
- PPO 工程实现细节（entropy bonus、value clip）

### 10.3 GAE 与奖励模型 → `chapter07_ppo/gae-reward-model.md`

**子大纲：**

- GAE 在 PPO 中的应用
- 奖励模型与 PPO 的接口（为第 15 章 RLHF 做铺垫）

### 10.4 长程任务与实验 → `chapter07_ppo/rl-long-horizon-planning.md` + `ppo-bipedal-walker.md` + `ppo-game-benchmark.md`

**子大纲：**

- 长程任务中的 PPO
- PPO 在 LLM RL 时代的位置（详见第 18 章 GRPO 家族）
- 实验：BipedalWalker 连续控制与游戏基准

---

## 第 11 章 连续控制与基于模型的深度 RL `[多篇章节]`

> **目录**：`chapter12_continuous_control/`

### 11.1 确定性策略梯度与 DDPG → `chapter12_continuous_control/intro.md`

**子大纲：**

- 确定性策略梯度（DPG）定理
- DDPG 算法：Actor-Critic + 经验回放 + 目标网络
- DDPG 的缺陷：Q 值过估计、超参敏感、训练不稳定

### 11.2 TD3 与 SAC → `chapter12_continuous_control/td3-sac.md`

**子大纲：**

- TD3 三招修补：Twin Q / 延迟策略更新 / 目标策略平滑
- Soft Actor-Critic：最大熵 RL 框架
- Soft Bellman 方程与自动温度调节
- 三大算法对比表与选型建议

### 11.3 Model-Based RL：Dyna、PETS、MBPO → `chapter12_continuous_control/model-based.md`

**子大纲：**

- 为什么用模型：样本效率的根本提升
- Dyna：模型作为数据增强
- PETS：概率轨迹采样与集成
- MBPO：模型策略迭代与短 horizon rollout

### 11.4 AlphaZero、MuZero 与 Dreamer V3 → `chapter12_continuous_control/search-world-models.md`

**子大纲：**

- AlphaZero：MCTS + 神经网络估值 + 自我对弈
- MuZero：隐式模型学习（表示 + 动力学 + 预测三网络）
- Dreamer V3：RSSM 世界模型 + 想象中训练 actor-critic
- Model-Based vs Model-Free 权衡表

---

# Part III · 高级 RL 方法（3 章，精简但深入）

## 第 12 章 离线强化学习与决策 Transformer `[多篇章节]`

> **目录**：`chapter13_offline_rl/`

### 12.1 离线 RL 挑战与经典方法 → `chapter13_offline_rl/intro.md`

**子大纲：**

- 离线 RL 的挑战：分布偏移与外推误差
- 悲观主义原则：CQL、IQL、BCQ
- AWAC 与 TD3+BC：保守约束的简化
- 与行为克隆（BC）的关系

### 12.2 Decision Transformer、Trajectory Transformer 与 Diffuser → `chapter13_offline_rl/sequence-modeling.md`

**子大纲：**

- Decision Transformer：RL 作为序列建模
- Trajectory Transformer：规划式采样
- Diffuser：扩散模型做决策
- LLM 时代的离线 RL（DPO 即离线 RL 的视角）

### 12.3 离线 RL 实验与 LLM 视角 → `chapter13_offline_rl/experiments.md`

**子大纲：**

- D4RL 基准与经典任务（HalfCheetah、AntMaze）
- CQL / IQL / DT 实验对比
- 从离线 RL 视角看 DPO/IPO 的统一性
- 衔接第 17 章 DPO 家族

---

## 第 13 章 模仿学习、反向 RL 与元 RL `[多篇章节]`

> **目录**：`chapter14_imitation_meta_rl/`

### 13.1 行为克隆与 DAgger → `chapter14_imitation_meta_rl/bc-dagger.md`

**子大纲：**

- 行为克隆（BC）与 covariate shift 问题
- DAgger：数据集聚合解决分布漂移
- 与 SFT 的连接：监督学习范式

### 13.2 逆向 RL 与 GAIL → `chapter14_imitation_meta_rl/irl-gail.md`

**子大纲：**

- 最大熵逆向 RL（MaxEnt IRL）
- GAIL：生成对抗模仿学习
- 与 RLHF 的连接：奖励学习

### 13.3 元 RL：MAML、RL²、PEARL、In-Context RL → `chapter14_imitation_meta_rl/meta-rl.md`

**子大纲：**

- MAML：模型不可知元学习
- RL²：用 RNN 隐式学习快速适应
- PEARL：概率元 RL
- In-Context RL 与 Algorithm Distillation `[DeepMind 2022]`

---

## 第 14 章 探索、多智能体与分层 RL `[多篇章节]`

> **目录**：`chapter15_exploration_marl_hierarchical/`

### 14.1 探索：ICM、RND、NGU、Agent57 → `chapter15_exploration_marl_hierarchical/intro.md`

**子大纲：**

- 探索-利用的根本张力（理论回顾）
- 内在好奇心（ICM）与随机网络蒸馏（RND）
- NGU 与 Agent57：episodic + life-long 内在奖励
- 与第 2 章 MAB 探索理论的呼应

### 14.2 多智能体 RL：CTDE、MADDPG、MAPPO → `chapter15_exploration_marl_hierarchical/marl.md`

**子大纲：**

- 多智能体 RL 的挑战：非平稳环境
- CTDE 框架：中心化训练、分布执行
- MADDPG：每个 agent 一个 critic
- MAPPO：多智能体 PPO 的工业实践

### 14.3 分层 RL 与生成式世界模型引子 → `chapter15_exploration_marl_hierarchical/hierarchical.md`

**子大纲：**

- 分层 RL 的动机：长程任务的层次分解
- Options 框架
- FeUdal Networks 与 HIRO
- 生成式世界模型作为 RL 环境（Genie 3 引子，详见第 31 章）

---

# Part IV · 大语言模型对齐与后训练（7 章）

## 第 15 章 RLHF 训练流水线 `[多篇章节]` `[v5.1 扩展]`

> **目录**：`chapter08_rlhf/`（10 个文件，目录名沿用旧编号 08，与章节号 15 不一致——历史遗留）

### 15.1 基座模型与指令对齐 → `chapter08_rlhf/base-model-to-assistant.md`

**子大纲：**

- 基座模型与指令对齐
- 现代三阶段范式：SFT → RLHF → RLVR

### 15.2 SFT 指令微调 → `chapter08_rlhf/imitation-learning-pipeline.md`

**子大纲：**

- SFT 指令微调
- 数据构造与格式

### 15.3 奖励建模：Bradley-Terry 模型 → `chapter08_rlhf/reward-function-design.md`

**子大纲：**

- 奖励建模：Bradley-Terry 模型
- Reward Model 训练流程

### 15.4 RL 微调（PPO 或 GRPO）→ `chapter08_rlhf/standard-rlhf-pipeline.md` + `ppo-rlhf-loop.md`

**子大纲：**

- 标准 RLHF 流水线
- PPO 在 RLHF 中的循环结构
- KL 约束与参考策略

### 15.5 双轨奖励与 Pre-PPO `[v5.1 扩展]` → `chapter08_rlhf/intro.md`

**子大纲：**

- 双轨奖励设计（Seed-Thinking，verifiable + pairwise）
- Pre-PPO：Prompt 选择策略避免 reward hacking

### 15.6 扩展实战与大规模训练 → `chapter08_rlhf/extended-practice.md` + `scaling-to-large-models.md`

**子大纲：**

- 扩展实战
- 大规模训练（含 Tülu 3 三阶段范式参考）

### 15.7 评测 → `chapter08_rlhf/evaluation.md`

**子大纲：**

- RLHF 训练后评测指标
- 安全性与对齐评估

### 15.8 动手实验：veRL + PPO 训练 GSM8K → `chapter08_rlhf/verl-ppo-gsm8k.md`

**子大纲：**

- veRL 框架上手
- GSM8K 数据集训练
- 训练曲线与结果分析

---

## 第 16 章 LLM RL 工业实战与分布式训练系统 `[多篇章节]` `[v5.1 扩展：从 PPO 转向 GRPO 现代流水线]`

> **设计理由**：2025-2026 Llama 4 / Qwen3 / DeepSeek V3.2 / GLM-4.6 已全面转向 GRPO/Dr.GRPO + RLVR，旧版以 PPO 经典实现为主的写法已过时。
>
> **合并说明**：原 第 36 章 分布式 RL 训练系统（veRL/OpenRLHF/async/MoE/万卡）合并到本章——训练框架与分布式系统本属同一主题，强行拆开造成内容重复。
>
> **目录**：`chapter17_llm_rl_industrial/` + `chapter36_distributed_rl_training/`（分布式子章节保留独立目录）

### 16.1 训练框架对比 → `chapter17_llm_rl_industrial/01-frameworks.md`

**子大纲：**

- 同步框架：veRL（字节主流）/ OpenRLHF（开源友好）/ TRL（HF 生态）/ NeMo-Aligner（NVIDIA）
- 异步框架：AReaL（清华+智谱）/ AgentRL（智谱+清华）/ SLIME / ROLL / LlamaRL
- 框架对比表与选型决策树

### 16.2 现代后训练流水线范式 → `chapter09_alignment/industrial-post-training.md`（沿用现有）

**子大纲：**

- DeepSeek-R1 多阶段：冷启动 SFT → 推理 RL → 拒绝采样 → 全场景 RL
- Llama 4：轻量 SFT → online RL → 轻量 DPO + pass@k 难度过滤
- Qwen3：Thinking Mode Fusion + Thinking Budget + GSPO
- GLM-4.5 / 4.6：难度课程 RL + Hybrid Thinking + RLCS 课程采样
- GLM-5（2026.02）：异步 Agent RL + DSA 稀疏注意力
- Seed-Thinking-v1.5：Dual-track reward + Pre-PPO + Hybrid reward

### 16.3 双轨奖励设计 → `chapter17_llm_rl_industrial/03-dual-reward.md`

**子大纲：**

- Verifiable Reward（Math、Code）
- Pairwise Preference Reward（开放对话）
- Pre-PPO：Prompt 选择策略避免 reward hacking
- Hybrid Reward：RTV + GenRM 组合

### 16.4 优化器与训练稳定性 → `chapter09_alignment/modern-industrial-practice.md`（沿用现有）

**子大纲：**

- AdamW 在 RL 训练中的稳定性问题
- MuonClip 优化器（Kimi K2）
- QK-clip：注意力数值稳定性
- KL 爆炸的早期信号与处理

### 16.5 分布式同步框架与 Rollout 引擎 → `chapter36_distributed_rl_training/intro.md`

**子大纲：**

- veRL 架构深度解析
- OpenRLHF / NeMo-Aligner / TRL 对比
- Rollout 引擎与 vLLM 集成
- GPU 内存优化：ZeRO、FSDP、Gradient Checkpointing
- 性能 profiling 与瓶颈分析

### 16.6 异步 RL 训练系统 → `chapter36_distributed_rl_training/async.md`

**子大纲：**

- 异步 RL 的 staleness 问题
- LlamaRL（Meta）：纯异步 pipeline
- AReaL（清华+智谱）：异构算力调度
- AgentRL（智谱+清华）：长轨迹 agent 异步训练
- SLIME / ROLL 框架对比

### 16.7 MoE + RL 与万卡集群 → `chapter36_distributed_rl_training/scale.md`

**子大纲：**

- MoE + RL 训练（DeepSeek V3、Step Flash、GLM-4.5）
- DualPipe 与 Best-Fit packing
- Expert 负载均衡在 RL 阶段的稳定性
- 万卡集群实践：通信、容错、checkpoint
- 万卡 RL 训练的成本/吞吐实测

### 16.8 工业实战：GSM8K 与 AIME → `chapter09_grpo_rlvr/verl-code-sandbox.md`（沿用现有）

**子大纲：**

- 实验：用 GRPO 训练 GSM8K
- 实验：用 DAPO 训练 AIME 2024
- 完整开源复现：Open-R1 / Sky-T1 / Tülu 3

### 16.9 中国对齐团队面试常见考点 → `chapter17_llm_rl_industrial/07-interview.md`

**子大纲：**

- PG → REINFORCE → TRPO → PPO → GRPO 完整推导链（智谱真题）
- DPO 家族 + DPO 正则化
- DeepSpeed vs Megatron 工程对比
- 训练资源消耗现场推算

---

## 第 17 章 偏好对齐：DPO 家族 `[多篇章节]` `[v5.1 扩展]`

> **目录**：`chapter02_dpo/`（3 个文件，旧编号 02）+ `chapter09_alignment/dpo-theory-and-family.md`

### 17.1 DPO 入门与推导 → `chapter02_dpo/intro.md` + `principles.md`

**子大纲：**

- DPO 的数学推导（从 RLHF 目标推导）
- DPO 训练动态分析

### 17.2 DPO 训练指标 → `chapter02_dpo/metrics.md`

**子大纲：**

- 训练监控指标
- reward margin、accuracy 等关键指标

### 17.3 DPO 原理、数学与家族选型 → `chapter09_alignment/dpo-theory-and-family.md`

**子大纲：**

- IPO：解决 DPO 的过拟合
- KTO：无需成对偏好数据
- SimPO：无参考策略方法
- DPO 正则化方法（智谱面试真题）
- Iterative DPO 与 ReST
- 自我对弈微调（SPIN）
- DPO 家族选型决策树

---

## 第 18 章 GRPO 家族、RLVR 与 Verifier 工程 `[多篇章节]` `[v5.1 完整重构]`

> **设计理由**：2025-2026 最大算法焦点。四家独立调研一致指出 v5 此章只列名字无算法细节。v5.1 按改进方向重新组织，覆盖 6+ 主流变体的算法差异。
>
> **合并说明**：原 第 23 章 RL Environments 与 Verifiers 设计合并到本章——Verifier 是 RLVR 的核心组件，RLVR 又是 GRPO 家族的训练范式，三者构成一个完整闭环，拆开造成交叉引用过多。
>
> **目录**：`chapter09_grpo_rlvr/`（7 个文件）+ `chapter23_rl_environments/`

### 18.1 GRPO 训练与核心机制 → `chapter09_grpo_rlvr/grpo-practice-and-mechanism.md`

**子大纲：**

- 从 PPO 到 GRPO：为什么去掉 Critic
- 群体归一化原理：同 prompt 多 rollout 相对优势
- KL 约束与参考策略实现

### 18.2 R1-Zero 范式（DAPO）→ `chapter09_grpo_rlvr/deepseek-dapo.md`

**子大纲：**

- DAPO（字节+清华 2025.03，arXiv:2503.14476 NeurIPS 2025）
  - Clip-Higher：解耦 $\epsilon_{low} \neq \epsilon_{high}$
  - Dynamic Sampling：过滤全对/全错样本
  - Token-level Loss：避免长 response 主导
  - Overlong Filtering + Soft Shaping
- Dr.GRPO（Liu et al. 2025，arXiv:2508.10355）：移除 std 与长度归一化
- DeepSeek V3.2 的 KL 调参：zero KL、自验证 RLVR、mHC 残差稳定性（arXiv:2512.02556）

### 18.3 RLVR 可验证奖励 → `chapter09_grpo_rlvr/rlvr.md`

**子大纲：**

- RLVR 定义：规则反馈替代人工标注
- RLVR 的奖励来源：数学验证器、单元测试、形式化证明
- RLVR 与 RLHF 的混合流水线

### 18.4 GRPO 改进家族（Dr.GRPO / GSPO / CISPO / VAPO / RPT）→ `chapter09_grpo_rlvr/grpo-family.md`

**子大纲：**

- GSPO（Zheng et al. 2025，Qwen3，arXiv:2507.18071）：序列级 IS ratio + 序列级 clip
- CISPO（MiniMax 2025.06，arXiv:2506.13585）：Clip IS 权重而非 token 更新，2× speedup
- VAPO（字节 Seed 2025.04，arXiv:2504.05118）：Value-based 反潮流，长 CoT 打败 GRPO
- REINFORCE++（Hu 2025）/ AREAL（异步）/ ASPO / DCPO 等小众变体
- DAPO vs CISPO 选型对比
- RPT（Microsoft 2025.06，arXiv:2506.08007）：Reinforcement Pre-Training 挑战预训练/后训练二分法
- 选型决策树：任务类型 → 算法推荐映射

### 18.5 RL Environments 作为新瓶颈 → `chapter23_rl_environments/intro.md`

**子大纲：**

- Anthropic $1B 投资 RL Environments（The Information 2025.09）
- Wing VC 数据：Anthropic 年花数千万美元，2026 扩展 3-5 倍
- Karpathy："RLVR 是 LLM 训练流水线的新主要阶段"
- Mechanize 给 RL environments 工程师 $500K 年薪
- Evals = RL Environments（Pash 2025）：评测即训练，训练即评测

### 18.6 Verifier 与 Sandbox 工程 → `chapter23_rl_environments/verifier-sandbox.md`

**子大纲：**

- Verifier 设计原则：正确性、效率、抗作弊
- 形式化 verifier vs 启发式 verifier
- Sandbox 工程：Docker 隔离、代码执行沙箱、网络白名单、资源配额
- 多 agent 并行 sandbox 管理
- 长程任务 harness：Anthropic Effective Harnesses（2025.11）、Karpathy "5-6 agents" 模式

### 18.7 异步 RL 与评测基准 → `chapter23_rl_environments/async-eval.md`

**子大纲：**

- 同步 RL 训练（veRL、TRL、OpenRLHF 传统模式）
- 异步 RL 训练的动机：rollout 与 training 解耦
- AReaL（清华+智谱，arXiv:2505.24298）：Staleness-enhanced PPO，2.77× 加速
- AgentRL（智谱+清华）：Cross-policy sampling、Task advantage normalization
- SLIME / ROLL / LlamaRL / PRIME-RL / TOPLOC + SHARDCAST
- 评测基准：CyberGym、SWE-bench、Terminal-Bench、τ-bench、BFCL、WebArena、Vending-Bench、BrowseComp
- 训练-评估循环工程化：Eval-driven RL、增量评测、数据污染检测（对应第 30 章）

### 18.8 动手：金融 API 工具调用 GRPO → `chapter09_grpo_rlvr/financial-tool-calling-grpo.md`

**子大纲：**

- 金融 API 数据集
- GRPO + Tool Use 训练流程
- 工具调用准确率评估

### 18.9 OPD 在线蒸馏 → `chapter09_grpo_rlvr/on-policy-distillation.md`

**子大纲：**

- On-Policy Distillation 原理
- 与 RL 的协同训练
- 实战收益

### 18.10 动手：veRL 代码生成 RL → `chapter09_grpo_rlvr/verl-code-sandbox.md`

**子大纲：**

- veRL 框架上手
- 代码生成 RL 训练
- Sandbox 与 verifier 工程

---

## 第 19 章 Reasoning Models：从 o1 到 Claude Opus 4.6 `[多篇章节]` `[v5.1 扩展]`

> **目录**：`chapter13_reasoning_models/`（6 个文件，旧编号 13，与章节号 19 不一致——历史遗留）

### 19.1 推理模型的兴起：从 o1 到推理即产品 → `chapter13_reasoning_models/emergence-and-o1.md`

**子大纲：**

- OpenAI o1 → o3 → o4 演进
- Competitive Programming with Large Reasoning Models（OpenAI 2025.02，arXiv:2502.06807）
- 推理能力作为"涌现现象"的实证

### 19.2 R1-Zero 范式：无 SFT 的纯 RL → `chapter13_reasoning_models/intro.md`

**子大纲：**

- DeepSeek-R1-Zero（Nature 2025）：直接从基座模型做 RL
- reflection、verification、aha moment 自发涌现
- R1-Zero 的开源工业级对照（DAPO / VAPO / Qwen3）
- DeepSeek-R1 完整训练流程

### 19.3 Test-time Compute Scaling → `chapter13_reasoning_models/test-time-scaling.md`

**子大纲：**

- Test-time compute vs Train-time compute 的权衡
- Gemini 3 Pro Deep Think（2025.10）/ 3.1 Deep Think（2026.02）
- 并行推理"思考层"叠加在 MoE 上
- IMO 2025 金牌、HLE 48.4%、ARC-AGI-2 84.6%

### 19.4 Hybrid Thinking 与思考预算 → `chapter13_reasoning_models/hybrid-thinking.md`

**子大纲：**

- 单模型同时支持 think/non-think 双模式
- DeepSeek V3.1（2025.08）：Hybrid 模式融合
- Qwen3（arXiv:2505.09388 §4.3）：Thinking Mode Fusion + Thinking Budget
- NoThinking + Best-of-N：不思考也能达到 thinking 水平（Ma et al. arXiv:2505.18681）
- Thinking Budget 控制推理深度的工程实现
- 长 CoT 压缩：Kimi k1.5 long2short RL

### 19.5 自适应思考 → `chapter13_reasoning_models/adaptive-thinking.md`

**子大纲：**

- Claude Opus 4.6 的自适应思考深度
- Opus 4.6 内部 AI Research Eval Suite（LLM training / Text-RL / Quadruped-RL 子任务，34× 人类加速）
- Anthropic 2026 80 页 Constitution 与推理能力

### 19.6 推理链的可读性与对齐 → `chapter13_reasoning_models/cot-visibility-alignment.md`

**子大纲：**

- 推理过程对齐（Reasoning Alignment）
- Hidden CoT vs Visible CoT 的工程权衡
- 推理链的安全过滤
- Hidden CoT 中的潜在欺骗问题

---

## 第 20 章 过程奖励模型与推理时搜索 `[多篇章节]` `[v5.1 完整重构]`

> **设计理由**：三家独立调研都指出 v5 此章仍以判别式 PRM 为主，缺生成式和形式化两条新主线。
>
> **目录**：`chapter14_prm_search/`（7 个文件，旧编号 14，与章节号 21 不一致——历史遗留）

### 20.1 Outcome vs Process 奖励 → `chapter14_prm_search/outcome-vs-process.md`

**子大纲：**

- Outcome Reward 的稀疏性问题
- Process Reward 的细粒度优势
- 为什么 PRM 在长 CoT 任务里不可替代

### 20.2 判别式 PRM（经典路线）→ `chapter14_prm_search/discriminative-prm.md`

**子大纲：**

- OpenAI "Let's Verify Step by Step"（Lightman et al. 2023，arXiv:2305.20050）
- PRM800K 数据集与人工标注
- PRM 作为 Re-ranking 模型
- 局限：标注成本高、泛化弱

### 20.3 生成式 PRM（新路线）→ `chapter14_prm_search/generative-prm.md`

**子大纲：**

- ThinkPRM（arXiv:2504.16828）：生成式 PRM 优于判别式
- 标签少 100 倍的关键：让 verifier 自己生成评价
- 验证器计算扩展（Verifier Compute Scaling）
- PRM 综述（arXiv:2510.08049）：生成式 vs 判别式对比

### 20.4 形式化 PRM（终极 verifier）→ `chapter14_prm_search/formal-prm.md`

**子大纲：**

- Lean4 / Coq 作为天然 verifier：零误判
- AlphaProof（DeepMind 2024.07，IMO 银牌）：AlphaZero + Lean
- AlphaGeometry 2（DeepMind）：几何题专用形式化
- DeepSeek-Prover-V2（2025.04，arXiv:2504.21801）：Lean4 + RL with binary reward，MiniF2F 88.9%
- 形式化 PRM 的代价：形式语言稀缺、领域受限

### 20.5 推理时搜索 → `chapter14_prm_search/inference-time-search.md`

**子大纲：**

- Beam Search over Thoughts
- MCTS over Thoughts：树形展开
- Tree of Thoughts（ToT）
- AlphaCodium：代码生成搜索
- rStar：自我对弈搜索
- PaCoRe（Step3-VL-10B，ACL 2026）：16 路并行 rollout 聚合，AIME 2025: 94.4
- GenRM 与 Verifier 模型：Generative Reward Model、LLM-as-Judge、Self-Rewarding

### 20.6 并行协调推理与总结 → `chapter14_prm_search/parallel-reasoning-and-summary.md`

**子大纲：**

- PaCoRe vs DeepThink vs MCTS 的对比
- 从深度扩展转向并行广度扩展 TTC
- PRM 家族选型决策

---

## 第 21 章 Constitutional AI 与 RLAIF `[多篇章节]`

> **目录**：`chapter22_cai_rlvr/`

### 21.1 Constitutional AI 与 RLAIF 框架 → `chapter22_cai_rlvr/intro.md`

**子大纲：**

- Constitutional AI 框架（Anthropic 2022，arXiv:2212.08073）
- RLAIF：用 AI 反馈替代人类标注
- 自我修正（Self-Correction）与自我奖励（Self-Rewarding）
- CAI vs RLHF 的成本与效果对比

### 21.2 HHH 原则与 Claude 实践 → `chapter22_cai_rlvr/hhh-practice.md`

**子大纲：**

- HHH 对齐原则：Helpful, Harmless, Honest
- Claude 训练中的 CAI 实际应用
- Anthropic 2026 80 页 Constitution 的工程意义
- 与第 30 章奖励黑客的呼应

### 21.3 RLAIF 工程化与宪法扩展 → `chapter22_cai_rlvr/rlaif-engineering.md`

**子大纲：**

- RLAIF 的 prompt 工程与反馈模型选择
- 大规模 Constitution 的演化机制
- 成本对比：RLHF / RLAIF / CAI 的标注预算
- Anthropic 2025-2026 公开 Constitution 的版本演进

---

# Part V · Agentic 强化学习（4 章，`[v5 核心新增]`）

> **设计理由**：2025-2026 真实工业需求集中在此。Anthropic Code RL JD 60% 内容围绕 agentic，OpenAI Operator、Claude Computer Use、SWE-Agent 都在快速演进。原书一章远远不够。
>
> **合并说明**：v5.1 把 Part V 压缩为 3 章——原 第 24 章 多轮交互 RL 合并到工具调用（多轮+工具紧耦合），原 第 27 章 Deep Research 合并到 Computer Use（同属 agentic 应用）。**v5.2 反向拆分**：原 第 24 章 Deep Research + Computer Use + 多智能体的三合一硬拼被拆回——Deep Research 信息检索、Computer Use GUI 控制、多智能体协作 是三个不同任务，强行合并反而每个都讲不透。v5.2 最终：第 22 章（工具+多轮+多智能体）/ 第 23 章（代码）/ 第 24 章（Deep Research）/ 第 25 章（Computer Use），共 4 章。
>
> **目录**：当前 Part V 内容散落在 `chapter10_agentic_rl/`（14 个文件）和 `chapter15_rl_based_swe/`（5 个文件）。

## 第 22 章 工具调用、多轮交互与多智能体 RL `[多篇章节]`

> **目录**：当前散落在 `chapter10_agentic_rl/`（multi-turn-rl.md / tool-use-agents.md / tool-use-and-trajectory.md / trajectory-synthesis.md / industrial-practice.md / industrial-evaluation.md / multi-agent-swarm.md）
>
> **合并说明**：原 第 24 章 多轮交互 RL 合并到本章——多轮 MDP 与工具调用在 agentic RL 里是同一套建模与工程问题（轨迹信用分配、长程奖励、用户模拟器），拆开造成概念重复。v5.2 进一步把 多智能体协作 / Agent Swarm 也并入本章——LLM-era 多智能体本质是"多轮 + 多角色"的扩展，与 22.1-22.2 的多轮 MDP 建模共用同一套轨迹信用分配框架。

### 22.1 多轮 MDP 建模与信用分配 → `chapter10_agentic_rl/multi-turn-rl.md`

**子大纲：**

- 从单轮到多轮：轨迹信用分配的根本变化
- 多轮 MDP 建模：状态包含完整对话历史
- 长程任务奖励设计：稀疏 vs 稠密、过程奖励（PRM）的角色
- 与第 20 章 PRM 的衔接

### 22.2 用户模拟器与多轮 RL 实验 → `chapter10_agentic_rl/multi-turn-rl.md`

**子大纲：**

- 用户模拟器设计：rule-based / model-based / human-in-loop
- 多轮 vs 单轮 RL 的工程差异：context 管理、reward 延迟
- 实验：多轮对话 RL 训练流程
- 评测：τ-bench 多轮工具调用基准

### 22.3 工具调用 RL 入门 → `chapter10_agentic_rl/tool-use-and-trajectory.md`

**子大纲：**

- Tool Use 的动作空间扩展
- Function Calling 的轨迹建模
- Tool Reward 设计：执行结果 + 调用合理性
- ReAct / ToolFormer 范式

### 22.4 Search-Augmented RL → `chapter10_agentic_rl/tool-use-agents.md`

**子大纲：**

- Search-R1（arXiv:2503.09516）
- R1-Searcher（arXiv:2503.05592）
- 检索增强的 RL 训练流程

### 22.5 Code Interpreter RL 与工业实战 → `chapter10_agentic_rl/industrial-practice.md` + `industrial-evaluation.md`

**子大纲：**

- SimpleTIR / ReTool / AFM
- 实验：工具调用 GRPO 训练
- 工业评测：BFCL、τ-bench

### 22.6 多智能体协作与 Agent Swarm → `chapter10_agentic_rl/multi-agent-swarm.md` `[v5.2 新增]`

**子大纲：**

- LLM-era 多智能体 vs 经典 MARL（与第 14 章 CTDE/MADDPG/MAPPO 的差异）
- Anthropic Orchestrator-worker 模式（90.2% 加速）
- Karpathy "5-6 agents" 编排范式
- Agent Swarm 产品：Kimi K2.5（2026.01，arXiv:2602.02276）、Step 3.7 Flash Advisor Mode
- Self-Play 多智能体训练（呼应第 32 章自我博弈）
- 实验：双 agent 协作任务训练

---

## 第 23 章 代码智能体强化学习 `[多篇章节]` `[v5.1 重构：RL-based 为主线]`

> **设计理由**：2025-2026 主线是 RL-based SWE（SWE-RL → CWM → DeepSWE → SSR），SFT-only 范式已过时。
>
> **目录**：`chapter15_rl_based_swe/`（5 个文件，旧编号 15，与章节号 23 不一致——历史遗留）

### 23.1 任务定义与基准 → `chapter15_rl_based_swe/intro.md`

**子大纲：**

- SWE-bench：软件工程任务标准（Live/Verified）
- SWE-bench-Lite / SWE-bench Multimodal
- 评测指标：Resolved %、Pass@k、Edit Distance

### 23.2 SWE-RL 与基础实验 → `chapter15_rl_based_swe/swe-bench-and-rlvr.md`

**子大纲：**

- 第一代 SFT-based SWE Agent（SWE-Gym、SWE-Smith）的局限
- SWE-RL（Meta 2025.02，arXiv:2502.18449 NeurIPS'25）：11M GitHub PR + 规则奖励
- Llama3-70B 在 SWE-bench Verified 41%，首次观测到 "aha moment"
- rLLM / DeepCoder：工业级 RL-based SWE Agent

### 23.3 Code World Model 与 DeepSWE → `chapter15_rl_based_swe/world-model-and-deep-swe.md`

**子大纲：**

- Code World Model（CWM，Meta 2025.09，arXiv:2510.02387）：32B dense，SWE-bench 65.8%
- DeepSWE（Luo et al. 2025）
- World Model 范式：agent 学会"预测代码执行结果"

### 23.4 Self-Play SWE-RL 与总结 → `chapter15_rl_based_swe/self-play-ssr-and-summary.md` + `meta-swe-rl.md`

**子大纲：**

- Self-play SWE-RL（SSR，Meta 2025.12，arXiv:2512.18552）：单一策略双角色
- 自生成训练数据飞轮
- 代码 Verifier 设计（单元测试、Code Repair、SWE-RM arXiv:2512.21919）
- 长程自主工程能力（Effective Harnesses、Progress tracking、Test ratchet）
- 中国实验室代码 Agent（Qwen3-Coder 20000 并行环境、DeepSeek Coder、CodeGeeX）

---

## 第 24 章 Deep Research 与浏览器智能体 `[多篇章节]` `[v5.2 拆分聚焦]`

> **目录**：`chapter10_agentic_rl/deep-research-agent.md`（拆分为 3 节）
>
> **设计理由**：v5.1 把 Deep Research 和 Computer Use 合并，但二者任务定位不同——Deep Research 是**信息检索与综合**（多步搜索 + 答案聚合），Computer Use 是**通用 GUI 控制**（截图 + 点击 + 桌面自动化）。共享"点击/滚动"动作空间不构成合并理由。v5.2 拆分为两个聚焦章节，让各自方法讲透。

### 24.1 Deep Research 任务定义与多步检索 → `chapter10_agentic_rl/deep-research-agent.md`

**子大纲：**

- Deep Research 任务定义：从单次问答到多步研究
- Open Domain QA、金融调研、学术综述等场景
- 多步检索策略：query 改写、迭代搜索、信息聚合
- 与 RAG 的根本区别：agency 与 planning

### 24.2 浏览器 RL 动作空间与 harness 工程 → `chapter10_agentic_rl/browser-rl-harness.md`

**子大纲：**

- 浏览器智能体动作空间：搜索、点击、滚动、提取、回退
- Harness 工程与进度跟踪（claude-progress.txt、feature_list.json）
- 长程任务的内存管理：上下文压缩、关键信息缓存
- 奖励设计：答案正确性、检索效率、步数惩罚
- 金融问答与 Open Domain QA Agent 实战

### 24.3 评测基准与开源项目 → `chapter10_agentic_rl/deep-research-eval.md`

**子大纲：**

- BrowseComp（Meta）：浏览器智能体基准
- xbench-DeepSearch：深度研究评测
- GAIA：通用 AI 助手基准
- 开源复现项目：GPT-Researcher、Stanford STORM、OpenResearcher
- 实验端到端：从浏览器环境到训练 Deep Research Agent

---

## 第 25 章 Computer Use 与 GUI Agent `[多篇章节]` `[v5.2 新章拆分]`

> **目录**：`chapter28_computer_use/`
>
> **设计理由**：从原 第 24 章拆出。Computer Use 是 2025-2026 大方向（Anthropic Computer Use、OpenAI Operator、Google Project Mariner、字节 UI-TARS-2、智谱 AutoGLM），值得独立成章讲透 GUI 控制的建模、训练与安全。

### 25.1 Computer Use 范式与 GUI Grounding RL → `chapter28_computer_use/intro.md`

**子大纲：**

- Computer Use 范式（Anthropic Computer Use / OpenAI Operator / Google Project Mariner）
- 核心动作空间：点击、滚动、键入、截图
- GUI Grounding RL：Set-of-Mark、视觉 grounding、动作映射
- 视觉理解与动作对齐
- 桌面、移动、Web 三种环境的差异

### 25.2 GUI Agent 训练实践 → `chapter28_computer_use/training.md`

**子大纲：**

- UI-TARS-2（字节 Seed 2025.09，arXiv:2509.02544）：Multi-Turn RL + 异步 rollout + Stateful envs + Value pretraining
- AutoGLM / Open-AutoGLM（智谱 2025.12）：Self-evolving online curriculum RL
- MobileRL（arXiv:2509.18119）/ ComputerRL（arXiv:2508.14040）
- CogAgent（智谱）
- 训练数据合成：trajectory bootstrapping、人工演示采集

### 25.3 指令层级与 Prompt Injection 防御 → `chapter28_computer_use/safety-swarm.md`

**子大纲：**

- 指令层级（OpenAI 2024.04，arXiv:2404.13208）：系统/开发者/用户/工具指令的权限级别
- GPT-5 Mini-R 把指令层级用作 RL 奖励（+0.11~0.21）
- Prompt Injection 防御与"内核模式"类比
- GUI 语境下的安全挑战：恶意网页、伪造 UI、跨应用攻击
- 与第 30 章奖励黑客的呼应（agent 被劫持是奖励黑客的特殊形态）

---

# Part VI · 多模态强化学习（4 章 `[v5.1 新增视觉生成 RL]`）

## 第 26 章 视觉语言模型 RL `[多篇章节]` `[v5.1 扩展]`

> **目录**：`chapter11_vlm_rl/`（8 个文件，旧编号 11，与章节号 25 不一致——历史遗留）

### 26.1 VLM RL 训练基础 → `chapter11_vlm_rl/intro.md` + `vlm-frameworks.md`

**子大纲：**

- 视觉-语言联合表征
- 多模态奖励信号来源
- 视觉 token 与文本 token 的 RL 处理
- 训练框架：EasyR1 / R1-V / Open-Vision-Reasoner / Perception-R1

### 26.2 视觉奖励与挑战 → `chapter11_vlm_rl/vlm-challenges.md`

**子大纲：**

- 视觉问答正确性奖励
- 视觉描述完整性奖励
- 视觉幻觉惩罚
- 视觉推理的 "Missing Trace" 问题

### 26.3 视觉反思 RL `[v5.1 新增]` → `chapter11_vlm_rl/qwen3-vl-reflection.md`

**子大纲：**

- Qwen3-VL（2025.11.26）：Reflection-driven visual re-attention
- 视觉 grounding 的自我纠错
- 反思机制 + RL 联合训练

### 26.4 中国多模态前沿 → `chapter11_vlm_rl/vlm-grpo-hands-on.md`

**子大纲：**

- Step3-VL-10B（arXiv:2601.09668）：1000+ RL iterations
- GLM-4.6V：RLCS 课程采样
- Seed1.5-VL（字节，arXiv:2505.07062）：20B-A200B MoE，GUI agent + 游戏 RL
- PaCoRe 16 路并行 rollout 聚合 vs MCTS over Thoughts 对比

### 26.5 实验：GeoQA 几何推理 → `chapter11_vlm_rl/easyr1-geoqa.md`

**子大纲：**

- EasyR1 框架上手
- GeoQA 数据集
- VLM GRPO 训练流程

---

## 第 27 章 音频与语音 RL `[多篇章节]` `[v5.1 扩展]`

> **设计理由**：v5 此章只有大纲点，缺 Step-Audio MGRD 等核心方法。
>
> **目录**：`chapter30_audio_rl/`

### 27.1 音频语言模型概览与 Step-Audio 系列 → `chapter30_audio_rl/intro.md`

**子大纲：**

- 音频 token 化方案（codec、semantic/acoustic 分离）
- 语音生成与文本生成的差异
- 实时推理的工程挑战
- Step-Audio 系列 `[中国独特方向]`：Step-Audio-R1（arXiv:2511.15848）
- MGRD（Modality-Grounded Reasoning Distillation）
- Acoustic-Grounded Reasoning / Mind-Paced Speaking / Dual-Brain Architecture

### 27.2 RLVR → RLHF 演进与音频奖励设计 → `chapter30_audio_rl/reward-design.md`

**子大纲：**

- Step-Audio-R1.5：从 RLVR 转向 RLHF for Audio Reasoning
- 声音自然度 + 推理能力的多目标 RL
- 韵律（prosody）自然度保留
- 音频奖励设计：内容正确性 / 韵律自然度（人类偏好建模）/ 实时性
- 实验：简单语音对话 RL

### 27.3 多模态音频 Agent 与未来方向 → `chapter30_audio_rl/future.md`

**子大纲：**

- 实时语音 Agent：GPT-4o Voice、GLM-4.6 Voice、Step-Audio-2
- 情感感知与语音风格控制 RL
- 与 VLM 的统一：音视联合 RL
- 中国实验室音频 RL 路线对比

---

## 第 28 章 具身智能与 VLA 模型 `[多篇章节]` `[v5.1 升级旗舰案例]`

> **设计理由**：旧版用 RT-2 已过时，新基准是 Gemini Robotics 1.5 + π0 + Embodied Thinking。
>
> **目录**：`chapter12_future_trends/embodied-intelligence/`（旧编号 12，与章节号 31 不一致——历史遗留）

### 28.1 VLA 模型概览 → `chapter12_future_trends/embodied-intelligence/index.md`

**子大纲：**

- π0（Physical Intelligence 2024）：diffusion policy + VLM
- RT-2（Google 2023，作历史背景）
- OpenVLA（开源旗舰）
- Gemini Robotics 1.5（DeepMind 2025.09，旗舰）：VLA + ER 双模型、Embodied Thinking 范式、跨本体迁移（Apollo / Spot）

### 28.2 机器人学习基础

**子大纲：**

- 观测空间（视觉、本体感觉、力觉）
- 动作空间（关节角度、末端执行器位姿）
- 奖励函数设计

### 28.3 Diffusion Policy 与多模态融合

**子大纲：**

- Diffusion 模型作为策略
- 多模态动作分布
- 视觉-语言-动作 token 化与跨模态对齐

### 28.4 Sim-to-Real 与遥操作

**子大纲：**

- 域随机化（Domain Randomization）
- Sim-to-Real transfer 技术
- System Identification
- 人类示范采集、行为克隆预训练、RL 微调

### 28.5 实验：OpenVLA + RL 微调桌面抓取任务

---

## 第 29 章 视觉生成 RL `[多篇章节]` `[v5.1 全新章节]`

> **设计理由**：字节 Seed 在视频生成 RL 上是 2025-2026 最大创新源。DanceGRPO 把 GRPO 适配 diffusion，Seedance 多维 RLHF，LongCat-Video 多奖励 stacking——中国实验室的全球领先方向。
>
> **目录**：`chapter11_vlm_rl/visual-generation-rl.md` + `video-generation-modern.md`（暂存于 VLM 目录，建议拆出）

### 29.1 视觉生成任务定义 → `chapter11_vlm_rl/visual-generation-rl.md`

**子大纲：**

- 文生视频（Text-to-Video）
- 图生视频（Image-to-Video）
- 视频编辑与续写
- Diffusion + RL 基础：Diffusion 作为策略网络、Rectified Flow 的 RL 适配、与文本 LLM RL 的根本差异

### 29.2 DanceGRPO `[字节 Seed 创新]` → `chapter11_vlm_rl/visual-generation-rl.md`

**子大纲：**

- DanceGRPO（2025.05，arXiv:2505.07818）：把 GRPO 适配到 diffusion/flow
- 算法核心：diffusion step 作为 RL 时间步
- 与 DDPO 等先前方法对比
- Unified across 4 foundation models

### 29.3 多奖励视频 RLHF → `chapter11_vlm_rl/video-generation-modern.md`

**子大纲：**

- Seedance 1.0（字节，arXiv:2506.09113）：Foundational reward / Motion reward / Aesthetic reward / Refiner RLHF
- LongCat-Video（字节 2025.10，arXiv:2510.22200）：GRPO + 多奖励 stacking、LoRA stacking
- 视频生成的奖励模型：VisionReward、多维度奖励分解、人类偏好对齐

### 29.4 物理感知视频生成与实验

**子大纲：**

- Hailuo-02（MiniMax，physics-aware NCR 架构）
- 物理规律作为内在奖励
- 时序一致性约束
- 实验：用 DanceGRPO 训练简单视频生成模型

---

# Part VII · 安全、评估与研究前沿（3 章）`[v5.2 合并原 Part VII + VIII]`

> **合并说明**：v5.1 时 Part VII（安全/评估）只有 1 章单独成 Part 结构尴尬。v5.2 把原 Part VIII（研究前沿，2 章）合并进 Part VII，重命名为"安全、评估与研究前沿"，共 3 章。原 第 35 章 RL 评估方法论已合并到 第 30 章奖励黑客；原 第 34 章 可扩展监督与红队测试（纯 AI 安全哲学/工程流程，非 RL 技术）已移除；原 第 36 章 分布式训练已合并到 第 16 章。

## 第 30 章 奖励黑客与 RL 评估 `[多篇章节]` `[v5.1 扩展]`

> **目录**：`chapter16_alignment_failures/`（5 个文件）+ `chapter35_rl_evaluation/`

### 30.1 经典失败模式 → `chapter16_alignment_failures/classical-failures.md` + `intro.md`

**子大纲：**

- 奖励黑客完整分类法：Specification Gaming / Reward Tampering / Goodhart's Law
- Anthropic 2025.11 分类（arXiv:2511.18397）

### 30.2 RLVR 的"假性收益" `[v5.1 新增]` → `chapter16_alignment_failures/modern-incidents.md`

**子大纲：**

- 数据污染实证（arXiv:2507.10532 AAAI 2026）：Qwen 在 MATH-500 上的 "spurious reward RLVR" 收益主要来自数据污染
- GRPO clipping bias 导致记忆激活
- 评估 RLVR 真实收益的方法论
- 抗污染的评测设计

### 30.3 工业失败案例 `[v5.1 新增]`

**子大纲：**

- GPT-4o 谄媚回滚（OpenAI 2025.04-05）：用户反馈奖励稀释安全奖励、48 小时回滚
- 字节 Seed RLHF 数据 scaling：Reward hacking 与多样性衰减、Pre-PPO prompt 选择策略

### 30.4 Anthropic 失准研究 → `chapter16_alignment_failures/sleeper-and-faking.md`

**子大纲：**

- School of Reward Hacks（Gao et al. 2025.08）
- 自然涌现的失准（Anthropic 2025.11，arXiv:2511.18397）：HHH 奖励作为缓解
- Sleeper Agents（Hubinger et al. 2024.01，arXiv:2401.05566）
- Alignment Faking（Greenblatt et al. 2024.12，arXiv:2412.14093）
- In-Context Scheming（Apollo 2024.12，arXiv:2412.04984）
- Sycophancy to Subterfuge（Anthropic 2024，arXiv:2406.10162）
- METR：Frontier Models Reward Hacking（Von Arx et al. 2025）

### 30.5 防御机制与总结 → `chapter16_alignment_failures/scaling-and-defenses.md`

**子大纲：**

- Preference Models 与 Reward Hack Classifier
- By Construction：从架构上防止 hacking
- 多 verifier 集成
- 形式化验证作为终极防线（对应第 20 章）

### 30.6 评估原则与污染鲁棒性 → `chapter35_rl_evaluation/intro.md`

**子大纲：**

- 评估基准设计原则
- 污染与泄漏检测（呼应 29.2 节 RLVR 假性收益）
- 提示敏感性分析
- 分布外鲁棒性
- 行为评估 vs 能力评估
- 长程任务评估的挑战

### 30.7 现代评估 Harness 与内部基准 → `chapter35_rl_evaluation/harness.md`

**子大纲：**

- 标准化评测 harness：lm-eval-harness、BigCode Eval、τ-bench、BFCL
- Anthropic 内部 AI Research Eval Suite（Opus 4.6：LLM training / Text-RL / Quadruped-RL 子任务，34× 人类加速）
- Claude 4.6 自我评估与对抗基线
- LiveCodeBench、SWE-bench Verified 等动态基准

---

---

## 第 31 章 进化式 LLM 搜索与生成式世界模型 `[多篇章节]` `[v5.1 全新]`

> **设计理由**：v5 完全缺失 AlphaEvolve 和 Genie 3 这两条 2025-2026 最前沿方向。
>
> **目录**：`chapter12_future_trends/`（部分内容已在 llm-driven-discovery.md 中）

### 31.1 AlphaEvolve 范式 → `chapter12_future_trends/llm-driven-discovery.md`

**子大纲：**

- AlphaEvolve（DeepMind 2025.05）：LLM 提出 diff + 自动评估器打分 + 进化算法挑选
- 首次发现矩阵乘法 23% 加速
- 改进 50 余个开放数学问题
- AlphaEvolve 算法架构：evolutionary search + LLM proposal
- 与传统 RL 的差异：不是 policy gradient，是 search + LLM
- LLM 时代搜索算法的新范式

### 31.2 生成式世界模型作为 RL 环境

**子大纲：**

- Genie 3（DeepMind 2025.08）：实时可交互世界模型、720p/24fps 生成、World memory 多分钟一致性
- 生成式环境 vs 真实环境
- 无限 RL 训练课程：agent 在生成世界中学习
- AGI 通用世界模型基础

### 31.3 递归自我改进

**子大纲：**

- Anthropic Funded Research / 递归自我改进（2026.04）：Claude 自身做 AI 研究、内部基准 52× 加速
- "Claude Mythos Preview" 模型
- RL 训练 AI 做 AI 研究的终极愿景

---

## 第 32 章 自我博弈、规模化趋势与未来方向 `[多篇章节]`

> **目录**：`chapter12_future_trends/`（部分内容已在 self-play-outlook/ 和 rl-scaling-outlook.md 中）

### 32.1 自我博弈基础与 LLM 自我博弈 → `chapter12_future_trends/self-play-outlook/index.md`

**子大纲：**

- AlphaGo → AlphaZero → MuZero 演进
- 自我对弈的收敛性
- Self-play 在围棋 / 国际象棋 / 星际争霸的应用
- LLM 自我博弈与 SPIN
- Self-play SWE-RL（SSR）（对应第 23 章）
- Multi-agent debate as self-play
- Mode collapse 与多样性保护

### 32.2 RL Scaling Laws 与 Foundation Model RL → `chapter12_future_trends/rl-scaling-outlook.md`

**子大纲：**

- RL Scaling Laws（类比 Chinchilla）
- 奖励信号 vs 数据量 vs 模型规模
- RLVR 的 scaling 极限
- Foundation Model 作为 RL 的起点
- RLHF / RLVR / RLAIF / Agent RL 的统一视角
- Foundation Model RL 的未来形态

### 32.3 In-Context RL 与未来十年 → `chapter12_future_trends/llm-multi-agent-rl/index.md`

**子大纲：**

- In-Context RL 与 Algorithm Distillation（DeepMind 2022）
- 元学习与持续学习
- Karpathy 的"AGI 还需十年"反思
- 开放问题：信用分配、长程规划、泛化、安全
- 中国 vs 美国实验室的差异化路线
- 从对话模型到自主智能体的跨越

---

# 附录（8 部分）

## 附录 A · 训练调试手册 `[多篇章节]` `[v5.1 扩展]`

> **目录**：`appendix_common_pitfalls/`

### A.1 数值稳定性与崩溃诊断 → `appendix_common_pitfalls/intro.md`

**子大纲：**

- 常见训练崩溃诊断
- 梯度异常检测
- KL 散度爆炸处理
- 训练崩溃复现 checklist

### A.2 优化器与系统稳定性 → `appendix_common_pitfalls/optimizer-stability.md`

**子大纲：**

- MuonClip + QK-clip 优化器稳定性（Kimi K2）
- MoE + RL 训练的 router 干扰问题
- 异步 RL 训练的 staleness 调优

### A.3 Agent / Long-trace 排查清单 → `appendix_common_pitfalls/agentic-failure.md`

**子大纲：**

- Reward Hacking 早期信号（agent 特有形式）
- Function Call 解析失败排查清单
- 长轨迹 OOM 诊断决策树

---

## 附录 B · 强化学习工程实践 `[多篇章节]` `[v5.1 大幅扩展]`

> **目录**：`appendix_industrial_training/`

### B.1 同步与异步训练系统 → `appendix_industrial_training/intro.md`

**子大纲：**

- 同步 RL 训练系统底座（veRL、TRL）
- 异步 RL 训练系统（AReaL、AgentRL、SLIME、ROLL、LlamaRL）
- Staleness、Cross-policy sampling 工程实现

### B.2 Agent 沙箱与评测工程 → `appendix_industrial_training/agentic-rl-infra.md`

**子大纲：**

- Agent 沙箱工程
- 评测基准工程
- 长轨迹 rollout 资源管理

### B.3 指标词典与实战练习 → `appendix_industrial_training/metrics-exercises.md`

**子大纲：**

- 训练指标词典
- MoE + RL 训练工程（DeepSeek V3、Step Flash、GLM-4.5）
- 工业实战练习

---

## 附录 C · 核心算法实现 `[多篇章节]` `[v5.1 扩展]`

> **目录**：`appendix_code_cheatsheet/`

### C.1 SFT / PPO / DPO 实现 → `appendix_code_cheatsheet/intro.md`

**子大纲：**

- SFT 与 KL 散度实现
- PPO 与 GAE 实现
- DPO 家族实现

### C.2 GRPO 家族与 RPT 实现 → `appendix_code_cheatsheet/grpo-family.md`

**子大纲：**

- GRPO 基础实现
- GRPO 改进家族实现（DAPO、Dr.GRPO、GSPO、CISPO、VAPO）
- RPT（Reinforcement Pre-Training）实现
- DanceGRPO 适配 diffusion/flow

### C.3 采样、注意力与优化器实现 → `appendix_code_cheatsheet/numerical.md`

**子大纲：**

- Softmax 与交叉熵实现
- 采样方法实现（top-k, top-p, min-p）
- 注意力机制实现（MHA、GQA、MLA、DSA 稀疏注意力）
- MuonClip + QK-clip 优化器实现
- PRM 训练实现（判别式 + 生成式 + 形式化 Lean4）

---

## 附录 D · 学习资源与复现项目 `[多篇章节]`

> **目录**：`appendix_resources/`

### D.1 论文与课程索引 → `appendix_resources/intro.md`

**子大纲：**

- 必读论文清单（按主题分类，100+ 篇）
- 视频课程索引（CS285、CS234、Hugging Face Course）

### D.2 开源复现项目 → `appendix_resources/open-projects.md`

**子大纲：**

- 开源代码库索引（veRL / OpenRLHF / TRL / trl-X）
- 复现项目推荐（Sky-T1、Open-R1、Tülu 3）
- 中国实验室开源复现追踪

---

## 附录 E · 数学基础 `[多篇章节]`

> **目录**：`appendix_math/`

### E.1 线性代数与概率统计 → `appendix_math/probability-linear.md`

**子大纲：**

- 线性代数（贝尔曼矩阵、函数逼近、收敛性）
- 概率与统计（回报、价值、采样估计、GAE）

### E.2 微积分与信息论 → `appendix_math/calculus-information.md`

**子大纲：**

- 微积分与优化（梯度、PG、PPO、Adam）
- 信息论（熵、KL、交叉熵、互信息）

---

## 附录 F · 论文阅读路线图 `[多篇章节]` `[新增]`

> **目录**：`appendix_paper_reading/`

### F.1 经典与深度 RL 必读 → `appendix_paper_reading/classical-deep-rl.md`

**子大纲：**

- 经典 RL 必读（Sutton、Watkins、Mnih）
- 深度 RL 必读（DQN、A3C、PPO、SAC）

### F.2 LLM RL 与安全研究必读 → `appendix_paper_reading/llm-rl-safety.md`

**子大纲：**

- LLM RL 必读（InstructGPT、CAI、DPO、GRPO、R1）
- 安全研究必读（Sleeper Agents、Alignment Faking、Reward Hacking）
- 2025-2026 前沿（DAPO、GSPO、CISPO、PRM、PaCoRe）

---

## 附录 G · GPU 小时估算表 `[多篇章节]` `[新增]`

> **目录**：`appendix_gpu_hours/`

### G.1 预训练与后训练成本 → `appendix_gpu_hours/intro.md`

**子大纲：**

- 不同模型规模的预训练成本
- SFT / RLHF / RLVR 各阶段成本
- DeepSeek / Qwen / Step 公开训练数据参考

### G.2 自训预算规划 → `appendix_gpu_hours/budget-planning.md`

**子大纲：**

- 自训模型预算规划
- 算力采购 vs 云租用对比
- 不同 RL 范式的 GPU 小时单价

---

## 附录 H · 符号表与算法索引 `[多篇章节]` `[新增]`

> **目录**：`appendix_terminology/`（已存在）

### H.1 符号与缩写表 → `appendix_terminology/intro.md`

**子大纲：**

- 全书符号统一表
- 缩写表（RLHF、RLVR、PRM、CAI...）

### H.2 算法索引 → `appendix_terminology/algorithm-index.md`

**子大纲：**

- 算法名称索引（GRPO、PPO、DPO、SAC...）
- 算法族关系图（PPO → GRPO → DAPO / GSPO / CISPO / VAPO）
- 算法-章节交叉引用表

---

# 完整章节统计 `[v5.1 修订]`

> **v5.1（0622）合并记录**：原 38 章经 7 处合并压缩为 31 章——
> ①第 1 章概览 → 序章（内容重复）
> ②第 23 章 Verifiers → 第 18 章 GRPO 家族（verifier 是 RLVR 组件）
> ③第 24 章多轮交互 → 第 22 章工具调用（紧耦合）
> ④第 27 章 Deep Research → 第 24 章 Computer Use（共享 GUI RL 建模）
> ⑤第 34+35 章 Scalable Oversight + RL 评估 → 第 30 章 奖励黑客（同主题）
> ⑥第 36 章分布式训练 → 第 16 章 LLM RL 工业实战（重叠巨大）
>
> **v5.2 反向拆分记录**：原 第 24 章（Deep Research + Computer Use + 多智能体）三合一硬拼被拆回——
> ⑦Deep Research → 独立第 24 章（信息检索与浏览器 RL）
> ⑧Computer Use → 独立第 25 章（GUI 控制与指令层级安全）
> ⑨多智能体协作 / Agent Swarm → 并入第 22 章 22.6 节（LLM-era 多智能体本质是多轮 + 多角色）
> Part V 由 3 章扩展为 4 章，全书从 31 章变 32 章。

| Part     | 主题                        | 章节数    | 章节性质             |
| -------- | --------------------------- | --------- | -------------------- |
| 0        | 序章 · 导论                 | 7 节      | 多篇章节（3 个文件） |
| I        | 基础与经典 RL               | 6         | 6 多篇               |
| II       | 深度 RL                     | 5         | 5 多篇               |
| III      | 高级 RL 方法                | 3         | 3 多篇               |
| IV       | LLM 对齐与后训练            | 7         | 7 多篇               |
| V        | **Agentic RL**              | **4**     | 4 多篇               |
| VI       | **多模态 RL（含视觉生成）** | **4**     | 4 多篇               |
| VII      | 安全、评估与研究前沿        | **3**     | 3 多篇               |
| **总计** |                             | **32 章** | **0 单篇 + 32 多篇** |
| 附录     | A-H                         | 8 部分    | 0 单篇 + 8 多篇      |

---

# 与现有书的对比 `[v5.1 修订]`

| 维度            | 当前书               | **v5.1 最终**                                                              |
| --------------- | -------------------- | -------------------------------------------------------------------------- |
| 总章节数        | 12                   | **32**                                                                     |
| 序章            | 哲学论述在前         | **0.1 先动手玩 CartPole + 未来剧透**                                       |
| Agentic 内容    | 1 章浅               | **4 章深入 + 第 25 章指令层级 / UI-TARS-2 / K2.5**                         |
| 多模态          | 1 章浅               | **4 章（VLM / 音频 / VLA / 视觉生成）**                                    |
| 安全/对齐研究   | 0                    | **1 章合并（奖励黑客 + 评估）**                                            |
| GRPO 家族       | DAPO 一个            | **6+ 变体算法细节（DAPO / Dr.GRPO / GSPO / CISPO / VAPO）+ Verifier 工程** |
| PRM             | 判别式为主           | **生成式（ThinkPRM）+ 形式化（Lean4 / AlphaProof）**                       |
| RL Environments | 0                    | **合并到第 18 章 GRPO 家族 + 异步 RL（AReaL / AgentRL）**                  |
| 视觉生成 RL     | 0                    | **独立章节（DanceGRPO / Seedance / LongCat）**                             |
| 工程系统        | 附录                 | **正文 1 章 + 附录扩展**                                                   |
| 实战代码        | 部分                 | **每章带 lab**                                                             |
| 中国实验室覆盖  | DeepSeek 一个        | **DeepSeek / Qwen / Kimi / Zhipu / Step / ByteDance / MiniMax 全覆盖**     |
| 真实论文出处    | 无                   | **每个主题有 arXiv 编号 + 官方 URL**                                       |
| 前沿方向        | 0                    | **AlphaEvolve / Genie 3 / 递归自我改进 / RPT**                             |
| **标题层级**    | **混乱（X.Y 歧义）** | **强制三层：Part / 章节 / 文章 / 子大纲**                                  |

---

# 落地建议 `[v5.1 更新]`

**Phase 1（立即，零风险）**：序章重构 + 标题教材化 + 拆分第 3 章 MDP

**Phase 2（本月，P0）**：补 Part IV 核心

- 第 18 章 GRPO 家族 + Verifier 工程完整重构（DAPO / Dr.GRPO / GSPO / CISPO / VAPO / RPT）
- 第 19 章 Reasoning 补 Hybrid Thinking + long2short + 涌现证据
- 第 20 章 PRM 升级到生成式 + 形式化

**Phase 3（下季，P0）**：补 Part V Agentic

- 第 23 章代码 agent 重构为 RL-based SWE 主线
- 第 24 章 Deep Research / 第 25 章 Computer Use 各自独立成章
- 第 22 章新增 22.6 多智能体协作与 Agent Swarm

**Phase 4（下半，P0）**：补 Part VI 多模态

- 第 27 章音频 RL 深化 MGRD
- 第 28 章 VLA 升级 Gemini Robotics 1.5
- 第 29 章视觉生成 RL（DanceGRPO / Seedance）`[全新]`

**Phase 5（持续，P1）**：Part VII 安全 + 前沿

- 第 30 章 Reward Hacking + 评估（合并章）补 GPT-4o 回滚 / 数据污染 / Seed scaling
- 第 31 章 AlphaEvolve / Genie 3 / 递归自我改进 `[全新]`

**Phase 6（长期，P1-P2）**：附录扩充

- 附录 A 补 MuonClip + QK-clip
- 附录 B 补异步 RL 系统
- 附录 C 补 DanceGRPO / RPT / PRM 实现

**Phase 7（持续）**：旧目录重命名（可选）

- `chapter01_cartpole/` → `chapter02_cartpole/`
- `chapter02_dpo/` → 合并到 `chapter18_dpo/`
- `chapter03_mdp/` → 拆分为 `chapter04_mdp/`、`chapter05_value/`、`chapter06_dp_mc_td/`、`chapter07_q_learning/`
- ...（详见各章"历史遗留"注释）

---

# 关键论文出处速查 `[v5.1 更新]`

```
# GRPO 家族
[DeepSeek-R1] Nature 2025. https://www.nature.com/articles/s41586-025-09422-z
[DeepSeek-V3] arXiv:2412.19437
[DeepSeek V3.2 / DSA] arXiv:2512.02556
[DAPO] Yu et al. 2025.03. arXiv:2503.14476 NeurIPS 2025
[Dr.GRPO] Liu et al. 2025. arXiv:2508.10355
[GSPO] Zheng et al. 2025.07. arXiv:2507.18071 (Qwen3)
[CISPO] MiniMax 2025.06. arXiv:2506.13585 (M1)
[VAPO] ByteDance Seed 2025.04. arXiv:2504.05118
[REINFORCE++] Hu 2025
[RPT] Microsoft 2025.06. arXiv:2506.08007

# Qwen / Kimi
[Qwen3 Tech Report] arXiv:2505.09388
[Qwen3-Coder] qwenlm.github.io/blog/qwen3-coder
[Qwen 数据污染] arXiv:2507.10532 AAAI 2026
[Kimi k1.5] arXiv:2501.12599
[Kimi K2] arXiv:2507.20534
[Kimi K2.5] arXiv:2602.02276 / kimi.com/blog/kimi-k2-5
[Ma et al. NoThinking] arXiv:2505.18681

# 智谱 GLM
[GLM-4.5 ARC] arXiv:2508.06471
[GLM-4.6] HuggingFace zai-org/GLM-4.6
[GLM-5] arXiv:2602.15763
[AReaL] arXiv:2505.24298 NeurIPS 2025
[AgentRL] arXiv:2510.04206
[AutoGLM] xiao9905.github.io/AutoGLM
[MobileRL] arXiv:2509.18119
[ComputerRL] arXiv:2508.14040

# StepFun
[Step3-VL-10B] arXiv:2601.09668
[Step-Audio-R1] arXiv:2511.15848
[Step 3.5 Flash] arXiv:2602.10604
[PaCoRe] github.com/stepfun-ai/PaCoRe (ACL 2026)

# ByteDance
[Seed-Thinking-v1.5] arXiv:2504.13914
[Seed1.5-VL] arXiv:2505.07062
[UI-TARS-2] arXiv:2509.02544
[Seedance 1.0] arXiv:2506.09113
[DanceGRPO] arXiv:2505.07818
[LongCat-Video] arXiv:2510.22200

# PRM
[Let's Verify Step by Step] Lightman et al. OpenAI 2023. arXiv:2305.20050
[ThinkPRM] arXiv:2504.16828
[PRM Survey] arXiv:2510.08049

# 形式化 RL
[AlphaProof + AlphaGeometry 2] deepmind.google/blog/ai-solves-imo-problems-at-silver-medal-level
[DeepSeek-Prover-V2] arXiv:2504.21801

# Agentic
[SWE-RL] Meta 2025.02. arXiv:2502.18449 NeurIPS 2025
[Code World Model] arXiv:2510.02387
[Self-play SWE-RL (SSR)] arXiv:2512.18552
[Search-R1] arXiv:2503.09516
[R1-Searcher] arXiv:2503.05592
[SWE-RM] arXiv:2512.21919
[Effective Harnesses] anthropic.com/engineering/effective-harnesses-for-long-running-agents 2025.11
[Multi-Agent Research System] anthropic.com/engineering/multi-agent-research-system 2025.06
[Anthropic Code RL JD] job-boards.greenhouse.io/anthropic/jobs/4613568008

# 安全与对齐
[Constitutional AI] Bai et al. Anthropic 2022. arXiv:2212.08073
[Sleeper Agents] Hubinger et al. 2024.01. arXiv:2401.05566
[Alignment Faking] Greenblatt et al. 2024.12. arXiv:2412.14093
[In-Context Scheming] Apollo 2024.12. arXiv:2412.04984
[Sycophancy to Subterfuge] Anthropic 2024. arXiv:2406.10162
[Natural Emergent Misalignment] MacDiarmid et al. Anthropic 2025.11. arXiv:2511.18397
[School of Reward Hacks] Gao et al. 2025.08
[METR Frontier Reward Hacking] Von Arx et al. 2025
[GPT-4o Sycophancy Rollback] openai.com/index/sycophancy-in-gpt-4o 2025.05
[Instruction Hierarchy] OpenAI 2024.04. arXiv:2404.13208

# Anthropic 与 OpenAI
[Anthropic Funded Research / 递归自我改进] anthropic.com/institute/recursive-self-improvement 2026.04
[Opus 4.6] anthropic.com/news/claude-opus-4-6
[Competitive Programming with LRM] OpenAI 2025.02. arXiv:2502.06807
[Weak-to-Strong Generalization] OpenAI 2023

# DeepMind
[AlphaEvolve] deepmind.google/blog/alphaevolve + 论文 PDF
[Genie 3] deepmind.google/blog/genie-3-a-new-frontier-for-world-models
[Gemini 3 Deep Think] blog.google Gemini 3
[Gemini Robotics 1.5] storage.googleapis.com/deepmind-media/.../Gemini-Robotics-1-5-Tech-Report.pdf

# Meta
[Llama 4] ai.meta.com/blog/llama-4-multimodal-intelligence
[Llama Guard 4] huggingface.co/meta-llama/Llama-Guard-4-12B

# 工业 / 经济
[Wing VC RL Environments Market] wing.vc/content/rl-environments-for-agentic-ai
[Karpathy 2025 Year in Review] karpathy.bearblog.dev
[Epoch AI RL Environments FAQ] epochai.substack.com/p/an-faq-on-reinforcement-learning
[Raschka State of LLMs 2025] magazine.sebastianraschka.com/p/state-of-llms-2025
[Raschka LLM Papers 2025] magazine.sebastianraschka.com/p/llm-research-papers-2025-list-one
[DeepSeek CRFM Transparency] crfm.stanford.edu/fmti/December-2025/company-reports/DeepSeek_FinalReport_FMTI2025.html

# 评测基准
[τ-bench] Salesforce 2024-2025
[BFCL] Berkeley Function Calling Leaderboard
[WebArena] webarena.dev
[CyberGym] arXiv:2506.02548
[Vending-Bench] arXiv:2502.15840

# Tülu 3
[Tülu 3] Allen AI 2024-2025
```

---

# 对用户的最终诚实声明

1. **38 章是大工程**，但这是 MIT 级别教材应有的体量（Sutton & Barto 14 章、CS285 11 讲、本书额外覆盖 LLM / Agentic / Multimodal）。
2. **不是"读完就能入职"**，真实 JD 还要求：SE 工程能力、生产调试、分布式系统经验、产品 sense。本书覆盖知识部分。
3. **每章应有 lab / 实验**，真正动手才能内化。本书的 `code/` 目录已有部分，需要扩充。
4. **持续更新**：2026 年会有新论文、新模型，本书需要每季小修、每年大修。
5. **写作工作量估算**：38 章 × 每章约 3000-5000 字 + 代码 = 约 15-20 万字 + 大量代码。预计全职 6-12 个月。
6. **本版（0622）的核心修正**：强制三层标题层级（Part / 章节 / 文章 / 子大纲），消除旧版 X.Y 编号歧义，让"单篇章节的内部小节"和"多篇章节的独立文章"在视觉上彻底分开。
