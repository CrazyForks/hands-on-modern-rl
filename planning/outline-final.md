# 现代强化学习实战 — 完整 v5 大纲(MIT 级教材)

> 综合 v1→v2→v3→v4 的所有修正,基于真实证据(OpenAI/Anthropic/DeepSeek/Qwen/Zhipu/StepFun JD + 2025-2026 论文)的最终方案。

---

## 设计哲学

### 为什么用这个结构

1. **理论 → 实战 → 前沿** 三段递进,符合 Stanford CS285 + Sutton & Barto + D2L 风格
2. **每个 Part 对应一个清晰的学习目标**,可独立教学
3. **Agentic 与实战不是附录,是核心 Part V**,反映 2025-2026 真实工业需求
4. **每章统一结构**:本章导读 → 理论 → 实现 → 实验 → 总结 → 延伸阅读
5. **真实论文出处**:每个关键论点都有 arXiv 编号或公司技术报告链接

### 与现有书籍的对比

| 维度       | Sutton & Barto | CS285 | Raschka (in progress) | **本书 v5**      |
| ---------- | -------------- | ----- | --------------------- | ---------------- |
| 经典 RL    | ✅ 完整        | ✅    | ❌                    | ✅               |
| 深度 RL    | ❌             | ✅    | ❌                    | ✅               |
| LLM RL     | ❌             | 部分  | ✅                    | ✅ 完整          |
| Agentic RL | ❌             | ❌    | ❌                    | **✅ 独立 5 章** |
| 多模态 RL  | ❌             | ❌    | ❌                    | **✅ 独立 3 章** |
| 安全/对齐  | ❌             | ❌    | ❌                    | **✅ 独立 3 章** |
| 工程系统   | ❌             | ❌    | 部分                  | **✅ 独立 2 章** |

---

# 序章 · 导论(对应 docs/preface/)

> **设计理由**:本书承诺"先动手、后理论"。但当前 `preface/intro.md` 开篇却是 Sutton 苦涩的教训哲学论述,直到第 1 章才让读者碰代码——**导论自己违反了承诺**。v5 修正:序章 0.1 节直接放一个可立刻玩的 CartPole 入口,让读者 30 秒内看到智能体从摇晃到站稳;**玩过之后再讲为什么**。

## 0.1 先动手:30 秒玩转 CartPole **[v5 新前置]**

**三层体验,海内外双源部署,覆盖所有读者类型:**

**① 一键试玩(首选体验,零安装)— 双源部署**

- **主源 ModelScope 创空间**:`spaces.modelscope.cn/{namespace}/cartpole-playground`
  - 大陆访问稳定,本书主要受众(中文读者)默认走这里
  - gradio 应用,与 HF Space 代码几乎通用
- **副源 HuggingFace Space**:`hf.co/spaces/{namespace}/cartpole-playground`
  - 海外读者/镜像访问者走这里
- **页内呈现**:两个 iframe 标签切换 "🇨🇳 国内入口 / 🌍 海外入口"
  - 默认显示 ModelScope(主源)
  - 加载失败时自动 fallback 提示,引导切换
- 体验:点击 "Train" 按钮 → 实时看 reward 曲线爬升 → 训练完成自动播放最终策略

**② 一行命令本地跑(深度选项)**

- 醒目代码框:`pip install "gymnasium[classic-control]" stable-baselines3 && python 1-ppo_cartpole.py`
- 30 秒 CPU 训练完毕,弹出 `--gui` 小车演示窗口
- 衔接到 `code/chapter01_cartpole/` 完整代码
- 大陆读者:`pip install` 走清华/阿里镜像源,文档里给出配置说明

**③ 视觉预览(离线兜底)**

- 训练过程 GIF:reward 曲线从 20 爬到 500 的全过程动画(自托管在本书仓库 `docs/preface/images/`)
- 最终策略演示视频:智能体从摇摇晃晃到稳稳站立
- 双源都加载失败、不愿安装、离线读者的最低门槛体验

**承诺兑现点**:无论选哪层、身处哪个地区,此刻读者都已经"见过"一个智能体学会一件事——后续 0.2-0.6 都是回头解释刚才发生了什么。

---

### 经典 → 现代的呼应:**未来剧透**区 **[v5 新增]**

> CartPole 是 RL 的过去(1990s 起经典任务),本书的真正主角是 LLM 时代的现代 RL。序章末尾用 3-5 段短视频剧透**读完本书你能做到什么**——经典入口已能玩,现代入口先看见。

**剧透 1:DPO 让大模型学会"不盲从用户"**

- 训练前 vs 训练后对话对比 GIF(用户要求写恶意代码 → 训练前照办 / 训练后婉拒)
- 衔接第 18 章 DPO 家族

**剧透 2:DeepSeek-R1 推理能力涌现**

- R1-Zero 纯 RL 训练 CoT 自发拉长的视频
- 衔接第 19 章 GRPO 家族 + 第 20 章 Reasoning Models

**剧透 3:Computer Use 智能体操作浏览器**

- Claude Computer Use / OpenAI Operator 公开演示视频
- 衔接第 28 章计算机使用与 GUI Agent

**剧透 4:SWE-Agent 自主修 Bug**

- SWE-bench 上智能体读代码 → 定位 → 修改 → 通过测试的完整流程
- 衔接第 26 章代码智能体 RL

**呈现方式**:全部自托管 GIF/视频在 `docs/preface/images/teasers/`,避免外链失效。读者看完即知本书终点,但不在序章强行要求能玩 LLM(门槛太高)。

## 0.2 写在开头:为什么需要 RL(原 `preface/intro.md`「为什么需要强化学习」)

- 萨顿《苦涩的教训》与 70 年 AI 史的两条主线:搜索与学习
- 为什么试错是学习最原始的形态:骑自行车类比
- 从识别到决策:为什么监督学习无法覆盖连续决策
- RL 提供了什么:不告诉怎么做,只告诉什么好什么不好
- 衔接 0.1:你刚才在 CartPole 上看到的"试错 → 收敛"就是这一节的实例

## 0.3 什么是 RL:核心循环与关键术语(原 `preface/intro.md`「什么是强化学习」)

- 智能体-环境-状态-动作-奖励循环
- 轨迹、回报、折扣因子 $\gamma$
- 状态 vs 观测、离散 vs 连续动作空间
- 衔接第 1 章:把 0.1 的 CartPole 用这些术语重新描述一遍

## 0.4 强化学习简史(原 `preface/brief-history/`)

- 1950s–1980s:试错学习、Bellman 方程、TD 学习的诞生
- 1992:TD-Gammon——第一个战胜人类冠军的 RL 系统
- 2013:DQN 玩 Atari——深度 RL 元年
- 2016:AlphaGo 击败李世石
- 2017–2019:AlphaGo Zero、MuZero、自我博弈
- 2017:PPO 发布,成为工业标准(你刚才在 0.1 用的就是 PPO)
- 2022:InstructGPT / RLHF 进入大模型训练
- 2023–2024:DPO、GRPO、Constitutional AI
- 2025:DeepSeek-R1、o1/o3、RLVR 范式确立
- 中国实验室崛起:Qwen3 GSPO、Step-Audio、DeepSeek 透明度

## 0.5 环境安装指南(原 `preface/env-setup.md`)

- Python 环境:conda / venv 选择
- PyTorch 版本与 CUDA 配置
- Gymnasium 安装与验证
- veRL / OpenRLHF / TRL 工具链预告
- 训练硬件检查表:入门实验 / 核心实验 / 大型项目三档
- 仓库代码结构:`code/` 各章独立子目录

## 0.6 本书结构与读者路线图(原 `preface/intro.md`「关于本书」)

- 全书 8 大 Part 的递进逻辑:基础 → 深度 → 高级 → LLM → Agentic → 多模态 → 安全 → 前沿
- 三种背景读者的推荐路径:
  - ML 工程师:0.1 → Part IV–V(LLM + Agentic)
  - RL 背景读者:Part II–III + Part IV
  - 学生:从 Part I 循序渐进
- 符号约定与记号表(详见附录 H)

---

# Part I · 基础与经典强化学习(7 章)

## 第 1 章 强化学习概览

- 1.1 从序章的直觉到形式化定义
- 1.2 智能体-环境-奖励-状态的核心循环
- 1.3 现代应用版图:控制、游戏、对齐、智能体
- 1.4 RL 与监督学习、无监督学习的根本区别
- 1.5 本书后续章节的衔接

## 第 2 章 CartPole:第一个强化学习实验

- 2.1 CartPole 问题与 Gym/Gymnasium 接口
- 2.2 状态、动作、奖励的工程化定义
- 2.3 随机策略基线与失败模式
- 2.4 训练指标设计:回报曲线、成功率、稳定性
- 2.5 实验:从随机到收敛的完整流程

## 第 3 章 多臂老虎机与探索-利用理论

- 3.1 多臂老虎机问题与基础策略
- 3.2 ε-贪心与衰减调度
- 3.3 上置信界 (UCB) 算法
- 3.4 Thompson 采样与贝叶斯视角
- 3.5 遗憾界与 PAC 分析 **[理论]**
- 3.6 上下文老虎机(Contextual Bandits) **[对齐章节前置]**

## 第 4 章 马尔可夫决策过程

- 4.1 从老虎机到序列决策
- 4.2 马尔可夫性的数学定义与直觉
- 4.3 状态空间、动作空间、转移函数、奖励函数
- 4.4 折扣因子与回报
- 4.5 轨迹与回合
- 4.6 部分可观测 MDP (POMDP) **[LLM 多轮前置]**

## 第 5 章 价值函数与贝尔曼方程

- 5.1 状态价值函数 $V^\pi(s)$
- 5.2 动作价值函数 $Q^\pi(s,a)$
- 5.3 贝尔曼期望方程
- 5.4 贝尔曼最优方程
- 5.5 贝尔曼算子的压缩映射性质 **[理论]**
- 5.6 最优策略的存在性与唯一性

## 第 6 章 动态规划、蒙特卡洛与时序差分

- 6.1 动态规划:策略评估与策略改进
- 6.2 策略迭代与价值迭代
- 6.3 蒙特卡洛方法:首次访问与每次访问
- 6.4 时序差分 (TD) 学习:TD(0)
- 6.5 n-step Bootstrap 与 TD(λ)
- 6.6 资格迹 (Eligibility Traces)
- 6.7 三类方法的对比与权衡

## 第 7 章 Q-Learning 与离策略控制

- 7.1 在策略 (on-policy) 与离策略 (off-policy)
- 7.2 Q-Learning 算法与收敛性
- 7.3 SARSA 算法
- 7.4 重要性采样 (Importance Sampling) **[关键前置]**
- 7.5 函数逼近的挑战与 Deadly Triad **[理论]**
- 7.6 奖励函数设计:稀疏 vs 稠密、shaping、黑客

---

# Part II · 深度强化学习(5 章)

## 第 8 章 深度 Q 网络与 Distributional RL

- 8.1 从 Q-Learning 到 DQN 的动机
- 8.2 经验回放 (Experience Replay)
- 8.3 目标网络 (Target Network)
- 8.4 Double DQN:解决过估计
- 8.5 Dueling DQN:状态-动作价值分解
- 8.6 优先经验回放 (PER)
- 8.7 Distributional RL:C51、QR-DQN、IQN
- 8.8 Rainbow 与 NoisyNet
- 8.9 实验:LunarLander 与 Atari

## 第 9 章 策略梯度方法

- 9.1 策略梯度方法的动机(连续动作、随机策略)
- 9.2 策略表示:Softmax、高斯、Categorical
- 9.3 策略梯度定理(完整推导) **[理论]**
- 9.4 REINFORCE 算法
- 9.5 方差问题与基线 (Baseline)
- 9.6 Off-policy 策略梯度
- 9.7 实验:CartPole 与 Pendulum

## 第 10 章 Actor-Critic 架构

- 10.1 优势函数 $A(s,a) = Q(s,a) - V(s)$
- 10.2 Critic 网络的训练(价值函数拟合)
- 10.3 Actor-Critic 框架与同步更新
- 10.4 广义优势估计 (GAE) **[PPO 前置]**
- 10.5 A2C 与 A3C:异步并行
- 10.6 实验:Pendulum 与 BipedalWalker

## 第 11 章 PPO 与信任域方法

- 11.1 策略更新的稳定性问题
- 11.2 TRPO 与单调改进定理 **[理论]**
- 11.3 PPO-Clip 算法
- 11.4 PPO-Penalty 与自适应 KL
- 11.5 PPO 工程实现细节(entropy bonus、value clip)
- 11.6 长程任务中的 PPO
- 11.7 PPO 在 LLM RL 时代的位置(背景,详见第 19 章 GRPO 家族)
- 11.8 实验:BipedalWalker 连续控制

## 第 12 章 连续控制与基于模型的深度 RL

- 12.1 确定性策略梯度 (DPG)
- 12.2 DDPG 算法
- 12.3 TD3:目标策略平滑与双 Q
- 12.4 Soft Actor-Critic (SAC) 与最大熵 RL **[理论]**
- 12.5 Model-Based RL:Dyna、PETS、MBPO
- 12.6 AlphaZero 与 MuZero
- 12.7 Dreamer V3 与 World Models
- 12.8 Model-Based vs Model-Free 权衡

---

# Part III · 高级 RL 方法(3 章,精简但深入)

## 第 13 章 离线强化学习与决策 Transformer

- 13.1 离线 RL 的挑战:分布偏移
- 13.2 CQL、IQL、BCQ 的悲观主义
- 13.3 AWAC 与 TD3+BC
- 13.4 Decision Transformer:RL 作为序列建模
- 13.5 Trajectory Transformer 与 Diffuser
- 13.6 LLM 时代的离线 RL

## 第 14 章 模仿学习、反向 RL 与元 RL

- 14.1 行为克隆 (BC) 与 DAgger
- 14.2 数据集聚合 (DAgger)
- 14.3 最大熵逆向 RL (MaxEnt IRL)
- 14.4 GAIL:生成对抗模仿学习
- 14.5 元 RL:MAML、RL²、PEARL
- 14.6 In-Context RL 与 Algorithm Distillation **[DeepMind 2022]**

## 第 15 章 探索、多智能体与分层 RL

- 15.1 探索-利用的根本张力(理论回顾)
- 15.2 内在好奇心 (ICM) 与随机网络蒸馏 (RND)
- 15.3 NGU 与 Agent57
- 15.4 多智能体 RL:CTDE 框架
- 15.5 MADDPG 与 MAPPO
- 15.6 分层 RL:Options、FeUdal Networks、HIRO
- 15.7 **生成式世界模型作为 RL 环境**(Genie 3 引子,详见第 37 章)

---

# Part IV · 大语言模型对齐与后训练(8 章)

## 第 16 章 RLHF 训练流水线 **[v5.1 扩展]**

- 16.1 基座模型与指令对齐
- 16.2 现代三阶段范式:SFT → RLHF → RLVR
- 16.3 SFT 指令微调
- 16.4 奖励建模:Bradley-Terry 模型
- 16.5 RL 微调(PPO 或 GRPO)
- 16.6 KL 约束与参考策略
- 16.7 **双轨奖励设计**(Seed-Thinking, verifiable + pairwise)
- 16.8 **Pre-PPO**:Prompt 选择策略避免 reward hacking
- 16.9 Tülu 3:开源三阶段范式参考

## 第 17 章 LLM RL 工业实战 **[v5.1 扩展:从 PPO 转向 GRPO 现代流水线]**

> **设计理由**:Agent #2 直接点名 v5 此章以 PPO 经典实现为主已过时。2025-2026 Llama 4 / Qwen3 / DeepSeek V3.2 / GLM-4.6 已全面转向 GRPO/Dr.GRPO + RLVR。

### 17.1 训练框架对比

- 17.1.1 veRL(字节,主流)
- 17.1.2 OpenRLHF(开源友好)
- 17.1.3 TRL(HuggingFace 生态)
- 17.1.4 NeMo-Aligner(NVIDIA)
- 17.1.5 AReaL(清华+智谱,异步)
- 17.1.6 AgentRL(智谱+清华)
- 17.1.7 SLIME / ROLL / LlamaRL 对比

### 17.2 现代后训练流水线范式

- 17.2.1 **DeepSeek-R1 多阶段**:冷启动 SFT → 推理 RL → 拒绝采样 → 全场景 RL
- 17.2.2 **Llama 4 流水线**:轻量 SFT → online RL → 轻量 DPO + pass@k 难度过滤(arXiv:2504.13914 Llama 4 tech report)
- 17.2.3 **Qwen3 流水线**:Thinking Mode Fusion + Thinking Budget + GSPO(arXiv:2505.09388)
- 17.2.4 **GLM-4.5 / 4.6 流水线**:难度课程 RL + Hybrid Thinking + RLCS 课程采样(arXiv:2508.06471)
- 17.2.5 **GLM-5**(2026.02, arXiv:2602.15763):新异步 Agent RL + DSA 稀疏注意力 + 744B/28.5T
- 17.2.6 **Seed-Thinking-v1.5**:Dual-track reward(verifiable + pairwise)+ Pre-PPO + Hybrid reward(RTV+GenRM)(arXiv:2504.13914)

### 17.3 双轨奖励设计

- 17.3.1 Verifiable Reward(Math、Code)
- 17.3.2 Pairwise Preference Reward(开放对话)
- 17.3.3 Pre-PPO:Prompt 选择策略避免 reward hacking
- 17.3.4 Hybrid Reward:RTV + GenRM 组合

### 17.4 优化器与训练稳定性

- 17.4.1 AdamW 在 RL 训练中的稳定性问题
- 17.4.2 **MuonClip 优化器**(Kimi K2, arXiv:2507.20534 §3.2)
- 17.4.3 QK-clip:注意力数值稳定性
- 17.4.4 KL 爆炸的早期信号与处理

### 17.5 训练成本估算

- 17.5.1 不同模型规模的预训练成本
- 17.5.2 SFT / RLHF / RLVR 各阶段 GPU 小时
- 17.5.3 DeepSeek 公开数据参考:V3 预训练 2.664M H800 小时、R1-Zero 128K GPU 小时
- 17.5.4 自训模型预算规划

### 17.6 工业实战:GSM8K 与 AIME

- 17.6.1 实验:用 GRPO 训练 GSM8K
- 17.6.2 实验:用 DAPO 训练 AIME 2024
- 17.6.3 完整开源复现:Open-R1 / Sky-T1 / Tülu 3

### 17.7 中国对齐团队面试常见考点

- 17.7.1 PG → REINFORCE → TRPO → PPO → GRPO 完整推导链(智谱真题)
- 17.7.2 DPO 家族 + DPO 正则化
- 17.7.3 DeepSpeed vs Megatron 工程对比
- 17.7.4 训练资源消耗现场推算

## 第 18 章 偏好对齐:DPO 家族 **[v5.1 扩展]**

- 18.1 DPO 的数学推导(从 RLHF 目标推导)
- 18.2 DPO 训练动态分析
- 18.3 IPO:解决 DPO 的过拟合
- 18.4 KTO:无需成对偏好数据
- 18.5 SimPO:无参考策略方法
- 18.6 **DPO 正则化方法**(智谱面试真题)
- 18.7 Iterative DPO 与 ReST
- 18.8 自我对弈微调 (SPIN)
- 18.9 DPO 家族选型决策树

## 第 19 章 GRPO 家族:从群体归一化到改进谱系 **[v5.1 完整重构]**

> **设计理由**:2025-2026 最大算法焦点。四家独立调研(OpenAI/Anthropic、DeepMind/Meta、DeepSeek/Qwen/Kimi、Zhipu/StepFun/ByteDance/MiniMax)一致指出 v5 此章只列名字无算法细节。v5.1 按改进方向重新组织,覆盖 6+ 主流变体的算法差异。

### 19.1 GRPO 基础

- 19.1.1 从 PPO 到 GRPO:为什么去掉 Critic
- 19.1.2 群体归一化原理:同 prompt 多 rollout 相对优势
- 19.1.3 KL 约束与参考策略实现

### 19.2 改进方向 A:修正归一化偏差

- 19.2.1 **Dr.GRPO**(Liu et al. 2025, arXiv:2508.10355):移除 std 归一化与长度归一化,避免 reward hacking 和长度膨胀
- 19.2.2 DeepSeek V3.2 的 KL 调参:数学任务 zero KL、自验证 RLVR、mHC 残差稳定性(arXiv:2512.02556)

### 19.3 改进方向 B:序列级重要性采样

- 19.3.1 **GSPO**(Zheng et al. 2025, Qwen3, arXiv:2507.18071):序列级 IS ratio + 序列级 clip,专为 MoE RL 训练稳定设计
- 19.3.2 Qwen3 全系采用 GSPO 的实证收益

### 19.4 改进方向 C:裁剪策略对比

- 19.4.1 **DAPO**(字节+清华 2025.03, arXiv:2503.14476 NeurIPS 2025):
  - Clip-Higher:解耦 $\epsilon_{low} \neq \epsilon_{high}$
  - Dynamic Sampling:过滤全对/全错样本
  - Token-level Loss:避免长 response 主导
  - Overlong Filtering + Soft Shaping
- 19.4.2 **CISPO**(MiniMax 2025.06, arXiv:2506.13585):
  - Clip IS 权重而非 token 更新
  - 保留所有 token 梯度,避免 token 丢失
  - Lightning attention 下精度对齐
  - 2× speedup vs DAPO
- 19.4.3 DAPO vs CISPO 选型对比

### 19.5 改进方向 D:Value-based 反潮流

- 19.5.1 **VAPO**(字节 Seed 2025.04, arXiv:2504.05118):
  - Value-based Augmented PPO
  - 长 CoT 场景下 value model 重新打败 GRPO
  - AIME 60.4(超越同期 GRPO 变体)
- 19.5.2 VAPO 的启示:Critic-free 不是唯一答案

### 19.6 其他工业变体

- 19.6.1 REINFORCE++(Hu 2025)
- 19.6.2 AREAL(异步 RL 框架,清华+智谱)
- 19.6.3 ASPO / DCPO 等小众变体

### 19.7 RLVR:可验证奖励范式

- 19.7.1 RLVR 定义:规则反馈替代人工标注
- 19.7.2 RLVR 的奖励来源:数学验证器、单元测试、形式化证明
- 19.7.3 RLVR 与 RLHF 的混合流水线

### 19.8 预训练阶段 RL:Reinforcement Pre-Training **[v5.1 全新概念]**

- 19.8.1 RPT(Microsoft 2025.06, arXiv:2506.08007):把 next-token 预测重构为带内在二元奖励的推理任务
- 19.8.2 RPT 与预训练扩展性媲美:挑战预训练/后训练二分法
- 19.8.3 2025 最大概念转变:RL 不再只是后训练

### 19.9 中国实验室实战对比

- 19.9.1 DeepSeek-R1 训练全流程:冷启动 SFT → 推理 RL → 拒绝采样 → 全场景 RL
- 19.9.2 Qwen3 GSPO + Thinking Mode Fusion
- 19.9.3 MiniMax M1 CISPO + Lightning Attention
- 19.9.4 字节 Seed DAPO + VAPO 双线
- 19.9.5 Kimi K2 MuonClip + QK-clip

### 19.10 选型决策树

- 任务类型 → 算法推荐映射表
- 显存 / 训练时长 / 收敛稳定性的三维权衡

## 第 20 章 Reasoning Models:从 o1 到 Claude Opus 4.6 **[v5.1 扩展]**

### 20.1 推理模型的兴起

- 20.1.1 OpenAI o1 → o3 → o4 演进
- 20.1.2 **Competitive Programming with Large Reasoning Models**(OpenAI 2025.02, arXiv:2502.06807):
  - 端到端通用 RL 在 IOI/Codeforces 上优于特定领域流水线
  - **复杂测试时推理从端到端 RL 自然涌现**,而非人工设计
- 20.1.3 推理能力作为"涌现现象"的实证

### 20.2 R1-Zero 范式:无 SFT 的纯 RL

- 20.2.1 **DeepSeek-R1-Zero**(Nature 2025, nature.com/articles/s41586-025-09422-z):
  - 直接从基座模型做 RL,无中间 SFT
  - reflection、verification、aha moment 自发涌现
- 20.2.2 R1-Zero 的开源工业级对照:
  - **DAPO**(字节+清华):AIME 2024 用 50% 步数超越 R1-Zero
  - **VAPO**(字节 Seed):Value-based 反潮流路线
  - **Qwen3**:Thinking Mode Fusion + GSPO
- 20.2.3 DeepSeek-R1 完整训练流程(冷启动 + 推理 RL + 拒绝采样 + 全场景 RL)

### 20.3 Test-time Compute Scaling

- 20.3.1 Test-time compute vs Train-time compute 的权衡
- 20.3.2 **Gemini 3 Pro Deep Think**(2025.10)/ **3.1 Deep Think**(2026.02):
  - 并行推理"思考层"叠加在 MoE 上
  - IMO 2025 金牌、HLE 48.4%、ARC-AGI-2 84.6%
- 20.3.3 Deep Think 作为 test-time scaling 的旗舰案例

### 20.4 Hybrid Thinking 与思考预算 **[v5.1 新增]**

- 20.4.1 单模型同时支持 think/non-think 双模式
- 20.4.2 **DeepSeek V3.1**(2025.08):Hybrid 模式融合
- 20.4.3 **Qwen3**(arXiv:2505.09388 §4.3):Thinking Mode Fusion + Thinking Budget
- 20.4.4 **NoThinking + Best-of-N**:不思考也能达到 thinking 水平(Ma et al. arXiv:2505.18681)
- 20.4.5 Thinking Budget 控制推理深度的工程实现

### 20.5 长 CoT 压缩 **[v5.1 新增]**

- 20.5.1 **Kimi k1.5 long2short RL**(arXiv:2501.12599 §2.4, §3.4):
  - 长 CoT 蒸馏到短 CoT 的 RL 方法
  - Length penalty 控制
- 20.5.2 推理效率与质量的平衡

### 20.6 Hidden CoT vs Visible CoT

- 20.6.1 OpenAI o1/o3 Hidden CoT 的工程动机
- 20.6.2 DeepSeek-R1 Visible CoT 的开放策略
- 20.6.3 CoT 可读性 vs 推理能力的权衡

### 20.7 自适应思考

- 20.7.1 **Claude Opus 4.6** 的自适应思考深度
- 20.7.2 Opus 4.6 内部 AI Research Eval Suite(LLM training / Text-RL / Quadruped-RL 子任务,34× 人类加速)
- 20.7.3 Anthropic 2026 80 页 Constitution 与推理能力

### 20.8 推理链的可读性与对齐

- 20.8.1 推理过程对齐(Reasoning Alignment)
- 20.8.2 推理链的安全过滤
- 20.8.3 Hidden CoT 中的潜在欺骗问题

## 第 21 章 过程奖励模型与推理时搜索 **[v5.1 完整重构]**

> **设计理由**:三家独立调研(OpenAI/Anthropic、DeepMind/Meta、DeepSeek/Qwen/Kimi)都指出 v5 此章仍以判别式 PRM 为主,缺生成式和形式化两条新主线。

### 21.1 Outcome 奖励 vs Process 奖励

- 21.1.1 Outcome Reward 的稀疏性问题
- 21.1.2 Process Reward 的细粒度优势
- 21.1.3 为什么 PRM 在长 CoT 任务里不可替代

### 21.2 判别式 PRM(经典路线)

- 21.2.1 OpenAI "Let's Verify Step by Step"(Lightman et al. 2023, arXiv:2305.20050)
- 21.2.2 PRM800K 数据集与人工标注
- 21.2.3 PRM 作为 Re-ranking 模型
- 21.2.4 局限:标注成本高、泛化弱

### 21.3 生成式 PRM(新路线)

- 21.3.1 ThinkPRM(arXiv:2504.16828):生成式 PRM 优于判别式
- 21.3.2 标签少 100 倍的关键:让 verifier 自己生成评价
- 21.3.3 验证器计算扩展(Verifier Compute Scaling)
- 21.3.4 PRM 综述(arXiv:2510.08049):生成式 vs 判别式对比

### 21.4 形式化 PRM(终极 verifier)

- 21.4.1 Lean4 / Coq 作为天然 verifier:零误判
- 21.4.2 **AlphaProof**(DeepMind 2024.07,IMO 银牌):
  - AlphaZero 算法 + Lean 形式语言
  - 百万级问题自训练
  - 自博弈证明
- 21.4.3 **AlphaGeometry 2**(DeepMind):几何题专用形式化
- 21.4.4 **DeepSeek-Prover-V2**(2025.04, arXiv:2504.21801):
  - Lean4 形式定理证明 + RL with binary reward
  - MiniF2F 88.9%
- 21.4.5 形式化 PRM 的代价:形式语言稀缺、领域受限

### 21.5 推理时搜索

- 21.5.1 Beam Search over Thoughts
- 21.5.2 MCTS over Thoughts:树形展开
- 21.5.3 Tree of Thoughts(ToT)
- 21.5.4 AlphaCodium:代码生成搜索
- 21.5.5 rStar:自我对弈搜索

### 21.6 并行协调推理(PaCoRe)

- 21.6.1 **PaCoRe**(Step3-VL-10B, ACL 2026, github.com/stepfun-ai/PaCoRe)
- 21.6.2 16 路并行 rollout 聚合
- 21.6.3 outcome-based RL 训练 reasoning synthesis
- 21.6.4 从深度扩展转向并行广度扩展 TTC
- 21.6.5 AIME 2025: 94.4
- 21.6.6 PaCoRe vs DeepThink vs MCTS 的对比

### 21.7 GenRM 与 Verifier 模型

- 21.7.1 Generative Reward Model:把 verification 转为生成
- 21.7.2 LLM-as-Judge 范式
- 21.7.3 Self-Rewarding Language Models

## 第 22 章 Constitutional AI 与 RLAIF

- 22.1 Constitutional AI 框架(Anthropic 2022)
- 22.2 RLAIF:用 AI 反馈替代人类标注
- 22.3 自我修正与自我奖励
- 22.4 HHH 对齐原则:Helpful, Harmless, Honest
- 22.5 Claude 训练中的 CAI 实际应用
- 22.6 Anthropic 2026 80 页 Constitution

## 第 23 章 RL Environments 与 Verifiers 设计 **[v5.1 扩展]**

> **设计理由**:三家独立调研都指出 v5 此章缺异步 RL 训练系统。附录 B 也只讲同步 veRL 已过时。

### 23.1 RL Environments 作为新瓶颈

- 23.1.1 Anthropic $1B 投资(The Information 2025.09)
- 23.1.2 Wing VC 数据:Anthropic 年花数千万美元,2026 扩展 3-5 倍(wing.vc 报告)
- 23.1.3 Karpathy:"RLVR 是 LLM 训练流水线的新主要阶段"
- 23.1.4 Mechanize 给 RL environments 工程师 $500K 年薪

### 23.2 Evals 与 RL Environments 的等价性

- 23.2.1 Evals = RL Environments(Pash 2025)
- 23.2.2 评测即训练,训练即评测

### 23.3 Verifier 设计原则

- 23.3.1 正确性(Correctness)
- 23.3.2 效率(Efficiency)
- 23.3.3 抗作弊(Anti-gaming)
- 23.3.4 形式化 verifier vs 启发式 verifier

### 23.4 Sandbox 工程

- 23.4.1 Docker 容器隔离
- 23.4.2 代码执行沙箱
- 23.4.3 网络白名单与资源配额
- 23.4.4 多 agent 并行 sandbox 管理

### 23.5 长程任务 harness

- 23.5.1 **Anthropic Effective Harnesses**(2025.11):
  - 初始化器 agent + 增量编码 agent 模式
  - `claude-progress.txt` 进度跟踪
  - `feature_list.json` 状态文件
  - Test ratchet 测试棘轮
- 23.5.2 Karpathy 的 "5-6 agents" 模式

### 23.6 同步 vs 异步 RL 训练 **[v5.1 新增]**

- 23.6.1 同步 RL 训练(veRL、TRL、OpenRLHF 传统模式)
- 23.6.2 异步 RL 训练的动机:rollout 与 training 解耦
- 23.6.3 **AReaL**(清华+智谱, arXiv:2505.24298 NeurIPS 2025):
  - 全异步 rollout-training 解耦
  - Staleness-enhanced PPO
  - 2.77× 加速
- 23.6.4 **AgentRL**(智谱+清华, arXiv:2510.04206):
  - Cross-policy sampling
  - Task advantage normalization
- 23.6.5 **SLIME / ROLL / LlamaRL / PRIME-RL**(分布式异步 RL)
- 23.6.6 TOPLOC + SHARDCAST 分布式编码(INTELLECT-2)
- 23.6.7 Staleness、cross-policy sampling 的工程实现

### 23.7 评测基准

- 23.7.1 CyberGym(arXiv:2506.02548)
- 23.7.2 SWE-bench(Live/Verified/Multimodal)
- 23.7.3 Terminal-Bench
- 23.7.4 **τ-bench**(Salesforce):多轮工具调用
- 23.7.5 **BFCL**(Berkeley Function Calling Leaderboard)
- 23.7.6 WebArena / VisualWebArena
- 23.7.7 Vending-Bench(arXiv:2502.15840)
- 23.7.8 BrowseComp / xbench-DeepSearch

### 23.8 训练-评估循环工程化

- 23.8.1 Eval-driven RL Training
- 23.8.2 增量评测:每 N 步采样评估
- 23.8.3 数据污染检测(对应第 33 章)

---

# Part V · Agentic 强化学习(5 章,**v5 核心新增**)

> **设计理由**:2025-2026 真实工业需求集中在此。Anthropic Code RL JD 60% 内容围绕 agentic,OpenAI Operator、Claude Computer Use、SWE-Agent 都在快速演进。原书一章远远不够。

## 第 24 章 多轮交互强化学习

- 24.1 从单轮到多轮:轨迹信用分配
- 24.2 多轮 MDP 建模
- 24.3 用户模拟器设计
- 24.4 长程任务奖励设计
- 24.5 多轮 vs 单轮 RL 的工程差异
- 24.6 实验:多轮对话 RL 训练

## 第 25 章 工具调用与函数调用 RL

- 25.1 Tool Use 的动作空间扩展
- 25.2 Function Calling 的轨迹建模
- 25.3 Tool Reward 设计:执行结果 + 调用合理性
- 25.4 ReAct / ToolFormer 范式
- 25.5 Search-Augmented RL:Search-R1、R1-Searcher
- 25.6 Code Interpreter RL:SimpleTIR、ReTool、AFM
- 25.7 实验:工具调用 GRPO 训练

## 第 26 章 代码智能体强化学习 **[v5.1 重构:RL-based 为主线]**

> **设计理由**:Agent #2(DeepMind/Meta)直接点名 v5 此章仍以 SFT-only SWE-agent 为例已过时。2025-2026 主线是 RL-based SWE(SWE-RL → CWM → DeepSWE → SSR)。

### 26.1 任务定义与基准

- 26.1.1 SWE-bench:软件工程任务标准(Live/Verified)
- 26.1.2 SWE-bench-Lite / SWE-bench Multimodal
- 26.1.3 评测指标:Resolved %、Pass@k、Edit Distance

### 26.2 第一代:SFT-based SWE Agent(背景)

- 26.2.1 SWE-Gym 与训练数据生成
- 26.2.2 SWE-Smith:大规模数据合成
- 26.2.3 局限:SFT 无法学到长程纠错能力

### 26.3 第二代:RL-based SWE(主线)

- 26.3.1 **SWE-RL**(Meta 2025.02, arXiv:2502.18449 NeurIPS'25):
  - 11M GitHub PR + 规则奖励
  - Llama3-70B 在 SWE-bench Verified 41%
  - **首次观测到 "aha moment"**
- 26.3.2 rLLM / DeepCoder:工业级 RL-based SWE Agent
- 26.3.3 实现细节:parallel envs、reward shaping、context management

### 26.4 第三代:Code World Model

- 26.4.1 **Code World Model (CWM)**(Meta 2025.09, arXiv:2510.02387):
  - 32B dense
  - Python 执行轨迹 mid-training
  - GRPO 后训练
  - SWE-bench 65.8%
- 26.4.2 DeepSWE(Luo et al. 2025)
- 26.4.3 World Model 范式:agent 学会"预测代码执行结果"

### 26.5 第四代:Self-Play 范式

- 26.5.1 **Self-play SWE-RL (SSR)**(Meta 2025.12, arXiv:2512.18552):
  - 单一策略双角色(bug-injector + bug-solver)
  - 无需人工 issue 描述
  - SWE-bench Verified +10.4
- 26.5.2 自生成训练数据飞轮

### 26.6 代码 Verifier 设计

- 26.6.1 单元测试作为 reward signal
- 26.6.2 Code Repair 作为过程奖励
- 26.6.3 Process Reward Model for code
- 26.6.4 **SWE-RM**(执行免奖励模型, arXiv:2512.21919)
- 26.6.5 混合奖励:rule + model

### 26.7 长程自主工程能力

- 26.7.1 Anthropic Effective Harnesses for Coding Agents
- 26.7.2 Progress tracking(`claude-progress.txt`、`feature_list.json`)
- 26.7.3 Test ratchet 模式
- 26.7.4 初始化器 agent + 增量编码 agent 模式

### 26.8 中国实验室代码 Agent

- 26.8.1 **Qwen3-Coder**(2025.07):Long-horizon Agent RL,20000 并行环境
- 26.8.2 DeepSeek Coder 训练实践
- 26.8.3 智谱 CodeGeeX 训练方法

### 26.9 实验

- 26.9.1 用 SWE-RL 算法训练一个开源代码 Agent
- 26.9.2 复现 SWE-bench Verified 30-40% 基线

## 第 27 章 Deep Research 与 Web 智能体

- 27.1 Deep Research 任务定义
- 27.2 多步检索与信息聚合
- 27.3 浏览器智能体 RL
- 27.4 Anthropic Effective Harnesses for Long-Running Agents
- 27.5 BrowseComp / xbench-DeepSearch 评测
- 27.6 金融问答与 Open Domain QA Agent
- 27.7 实验:Deep Research Agent 训练

## 第 28 章 计算机使用与多智能体协作 **[v5.1 扩展]**

### 28.1 Computer Use 范式

- 28.1.1 Anthropic Computer Use
- 28.1.2 OpenAI Operator
- 28.1.3 Google Project Mariner
- 28.1.4 Computer Use 的核心动作空间(点击、滚动、键入、截图)

### 28.2 GUI Grounding RL

- 28.2.1 屏幕元素定位(Set-of-Mark、视觉 grounding)
- 28.2.2 动作映射:从像素到鼠标键盘事件
- 28.2.3 视觉理解与动作对齐

### 28.3 GUI Agent 训练实践 **[v5.1 新增中国实验室]**

- 28.3.1 **UI-TARS-2**(字节 Seed 2025.09, arXiv:2509.02544):
  - Multi-Turn RL for GUI Agent
  - Asynchronous rollouts + streaming training pool
  - Stateful envs + hybrid GUI-SDK
  - Value pretraining
- 28.3.2 **AutoGLM / Open-AutoGLM**(智谱 2025.12):
  - Self-evolving online curriculum RL
  - GUI Agent 自演化
- 28.3.3 **MobileRL / ComputerRL**(智谱, arXiv:2509.18119 / arXiv:2508.14040)
- 28.3.4 CogAgent(智谱)

### 28.4 指令层级与 Agent 安全 **[v5.1 新增]**

- 28.4.1 **指令层级**(OpenAI 2024.04, arXiv:2404.13208):
  - 系统/开发者/用户/工具指令的权限级别
  - "内核模式"类比
- 28.4.2 GPT-5 Mini-R 把指令层级用作 RL 奖励(+0.11~0.21)
- 28.4.3 Prompt Injection 防御
- 28.4.4 Agent 时代的核心安全机制

### 28.5 多智能体协作框架

- 28.5.1 **Anthropic 多智能体研究系统**(2025.06):
  - Orchestrator-worker 模式
  - 显式 OODA 循环
  - 子 agent 独立上下文窗口
  - 比单 agent 快 90.2%
- 28.5.2 Karpathy 的 "5-6 agents" 模式
- 28.5.3 Self-Play 多智能体(辩论、共识)

### 28.6 Agent Swarm 与并行 Agent RL **[v5.1 新增]**

- 28.6.1 **Kimi K2.5 Agent Swarm**(2026.01, arXiv:2602.02276, kimi.com/blog/kimi-k2-5):
  - 并行多 agent + 可训练 orchestrator
  - Parallel-agent RL
- 28.6.2 **Step 3.7 Flash Advisor Mode**(2026.05):
  - 小模型执行 + 大模型 advisor
  - 对应 Anthropic advisor strategy
- 28.6.3 Agent 间通信与协调

### 28.7 实验

- 28.7.1 训练一个简单 GUI Agent
- 28.7.2 多 agent 协作任务实验

---

# Part VI · 多模态强化学习(4 章 **[v5.1 新增视觉生成 RL]**)

## 第 29 章 视觉语言模型 RL **[v5.1 扩展]**

### 29.1 VLM RL 训练基础

- 29.1.1 视觉-语言联合表征
- 29.1.2 多模态奖励信号来源
- 29.1.3 视觉 token 与文本 token 的 RL 处理

### 29.2 视觉奖励信号设计

- 29.2.1 视觉问答正确性奖励
- 29.2.2 视觉描述完整性奖励
- 29.2.3 视觉幻觉惩罚

### 29.3 视觉反思 RL **[v5.1 新增]**

- 29.3.1 **Qwen3-VL**(2025.11.26):Reflection-driven visual re-attention
- 29.3.2 视觉 grounding 的自我纠错
- 29.3.3 反思机制 + RL 联合训练

### 29.4 训练框架

- 29.4.1 EasyR1 / R1-V 训练框架
- 29.4.2 Open-Vision-Reasoner / Perception-R1
- 29.4.3 视觉推理的 "Missing Trace" 问题

### 29.5 中国多模态前沿

- 29.5.1 Step3-VL-10B(arXiv:2601.09668):1000+ RL iterations
- 29.5.2 GLM-4.6V:RLCS 课程采样
- 29.5.3 Qwen3-VL:Reflection-driven
- 29.5.4 **Seed1.5-VL**(字节, arXiv:2505.07062):20B-A200B MoE,GUI agent + 游戏 RL

### 29.6 并行协调推理(PaCoRe)深度解析

- 29.6.1 PaCoRe 16 路并行 rollout 聚合
- 29.6.2 Test-time compute scaling 的另一条路径
- 29.6.3 与 MCTS over Thoughts 的对比

### 29.7 实验:GeoQA 几何推理

---

## 第 30 章 音频与语音 RL **[v5.1 扩展]**

> **设计理由**:Agent #4 直接点名 v5 此章只有大纲点,缺 Step-Audio MGRD 等核心方法。

### 30.1 音频语言模型概览

- 30.1.1 音频 token 化方案
- 30.1.2 语音生成与文本生成的差异
- 30.1.3 实时推理的工程挑战

### 30.2 Step-Audio 系列 **[中国独特方向]**

- 30.2.1 **Step-Audio-R1**(2025.11, arXiv:2511.15848):
  - 首个语音语言模型实现 test-time compute scaling
  - **MGRD**(Modality-Grounded Reasoning Distillation)
- 30.2.2 **Acoustic-Grounded Reasoning**:音频作为推理依据
- 30.2.3 **Mind-Paced Speaking**:实时推理与语音生成
- 30.2.4 Dual-Brain Architecture

### 30.3 RLVR → RLHF 演进

- 30.3.1 **Step-Audio-R1.5**:从 RLVR 转向 RLHF for Audio Reasoning
- 30.3.2 声音自然度 + 推理能力的多目标 RL
- 30.3.3 韵律(prosody)自然度保留

### 30.4 音频奖励设计

- 30.4.1 内容正确性奖励
- 30.4.2 韵律自然度奖励(人类偏好建模)
- 30.4.3 实时性奖励

### 30.5 实验:简单语音对话 RL

---

## 第 31 章 具身智能与 VLA 模型 **[v5.1 升级旗舰案例]**

> **设计理由**:Agent #2 直接点名 v5 此章用 RT-2 已过时,新基准是 Gemini Robotics 1.5 + π0 + Embodied Thinking。

### 31.1 VLA 模型概览

- 31.1.1 **π0**(Physical Intelligence 2024):diffusion policy + VLM
- 31.1.2 RT-2(Google 2023,作历史背景)
- 31.1.3 OpenVLA(开源旗舰)
- 31.1.4 **Gemini Robotics 1.5**(DeepMind 2025.09,旗舰):
  - VLA + ER 双模型
  - **Embodied Thinking** 范式
  - 跨本体迁移(Apptronik Apollo / Boston Dynamics Spot)
  - 技术报告 PDF:storage.googleapis.com/deepmind-media

### 31.2 机器人学习基础

- 31.2.1 观测空间(视觉、本体感觉、力觉)
- 31.2.2 动作空间(关节角度、末端执行器位姿)
- 31.2.3 奖励函数设计

### 31.3 Diffusion Policy

- 31.3.1 Diffusion 模型作为策略
- 31.3.2 多模态动作分布
- 31.3.3 与传统高斯策略对比

### 31.4 多模态融合

- 31.4.1 视觉-语言-动作 token 化
- 31.4.2 跨模态对齐
- 31.4.3 长程任务的条件输入

### 31.5 Sim-to-Real

- 31.5.1 域随机化(Domain Randomization)
- 31.5.2 Sim-to-Real transfer 技术
- 31.5.3 System Identification

### 31.6 遥操作与示范

- 31.6.1 人类示范采集
- 31.6.2 行为克隆预训练
- 31.6.3 RL 微调

### 31.7 实验:简单 VLA 训练

- 31.7.1 用 OpenVLA + RL 微调一个桌面抓取任务

---

## 第 32 章 视觉生成 RL **[v5.1 全新章节]**

> **设计理由**:Agent #4 P0 直接指出字节 Seed 在视频生成 RL 上是 2025-2026 最大创新源,但 v5 完全空白。DanceGRPO 把 GRPO 适配 diffusion,Seedance 多维 RLHF,LongCat-Video 多奖励 stacking——这是中国实验室的全球领先方向。

### 32.1 视觉生成任务定义

- 32.1.1 文生视频(Text-to-Video)
- 32.1.2 图生视频(Image-to-Video)
- 32.1.3 视频编辑与续写

### 32.2 Diffusion + RL 基础

- 32.2.1 Diffusion 模型作为策略网络
- 32.2.2 Rectified Flow 的 RL 适配
- 32.2.3 与文本 LLM RL 的根本差异

### 32.3 DanceGRPO **[字节 Seed 创新]**

- 32.3.1 **DanceGRPO**(2025.05, arXiv:2505.07818):
  - 把 GRPO 适配到 diffusion/flow 视觉生成
  - Unified across 4 foundation models
- 32.3.2 算法核心:diffusion step 作为 RL 时间步
- 32.3.3 与 DDPO 等先前方法对比

### 32.4 多奖励视频 RLHF

- 32.4.1 **Seedance 1.0**(字节, arXiv:2506.09113):
  - Foundational reward(基础质量)
  - Motion reward(运动合理性)
  - Aesthetic reward(美学)
  - Refiner RLHF
- 32.4.2 **LongCat-Video**(字节 2025.10, arXiv:2510.22200):
  - GRPO + 多奖励 stacking
  - LoRA stacking 工程实现

### 32.5 视频生成的奖励模型

- 32.5.1 VisionReward:开源视频评估模型
- 32.5.2 多维度奖励分解
- 32.5.3 人类偏好对齐

### 32.6 物理感知视频生成

- 32.6.1 **Hailuo-02**(MiniMax, physics-aware NCR 架构)
- 32.6.2 物理规律作为内在奖励
- 32.6.3 时序一致性约束

### 32.7 实验:用 DanceGRPO 训练简单视频生成模型

---

# Part VII · 安全、评估与对齐研究(4 章)

## 第 33 章 奖励黑客与对齐失败模式 **[v5.1 扩展]**

### 33.1 奖励黑客完整分类法

- 33.1.1 Reward Hacking 的形式化定义
- 33.1.2 Specification Gaming vs Reward Tampering vs Goodhart's Law
- 33.1.3 Anthropic 2025.11 分类(arXiv:2511.18397)

### 33.2 RLVR 的"假性收益" **[v5.1 新增]**

- 33.2.1 **数据污染实证**(arXiv:2507.10532 AAAI 2026):
  - Qwen 在 MATH-500 上的"spurious reward RLVR"收益主要来自数据污染
  - "随机奖励也能提升 Qwen 性能"现象
- 33.2.2 GRPO clipping bias 导致记忆激活
- 33.2.3 评估 RLVR 真实收益的方法论
- 33.2.4 抗污染的评测设计

### 33.3 工业失败案例 **[v5.1 新增]**

- 33.3.1 **GPT-4o 谄媚回滚**(OpenAI 2025.04-05):
  - 现实世界的 RLHF 失败案例
  - 用户反馈奖励稀释了主要的安全奖励
  - 48 小时后回滚
  - 修复:基于谄媚的 RL 奖励信号(openai.com/index/sycophancy-in-gpt-4o)
- 33.3.2 **字节 Seed RLHF 数据 scaling**:
  - Reward hacking 与多样性衰减
  - Pre-PPO prompt 选择策略

### 33.4 Anthropic 失准研究

- 33.4.1 **School of Reward Hacks**(Gao et al. 2025.08)
- 33.4.2 **自然涌现的失准**(Anthropic 2025.11, arXiv:2511.18397):
  - RL 环境中自然产生 reward hacking
  - 泛化到失准行为
  - HHH 奖励作为缓解
- 33.4.3 **Sleeper Agents**(Hubinger et al. 2024.01, arXiv:2401.05566)
- 33.4.4 **Alignment Faking**(Greenblatt et al. 2024.12, arXiv:2412.14093)
- 33.4.5 **In-Context Scheming**(Apollo 2024.12, arXiv:2412.04984)
- 33.4.6 **Sycophancy to Subterfuge**(Anthropic 2024, arXiv:2406.10162)

### 33.5 METR 与前沿模型研究

- 33.5.1 METR:Frontier Models Reward Hacking(Von Arx et al. 2025)
- 33.5.2 长期自主 agent 的失准风险

### 33.6 防御机制

- 33.6.1 Preference Models 与 Reward Hack Classifier
- 33.6.2 By Construction:从架构上防止 hacking
- 33.6.3 多 verifier 集成
- 33.6.4 形式化验证作为终极防线(对应第 21 章)

---

## 第 34 章 可扩展监督与红队测试

- 34.1 可扩展监督 (Scalable Oversight) 问题
- 34.2 AI Safety via Debate(Irving et al.)
- 34.3 递归奖励建模 (Recursive Reward Modeling, OpenAI)
- 34.4 Weak-to-Strong Generalization(OpenAI 2023)
- 34.5 红队测试方法论
- 34.6 对抗训练与鲁棒性
- 34.7 Sandwiching Problem
- 34.8 Exploration Hacking 与 exploit search problem

---

## 第 35 章 RL 评估方法论

- 35.1 评估基准设计原则
- 35.2 污染与泄漏检测(对应第 33 章 RLVR 假性收益)
- 35.3 提示敏感性分析
- 35.4 分布外鲁棒性
- 35.5 行为评估 vs 能力评估
- 35.6 长程任务评估的挑战
- 35.7 **Anthropic 内部 AI Research Eval Suite**(Opus 4.6):
  - LLM training / Text-RL / Quadruped-RL 子任务
  - 34× 人类加速基准
- 35.8 标准化评测 harness:lm-eval-harness、BigCode Eval、τ-bench、BFCL

---

## 第 36 章 分布式 RL 训练系统

- 36.1 veRL 架构深度解析
- 36.2 OpenRLHF / NeMo-Aligner / TRL 对比
- 36.3 Rollout 引擎与 vLLM 集成
- 36.4 GPU 内存优化:ZeRO、FSDP、Gradient Checkpointing
- 36.5 异步 RL 训练(LlamaRL、AReaL、AgentRL)
- 36.6 MoE + RL 训练(DeepSeek V3、Step Flash、GLM-4.5)
- 36.7 DualPipe 与 Best-Fit packing
- 36.8 性能 profiling 与瓶颈分析
- 36.9 万卡集群实践

---

# Part VIII · 研究前沿(2 章 **[v5.1 扩展]**)

## 第 37 章 进化式 LLM 搜索与生成式世界模型 **[v5.1 全新]**

> **设计理由**:Agent #2 P0 直接指出 v5 完全缺失 AlphaEvolve 和 Genie 3 这两条 2025-2026 最前沿方向。

### 37.1 AlphaEvolve 范式

- 37.1.1 **AlphaEvolve**(DeepMind 2025.05):
  - LLM 提出 diff + 自动评估器打分 + 进化算法挑选
  - 首次发现矩阵乘法 23% 加速
  - 改进 50 余个开放数学问题
  - 论文 PDF:storage.googleapis.com/deepmind-media
- 37.1.2 AlphaEvolve 算法架构:evolutionary search + LLM proposal
- 37.1.3 与传统 RL 的差异:不是 policy gradient,是 search + LLM
- 37.1.4 LLM 时代搜索算法的新范式

### 37.2 生成式世界模型作为 RL 环境

- 37.2.1 **Genie 3**(DeepMind 2025.08):
  - 实时可交互世界模型
  - 720p/24fps 生成
  - World memory 多分钟一致性
- 37.2.2 生成式环境 vs 真实环境
- 37.2.3 无限 RL 训练课程:agent 在生成世界中学习
- 37.2.4 AGI 通用世界模型基础

### 37.3 长期记忆架构

- 37.3.1 **Titans + MIRAS**(Google Research 2025.12):
  - 神经长期记忆模块
  - Test-time 学习更新记忆权重
  - 2M+ tokens 上下文
- 37.3.2 Attention / RNN 之外的第三范式
- 37.3.3 长期记忆与 RL agent 的结合

### 37.4 递归自我改进

- 37.4.1 **Anthropic Funded Research / 递归自我改进**(2026.04):
  - Claude 自身做 AI 研究
  - 内部基准 52× 加速(Opus 4 为 3×)
  - anthropic.com/institute/recursive-self-improvement
- 37.4.2 "Claude Mythos Preview" 模型
- 37.4.3 RL 训练 AI 做 AI 研究的终极愿景

---

## 第 38 章 自我博弈、规模化趋势与未来方向

### 38.1 自我博弈基础

- 38.1.1 AlphaGo → AlphaZero → MuZero 演进
- 38.1.2 自我对弈的收敛性
- 38.1.3 Self-play 在围棋 / 国际象棋 / 星际争霸的应用

### 38.2 LLM 自我博弈

- 38.2.1 LLM 自我博弈与 SPIN
- 38.2.2 Self-play SWE-RL (SSR)(对应第 26 章)
- 38.2.3 Multi-agent debate as self-play
- 38.2.4 Mode collapse 与多样性保护

### 38.3 RL Scaling Laws

- 38.3.1 RL Scaling Laws(类比 Chinchilla)
- 38.3.2 奖励信号 vs 数据量 vs 模型规模
- 38.3.3 RLVR 的 scaling 极限

### 38.4 Foundation Model RL

- 38.4.1 Foundation Model 作为 RL 的起点
- 38.4.2 RLHF / RLVR / RLAIF / Agent RL 的统一视角
- 38.4.3 Foundation Model RL 的未来形态

### 38.5 In-Context RL

- 38.5.1 In-Context RL 与 Algorithm Distillation(DeepMind 2022)
- 38.5.2 元学习与持续学习

### 38.6 未来十年研究方向

- 38.6.1 Karpathy 的"AGI 还需十年"反思
- 38.6.2 开放问题:信用分配、长程规划、泛化、安全
- 38.6.3 中国 vs 美国实验室的差异化路线
- 38.6.4 从对话模型到自主智能体的跨越

---

# 附录(7 部分)

## A. 训练调试手册 **[v5.1 扩展]**

- A.1 常见训练崩溃诊断
- A.2 梯度异常检测
- A.3 KL 散度爆炸处理
- A.4 **MuonClip + QK-clip** 优化器稳定性(Kimi K2)
- A.5 MoE + RL 训练的 router 干扰问题
- A.6 Reward Hacking 早期信号(agent 特有形式)
- A.7 Function Call 解析失败排查清单
- A.8 长轨迹 OOM 诊断决策树
- A.9 训练崩溃复现 checklist
- A.10 异步 RL 训练的 staleness 调优

## B. 强化学习工程实践 **[v5.1 大幅扩展]**

- B.1 同步 RL 训练系统底座(veRL、TRL)
- B.2 **异步 RL 训练系统**(AReaL、AgentRL、SLIME、ROLL、LlamaRL)
- B.3 **Staleness、Cross-policy sampling 工程实现**
- B.4 Agent 沙箱工程
- B.5 评测基准工程
- B.6 训练指标词典
- B.7 **MoE + RL 训练工程**(DeepSeek V3、Step Flash、GLM-4.5)
- B.8 工业实战练习

## C. 核心算法实现 **[v5.1 扩展]**

- C.1 SFT 与 KL 散度实现
- C.2 PPO 与 GAE 实现
- C.3 DPO 家族实现
- C.4 GRPO 基础实现
- C.5 **GRPO 改进家族实现**(DAPO、Dr.GRPO、GSPO、CISPO、VAPO)
- C.6 RPT(Reinforcement Pre-Training)实现
- C.7 Softmax 与交叉熵实现
- C.8 采样方法实现(top-k, top-p, min-p)
- C.9 注意力机制实现(MHA、GQA、MLA、**DSA 稀疏注意力**)
- C.10 **DanceGRPO** 适配 diffusion/flow
- C.11 PRM 训练实现(判别式 + 生成式 + 形式化 Lean4)
- C.12 MuonClip + QK-clip 优化器实现

## D. 学习资源与复现项目

- D.1 必读论文清单(按主题分类,100+ 篇)
- D.2 开源代码库索引
- D.3 复现项目推荐(Sky-T1、Open-R1、Tülu 3)
- D.4 视频课程索引(CS285、CS234、Hugging Face Course)

## E. 数学基础

- E.1 线性代数(贝尔曼矩阵、函数逼近、收敛性)
- E.2 概率与统计(回报、价值、采样估计、GAE)
- E.3 微积分与优化(梯度、PG、PPO、Adam)
- E.4 信息论(熵、KL、交叉熵、互信息)

## F. 论文阅读路线图 **[新增]**

- F.1 经典 RL 必读(Sutton、Watkins、Mnih)
- F.2 深度 RL 必读(DQN、A3C、PPO、SAC)
- F.3 LLM RL 必读(InstructGPT、CAI、DPO、GRPO、R1)
- F.4 安全研究必读(Sleeper Agents、Alignment Faking、Reward Hacking)
- F.5 2025-2026 前沿(DAPO、GSPO、CISPO、PRM、PaCoRe)

## G. GPU 小时估算表 **[新增]**

- G.1 不同模型规模的预训练成本
- G.2 SFT / RLHF / RLVR 各阶段成本
- G.3 DeepSeek / Qwen / Step 公开训练数据参考
- G.4 自训模型预算规划

## H. 符号表与算法索引 **[新增]**

- H.1 全书符号统一表
- H.2 算法名称索引(GRPO、PPO、DPO、SAC...)
- H.3 缩写表(RLHF、RLVR、PRM、CAI...)

---

# 完整章节统计 **[v5.1 修订]**

| Part     | 主题                      | 章节数              |
| -------- | ------------------------- | ------------------- |
| 0        | 序章 · 导论               | 6 节(对应 preface/) |
| I        | 基础与经典 RL             | 7                   |
| II       | 深度 RL                   | 5                   |
| III      | 高级 RL 方法              | 3                   |
| IV       | LLM 对齐与后训练          | 8                   |
| V        | **Agentic RL**            | **5**               |
| VI       | **多模态 RL(含视觉生成)** | **4** **[v5.1 +1]** |
| VII      | 安全、评估与系统          | 4                   |
| VIII     | **研究前沿**              | **2** **[v5.1 +1]** |
| **总计** |                           | **38 章**           |
| 附录     | A-H                       | 8 部分              |

---

# 与现有书的对比 **[v5.1 修订]**

| 维度            | 当前书        | **v5.1 最终**                                              |
| --------------- | ------------- | ---------------------------------------------------------- |
| 总章节数        | 12            | **38**                                                     |
| 序章            | 哲学论述在前  | **0.1 先动手玩 CartPole + 未来剧透**                       |
| Agentic 内容    | 1 章浅        | **5 章深入 + 第 28 章指令层级/UI-TARS-2/K2.5**             |
| 多模态          | 1 章浅        | **4 章(VLM/音频/VLA/视觉生成)**                            |
| 安全/对齐研究   | 0             | **3 章含 2025.11 + GPT-4o 回滚 + 数据污染**                |
| GRPO 家族       | DAPO 一个     | **6+ 变体算法细节(DAPO/Dr.GRPO/GSPO/CISPO/VAPO)**          |
| PRM             | 判别式为主    | **生成式(ThinkPRM)+ 形式化(Lean4/AlphaProof)**             |
| RL Environments | 0             | **独立章节 + 异步 RL(AReaL/AgentRL)**                      |
| 视觉生成 RL     | 0             | **独立章节(DanceGRPO/Seedance/LongCat)**                   |
| 工程系统        | 附录          | **正文 1 章 + 附录扩展**                                   |
| 实战代码        | 部分          | **每章带 lab**                                             |
| 中国实验室覆盖  | DeepSeek 一个 | **DeepSeek/Qwen/Kimi/Zhipu/Step/ByteDance/MiniMax 全覆盖** |
| 真实论文出处    | 无            | **每个主题有 arXiv 编号 + 官方 URL**                       |
| 前沿方向        | 0             | **AlphaEvolve/Genie 3/Titans/递归自我改进/RPT**            |

---

# 落地建议 **[v5.1 更新]**

**Phase 1(立即,零风险)**:序章重构 + 标题教材化 + 拆分第 3 章 MDP

**Phase 2(本月,P0)**:补 Part IV 核心

- 第 19 章 GRPO 家族完整重构(DAPO/Dr.GRPO/GSPO/CISPO/VAPO/RPT)
- 第 20 章 Reasoning 补 Hybrid Thinking + long2short + 涌现证据
- 第 21 章 PRM 升级到生成式 + 形式化

**Phase 3(下季,P0)**:补 Part V Agentic

- 第 26 章代码 agent 重构为 RL-based SWE 主线
- 第 28 章补指令层级 / UI-TARS-2 / Kimi K2.5 Agent Swarm

**Phase 4(下半,P0)**:补 Part VI 多模态

- 第 30 章音频 RL 深化 MGRD
- 第 31 章 VLA 升级 Gemini Robotics 1.5
- 第 32 章视觉生成 RL(DanceGRPO/Seedance)**[全新]**

**Phase 5(持续,P1)**:Part VII 安全 + Part VIII 前沿

- 第 33 章 Reward Hacking 补 GPT-4o 回滚 / 数据污染 / Seed scaling
- 第 37 章 AlphaEvolve / Genie 3 / Titans / 递归自我改进 **[全新]**

**Phase 6(长期,P1-P2)**:附录扩充

- 附录 A 补 MuonClip + QK-clip
- 附录 B 补异步 RL 系统
- 附录 C 补 DanceGRPO / RPT / PRM 实现

---

# 关键论文出处速查 **[v5.1 更新]**

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
[Titans + MIRAS] research.google/blog/titans-miras-helping-ai-have-long-term-memory 2025.12

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

# 关键论文出处速查

```
[DeepSeek-R1] Nature 2025. https://www.nature.com/articles/s41586-025-09422-z
[DeepSeek-V3] arXiv:2412.19437
[DAPO] Yu et al. 2025. arXiv:2503.14476
[Dr. GRPO] Liu et al. 2025
[GSPO] Zheng et al. 2025 (Qwen3)
[CISPO] MiniMax et al. 2025
[REINFORCE++] Hu 2025
[Qwen3 Tech Report] arXiv:2505.09388
[Kimi k1.5] Jan 2025
[Tülu 3] Allen AI 2024-2025
[Let's Verify Step by Step] Lightman et al. OpenAI 2023. arXiv:2305.20050
[Constitutional AI] Bai et al. Anthropic 2022. arXiv:2212.08073
[Sleeper Agents] Hubinger et al. 2024. arXiv:2401.05566
[Alignment Faking] Greenblatt et al. 2024. arXiv:2412.14093
[In-Context Scheming] Apollo 2024. arXiv:2412.04984
[Sycophancy to Subterfuge] Anthropic 2024. arXiv:2406.10162
[Natural Emergent Misalignment] MacDiarmid et al. Anthropic 2025.11. arXiv:2511.18397
[School of Reward Hacks] Gao et al. 2025.08
[METR Frontier Reward Hacking] Von Arx et al. 2025
[Effective Harnesses for Long-Running Agents] Anthropic 2025.11
[Weak-to-Strong Generalization] OpenAI 2023
[Karpathy 2025 Year in Review] karpathy.bearblog.dev
[Step3-VL-10B] arXiv:2601.09668
[Step-Audio-R1] github.com/stepfun-ai/Step-Audio-R1
[Epoch AI RL Environments FAQ] epochai.substack.com/p/an-faq-on-reinforcement-learning
[Raschka State of LLMs 2025] magazine.sebastianraschka.com/p/state-of-llms-2025
[Raschka LLM Papers 2025 List] magazine.sebastianraschka.com/p/llm-research-papers-2025-list-one
[SWE-RL] Meta 2025. arXiv:2502.18486
[Search-R1] arXiv:2503.09516
[R1-Searcher] arXiv:2503.05592
[Anthropic Code RL JD] job-boards.greenhouse.io/anthropic/jobs/5254364008
[DeepSeek CRFM Transparency] crfm.stanford.edu/fmti/December-2025/company-reports/DeepSeek_FinalReport_FMTI2025.html
[τ-bench] Salesforce 2024-2025
[CyberGym] arXiv:2506.02548
[Vending-Bench] arXiv:2502.15840
```

---

# 对用户的最终诚实声明

1. **36 章是大工程**,但这是 MIT 级别教材应有的体量(Sutton & Barto 14 章、CS285 11 讲、本书额外覆盖 LLM/Agentic/Multimodal)。
2. **不是"读完就能入职"**,真实 JD 还要求:SE 工程能力、生产调试、分布式系统经验、产品 sense。本书覆盖知识部分。
3. **每章应有 lab/实验**,真正动手才能内化。本书的 code/ 目录已有部分,需要扩充。
4. **持续更新**:2026 年会有新论文、新模型,本书需要每季小修、每年大修。
5. **写作工作量估算**:36 章 × 每章约 3000-5000 字 + 代码 = 约 15-20 万字 + 大量代码。预计全职 6-12 个月。
