# 现代强化学习实战 — v5 大纲（结构骨架）

> 本文件验证"Part / 单篇文章 / 文章内子大纲"三层是否清晰区分。确认格式后再扩展到全部 38 章。

## 标题层级规则（强制统一）

| 层级       | 用什么       | 含义                                                                             |
| ---------- | ------------ | -------------------------------------------------------------------------------- |
| **Part**   | `#`          | 大章（Part I、Part II、序章…）                                                   |
| **章节**   | `##`         | 一个教学单元，对应一个目录 `chapterNN_xxx/`，必须标 `[单篇文章]` 或 `[多篇章节]` |
| **文章**   | `###`        | 仅 `[多篇章节]` 下出现，一个 `.md` 文件                                          |
| **子大纲** | `-` 缩进列表 | 该文章内部要展开的 H2/H3 要点                                                    |

**强制规则：**

- 单篇章节下**不允许**用 `###`（避免误导读者以为是独立文件）
- 多篇章节下**必须**用 `###` 标每个文件
- 设计理由、备注一律放进 `>` 引用块，不混入正文
- `[v5 新增]`、`[v5.1 扩展]` 这类版本标签放进章节标题后的方括号，不污染层级

---

## 阅读约定

| 标记         | 含义                                                                 |
| ------------ | -------------------------------------------------------------------- |
| `[单篇文章]` | 这个章节只有 1 个 `.md` 文件，下方"内部小节"是这同一篇文章里的 H2/H3 |
| `[多篇章节]` | 这个章节有多个 `.md` 文件，下方每一项是一个独立文章                  |
| `📄`         | 文章（一个 `.md` 文件）                                              |
| `→ 路径`     | 文件实际位置（含旧目录名历史遗留）                                   |
| `子大纲:`    | 该文章内部要展开的 H2/H3 要点                                        |

---

# 序章 · 导论 `[多篇章节]`

> **设计理由**：本书承诺"先动手、后理论"。但旧版 `preface/intro.md` 开篇却是 Sutton 苦涩的教训哲学论述，直到第 1 章才让读者碰代码——导论自己违反了承诺。v5 修正：序章首节直接放可立刻玩的 CartPole 入口，30 秒内看到智能体从摇晃到站稳；**玩过之后再讲为什么**。
>
> **目录**：`docs/preface/`（3 个文件覆盖 0.1-0.6 共 6 节）

### 写在开头（含 0.1-0.4 + 0.6）→ `docs/preface/intro.md`

**内部小节：**

- **0.1 先动手：30 秒玩转 CartPole** `[v5 新前置]`
  - ① 一键试玩（零安装）：ModelScope 主源 + HuggingFace 副源双部署
  - ② 一行命令本地跑：`pip install gymnasium[...] stable-baselines3 && python 1-ppo_cartpole.py`
  - ③ 视觉预览（离线兜底）：训练 GIF + 最终策略演示视频
- **0.2 未来剧透** `[v5 新增]`：DPO 拒答恶意请求 / R1 推理涌现 / Computer Use / SWE-Agent 修 Bug
- **0.3 为什么需要 RL**：萨顿《苦涩的教训》/ 试错是学习最原始形态 / 监督→决策的根本差异
- **0.4 什么是 RL**：智能体-环境-状态-动作-奖励循环 / 轨迹·回报·折扣 / 状态 vs 观测
- **0.6 本书结构与读者路线图**：8 大 Part 递进 / ML 工程师·RL 背景·学生三种路径 / 符号约定

### 强化学习简史（0.5）→ `docs/preface/brief-history/index.md`

**内部小节：**

- 1950s-1980s：试错学习、Bellman 方程、TD 学习诞生
- 1992：TD-Gammon 首次战胜人类冠军
- 2013：DQN 玩 Atari——深度 RL 元年
- 2016：AlphaGo 击败李世石
- 2017-2019：AlphaGo Zero、MuZero、自我博弈
- 2017：PPO 发布（0.1 玩的就是 PPO）
- 2022：InstructGPT / RLHF 进入大模型训练
- 2023-2024：DPO、GRPO、Constitutional AI
- 2025：DeepSeek-R1、o1/o3、RLVR 范式确立
- 中国实验室崛起：Qwen3 GSPO、Step-Audio、DeepSeek 透明度

### 环境安装指南（0.7）→ `docs/preface/env-setup.md`

**内部小节：**

- Python 环境：conda / venv 选择
- PyTorch 版本与 CUDA 配置
- Gymnasium 安装与验证
- veRL / OpenRLHF / TRL 工具链预告
- 训练硬件检查表：入门 / 核心 / 大型项目三档
- 仓库代码结构：`code/` 各章独立子目录

---

# Part I · 基础与经典强化学习（7 章）

## 第 1 章 强化学习概览 `[单篇文章]`

📄 文件：`chapter00_overview/intro.md`

**内部小节：**

- 1.1 从序章的直觉到形式化定义
- 1.2 智能体-环境-奖励-状态的核心循环
- 1.3 现代应用版图：控制、游戏、对齐、智能体
- 1.4 RL 与监督学习、无监督学习的根本区别
- 1.5 本书后续章节的衔接

---

## 第 2 章 CartPole：第一个强化学习实验 `[多篇章节]`

> 现有目录：`chapter01_cartpole/`（目录名沿用旧编号，与章节号 2 不一致——历史遗留，暂不重命名）

### 2.1 CartPole 入门与原理 → `chapter01_cartpole/intro.md` + `principles.md`

**子大纲：**

- CartPole 问题与 Gym/Gymnasium 接口
- 状态、动作、奖励的工程化定义
- 随机策略基线与失败模式

### 2.2 训练指标设计 → `chapter01_cartpole/metrics.md`

**子大纲：**

- 回报曲线、成功率、稳定性
- 实验：从随机到收敛的完整流程

---

## 第 3 章 多臂老虎机与探索-利用理论 `[单篇文章]`

📄 文件：`chapter03_bandits/intro.md`

**内部小节：**

- 3.1 多臂老虎机问题与基础策略
- 3.2 ε-贪心与衰减调度
- 3.3 上置信界（UCB）算法
- 3.4 Thompson 采样与贝叶斯视角
- 3.5 遗憾界与 PAC 分析
- 3.6 上下文老虎机（Contextual Bandits）

---

# Part IV · 大语言模型对齐与后训练（8 章）

## 第 17 章 LLM RL 工业实战 `[多篇章节]` `[v5.1 扩展：从 PPO 转向 GRPO 现代流水线]`

> **设计理由**：2025-2026 Llama 4 / Qwen3 / DeepSeek V3.2 / GLM-4.6 已全面转向 GRPO/Dr.GRPO + RLVR，旧版以 PPO 经典实现为主的写法已过时。
>
> **目录**：`chapter17_llm_rl_industrial/`（目前仅 intro.md）；部分内容散落在 `chapter09_alignment/` 和 `chapter09_grpo_rlvr/` 下，需要整合

### 17.1 训练框架对比 → `chapter17_llm_rl_industrial/01-frameworks.md`

**子大纲：**

- 同步框架：veRL（字节主流）/ OpenRLHF（开源友好）/ TRL（HF 生态）/ NeMo-Aligner（NVIDIA）
- 异步框架：AReaL（清华+智谱）/ AgentRL（智谱+清华）/ SLIME / ROLL / LlamaRL
- 框架对比表与选型决策树

### 17.2 现代后训练流水线范式 → `chapter17_llm_rl_industrial/02-pipelines.md`

**子大纲：**

- DeepSeek-R1 多阶段：冷启动 SFT → 推理 RL → 拒绝采样 → 全场景 RL
- Llama 4：轻量 SFT → online RL → 轻量 DPO + pass@k 难度过滤
- Qwen3：Thinking Mode Fusion + Thinking Budget + GSPO
- GLM-4.5 / 4.6：难度课程 RL + Hybrid Thinking + RLCS 课程采样
- GLM-5（2026.02）：异步 Agent RL + DSA 稀疏注意力
- Seed-Thinking-v1.5：Dual-track reward + Pre-PPO + Hybrid reward

### 17.3 双轨奖励设计 → `chapter17_llm_rl_industrial/03-dual-reward.md`

**子大纲：**

- Verifiable Reward（Math、Code）
- Pairwise Preference Reward（开放对话）
- Pre-PPO：Prompt 选择策略避免 reward hacking
- Hybrid Reward：RTV + GenRM 组合

### 17.4 优化器与训练稳定性 → `chapter09_alignment/modern-industrial-practice.md`（沿用现有）

**子大纲：**

- AdamW 在 RL 训练中的稳定性问题
- MuonClip 优化器（Kimi K2）
- QK-clip：注意力数值稳定性
- KL 爆炸的早期信号与处理

### 17.5 训练成本估算 → `chapter17_llm_rl_industrial/05-cost.md`

**子大纲：**

- 不同模型规模的预训练成本
- SFT / RLHF / RLVR 各阶段 GPU 小时
- DeepSeek 公开数据参考：V3 预训练 2.664M H800 小时、R1-Zero 128K GPU 小时
- 自训模型预算规划

### 17.6 工业实战：GSM8K 与 AIME → `chapter09_grpo_rlvr/verl-code-sandbox.md`（沿用现有）

**子大纲：**

- 实验：用 GRPO 训练 GSM8K
- 实验：用 DAPO 训练 AIME 2024
- 完整开源复现：Open-R1 / Sky-T1 / Tülu 3

### 17.7 中国对齐团队面试常见考点 → `chapter17_llm_rl_industrial/07-interview.md`

**子大纲：**

- PG → REINFORCE → TRPO → PPO → GRPO 完整推导链（智谱真题）
- DPO 家族 + DPO 正则化
- DeepSpeed vs Megatron 工程对比
- 训练资源消耗现场推算
