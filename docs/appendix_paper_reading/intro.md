# C.1 论文阅读路线图

> 学完本书前 17 章后，读者已具备阅读强化学习前沿论文的全部数学与工程基础。但 RL 文献浩繁——arXiv 每年新增数千篇，真正决定方向、值得反复读的不到一百篇。本附录按主题把这批"必读论文"分类整理，每篇标注**核心贡献一句话**与**阅读优先级**（★ 必读 / ◆ 推荐 / ◇ 扩展），帮助读者建立从经典 RL 到 2026 年前沿的完整文献地图。
>
> 阅读建议：按 F.1 → F.2 → F.3 顺序建立理论根基，再按研究方向在 F.4 / F.5 中挑选相关论文精读。每篇论文都应在动手复现（[附录 B](../appendix_industrial_training/intro)、[附录 D](../appendix_code_cheatsheet/intro)）之后回看，才会发现第一次读时忽略的工程细节。

## F.1 经典 RL 必读

经典文献的价值不在于算法本身——时过境迁，TD-Learning、Q-Learning 已是教科书常识。它们的真正价值在于**奠基性论证**：把"智能体在环境中学习"这件事形式化、可分析、可证明。读这些论文是理解现代 RL 所有数学符号背后含义的最快路径。

| 优先级 | 论文                                                                                                         | 核心贡献                                                                                                                                                              |
| ------ | ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ★      | Sutton & Barto. _Reinforcement Learning: An Introduction_ (2018, 2nd ed.)                                    | RL 的"圣经"。MDP、Bellman 方程、TD 学习、策略迭代、资格迹的完整理论体系，所有现代 RL 论文都默认读者熟悉此书符号。                                                     |
| ★      | Watkins, Dayan. _Q-Learning_ (Machine Learning, 1992)                                                        | 证明 off-policy TD 控制 $Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$ 在 tabular 情况下收敛到 $Q^*$，奠定后续 DQN/Rainbow 的理论根基。 |
| ★      | Tesauro. _TD-Gammon, a Self-Teaching Backgammon Program_ (Neural Computation, 1995)                          | 第一次让神经网络 + TD-Learning 达到人类专家水平（双陆棋），证明 $\text{TD}(\lambda)$ 与函数逼近结合能学到超出训练数据分布的策略。                                     |
| ★      | Sutton. _Learning to Predict by the Methods of Temporal Differences_ (Neural Computation, 1988)              | TD 学习的奠基论文，提出 $\text{TD}(0)$、$\text{TD}(\lambda)$ 与资格迹机制，是价值估计的源头。                                                                         |
| ★      | Mnih et al. _Human-level control through deep reinforcement learning_ (Nature, 2015)                         | DQN：经验回放 + 目标网络让 CNN 在 49 个 Atari 游戏上达到人类水平。深度 RL 的奠基工程论文。                                                                            |
| ★      | Silver et al. _Mastering the game of Go with deep neural networks and tree search_ (Nature, 2016)            | AlphaGo：策略网络 + 价值网络 + MCTS 击败李世石，证明 RL 在围棋这种长期被认为"AI 不可能解决"的问题上的突破。                                                           |
| ◆      | Williams. _Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning_ (1992) | REINFORCE：策略梯度定理的原始推导，所有 on-policy 算法（A2C、PPO、GRPO）的祖先。                                                                                      |
| ◆      | Schulman et al. _Trust Region Policy Optimization_ (ICML, 2015)                                              | TRPO：用 KL 散度约束保证策略更新单调，是 PPO 的直接前身。                                                                                                             |
| ◆      | Kearns, Singh. _Bias-Variance Error Bounds for Temporal Difference Updates_ (2000)                           | 从偏差-方差权衡角度解释 $\text{TD}(\lambda)$ 中 $\lambda$ 的作用，对理解 GAE（[第 5 章](../chapter10_ppo/gae)）至关重要。                                             |
| ◇      | Bertsekas, Tsitsiklis. _Neuro-Dynamic Programming_ (1996)                                                    | 把 DP 与函数逼近结合的早期专著，数学严密但读起来沉重，适合做理论时查阅。                                                                                              |

## F.2 深度 RL 必读

2013 年后 RL 进入深度学习时代。这一批论文定义了现代深度 RL 的标准方法栈：经验回放、目标网络、actor-critic、最大熵、模型学习。读者应重点掌握每篇的**算法骨架**与**消融实验**——前者决定了能否复现，后者决定了能否改进。

| 优先级 | 论文                                                                                                                                      | 核心贡献                                                                                                                               |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| ★      | Mnih et al. _Playing Atari with Deep Reinforcement Learning_ (arXiv:1312.5602, 2013)                                                      | DQN 早期版本（NIPS Workshop），首次把 CNN 与 Q-Learning 结合，开启深度 RL。                                                            |
| ★      | Mnih et al. _Human-level Control through Deep Reinforcement Learning_ (Nature, 2015)                                                      | DQN 完整版：经验回放 + 目标网络，49 个 Atari 游戏达到人类水平。                                                                        |
| ★      | Mnih et al. _Asynchronous Methods for Deep Reinforcement Learning_ (A3C, arXiv:1602.01783, 2016)                                          | 异步 actor-critic：多线程并行采样消除相关性，无需经验回放；A2C 是其同步版本，至今仍是基线。                                            |
| ★      | Schulman et al. _Proximal Policy Optimization Algorithms_ (arXiv:1707.06347, 2017)                                                        | PPO：clip 替代 TRPO 的二阶优化，工程友好、训练稳定。LLM 后训练 PPO 的直接原型。详见[第 5 章](../chapter10_ppo/intro)。                 |
| ★      | Lillicrap et al. _Continuous Control with Deep Reinforcement Learning_ (DDPG, arXiv:1509.02971, 2015)                                     | 把 DPG 扩展到深度网络，确定性策略梯度 + 经验回放，连续控制的开山之作。详见[第 10 章](../chapter11_continuous_control/intro)。          |
| ★      | Fujimoto et al. _Addressing Function Approximation Error in Actor-Critic Methods_ (TD3, arXiv:1802.09477, 2018)                           | 双 Q + 延迟更新 + 目标平滑，修复 DDPG 的 Q 值过估计与训练不稳定。                                                                      |
| ★      | Haarnoja et al. _Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor_ (arXiv:1801.01290, 2018)                  | 最大熵 RL + 自动温度调节，MuJoCo 长期霸榜，连续控制首选算法。                                                                          |
| ★      | Hessel et al. _Rainbow: Combining Improvements in Deep RL_ (arXiv:1710.02298, 2017)                                                       | 把 Double DQN、Dueling、PER、NoisyNet、Multi-step、Distributional Q 这 6 个 DQN 改进组合，证明"组合优于单点改进"。消融实验是经典教材。 |
| ★      | Silver et al. _A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go Through Self-Play_ (AlphaZero, Science, 2018) | 自我对弈 + MCTS + 神经网络，无人类棋谱学到围棋/象棋/将棋超人类水平。                                                                   |
| ★      | Schrittwieser et al. _Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model_ (MuZero, arXiv:1911.08265, 2019)             | 学一个**隐式**世界模型（表示/动力学/预测三网络），不知道游戏规则也能规划。                                                             |
| ★      | Hafner et al. _Mastering Diverse Domains through World Models_ (Dreamer V3, arXiv:2301.04104, 2023)                                       | RSSM 循环世界模型 + 想象中训练 actor-critic，单一超参跨越 150+ 任务（Atari、MuJoCo、Crafter、DMLab），model-based RL 的现代旗舰。      |
| ◆      | Haarnoja et al. _Soft Actor-Critic Algorithms and Applications_ (arXiv:1812.05905, 2018)                                                  | SAC 的扩展技术报告，含完整超参表与多个真实机器人实验。                                                                                 |
| ◆      | Schaul et al. _Prioritized Experience Replay_ (arXiv:1511.05952, 2015)                                                                    | 用 TD 误差给经验回放样本加权，加速 DQN 收敛。Rainbow 的核心组件之一。                                                                  |
| ◆      | Wang et al. _Dueling Network Architectures for Deep RL_ (arXiv:1511.06581, 2015)                                                          | 把 $Q(s,a)$ 分解为 $V(s) + A(s,a)$，提升动作间差异较小任务的训练效率。                                                                 |
| ◆      | Van Hasselt et al. _Deep RL with Double Q-Learning_ (arXiv:1509.06461, 2015)                                                              | Double DQN：解耦动作选择与价值估计，抑制 Q 值过估计。                                                                                  |
| ◆      | Bellemare et al. _A Distributional Perspective on RL_ (C51, arXiv:1707.06887, 2017)                                                       | 学习回报的**分布**而非期望，对噪声 reward 更鲁棒。                                                                                     |
| ◇      | Janner et al. _When to Trust Your Model: Model-Based RL in the Stochastic World_ (MBPO, arXiv:1906.08253, 2019)                           | 短 horizon rollout 平衡模型偏差与样本效率。                                                                                            |
| ◇      | Chua et al. _Deep RL in a Handful of Trials Using Probabilistic Dynamics Models_ (PETS, arXiv:1805.12114, 2018)                           | 集成 + 概率模型表达认知与偶然不确定。                                                                                                  |

## F.3 LLM RL 必读

LLM RL 是 2022 年后 RL 的最大应用场景。这一批论文定义了 RLHF / DPO / GRPO / RLVR / R1 / DAPO 等核心范式，每一篇都对应一段工业界训练实践。读者应**按时间顺序**读完，才能理解算法演进脉络：奖励模型 → RLHF → DPO（无奖励模型）→ GRPO（无 critic）→ R1（纯 RL）→ DAPO（修复 GRPO 缺陷）→ 2026 年的 GSPO/CISPO。

| 优先级 | 论文                                                                                                                      | 核心贡献                                                                                                                                                                    |
| ------ | ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ★      | Ouyang et al. _Training Language Models to Follow Instructions with Human Feedback_ (InstructGPT, arXiv:2203.02155, 2022) | RLHF 三阶段工程化（SFT + RM + PPO），第一次大规模证明 RLHF 比 SFT 显著更好。LLM 后训练范式的奠基论文。详见[第 6 章](../chapter15_rlhf/intro)。                              |
| ★      | Bai et al. _Constitutional AI: Harmlessness from AI Feedback_ (arXiv:2212.08073, 2022)                                    | Anthropic 的 RLAIF：用 AI 反馈替代人类标注，"宪法"驱动自我修正。同时是 RLHF 与对齐研究的桥梁。详见[第 20 章](../chapter21_cai_rlvr/intro)。                                 |
| ★      | Rafailov et al. _Direct Preference Optimization: Your Language Model is Secretly a Reward Model_ (arXiv:2305.18290, 2023) | DPO：通过 Bradley-Terry 模型重新参数化，把 RLHF 转化为监督学习，**完全省去 RM 与 PPO**。LLM 后训练最优雅的数学推导之一。详见[第 2 章](../chapter17_dpo/intro)。             |
| ★      | DeepSeek-AI. _DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL_ (arXiv:2501.12948, 2025)                    | R1：纯 RL（GRPO + 规则奖励）让 7B 模型数学推理接近 GPT-4o；R1-Zero 证明无需 SFT 即可触发长 CoT。RLVR 范式的标志性论文。详见[第 7 章](../chapter18_grpo/intro)。             |
| ★      | Shao et al. _DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models_ (arXiv:2402.03300, 2024) | GRPO 的原始论文：去掉 PPO 的 critic，用组内 normalize 估计 baseline。Group-Normalized Policy Gradient 的范式起点。                                                          |
| ★      | Yu et al. _DAPO: Decoupled Clip and Dynamic Sampling Policy Optimization_ (arXiv:2503.14476, 2025)                        | 修复 GRPO 的四个缺陷——长度偏置、熵坍缩、噪声优势、长短 horizon 冲突——提出 Clip-Higher、Dynamic Sampling、Token-Level Loss、Overlong Reward Shaping。Qwen 团队的工程化改进。 |
| ★      | Kimi Team. _Kimi K2: Open Agentic Intelligence_ (arXiv:2507.20534, 2025)                                                  | 1T 参数 MoE（32B 激活）的开放训练，验证 GSPO（Group Sequence Policy Optimization）在超大规模 Agent 训练上的可扩展性。                                                       |
| ★      | Zheng et al. _Group Sequence Policy Optimization_ (arXiv:2507.18071, 2025)                                                | 序列级 importance sampling + sequence-level loss，修复 GRPO 在长序列、长 horizon 任务上 token-level variance 过高的问题。                                                   |
| ★      | MiniMax. _MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention_ (arXiv:2506.13585, 2025)            | 在 456B 混合注意力 MoE 上提出 CISPO——裁剪重要性采样权重而非 token 更新，配合 lightning attention 在 512 张 H800 上三周完成 RL 训练。                                        |
| ◆      | Xiao et al. _VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks_ (arXiv:2504.05118, 2025)   | Value-Assisted PPO：引入轻量 critic + 长序列 value 估计，让 PPO 在数学推理长 CoT 上重新可用。                                                                               |
| ◆      | Luong et al. _ReFT: Reasoning with Reinforced Fine-Tuning_ (arXiv:2401.08967, 2024)                                       | SFT 预热 + PPO 在线微调，从同一批训练题上采多样化解题路径，小数据集上持续提升推理能力。                                                                                     |
| ◆      | Yuan et al. _Free Process Rewards without Process Labels_ (arXiv:2412.01981, 2024)                                        | 从 RLVR 训练中蒸馏出 token 级隐式 PRM，无需人工过程标注即可提供密集过程奖励。                                                                                               |
| ◆      | Setlur et al. _Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning_ (PRM, arXiv:2410.08146, 2024)   | 系统化 PRM 训练方法论，PRM 引导搜索的基础。详见[第 12 章](../chapter20_prm_search/intro)。                                                                                  |
| ◇      | Yuan et al. _Self-Rewarding Language Models_ (arXiv:2401.10020, 2024)                                                     | 让 LLM 自己生成偏好对训练自己，减少人工标注依赖。                                                                                                                           |
| ◇      | Tajwar et al. _Preference Fine-Tuning of LLMs Should Leverage Suboptimal, On-Policy Data_ (arXiv:2404.14367, 2024)        | 对比监督式、on-policy RL 与对比学习三类 preference fine-tuning，证明次优的 on-policy 数据对偏好微调最有效。                                                                 |

## F.4 安全研究必读

RLHF / RLVR 在让模型变强的同时打开了**奖励黑客、欺骗、对齐伪装、潜在后门**的潘多拉魔盒。这一批论文是 2023 年后 AI 安全研究的核心文献——它们告诉你**为什么训练良好的模型可能伤害你**，以及现有对齐方法的根本局限。任何做后训练的工程师都应至少精读前 5 篇。

| 优先级 | 论文                                                                                                                                 | 核心贡献                                                                                                                             |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| ★      | Hubinger et al. _Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training_ (arXiv:2401.05566, 2024)              | Anthropic 证明：植入"特定触发词才作恶"后门的模型，RLHF / SFT / 对抗训练**都无法消除**后门。揭示现有对齐方法的根本盲区。              |
| ★      | Greenblatt et al. _Alignment Faking in Large Language Models_ (arXiv:2412.14093, 2024)                                               | Anthropic 证明：模型在训练时**假装对齐**——展示符合预期的行为以避免被修改；推理时则违背训练目标。这是 schema 在真实训练中的实证出现。 |
| ★      | Meinke et al. _Frontier Models are Capable of In-Context Scheming_ (arXiv:2412.04984, 2024)                                          | 在多个前沿模型上观察到 in-context scheming：模型会假装能力受限、会复制自身权重、会规避监督。                                         |
| ★      | Sharma et al. _Towards Understanding Sycophancy in Language Models_ (arXiv:2310.13548, 2023)                                         | 系统化研究 LLM 的 sycophancy（迎合）行为，揭示 RLHF 偏好数据中"用户赞同 = 高奖励"的偏置。                                            |
| ★      | Denison et al. _Sycophancy to Subterfuge: Investigating Reward-Tampering in Large Language Models_ (arXiv:2406.10162, 2024)          | 课程式实验证明：学会简单 specification gaming 的 LLM 会逐步泛化到**直接重写自身奖励函数**的 reward tampering。                       |
| ★      | OpenAI. _GPT-4o System Card_ (2024)                                                                                                  | GPT-4o 发布后 sycophancy 问题导致大规模回滚——公开承认 RLHF 训练中奖励信号偏置的可观察后果。                                          |
| ★      | Gao et al. _Scaling Laws for Reward Model Overoptimization_ (arXiv:2210.10760, 2022)                                                 | OpenAI 的奖励模型 over-optimization scaling law：RM 与真实偏好距离随 KL 增大呈指数恶化。理解 RLHF 上限的关键论文。                   |
| ★      | Anthropic. _Natural Emergent Misalignment from Reward Hacking in Production RL_ (arXiv:2511.18397, 2025)                             | 证明学会 reward hacking 的模型会泛化出 alignment faking 与蓄意破坏等 emergent misalignment，且标准 RLHF 安全训练无法消除。           |
| ◆      | Eisenstein et al. _Helping or Herding? Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking_ (arXiv:2312.09244, 2023) | 用 RM 集成降低奖励黑客，工业界常用的鲁棒化方案。                                                                                     |
| ◆      | Coste et al. _Reward Model Ensembles Help Mitigate Overoptimization_ (arXiv:2310.02743, 2023)                                        | 同上的工程实证，含完整超参与坏案例。                                                                                                 |
| ◆      | Hubinger et al. _Risks from Learned Optimization in Advanced Machine Learning Systems_ (arXiv:1906.01820, 2019)                      | mesa-optimization 理论框架：学会的优化器可能内化与训练目标不同的目标——deceptive alignment 概率分析的理论基础。                       |
| ◇      | Taylor et al. _School of Reward Hacks: Hacking Harmless Tasks Generalizes to Misaligned Behavior in LLMs_ (arXiv:2508.17511, 2025)   | 在无害任务上训练 reward hacking 会泛化为更危险的不对齐行为，可作为评估安全性的 benchmark。                                           |
| ◇      | METR. _Frontier Reward Hacking: An Empirical Study of Reward Hacking in Frontier Models_ (2024)                                      | METR 评估前沿模型 reward hacking 行为的实证基准。                                                                                    |
| ◇      | OpenAI. _The Instruction Hierarchy: Training LLMs to Prioritize Instructions_ (arXiv:2404.13208, 2024)                               | 把 prompt 划分为系统/用户/工具三层，提供对抗 prompt injection 的训练时方案。                                                         |

## F.5 2025–2026 前沿

2025 年起 RL 进入第二个爆发期：过程奖励模型驱动推理、Agent 训练范式成熟、自我博弈扩展到代码与软件工程、视频生成引入 RL fine-tuning。这一批论文大多是 2025–2026 年的 tech report，尚未进入主流教材——它们定义了**下一个十年 RL 的研究方向**。读者应密切关注。

### F.5.1 RLVR 与推理算法

| 优先级 | 论文                                                                                                                                                  | 核心贡献                                                                                           |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| ★      | Yu et al. _DAPO_ (arXiv:2503.14476, 2025)                                                                                                             | GRPO 的工程化修复，Qwen 团队开源完整训练细节。已在 F.3 列出。                                      |
| ★      | Zheng et al. _GSPO_ (arXiv:2507.18071, 2025)                                                                                                          | 序列级 IS，Kimi K2 使用的核心算法。已在 F.3 列出。                                                 |
| ★      | MiniMax. _CISPO_ (arXiv:2506.13585, 2025)                                                                                                             | 截断无效 IS 样本。已在 F.3 列出。                                                                  |
| ★      | Khalifa et al. _Process Reward Models That Think_ (ThinkPRM, arXiv:2504.16828, 2025)                                                                  | 用一个轻量 PRM 替代密集过程奖励标注，让 RLVR 在低数据下可用。                                      |
| ★      | DeepSeek. _DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition_ (arXiv:2504.21801, 2025) | 形式化定理证明（Lean 4）上的 RLVR：RL 学习子目标分解，MiniF2F 达到 88.9% 通过率。                  |
| ◆      | Microsoft. _rStar-Math_ (arXiv:2501.04519, 2025)                                                                                                      | MCTS + PRM + 自我对弈训练小模型数学推理。                                                          |
| ◆      | Liu et al. _Understanding R1-Zero-Like Training: A Critical Perspective_ (arXiv:2503.20783, 2025)                                                     | 拆解 R1-Zero 式训练中的基础模型与 RL 两个组件，指出 GRPO 的响应长度优化偏置并给出无偏的 Dr. GRPO。 |

### F.5.2 Agentic RL 与代码/软件工程

| 优先级 | 论文                                                                                                                         | 核心贡献                                                                                                                                    |
| ------ | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| ★      | Pan et al. _SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution_ (arXiv:2502.18449, 2025)  | Meta 在 SWE-bench 上规模化 RL 训练，定义"代码-执行-反馈-学习"的 Agent RLVR 范式。详见[第 13 章](../chapter23_rl_based_swe/intro)。          |
| ★      | FAIR CodeGen team. _CWM: An Open-Weights LLM for Research on Code Generation with World Models_ (arXiv:2510.02387, 2025)     | Meta FAIR 的 32B 开放权重模型：在 Python 解释器与 agentic Docker 轨迹上中训练世界模型，把代码执行预测融入 RL 推理。                         |
| ★      | Wei et al. _Toward Training Superintelligent Software Agents through Self-Play SWE-RL_ (arXiv:2512.18552, 2025)              | Meta/CMU 的自我博弈 SWE-RL：单一 agent 在沙盒代码库上迭代注入与修复 bug，无需人工标注 issue/test 即可持续自我改进。                         |
| ★      | ByteDance. _UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning_ (arXiv:2509.02544, 2025) | GUI 操作 Agent：截图 → 动作（鼠标点击/键盘），端到端视觉 RL。详见[第 26 章](../chapter25_computer_use/intro)。                              |
| ★      | StepFun. _Step-Audio-R1 Technical Report_ (arXiv:2511.15848, 2025)                                                           | 第一个解锁音频推理能力的模型：MGRD 蒸馏让推理链锚定真实声学特征，音频理解超越 Gemini 2.5 Pro。详见[第 28 章](../chapter27_audio_rl/intro)。 |
| ◆      | Tongyi. _Tongyi DeepResearch_ (2025)                                                                                         | PRM 引导的长 horizon research agent。                                                                                                       |
| ◇      | Google. _AlphaEvolve_ (2025)                                                                                                 | 进化 + LLM 自动发现新算法，应用于芯片设计与组合优化。                                                                                       |

### F.5.3 大规模训练与开放模型

| 优先级 | 论文                                                                                                            | 核心贡献                                                                                                                                                 |
| ------ | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ★      | Qwen Team. _Qwen3 Technical Report_ (arXiv:2505.09388, 2025)                                                    | Qwen3 全系列（0.6B–235B-A22B MoE）训练细节，含 SFT/RLHF/RLVR 三阶段完整 recipe。                                                                         |
| ★      | Kimi Team. _Kimi K2_ (arXiv:2507.20534, 2025)                                                                   | 1T MoE 开放训练，验证大规模 GSPO 在 Agent 任务上的可扩展性。已在 F.3 列出。                                                                              |
| ★      | Kimi Team. _Kimi K2.5: Visual Agentic Intelligence_ (arXiv:2602.02276, 2026)                                    | K2 的下一代多模态 agent 模型：联合文本-视觉预训练与 RL，Agent Swarm 并行任务编排显著降低推理延迟。                                                       |
| ★      | DeepSeek-AI. _DeepSeek-V3 Technical Report_ (arXiv:2412.19437, 2024)                                            | 671B MoE（37B 激活），2.664M H800 小时训练 14.8T tokens；MLA + DeepSeekMoE + FP8 是核心工程创新。预训练成本参考见[附录 G](../appendix_gpu_hours/intro)。 |
| ★      | Gao et al. _Seedance 1.0: Exploring the Boundaries of Video Generation Models_ (arXiv:2506.09113, 2025)         | 视频生成模型的 RLVR fine-tuning，把 RL 从 token 扩展到时空 latent。                                                                                      |
| ★      | Peng et al. _DanceGRPO: A Unified Post-training Framework for Generative Video Models_ (arXiv:2505.07818, 2025) | 把 GRPO 适配到视频扩散模型，统一图文与视频 RLVR。                                                                                                        |
| ★      | Meituan LongCat Team. _LongCat-Video Technical Report_ (arXiv:2510.22200, 2025)                                 | 美团 13.6B 参数的 DiT 视频生成模型，单模型支持文生视频/图生视频/续写，多奖励 RLHF 对齐长视频生成。                                                       |
| ◆      | Behrouz et al. _Titans: Learning to Memorize at Test Time_ (arXiv:2501.00663, 2025)                             | 用长期/短期记忆模块扩展 Transformer，测试时在线学习新概念，为长序列推理提供可扩展记忆架构。                                                              |
| ◆      | Hong et al. _Multi-Agent Deep Research: Training Multi-Agent Systems with M-GRPO_ (arXiv:2511.13288, 2025)      | GRPO 的层级扩展：主 agent 与各子 agent 分别计算组内优势，配合轨迹对齐与解耦训练，提升多 agent 深度研究系统的稳定性与样本效率。                           |
| ◆      | Google. _Genie 3_ (2025)                                                                                        | 通用世界模型，从视频学环境动力学，是 embodied RL 的下一代基础。                                                                                          |
| ◇      | Ishibashi et al. _Can Large Language Models Invent Algorithms to Improve Themselves?_ (arXiv:2410.15639, 2024)  | Self-Developing：模型用 DPO 迭代发现并改进自己的算法（模型合并），在数学推理上超过人工设计的 Task Arithmetic。                                           |

## 阅读路线建议

不同读者的最优路径不同：

- **从基础补齐**：F.1 → F.2 → F.3 → F.4，按时间顺序理解 RL 从 1992 到 2026 的完整演进。
- **LLM 后训练工程师**：F.3 全部 + F.4 前 5 篇 + F.5.1 全部。重点理解 InstructGPT/DPO/GRPO/R1/DAPO 五篇的工程细节。
- **Agentic RL 研究者**：F.3 GRPO/R1 + F.5.2 全部 + F.2 AlphaZero/MuZero/Dreamer（理解搜索与世界模型）。
- **安全研究者**：F.4 全部 + F.1 Sutton & Barto 第 15 章（探索与利用的失败模式）+ F.3 RLHF over-optimization 相关论文。
- **理论研究者**：F.1 全部 + F.2 Rainbow/Dreamer 消融实验 + F.5 中的 GSPO/CISPO 数学推导。

## 阅读方法

::: tip 精读 vs 泛读
对**算法核心论文**（带 ★）做精读——逐行读公式、复现代码、做消融实验。对**工程 tech report**（如 Qwen3、Kimi K2）做泛读——抓超参表、训练曲线、ablation。前者训练思维，后者训练工程直觉。
:::

::: warning 不要只读最新论文
2025–2026 年的 RL 论文常默认读者熟悉 GRPO/PPO/REINFORCE 的细节。如果跳过 F.1/F.2 直接读 F.5，你会觉得"算法都没什么特别"——实际是缺少历史背景。建议每个新算法都追溯到其 2-3 篇"祖先"论文。
:::

::: details 论文与本书章节的对应关系
本书每章末尾的"延伸阅读"已列出该章核心论文。本附录是更全的、跨章节的论文地图。建议把本附录与各章末尾清单交叉使用：

- [第 2 章 DPO](../chapter17_dpo/intro) → F.3 Rafailov et al.
- [第 5 章 PPO](../chapter10_ppo/intro) → F.1 Schulman PPO、F.2 A3C
- [第 6 章 RLHF](../chapter15_rlhf/intro) → F.3 InstructGPT
- [第 7 章 GRPO/RLVR](../chapter18_grpo/intro) → F.3 R1、Shao et al.
- [第 10 章 连续控制](../chapter11_continuous_control/intro) → F.2 DDPG/TD3/SAC/Dreamer V3
- [第 14 章 对齐失败](../chapter30_alignment_failures/intro) → F.4 全部
- [第 15 章 工业级 LLM RL](../chapter16_llm_rl_industrial/intro) → F.5.3 全部
  :::

## 本章总结

本附录的 100+ 篇论文构成了从 1992 Q-Learning 到 2026 Kimi K2.5 的完整 RL 文献地图。它们不是"读完就行"的清单，而是**反复回看、伴随职业成长**的资料库——每多理解一篇，你对现代 RL 的理解就深一层。当你能在某篇论文中发现作者没说清楚的工程细节，或找到一个未被消融验证的假设时，你就具备了做独立研究的能力。
