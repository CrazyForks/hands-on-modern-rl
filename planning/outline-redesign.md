# 大纲重构方案 v3:基于真实证据(2025-2026 论文 + Anthropic/OpenAI JD)

> v2 是基于训练数据推断的"理想教材"。v3 是基于实际搜索到的 2025-2026 论文与 Anthropic Code RL JD 的**真实证据**修订版,并诚实标注 v2 的过度推断。

---

## 一、证据来源(真实可点击)

### 1.1 真实的 Anthropic 招聘 JD

**Research Engineer, Code RL**(job-boards.greenhouse.io/anthropic/jobs/5254364008):

- "Pioneering fundamental RL research for large language models"
- "Building scalable RL infrastructure and training methodologies"
- "Design RL environments and coding tasks"
- "Build the reward signals and verifiers that capture what 'good code' means"
- "Long-horizon autonomous engineering"
- "Agentic coding behaviors"
- "High-performance code for accelerators"
- **必备**:"Strong software-engineering skills and deep Python expertise, including async/concurrent programming"
- **加分**:RLHF / post-training / LLM finetuning;coding agents / code-execution sandboxes / eval harnesses / verifiers / developer tooling;program analysis / testing / verification / compilers / formal methods;PyTorch + large-scale distributed training + performance profiling

### 1.2 真实的 2025-2026 关键论文

| 论文                                                              | 时间              | 出处                                   |
| ----------------------------------------------------------------- | ----------------- | -------------------------------------- |
| DeepSeek-R1 (Nature)                                              | 2025.01           | nature.com/articles/s41586-025-09422-z |
| DAPO (Yu et al.)                                                  | 2025.03           | arXiv:2503.14476                       |
| Dr. GRPO (Liu et al.)                                             | 2025              | arXiv                                  |
| GSPO (Zheng et al.)                                               | 2025              | arXiv                                  |
| CISPO (MiniMax)                                                   | 2025              | arXiv                                  |
| REINFORCE++ (Hu)                                                  | 2025              | arXiv                                  |
| **Natural Emergent Misalignment from Reward Hacking** (Anthropic) | **2025.11**       | **arXiv:2511.18397**                   |
| Sleeper Agents (Hubinger et al.)                                  | 2024.01           | arXiv:2401.05566                       |
| Alignment Faking (Greenblatt et al.)                              | 2024.12           | arXiv:2412.14093                       |
| In-Context Scheming (Apollo)                                      | 2024.12           | arXiv:2412.04984                       |
| Sycophancy to Subterfuge (Anthropic)                              | 2024              | arXiv:2406.10162                       |
| METR: Frontier Models Reward Hacking                              | 2025              | METR Blog                              |
| School of Reward Hacks (Gao et al.)                               | 2025.08           | arXiv                                  |
| Karpathy 2025 Year in Review                                      | 2025.12           | karpathy.bearblog.dev                  |
| Tülu 3 (Allen AI)                                                 | 2024-2025         | arXiv                                  |
| Anthropic: Effective Harnesses for Long-Running Agents            | 2025.11           | anthropic.com                          |
| OpenAI o1/o3 introduction                                         | 2024.09 / 2025.01 | openai.com                             |
| Competitive Programming with Large Reasoning Models               | 2025.02           | OpenAI                                 |
| Reinforcement Pre-Training (Lambert)                              | 2025.06           | arXiv                                  |

### 1.3 真实的行业信号

- **Anthropic 拟投入 $1B 在 RL environments**(2025.09 The Information 报道)
- **Mechanize 给 RL environments 工程师 $500K 年薪**(TechCrunch 2025.09)
- **Karpathy**:"RLVR 是 LLM 训练流水线的'新主要阶段'"
- **Post-training 岗位薪资**:OpenAI/Anthropic/DeepMind IC $200K-$312K,Senior $400K+
- **岗位增长**:2025.01-2026.03 间 RLHF/post-training JD 增长 3 倍

---

## 二、v2 的诚实评估

### 2.1 v2 猜对的部分(有真出处)

| v2 主张                         | 真实证据                                      |
| ------------------------------- | --------------------------------------------- |
| PRM 章节重要                    | Lightman et al. 2023 OpenAI,o1/o3 训练核心    |
| Constitutional AI / RLAIF       | Bai et al. 2022 Anthropic,Claude 训练实际使用 |
| Sleeper Agents 章节重要         | Hubinger et al. 2024.01(真论文)               |
| Alignment Faking                | Greenblatt et al. 2024.12(真论文)             |
| Reward Hacking 章节重要         | Anthropic 2025.11 论文(很新很关键)            |
| DPO 家族完整覆盖(KTO/IPO/SimPO) | 全部真实存在                                  |
| GRPO / RLVR 核心                | DeepSeek-R1 Nature 论文已确认                 |
| Test-time Compute Scaling       | OpenAI o1/o3 系列确认                         |

### 2.2 v2 严重低估的部分(被 JD 证据打脸)

**🔴 最大盲区:RL Environments 设计**

- Anthropic $1B 投资、Mechanize $500K 工资、Karpathy 称之为"新主要阶段"
- v2 只在附录 B 工程实践里一笔带过
- **必须升级为独立核心章节**

**🟡 工程能力被低估**

- Anthropic JD 强调:async/concurrent Python、性能 profiling、分布式训练
- v2 把"工程"放附录,实际是 JD 评分占比最大的部分
- 一个 candidate 看完 v2 知道算法但写不出能 scale 的 trainer,过不了面试

**🟡 Reasoning Models 没单独成章**

- o1、R1、Claude Opus 4.6、Gemini 3.1 Pro 都已是独立产品类别
- v2 把它塞在"Test-time Compute"一节,严重不够
- **应单独成章,讲解 SFT→RLHF→RLVR 三阶段**

**🟡 GRPO 改进家族覆盖太薄**

- 2025-2026 至少 6 个主流变体:DAPO、Dr.GRPO、GSPO、CISPO、REINFORCE++、AREAL
- 还有 SPO、BSPO、TOPR、GPPO、M2PO
- v2 只点名 DAPO 一个

### 2.3 v2 过度推断的部分(学术深度对工业 JD 价值低)

**⚠️ Hierarchical RL / Meta-RL / MARL / IRL & GAIL**

- 没有任何找到的 JD 提及这些
- Karpathy 2025 年终总结也没强调
- 这些是学术教材(Sutton & Barto / CS285)的内容,但对 OpenAI/Anthropic 入职**帮助有限**
- **建议降级为附录或合并为一章**

**⚠️ "读完直接入职 OpenAI/Anthropic"**

- 诚实承认:这是过度营销
- 真实 JD 还要:强 SE 背景、生产调试经验、分布式系统经验、产品 sense
- 这些不能靠读教材获得
- v3 应改为"读完能通过技术面试的 RL 知识部分"

### 2.4 v2 排序错误

- 把 PRM 放第 27 章 → 应该是 LLM RL 篇核心章节之一(工业实战强度)
- 把"Test-time Compute"独立章节 → 应并入 Reasoning Models 章
- 把 Offline RL 放 Part V 前沿 → 实际是 2025 工业实践重要组成(Anthropic Code RL 间接相关)

---

## 三、v3 提案(基于真实证据)

### 设计原则

1. **JD 驱动**:以 Anthropic Code RL JD 的实际要求为锚
2. **2025 真实趋势**:RL Environments、Reasoning Models、Reward Hacking
3. **删除低 ROI 学术内容**:Hierarchical / Meta-RL / MARL / IRL 合并
4. **工程能力升级**:从附录提到正文
5. **GRPO 家族完整覆盖**:DAPO / Dr.GRPO / GSPO / CISPO / REINFORCE++

### v3 章节结构(28 章 / 6 篇)

---

#### Part I · 基础与经典 RL(8 章)— 与 v2 一致

1. 强化学习概览
2. CartPole:第一个强化学习实验
3. 多臂老虎机与探索-利用理论
4. 马尔可夫决策过程
5. 价值函数与贝尔曼方程
6. 动态规划、蒙特卡洛与时序差分
7. Q-Learning 与离策略控制
8. 奖励函数设计

---

#### Part II · 深度强化学习(6 章)

9. 深度 Q 网络与 Distributional RL
10. 策略梯度方法
11. Actor-Critic 架构
12. PPO 与信任域方法
13. 连续控制深度方法(DDPG / TD3 / SAC)
14. 基于模型的深度 RL(MuZero / Dreamer)

---

#### Part III · 高级 RL 方法(精简为 2 章,v2 是 6 章)

15. 离线强化学习与决策 Transformer(CQL / IQL / Decision Transformer / Diffuser)
16. 模仿学习、反向 RL 与元 RL 合集 **[合并]**

- 行为克隆、DAgger
- MaxEnt IRL、GAIL
- MAML、RL²、Algorithm Distillation

> **取舍**:Hierarchical RL 和 MARL 不单独成章,在 Part III 末尾设 "延伸阅读" 子节即可。理由:真实 JD 没要求,Anthropic/OpenAI 2025 论文也不强调。

---

#### Part IV · LLM 对齐与后训练(8 章 — v2 的核心,扩充)

17. RLHF 训练流水线(SFT → RLHF → RLVR 三阶段)
18. PPO-RLHF 工业实战
19. 偏好对齐:DPO 家族(DPO / IPO / KTO / SimPO / Iterative DPO / SPIN)
20. **GRPO 与可验证奖励:从基础到改进家族** **[v3 重点扩充]**

- 20.1 GRPO 群体归一化原理
- 20.2 RLVR 范式
- 20.3 DAPO:非对称裁剪、动态采样、token-level loss、overlong shaping、no KL
- 20.4 Dr. GRPO:移除 std 归一化
- 20.5 GSPO:序列级重要性采样
- 20.6 CISPO:裁剪 IS 权重而非 token 更新
- 20.7 REINFORCE++ 与 AREAL
- 20.8 DeepSeek V3.2 的 KL 调参技巧
- 20.9 选型决策树

21. **Reasoning Models:从 o1 到 Claude Opus 4.6** **[v3 新增独立章]**

- 21.1 推理模型的兴起(o1 → o3 → o4)
- 21.2 R1-Zero 范式:无 SFT 的纯 RL
- 21.3 DeepSeek-R1 训练全流程
- 21.4 Test-time Compute Scaling
- 21.5 Claude Opus 4.6 的自适应思考
- 21.6 Hidden CoT vs Visible CoT

22. 过程奖励模型与推理时搜索(PRM / MCTS over Thoughts / Tree of Thoughts / rStar)
23. **Constitutional AI 与 RLAIF**(Anthropic 对齐范式)
24. Agentic 强化学习(多轮交互、工具调用、SWE-bench、Deep Research)

---

#### Part V · 安全、评估与对齐研究(3 章)

25. **奖励黑客与对齐失败模式** **[v3 重点更新到 2025.11 论文]**

- 25.1 奖励黑客分类(Anthropic 2025.11 arxiv 2511.18397)
- 25.2 School of Reward Hacks(Gao et al. 2025)
- 25.3 Sleeper Agents(Hubinger et al. 2024)
- 25.4 Alignment Faking(Greenblatt et al. 2024)
- 25.5 In-Context Scheming(Apollo 2024)
- 25.6 Sycophancy to Subterfuge(Anthropic 2024)
- 25.7 自然涌现的失准(reward hacking 导致)
- 25.8 防御:preference models、reward hack classifier

26. **可扩展监督与红队**

- Scalable Oversight
- AI Safety via Debate
- Weak-to-Strong Generalization(OpenAI 2023)
- 红队测试方法论

27. **RL 评估方法论**(evals = RL environments 等价性,Pash 2025)

---

#### Part VI · RL 工程与系统(v3 全新章节,JD 核心)

28. **RL Environments 与 Verifiers 设计** **[v3 核心新增]**

- 28.1 RL Environments 作为新瓶颈(Anthropic $1B,Karpathy)
- 28.2 Verifier 设计原则
- 28.3 Evals 与 RL Environments 的等价性
- 28.4 Sandbox 工程(Docker、code execution)
- 28.5 长程任务 harness(Anthropic 2025.11 Effective Harnesses)
- 28.6 多 agent 并行(Karpathy "5-6 agents")
- 28.7 评测基准:CyberGym、SWE-bench、Terminal-Bench、Prime Intellect Hub

29. **分布式 RL 训练系统**

- 29.1 veRL / OpenRLHF / TRL / NeMo-Aligner 对比
- 29.2 Rollout 引擎与 vLLM 集成
- 29.3 异步 RL 训练(LlamaRL 2025)
- 29.4 GPU 内存优化:ZeRO、FSDP、Gradient Checkpointing
- 29.5 性能 profiling 与瓶颈分析
- 29.6 大规模训练调试实战

---

#### Part VII · 研究前沿(精简为 3 章)

30. 视觉语言模型 RL(VLM-GRPO、EasyR1、GeoQA)
31. 具身智能与多模态(VLA:π0、RT-2、OpenVLA、Diffusion Policy)
32. 自我博弈与规模化趋势(AlphaGo → MuZero → LLM Self-Play、RL Scaling Laws)

---

## 四、对比总结表

| 维度                   | 当前 v1   | v2(理想推断)   | **v3(证据驱动)**             |
| ---------------------- | --------- | -------------- | ---------------------------- |
| 章节数                 | 12        | 38             | **32**(精简学术,扩充工程)    |
| RL Environments 重视   | ❌        | ❌             | **✅ 核心章节**              |
| Reasoning Models 独立  | ❌        | 一节           | **✅ 独立章节**              |
| GRPO 家族完整          | DAPO 一个 | DAPO + Dr.GRPO | **✅ 6+ 变体**               |
| 工程能力位置           | 附录      | 附录           | **✅ Part VI 正文**          |
| Reward Hacking 时效    | 2024 论文 | 2024 论文      | **✅ 2025.11 论文**          |
| Hierarchical/Meta/MARL | 散落      | 3 章学术       | **减为 1 章合并**            |
| "直接入职"承诺         | ❌        | ❌(夸大)       | **改为"通过技术面 RL 部分"** |
| 真实 JD 验证           | 无        | 无             | **✅ Anthropic Code RL JD**  |

---

## 五、迁移路径(基于真实证据优先级)

**Phase 1**(立即,零风险)— 标题教材化 + 拆分第 3 章 MDP

**Phase 2**(本月,中风险)— 移走 DPO,补 §20 GRPO 家族完整化(已有部分内容)

**Phase 3**(本季,高 ROI)— 新增 v3 的 Part IV 关键章:

- §20 GRPO 改进家族
- §21 Reasoning Models 独立章
- §22 PRM 与推理搜索
- §23 Constitutional AI
- §25 Reward Hacking(更新到 2025.11)

**Phase 4**(下季,JD 核心)— 新增 Part VI RL 工程:

- §28 RL Environments 与 Verifiers(最重要,JD 验证)
- §29 分布式训练系统

**Phase 5**(持续)— Part III 高级 RL、Part VII 前沿研究

---

## 六、关键引用(供写作时直接使用)

```
[DeepSeek-R1] Guo et al. 2025. Nature. https://www.nature.com/articles/s41586-025-09422-z
[DAPO] Yu et al. 2025. arXiv:2503.14476
[GSPO] Zheng et al. 2025.
[CISPO] MiniMax et al. 2025.
[Dr.GRPO] Liu et al. 2025.
[REINFORCE++] Hu 2025.
[Anthropic Reward Hacking] MacDiarmid et al. 2025. arXiv:2511.18397
[Anthropic Sleeper Agents] Hubinger et al. 2024. arXiv:2401.05566
[Alignment Faking] Greenblatt et al. 2024. arXiv:2412.14093
[In-Context Scheming] Apollo Research 2024. arXiv:2412.04984
[Sycophancy to Subterfuge] Anthropic 2024. arXiv:2406.10162
[METR Reward Hacking] Von Arx et al. 2025.
[School of Reward Hacks] Gao et al. 2025.
[Karpathy 2025 Year in Review] karpathy.bearblog.dev
[Anthropic Effective Harnesses] anthropic.com 2025.11
[Anthropic Code RL JD] job-boards.greenhouse.io/anthropic/jobs/5254364008
[Epoch AI RL Environments FAQ] epochai.substack.com/p/an-faq-on-reinforcement-learning
[Raschka State of LLMs 2025] magazine.sebastianraschka.com/p/state-of-llms-2025
[Raschka LLM Papers 2025 List] magazine.sebastianraschka.com/p/llm-research-papers-2025-list-one
```

---

## 七、对用户的诚实总结

1. **v2 的方向是对的**,真实证据确认了 PRM、CAI、Sleeper Agents、Reward Hacking、GRPO/RLVR 这些主题确实是 2025 工业重点。
2. **v2 最大的盲区是 RL Environments**,这是 2025 行业最大投资方向(Anthropic $1B),v3 补为独立章节。
3. **v2 学术偏重过高**(Hierarchical/Meta/MARL),这些对工业入职帮助小,v3 精简。
4. **v2 工程部分太弱**,JD 强调 async Python、profiling、分布式,v3 升级到正文。
5. **"读完直接入职"是过度营销**,真实 JD 还要 SE 经验和生产调试能力,任何教材都无法替代。
6. **GRPO 家族远比 v2 完整**:至少 6 个主流变体在 2025 出现,必须完整覆盖。

---

# v4 增量:中国实验室证据(2025-2026)

> 搜索范围扩展到国内主要实验室(DeepSeek、Qwen、智谱 Zhipu、阶跃星辰 StepFun),发现 v3 在以下方向需要强化。

## 八、中国实验室真实证据

### 8.1 智谱 GLM 对齐团队面试真题(牛客网真实面经)

**[来源]** nowcoder.com 智谱 AI 话题区,多位候选人的真实一面/二面记录

**重点考察内容**:

- **PG → REINFORCE → TRPO → PPO 完整推导链**(从 RL 角度优化 PPO)
- **DPO 家族 + DPO 正则化方法**(优化 DPO 的方向)
- **DeepSeek GRPO 与 PPO 的对比**
- **手撕 transformer decoder block**
- **DeepSpeed 与 Megatron 的对比**
- **PPO/DPO 训练资源消耗估算**(面试官现场推算 GPU 小时)
- 反馈:"如果只做过 SFT 的基本就不用浪费时间投递了"

→ **v4 启示**:本书必须涵盖

- ✅ 完整 PG 推导链(已有,需强化为独立理论小节)
- ✅ DPO 正则化方法(v3 §19 需扩充)
- ✅ 训练成本估算章节(v3 §29 工程章需新增"成本估算"小节)

### 8.2 DeepSeek 真实训练数据(Stanford CRFM 透明度报告)

**[来源]** crfm.stanford.edu/fmti/December-2025/company-reports/DeepSeek_FinalReport_FMTI2025.html

- DeepSeek-V3 预训练:**2.664M H800 GPU 小时,14.8T tokens**
- DeepSeek-R1-Zero:**648 H800 GPUs × 198 小时 = 128K GPU 小时**
- DeepSeek-R1(完整多阶段):**648 H800 GPUs × ~80 小时**
- V3 + R1 总计:**2.8M GPU 小时,67 天**
- 训练模块:加载 actor + critic(可选),支持 PPO/GRPO/DPO
- Best-Fit 数据 packing,DualPipe 算法

→ **v4 启示**:

- ✅ 训练成本估算必须有具体数字参考(本书可加附录"GPU 小时估算表")
- ✅ DualPipe、Best-Fit packing 这些工程实现应进 §29 工程章

### 8.3 Qwen3 训练方法(真实技术报告)

**[来源]** Qwen3 Technical Report (arXiv:2505.09388)、Qwen3-Thinking-2507、Qwen2.5-Math

- **Qwen3 使用 GSPO**(Group Sequence Policy Optimization,Zheng et al. 2025)— 序列级重要性采样
- **Qwen2.5-Math**:self-improvement 路径
- **Qwen3-Thinking-2507**:AIME 86.7,LiveCodeBench 74.1(超越 o3/o4-mini)
- **数据污染问题**(arXiv:2507.10532):Qwen 因预训练包含 benchmark 答案,RL 收益部分来自记忆激活而非推理泛化

→ **v4 启示**:

- ✅ GSPO 必须在 GRPO 家族章重点讲(v3 §20.5 已有)
- ✅ 数据污染问题应进 §25 reward hacking 章
- ⚠️ Qwen3 "随机奖励也能提升"现象 — 揭示 RLVR 的微妙性

### 8.4 StepFun(阶跃星辰)独特方向

**[来源]** static.stepfun.com/blog/step-3.5-flash、github.com/stepfun-ai/Step-Audio-R1、arXiv:2601.09668 (Step3-VL-10B)

**真实岗位需求**(BOSS 直聘 + 牛客 + 阶跃官网):

- AI Infra 工程师:训练框架、推理加速、分布式系统
- Kernel 开发、MoE 通信优化、万卡集群
- 语音/多模态优先方向
- StepStar 2026 校招(本硕博顶尖)

**Step 独特技术贡献**:

- **Step-Audio-R1**:首个语音语言模型实现 test-time compute scaling
- **Step-Audio-R1.5**:从 RLVR 转向 **RLHF for Audio Reasoning**(声音自然度 + 推理)
- **Step3-VL-10B**:
  - **SeRe**(Sequential Reasoning):标准 CoT,64K context
  - **PaCoRe**(Parallel Coordinated Reasoning):**16 路并行 rollout 聚合**,128K context,test-time compute 扩展
  - AIME 2025: 94.4(PaCoRe 模式)
- **Step 3.5 Flash**:MoE 196B/11B active,AIME 97.3,推理 350 tokens/s
- **Deep Research**:多智能体架构

→ **v4 启示**:

- ✅ **PaCoRe 等并行协调推理**应进 §22 PRM/推理搜索章(全新的 test-time scaling 方法)
- ✅ **音频 RL** 是国内特色方向,§31 具身/多模态章应单独小节
- ✅ **MoE + RL** 的工程优化应进 §29 分布式训练
- ✅ **多模态 RL 比美国更前沿**(Step3-VL-10B 在 AIME 2025 击败 GLM-4.6V/Qwen3-VL)

### 8.5 中国实验室特色(对比 OpenAI/Anthropic)

| 维度     | OpenAI/Anthropic                           | 中国实验室(DeepSeek/Qwen/Zhipu/Step)        |
| -------- | ------------------------------------------ | ------------------------------------------- |
| 重点     | RL Environments、Safety、Constitutional AI | **MoE 训练、多模态 RL、Kernel 优化**        |
| 推理范式 | o1/o3 Hidden CoT                           | R1 Visible CoT + **PaCoRe 并行推理**        |
| 多模态   | 偏文本 + 视觉                              | **音频 RL、视觉推理、GUI Agent**            |
| 训练成本 | 商业不公开                                 | **公开 GPU 小时**(DeepSeek 透明度最高)      |
| 面试重点 | async Python、verifiers、sandboxes         | **PG 推导链、DPO 家族、DeepSpeed/Megatron** |
| 工程岗   | RL Environments $500K                      | **Kernel、MoE 通信、万卡集群**              |

---

## 九、v4 新增章节建议

基于中国实验室证据,在 v3 基础上**额外补充**:

### v4 §20.10 中国实验室的 GRPO 实战变体

- Qwen3 GSPO(序列级 IS)
- DeepSeek V3.2 的 KL 调参(数学任务 zero KL)
- Dr. GRPO 在 Qwen 系列的应用

### v4 §22.6 并行协调推理(PaCoRe)— **全新小节**

- Step3-VL-10B 的 16 路并行 rollout 聚合
- Test-time compute scaling 的另一条路径
- 与 MCTS over Thoughts 的对比

### v4 §25.9 数据污染与 RLVR 的微妙性

- "随机奖励也能提升 Qwen 性能"现象(arXiv:2507.10532)
- GRPO 的 clipping bias 导致记忆激活
- 评估 RLVR 真实收益的方法论

### v4 §29.7 MoE 训练与 RL 集成 **[中国实验室核心]**

- DeepSeek-V3 MoE 架构
- 万卡集群上的 RL 训练
- DeepSpeed / Megatron / DualPipe / Best-Fit packing
- 训练成本估算实战(参考 DeepSeek 公开数据)

### v4 §31.4 音频 RL **[新增小节]**

- Step-Audio-R1:首个 test-time compute scaling 语音模型
- 从 RLVR 到 RLHF for Audio 的迁移
- 韵律自然度与推理能力的平衡

### v4 §31.5 多模态 RL 的中国前沿

- Step3-VL-10B、GLM-4.6V、Qwen3-VL 的对比
- 视觉推理 RL 的"missing trace"问题(Step 论文)
- Acoustic-Grounded Reasoning(Step-Audio R1.1)

---

## 十、中国实验室关键引用

```
[Qwen3 Technical Report] Yang et al. 2025. arXiv:2505.09388
[Qwen3-Thinking-2507] Hugging Face Qwen3-235B-A22B-Thinking-2507
[Step3-VL-10B Technical Report] arXiv:2601.09668
[Step-Audio-R1] github.com/stepfun-ai/Step-Audio-R1
[Step 3.5 Flash] static.stepfun.com/blog/step-3.5-flash
[DeepSeek-R1 Nature] Guo et al. 2025. nature.com/articles/s41586-025-09422-z
[DeepSeek-V3 Tech Report] DeepSeek-AI 2024. arXiv:2412.19437
[DeepSeek CRFM Transparency] crfm.stanford.edu/fmti/December-2025/company-reports/DeepSeek_FinalReport_FMTI2025.html
[智谱 GLM 对齐面试真题] nowcoder.com/creation/subject/da767c9233384be9a2992ee3d1946518
[Qwen Contamination Study] arXiv:2507.10532 (Reasoning or Memorization?)
[Olmo 3 GRPO improvements] magazine.sebastianraschka.com/p/state-of-llms-2025
```

---

## 十一、综合 v4 调整总结

| v3 章节            | v4 调整                                      | 理由                              |
| ------------------ | -------------------------------------------- | --------------------------------- |
| §20 GRPO 家族      | **新增 §20.10 中国实验室变体**               | Qwen3 GSPO、DeepSeek V3.2 KL 调参 |
| §22 PRM/搜索       | **新增 §22.6 PaCoRe 并行协调推理**           | Step3-VL 独创方法                 |
| §25 Reward Hacking | **新增 §25.9 数据污染与 RLVER 微妙性**       | Qwen 实证发现                     |
| §29 分布式训练     | **新增 §29.7 MoE + RL 集成**                 | DeepSeek/Step 核心工程            |
| §31 具身/多模态    | **新增 §31.4 音频 RL、§31.5 中国多模态前沿** | Step-Audio、Step3-VL              |

**结论**:中国实验室证据强化了 v3 的以下判断:

1. GRPO 家族完整覆盖是必要的(中国实验室是 GRPO 改进主要贡献者)
2. MoE + RL 工程必须深入(中国比美国更重视)
3. 多模态 RL 比预想的更成熟(应提早进正文)
4. 训练成本估算应成为附录标准内容

但中国实验室也带来 v3 没覆盖的新方向:

- **音频 RL**(Step-Audio 独特)
- **PaCoRe 并行协调推理**(Step 独创 test-time scaling)
- **数据污染与 RLVER 真实收益的微妙性**(Qwen 揭示的问题)
