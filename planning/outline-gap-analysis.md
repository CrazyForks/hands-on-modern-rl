# v5 大纲 Gap 分析(2026.06 四家调研汇总)

> 基于 OpenAI/Anthropic、DeepMind/Meta、DeepSeek/Qwen/Kimi、Zhipu/StepFun/ByteDance/MiniMax 四组独立子 agent 的真实 URL 调研结果。

---

## 一、四个独立 agent 交叉验证的极高置信度缺口

### Gap 1 · GRPO 改进家族完整化(P0,四家齐声)

**v5 现状**:第 19 章 GRPO 家族只覆盖 GRPO、DAPO、Dr.GRPO、GSPO、CISPO、REINFORCE++、AREAL 七个变体的名字,无算法细节对比。

**四家证据**:

| Agent                 | 指出                                                                                       |
| --------------------- | ------------------------------------------------------------------------------------------ |
| #1 OpenAI/Anthropic   | DAPO 显式命名,补 Clip-Higher/Dynamic Sampling/Token-level Loss/Overlong Filtering 四项技巧 |
| #2 DeepMind/Meta      | Dr.GRPO 移除 std 归一化和长度归一化,避免 reward hacking                                    |
| #3 DeepSeek/Qwen/Kimi | GSPO 序列级 IS + clip,Qwen3 全系基石;DAPO 工程技巧深化                                     |
| #4 中国实验室组       | 完整谱系 CISPO/DAPO/VAPO/GSPO/ASPO,缺 **VAPO**(字节 Seed value-based 反潮流)               |

**修订建议**:第 19 章重构为:

- 19.1 GRPO 群体归一化原理
- 19.2 GRPO 的两类修正方向
  - 19.2.1 移除归一化偏差:Dr.GRPO
  - 19.2.2 序列级 IS:GSPO(Qwen3)
  - 19.2.3 裁剪策略对比:Clip-Higher(DAPO) vs Clip IS-weight(CISPO,MiniMax)
  - 19.2.4 Value-based 反潮流:VAPO(字节 Seed)
- 19.3 工程技巧合集
  - 19.3.1 Dynamic Sampling(过滤全对/全错)
  - 19.3.2 Token-level Loss(避免长 response 主导)
  - 19.3.3 Overlong Filtering + Soft Shaping
  - 19.3.4 No KL / 精度对齐(CISPO lightning attention)
- 19.4 RLVR 范式
- 19.5 中国实验室变体实战对比
- 19.6 选型决策树

---

### Gap 2 · PRM 升级:生成式 + 形式化(P0,三家齐声)

**v5 现状**:第 21 章 PRM 仍以判别式 OpenAI "Let's Verify Step by Step" 为主。

**三家证据**:

| Agent | 指出                                                                           |
| ----- | ------------------------------------------------------------------------------ |
| #1    | 生成式 PRM(ThinkPRM)优于判别式,标签少 100 倍;验证器计算扩展是新维度            |
| #2    | AlphaProof + AlphaGeometry 2 用 Lean 形式语言 + AlphaZero MCTS 自训练,IMO 银牌 |
| #3    | DeepSeek-Prover-V2 Lean4 形式定理证明 RL,MiniF2F 88.9%                         |

**修订建议**:第 21 章重构为:

- 21.1 Outcome 奖励 vs Process 奖励
- 21.2 判别式 PRM:OpenAI "Let's Verify Step by Step"
- 21.3 生成式 PRM:ThinkPRM(arXiv:2504.16828)
- 21.4 形式化 PRM:Lean4/Coq 作为天然 verifier
  - AlphaProof(DeepMind,IMO 银牌)
  - DeepSeek-Prover-V2(MiniF2F 88.9%)
- 21.5 推理时搜索:Beam Search、MCTS over Thoughts、Tree of Thoughts
- 21.6 PaCoRe:并行协调推理(Step3-VL 独创)
- 21.7 GenRM 与 Verifier 模型

---

### Gap 3 · Reward Hacking / 评估陷阱(P0,三家齐声)

**v5 现状**:第 32 章已覆盖 Anthropic 2025.11 emergent misalignment,但缺数据污染和 GPT-4o 回滚案例。

**三家证据**:

| Agent | 指出                                                                            |
| ----- | ------------------------------------------------------------------------------- |
| #1    | GPT-4o 谄媚回滚(2025.04-05,真实 RLHF 失败案例 + 根因分析)                       |
| #2    | Anthropic 2025.11 emergent misalignment + HHH 缓解                              |
| #3    | 数据污染与 RLVR 评估陷阱(arXiv:2507.10532,Qwen 在 MATH-500 的"spurious reward") |
| #4    | 字节 Seed RLHF 数据 scaling:reward hacking + 多样性衰减                         |

**修订建议**:第 32 章新增子节:

- 32.10 RLVR 的"假性收益":数据污染实证(Qwen 案例)
- 32.11 工业失败案例:GPT-4o 谄媚回滚根因分析
- 32.12 字节 Seed RLHF scaling 中的多样性衰减

---

### Gap 4 · Agentic RL 工程基础设施(P0,三家齐声)

**v5 现状**:第 23 章 RL Environments 提了 Anthropic Effective Harnesses,但没讲异步 RL 训练系统。

**三家证据**:

| Agent | 指出                                                                             |
| ----- | -------------------------------------------------------------------------------- |
| #1    | Anthropic 多智能体研究系统(orchestrator-worker + OODA + 独立上下文,90.2% 加速)   |
| #3    | Agentic Rollout 基础设施(Qwen3-Coder 20000 envs,Kimi K2 agentic rollout infra)   |
| #4    | AReaL(清华+智谱,2.77x 加速)+ AgentRL(智谱+清华)+ SLIME + ROLL 四大异步系统已开源 |

**修订建议**:第 23 章和附录 B 都要补:

- 23.9 同步 vs 异步 RL 训练
- 23.10 异步系统开源对比:AReaL / AgentRL / SLIME / ROLL / LlamaRL
- 附录 B.2 异步 rollout-training 解耦:staleness、cross-policy sampling

---

## 二、全新缺口(v5 任何章节都没覆盖)

### Gap 5 · Reinforcement Pre-Training (RPT)(P0,Agent #1)

**缺口**:把 next-token 预测重构为带内在二元奖励的推理任务,可与预训练扩展性媲美。挑战预训练/后训练二分法,是 2025 最大概念转变。

**出处**:arXiv:2506.08007(Microsoft 2025.06)

**建议位置**:第 19 章末新增"预训练阶段 RL" 或 Part VIII 前沿章

---

### Gap 6 · AlphaEvolve 范式(P0,Agent #2)

**缺口**:LLM 提 diff + 自动评估器打分 + 进化算法挑选。DeepMind 2025.05 发现矩阵乘法 23% 加速、改进 50 余个开放数学问题。不属于传统 RL 也不属于 SFT,是 LLM 时代搜索的新范式。

**出处**:deepmind.google/blog/alphaevolve + 论文 PDF

**建议位置**:Part VIII 前沿章新增"进化式 LLM 搜索"

---

### Gap 7 · 指令层级 Instruction Hierarchy(P0,Agent #1)

**缺口**:Agent 安全的"内核模式"类比。OpenAI 2024.04 提出,GPT-5 Mini-R 用作 RL 奖励获得 +0.11~0.21 提升。Agent 时代的核心安全机制。

**出处**:openai.com/index/instruction-hierarchy-challenge, arXiv:2404.13208

**建议位置**:第 28 章 Computer Use 或第 32 章安全新增子节

---

### Gap 8 · VAPO Value-based 反潮流(P0,Agent #4)

**缺口**:字节 Seed 2025.04 提出 Value-based Augmented PPO,在长 CoT 场景下 value model 重新打败 GRPO,AIME 60.4。这是 GRPO 家族里**反潮流**的方向(critic-free → critic-resurrected)。

**出处**:arXiv:2504.05118

**建议位置**:第 19 章 GRPO 家族(见 Gap 1)

---

### Gap 9 · 视频生成 RL(P0,Agent #4)

**缺口**:字节 Seed 独有方向,DanceGRPO 把 GRPO 适配 diffusion/flow 视觉生成;Seedance 多维 RLHF(Foundational+Motion+Aesthetic);LongCat-Video GRPO+多奖励。v5 完全空白。

**出处**:

- DanceGRPO arXiv:2505.07818
- Seedance arXiv:2506.09113
- LongCat-Video arXiv:2510.22200

**建议位置**:Part VI 多模态新增第 32 章"视觉生成 RL"(原 Part VI 后移)

---

### Gap 10 · 音频 RL(P0,Agent #4 + Agent #3 已覆盖但深度不够)

**缺口**:Step-Audio-R1 的 MGRD(Modality-Grounded Reasoning Distillation)、Acoustic-Grounded Reasoning、RLVR→RLHF 演进。v5 第 30 章只有大纲点。

**出处**:arXiv:2511.15848

**建议位置**:Part VI 第 30 章扩充为独立完整章

---

### Gap 11 · GLM 系列训练范式(P1,Agent #4)

**缺口**:**v5 整本书无一个智谱引用**。GLM-4.5 ARC(arXiv:2508.06471,MoE 355B/32A,难度课程 RL)、GLM-4.6 异步 RL + RLCS、GLM-5(arXiv:2602.15763)。

**出处**:

- GLM-4.5 arXiv:2508.06471
- GLM-4.6 HuggingFace zai-org/GLM-4.6
- GLM-5 arXiv:2602.15763
- AReaL arXiv:2505.24298

**建议位置**:第 17 章后训练工业实战新增 GLM 案例

---

### Gap 12 · Kimi K2 / K2.5 / Agent Swarm(P1,Agent #3)

**缺口**:MuonClip 优化器 + RL 稳定性、K2 Thinking 200-300 步 tool calling、K2.5 Agent Swarm 并行多 agent + 可训练 orchestrator + parallel-agent RL。

**出处**:

- K2 arXiv:2507.20534
- K2.5 arXiv:2602.02276, kimi.com/blog/kimi-k2-5

**建议位置**:第 28 章多 agent 协作(详见 Gap 4)

---

### Gap 13 · Hybrid Thinking + Thinking Budget(P0,Agent #3)

**缺口**:DeepSeek V3.1、Qwen3 单模型同时支持 think/non-think 模式;thinking budget 控制推理深度;NoThinking + best-of-N 可达 thinking 水平(Ma et al. arXiv:2505.18681)。

**出处**:

- Qwen3 报告 §4.3
- DeepSeek V3.1 api-docs.deepseek.com/updates
- Ma et al. arXiv:2505.18681

**建议位置**:第 20 章 Reasoning Models 新增"思考模式融合"子节

---

### Gap 14 · Kimi long2short RL(P0,Agent #3)

**缺口**:长 CoT 蒸馏到短 CoT 的 RL 方法,length penalty 控制。

**出处**:arXiv:2501.12599 §2.4, §3.4

**建议位置**:第 20 章 Reasoning 新增"长 CoT 压缩"子节

---

### Gap 15 · Self-play SWE-RL (SSR)(P0,Agent #2)

**缺口**:Meta 2025.12 单策略双角色(bug-injector + bug-solver),无需人工 issue,SWE-bench Verified +10.4。

**出处**:arXiv:2512.18552

**建议位置**:第 26 章代码智能体新增"self-play 范式"子节

---

### Gap 16 · Code World Model (CWM) + DeepSWE(P1,Agent #2)

**缺口**:Meta CWM(32B dense,Python 执行轨迹 mid-training + GRPO 后训练,SWE-bench 65.8%)。代码 agent 从 SFT-only 转向 RL-based 的主线工作。

**出处**:arXiv:2510.02387

**建议位置**:第 26 章代码智能体

---

### Gap 17 · 生成式 vs 判别式 PRM(P1,Agent #1)

**已在 Gap 2 中合并**

---

### Gap 18 · GPT-4o 谄媚回滚(P1,Agent #1)

**已在 Gap 3 中合并**

---

### Gap 19 · Anthropic 多智能体研究系统(P1,Agent #1)

**已在 Gap 4 中合并**

---

### Gap 20 · Genie 3 生成式世界模型作为 RL 环境(P1,Agent #2)

**缺口**:DeepMind 2025.08 实时可交互世界模型,720p/24fps,作为 AGI 通用世界模型基础,提供无限 RL 训练课程。

**出处**:deepmind.google/blog/genie-3

**建议位置**:Part III 第 15 章探索/分层 RL 新增"生成式世界模型" 或 Part VIII 前沿

---

### Gap 21 · Llama 4 流水线(P1,Agent #2)

**缺口**:轻量 SFT → online RL → 轻量 DPO;Behemoth 剪枝 95% SFT 数据;RL pass@k 难度过滤。

**出处**:ai.meta.com/blog/llama-4

**建议位置**:第 17 章后训练工业实战

---

### Gap 22 · 竞赛编程论文作为推理涌现证据(P1,Agent #1)

**缺口**:o3 复杂测试时策略是端到端 RL 自然涌现,非人工设计。

**出处**:arXiv:2502.06807(OpenAI 2025.02)

**建议位置**:第 20 章 Reasoning §20.1 或 §20.4

---

### Gap 23 · Titans + MIRAS 长期记忆(P2,Agent #2)

**出处**:research.google/blog/titans-miras-helping-ai-have-long-term-memory

**建议位置**:Part VIII 前沿

---

### Gap 24 · DeepSeek V3.2 / Speciale(P2,Agent #2 + #3)

**缺口**:DSA 稀疏注意力 + 自验证/自精炼 RLVR + mHC 残差稳定性,AIME25 97%。

**出处**:arXiv:2512.02556, magazine.sebastianraschka.com/p/technical-deepseek

**建议位置**:第 19 章 GRPO 家族最新案例

---

### Gap 25 · RL Environments 市场经济数据(P2,Agent #1)

**出处**:wing.vc/.../rl-environments-for-agentic-ai

**建议位置**:第 23 章 §23.1

---

### Gap 26 · 递归自我改进 / Anthropic Funded Research(P2,Agent #1)

**出处**:anthropic.com/institute/recursive-self-improvement

**建议位置**:Part VIII 前沿或序章"未来剧透"

---

### Gap 27 · MuonClip 优化器(P1,Agent #3)

**缺口**:Kimi K2 的 MuonClip 优化器 + QK-clip 用于 RL 训练稳定性。

**出处**:arXiv:2507.20534 §3.2

**建议位置**:Part II 第 11 章 PPO 训练细节 或附录 A 调试手册

---

### Gap 28 · UI-TARS-2 + AutoGLM GUI Agent RL(P1,Agent #4)

**出处**:

- UI-TARS-2 arXiv:2509.02544
- AutoGLM xiao9905.github.io/AutoGLM

**建议位置**:第 28 章 Computer Use

---

### Gap 29 · MoE + RL 训练工程(P1,Agent #4)

**出处**:Step 3.5 Flash arXiv:2602.10604, GLM-4.5

**建议位置**:附录 B 工业训练章

---

### Gap 30 · Seed-Thinking 双轨奖励 / Pre-PPO(P1,Agent #4)

**出处**:arXiv:2504.13914, seed.bytedance.com RLHF scaling

**建议位置**:第 16 章 RLHF 流水线 或第 17 章工业实战

---

## 三、过时内容警告(必须更新)

### 过时 1 · Part IV 以 PPO 为主(Agent #2)

**问题**:Llama 4 / Qwen3 / DeepSeek V3.2 已全面转向 GRPO/Dr.GRPO + RLVR。

**修订**:Part IV 重心从"PPO 经典实现"调整为"GRPO 家族为主,PPO 作历史背景"。

### 过时 2 · Part V 代码章以 SFT-only SWE-agent(Agent #2)

**问题**:主线应是 RL-based SWE(SWE-RL / CWM / DeepSWE / SSR)。

**修订**:第 26 章重构,把 SFT-based 作为背景,RL-based 作为主线。

### 过时 3 · Part VI VLA 用 RT-2(Agent #2)

**问题**:新基准是 Gemini Robotics 1.5 / π0 / OpenVLA。

**修订**:第 31 章 VLA 升级 RT-2 为背景,新旗舰案例用 Gemini Robotics 1.5 + Embodied Thinking。

### 过时 4 · R1-Zero 范式只覆盖 DeepSeek(Agent #4)

**问题**:应补 DAPO(字节)和 VAPO(字节 Seed)作为 R1-Zero 路线的工业级开源对照。

**修订**:第 20 章 R1-Zero 范式扩展为多开源实现对比。

### 过时 5 · 附录 B 训练系统只讲同步 veRL(Agent #4)

**问题**:AReaL / AgentRL / SLIME / ROLL 四大异步系统都已开源。

**修订**:附录 B.1 扩充同步 vs 异步对比。

---

## 四、修订优先级矩阵

| 优先级 | Gap 编号       | 简述                                            | 工作量 |
| ------ | -------------- | ----------------------------------------------- | ------ |
| **P0** | 1, 8           | GRPO 家族重构(含 VAPO)                          | 大     |
| **P0** | 2              | PRM 升级(生成式 + 形式化)                       | 中     |
| **P0** | 3              | Reward Hacking / 评估陷阱(补 GPT-4o + 数据污染) | 中     |
| **P0** | 4              | Agentic RL 工程基础设施(异步系统)               | 中     |
| **P0** | 5              | Reinforcement Pre-Training(RPT)                 | 小     |
| **P0** | 6              | AlphaEvolve 范式                                | 中     |
| **P0** | 7              | 指令层级 Instruction Hierarchy                  | 小     |
| **P0** | 9              | 视频生成 RL(DanceGRPO/Seedance)                 | 大     |
| **P0** | 10             | 音频 RL 深化(Step-Audio MGRD)                   | 中     |
| **P0** | 13, 14         | Hybrid Thinking + long2short                    | 中     |
| **P0** | 15             | Self-play SWE-RL (SSR)                          | 中     |
| **P1** | 11             | GLM 系列训练范式                                | 中     |
| **P1** | 12             | Kimi K2/K2.5 + Agent Swarm                      | 中     |
| **P1** | 16             | Code World Model + DeepSWE                      | 中     |
| **P1** | 20             | Genie 3 生成式世界模型                          | 中     |
| **P1** | 21             | Llama 4 流水线                                  | 小     |
| **P1** | 22             | 竞赛编程论文推理涌现                            | 小     |
| **P1** | 27             | MuonClip 优化器                                 | 小     |
| **P1** | 28             | UI-TARS-2 + AutoGLM                             | 中     |
| **P1** | 29             | MoE + RL 训练工程                               | 中     |
| **P1** | 30             | Seed-Thinking 双轨奖励                          | 小     |
| **P2** | 23, 24, 25, 26 | Titans/V3.2/市场数据/递归改进                   | 各小   |

---

## 五、v5.1 修订版章节增减建议

**新增章节**:

- 第 19 章重构:GRPO 家族完整化(含 VAPO、Dr.GRPO、GSPO、CISPO、DAPO 谱系)
- 第 21 章重构:PRM 升级(生成式 ThinkPRM + 形式化 Lean4)
- 第 20 章新增子节:Hybrid Thinking + long2short + 推理涌现证据
- 第 23 章新增:同步 vs 异步 RL 系统 + 经济数据
- 第 26 章重构:RL-based SWE 为主线(SWE-RL/CWM/DeepSWE/SSR)
- 第 28 章新增:指令层级 + UI-TARS-2 + AutoGLM
- 第 31 章重构:VLA 升级 Gemini Robotics 1.5 + π0
- 第 32 章新增:数据污染 + GPT-4o 回滚 + Seed RLHF scaling
- **Part VI 新增第 32 章**:视觉生成 RL(DanceGRPO/Seedance)
- Part VIII 新增:AlphaEvolve + Genie 3 + Titans + 递归自我改进

**附录扩充**:

- 附录 A 调试手册:补 MuonClip + QK-clip
- 附录 B 工程实践:补 AReaL/AgentRL/SLIME/ROLL 异步系统 + MoE+RL 工程

**章节数变化**:

- v5 原计划:36 章
- v5.1 修订后:约 37-38 章(新增"视觉生成 RL"独立章)

---

## 六、关键发现总结

1. **GRPO 家族是 2025-2026 最大算法焦点**:四家独立调研都点名,5+ 主流变体各有创新点,v5 第 19 章必须重构。
2. **中国实验室在 RL 工程化上全球领先**:异步训练系统(AReaL/AgentRL)、MoE+RL 工程、视频生成 RL、音频 RL 都是中国首发。
3. **v5 多个章节"代表案例"已过时**:PPO 主线、SFT-only SWE-agent、RT-2 VLA 必须更新。
4. **形式化验证是 PRM 的下一站**:AlphaProof + DeepSeek-Prover-V2 把 Lean4 作为天然 verifier。
5. **预训练和后训练边界正在消失**:Reinforcement Pre-Training 是 2025 最大概念转变。
6. **Reward Hacking 研究已成熟**:GPT-4o 谄媚回滚、Anthropic 2025.11、Qwen 数据污染形成完整证据链。
7. **v5 整本书无智谱引用**:必须补 GLM-4.5/4.6/5 系列,这是中国对齐团队的真实工程实践。
