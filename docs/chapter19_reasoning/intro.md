# 17.2 R1-Zero 纯 RL 训练

上一章我们把 GRPO 改进家族的五条路线放在一起对比——Dr.GRPO、GSPO、CISPO、VAPO、RPT。这些算法层面的改进回答的是"怎么训得更稳、更快"。但整个 2025 年 RL 领域最深刻的变革，其实发生在更高的层面：**RL 不仅改变训练，还重塑了模型在推理阶段的行为**。

这一章的主角是**推理模型（Reasoning Model）**——一类把"思考"显式写到输出里、用 RL 把思考过程本身当作优化目标的大模型。从 OpenAI o1（2024.09）到 Claude Opus 4.6（2025），短短一年多时间里，推理模型从实验室原型变成了工业级产品，重新定义了"大模型能做什么"。

## 这一章要回答的问题

读完这一章，你应该能清楚地回答：

- **推理模型与传统 LLM 的本质区别是什么**——仅仅是"输出更长的 CoT"吗？
- **OpenAI o1/o3/o4 的演化路径**——Hidden CoT 为什么是工程必然？
- **Test-time Compute Scaling** 的理论依据——为什么"推理时多算"比"训练时多算"在某些任务上更划算？
- **Hybrid Thinking 与思考预算**——为什么不能让模型永远思考？怎么控制？
- **长 CoT 压缩（long2short）**——Kimi k1.5 怎么把 10K token 的推理压缩到 2K？
- **自适应思考**——Claude Opus 4.6 怎么决定"这道题该想多久"？
- **Hidden vs Visible CoT** 的对齐含义——推理过程要不要让用户看见？

## 章节地图

```text
19.1 推理模型的兴起
     ├── OpenAI o1 → o3 → o4 的演化
     ├── Competitive Programming 论文的涌现证据
     └── 推理能力是"涌现"还是"激活"
19.3 Test-time Compute Scaling
     ├── 训练算力 vs 推理算力的权衡
     ├── Snell et al. 的 scaling law
     └── Gemini 3 Pro Deep Think 的并行思考
19.4 Hybrid Thinking 与思考预算
     ├── DeepSeek V3.1 的双模式融合
     ├── Qwen3 Thinking Mode Fusion
     └── NoThinking + Best-of-N 的反直觉发现
19.6 推理链的可读性与对齐
     ├── OpenAI 隐藏推理的工程动机
     ├── DeepSeek-R1 开放推理的策略
     └── 可读性与对齐的取舍
19.5 自适应思考
     ├── Claude Opus 4.6 的自适应深度
     ├── Anthropic 的 Constitution 与推理能力
     └── 推理链中的欺骗与对齐
```

## 与其他章的关系

这一章假定你已经读过：

- [第 16 章 GRPO 改进家族](../chapter18_grpo/grpo-family)——R1-Zero 与 DAPO 的训练细节
- [第 16 章 DeepSeek-R1 与 DAPO](../chapter18_grpo/deepseek-dapo)——纯 RL 训练范式的具体实现
- [第 13 章 RLHF](../chapter15_rlhf/base-model-to-assistant)——奖励信号的基础知识

本章后续会指向：

- [第 18 章 PRM 与推理时搜索](../chapter20_prm_search/outcome-vs-process)——过程奖励与树搜索
- [第 28 章奖励黑客与对齐失败](../chapter30_alignment_failures/classical-failures)——推理模型特有的 reward hacking
- [第 20 章 Agentic RL](../chapter22_agentic/overview)——推理模型作为智能体的"大脑"

## 一个直觉性的开场

在进入正式内容前，先建立两个关键直觉：

**直觉一：推理模型不是"CoT prompt 工程的工业化"**。传统 CoT（Chain-of-Thought）prompting 是在推理阶段给模型加一句"让我们一步一步思考"——这只是激活了预训练阶段已经学到的推理能力。而推理模型是通过 RL 训练，**让模型自己学会什么时候该思考、思考多久、什么时候该停下来给出答案**。这两者的差别，就像"提醒一个学生会做题"和"训练一个学生学会解题方法论"。

**直觉二：推理模型的本质是"把推理变成可优化的目标"**。在 RLHF 时代，模型优化的目标是"回答让人满意"——这个目标对推理过程没有直接约束。在推理模型时代，模型优化的目标是"通过推理得到正确答案"——这个目标直接塑造了推理过程本身。这就是为什么 R1-Zero 能涌现出 aha moment：奖励信号选择了"会反思的推理路径"，反思行为就被强化保留下来。

带着这两个直觉，我们进入 19.1。
