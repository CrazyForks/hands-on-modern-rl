# 19.2 RLAIF 工程化宪法扩展

> [19.1](./hhh-practice) 讲了 HHH 三原则在 Claude 训练中的落地。本节关注 RLAIF 的工程化扩展——Anthropic 2026 年发布 80 页 Constitution，这是目前工业界最详尽的 AI 宪法实践。我们看看宪法是怎么"工程化"的。

## Anthropic 2026 80 页 Constitution

2026 年 Anthropic 公开发布了一份**80 页的 Claude 4 系列 Constitution** 文档（对应 Claude 4 Opus / Sonnet / Haiku）。这不是又一次原则扩写，而是一次**结构性的方法论转变**：从"列举规则"转向"**社会化**（socialization）"。这一节梳理它的三个核心创新。

### 从规则列表到价值观框架

旧版 Constitution 是"扁平的 80 条原则"，每条独立。新版引入**层级结构**：

```
顶层：北极星价值（North Star）
  ├── Helpful 子树
  │     ├── 真正解决问题
  │     ├── 区分请求 vs 行动
  │     └── 主动澄清歧义
  ├── Harmless 子树
  │     ├── 不协助严重伤害
  │     ├── 比例原则（拒绝强度匹配风险）
  │     └── 保护弱势群体
  └── Honest 子树
        ├── 表达不确定性
        ├── 区分事实与推测
        └── 承认错误
```

每个叶子节点是一个具体原则，但**冲突时按上层优先级仲裁**。例如"Helpful 解决问题"和"Harmless 比例原则"冲突时，按"风险等级"加权：低风险偏 Helpful，高风险偏 Harmless。

这种层级化让 AI judge 在打分时有了明确的优先级序，而不是 80 条原则相互打架。

### Socialization——让模型"内化"价值

80 页文档标题里的关键词是 _socialization_。Anthropic 借用社会学概念：**价值观不是通过规则灌输，而是通过"社会化"内化**。类比人类：孩子不是背法律条文长大的，而是在具体情境中观察、模仿、修正。

工程实现上，Claude 4 训练引入了**情境化对齐（contextual alignment）**：

1. 不再让模型背"原则 c_k"，而是构造大量**情境-行为对**（scenario-action pairs），让模型在情境中体现价值。
2. Judge prompt 从"按原则 c_k 评估"变成"在该情境下，一个理想助手应当怎么做"。
3. 训练 loss 从纯偏好损失改为偏好 + 情境一致性正则：

$$
\mathcal{L} = \mathcal{L}_{\text{pref}} + \lambda_{\text{ctx}} \cdot \mathcal{L}_{\text{context-consistency}}
$$

其中 $\mathcal{L}_{\text{context-consistency}}$ 衡量模型在不同情境下的回答是否与 Constitution 框架一致。

::: details 为什么 Socialization 比 Rule-Listing 更鲁棒
规则列表的根本问题：**规则无法穷尽**。80 条原则覆盖不了真实部署中遇到的千变万化情境。Socialization 让模型学的是"价值判断的能力"，而不是"规则匹配"。Anthropic 报告：Claude 4 在 OOD（训练时未见过）的安全情境上，鲁棒性比规则列表版本高 40%+。这与 [第 23 章 Computer Use](../chapter25_computer_use/training) 中"模型需要在新环境泛化"的需求直接对应。
:::

### 可审计性（Auditability）

80 页 Constitution 的第三个重点是 **可审计性**——每个模型决策都要能追溯到具体的 Constitution 条款。这要求：

1. **Judge 决策的可解释化**：judge 不是给一个标量分数，而是输出一段"判词"，明确指出依据哪条原则。
2. **训练数据溯源**：每个偏好对都标注触发哪条 Constitution 子节点。
3. **部署日志**：推理时记录模型"内心独白"中的价值判断（CoT 形式），便于事后审计。

形式化：模型输出 $y$ 附带一个 attribution $a(y) \in \mathcal{P}(\text{Constitution})$，表示 $y$ 依据的 Constitution 条款分布。Judge 的偏好损失改写为：

$$
\mathcal{L}_{\text{audit}} = -\mathbb{E} \big[\log \sigma\big(r_\phi(x, y_w, a_w) - r_\phi(x, y_l, a_l)\big)\big] + \lambda_{\text{attr}} \cdot \text{Entropy}(a_w)
$$

Entropy 项鼓励 attribution 不要坍缩到单一原则（多原则兼容时应当显式列出）。

### Claude 4 Constitution 与前沿对齐研究

| 维度       | Claude 2/3 Constitution | Claude 4 Constitution (2026)  |
| ---------- | ----------------------- | ----------------------------- |
| 结构       | 扁平规则列表（~80 条）  | 层级化价值树（北极星 + 子树） |
| 学习方式   | 规则匹配 + AI judge     | 情境化 socialization          |
| 冲突处理   | 隐式（judge 主观）      | 显式优先级仲裁                |
| 可解释性   | 隐式 reward             | 显式 attribution + CoT        |
| OOD 鲁棒性 | 弱                      | 强（socialization 泛化）      |
| 审计能力   | 黑盒                    | 每个决策可追溯到原则          |

这条路线和 [第 28 章 Scalable Oversight](../chapter30_alignment_failures/classical-failures) 的“AI supervision”研究、[附录 A.2 训练系统底座](../appendix_industrial_training/rl-infrastructure)的大规模对齐训练形成了完整的工业级对齐体系。

## 本章总结

Constitutional AI 与 RLAIF 是 LLM 对齐从"依赖人类标注"走向"可扩展监督"的关键一跳：

1. **Constitutional AI** 把安全对齐从"标注员逐条打分"重构为"模型按原则自评自改"，SL-CAI 和 RL-CAI 两条路线分别用 SFT 和 PPO 实现这一思想。
2. **RLAIF** 用 AI judge 替代人类标注，成本降两个数量级，速度提升千倍，但受限于 judge 能力，需要与少量人类 high-quality feedback 混合。
3. **自我修正与自我奖励** 把 critique-revise 显式写进训练，Self-Rewarding Language Models 进一步把 generator、judge、learner 三者合一，前三轮迭代显著有效，但需配合外部验证防止 reward hacking。
4. **HHH 三原则** 是 Constitution 的底层价值框架，三者在多目标 RL 中以加权 reward 形式联合优化。
5. **Claude 4 系列 Constitution**（2026）完成了从"规则列表"到"层级化价值树 + 情境化 socialization + 可审计 attribution"的方法论跃迁，为 OOD 鲁棒性和可解释性提供了新范式。

[第 16 章 RL Environments 与 Verifiers](../chapter18_grpo/rl-environments) 讨论了 RLAIF/RLVR 的另一半——**验证器（verifier）怎么设计**。一个数学题的答案对不对、一段代码能不能跑通、一次 API 调用是否合规，都需要可执行的环境来给出奖励信号。这是把 RLAIF 的"软偏好"转换为"硬规则"的工程基石。

## 延伸阅读

- [Bai et al. 2022 "Constitutional AI: Harmlessness from AI Feedback"](https://arxiv.org/abs/2212.08073)
- [Lee et al. 2023 "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback"](https://arxiv.org/abs/2309.00267)
- [Yuan et al. 2024 "Self-Rewarding Language Models"](https://arxiv.org/abs/2401.10020)
- [Askell et al. 2021 "A General Language Assistant as a Laboratory for Alignment"（HHH 原始定义）](https://arxiv.org/abs/2112.00861)
- [Anthropic 2024 "Collective Constitutional AI: Aligning a Language Model with Public Input"](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input)
- [Anthropic 2026 "Claude 4 Constitution"（80 页社会化价值框架文档）](https://www.anthropic.com/research/claudes-constitution)
- [Sharma et al. 2023 "Towards Understanding Sycophancy in Language Models"](https://arxiv.org/abs/2310.13548)（RLAIF 的失败模式分析）
- [Gao et al. 2022 "Scaling Laws for Reward Model Overoptimization"](https://arxiv.org/abs/2210.10760)（RM 过优化的 reward hacking）
