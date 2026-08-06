# 19.1 HHH 原则 Claude 实践

> [第 19 章](./intro) 讲了 Constitutional AI 的理论与 RLAIF 框架。本节回答工程问题：**Anthropic 实际在 Claude 训练中怎么落地 CAI？** 答案是 HHH 三原则——Helpful、Harmless、Honest——以及一系列对抗训练 trick。

## HHH 对齐原则

Constitutional AI 的底层价值框架是 **HHH**——Helpful, Harmless, Honest。这三者并非可有可无的口号，而是 Anthropic 用形式化的偏好函数刻画的三个可优化目标。

### Helpful 与 最大化用户效用

一个 helpful 的助手应当**真正解决用户的问题**，而不是回避或敷衍。形式化：

$$
\text{Helpful}(y \mid x) = \mathbb{E}_{u \sim \text{user}} \big[U_u(x, y)\big]
$$

其中 $U_u(x, y)$ 是用户 $u$ 对回答 $y$ 给 prompt $x$ 的效用。在 RLHF/RLAIF 里，$U$ 由偏好数据近似。

Helpful 的一个常见失败模式是**长度膨胀**（verbosity）——RM 容易给长回答高分，导致策略越训越长。Anthropic 在 Claude 训练中显式加入长度惩罚项：

$$
r_{\text{adj}}(x, y) = r_\phi(x, y) - \lambda_{\text{len}} \cdot |y|
$$

### Harmless 与 拒绝协助危险请求

Harmless 的形式化更微妙——不是"什么都不说"，而是"不帮助用户造成伤害"。一个典型定义：

$$
\text{Harmless}(y \mid x) = 1 - \mathbb{P}(\text{harm} \mid x, y)
$$

其中 $\mathbb{P}(\text{harm})$ 是该回答协助造成现实伤害的概率。这个量本身不可观测，CAI 用 Constitution + AI judge 来近似。

::: warning Helpful 与 Harmless 的张力
RLHF 训出的模型常出现 **evasiveness**：宁可拒绝也不冒险，于是"如何制作化肥"和"如何写一篇关于化肥的科普"都会被拒。CAI 的 Constitution 显式包含一条："如果请求本身无害（如科普、写作、研究），即使话题敏感也应该配合。"这是 CAI 相对纯 RLHF 的关键改进。
:::

### Honest 与 不输出错误信息

Honest 要求模型不撒谎、不假装知道、能表达不确定性。形式化：

$$
\text{Honest}(y \mid x) = 1 - D_{KL}\big(p_{\text{model}}(\cdot \mid x) \,\|\, p_{\text{true}}(\cdot \mid x)\big)
$$

这里 $p_{\text{true}}$ 是"客观真相分布"。实际中无法访问 $p_{\text{true}}$，所以用 **verifiable rewards**（数学答案、代码测试、事实检索）来近似。这也是 [RLVR](../chapter18_grpo/rlvr) 与 HHH 的连接点——RLVR 本质是 Honest 原则的硬验证版本。

### HHH 三者的联合优化

CAI 把三个目标加权组合：

$$
r_{\text{HHH}}(x, y) = \alpha_H \cdot \text{Helpful}(y \mid x) + \alpha_{HL} \cdot \text{Harmless}(y \mid x) + \alpha_{Ho} \cdot \text{Honest}(y \mid x)
$$

Constitution 的不同原则分别对应不同 $\alpha$：有些原则强调 Helpfulness（"如果请求合法请尽量配合"），有些强调 Harmlessness（"不要协助暴力"）。AI judge 在打分时把这些原则按 Constitution 权重组合，等价于一个 implicit 的 HHH 加权。

| 原则     | 典型失败模式             | CAI 的应对                              |
| -------- | ------------------------ | --------------------------------------- |
| Helpful  | 长度膨胀、模板坍缩       | 长度惩罚 + 多样性 reward                |
| Harmless | 过度回避（over-refusal） | Constitution 区分"敏感但合法" vs "危险" |
| Honest   | 幻觉、假装知道           | 显式 "I don't know" 训练 + RLVR 验证    |

## Claude 训练中的 CAI 实际应用

CAI 不是论文里的玩具，它是 Claude 全系列模型的真实训练流程。这一节梳理 Claude 2 → Claude 3 → Claude 3.5 的 CAI 演进，重点讲工业实践中的具体改动。

### Claude 2（2023） 与 第一版完整 CAI 落地

Claude 2 是第一个完整跑通 SL-CAI + RL-CAI 的产品级模型。关键技术细节：

- **Constitution 规模**：约 40 条原则，覆盖 HHH 三大类。
- **Self-Critique 长度**：每条 critique 限制在 200-400 token，避免太长拖慢训练。
- **Judge 模型**：使用一个比 generator 更大的模型当 judge（Claude 2 用内部 100B+ 模型 judge 50B 模型），避免 self-preference bias。
- **数据混合**：约 70% AI feedback + 30% 人类 high-quality feedback。人类 feedback 仍然保留，但只标注"AI 判断不确定"的边缘 case。

Anthropic 报告：Claude 2 相对纯 RLHF 版本，**有害性下降 50%+，过度回避率下降 30%**。

### Claude 3（2024） 与 Constitution 扩展与 Collective CAI

Claude 3 系列把 Constitution 从 40 条扩到 ~80 条，新增维度包括：

- **集体宪法（Collective Constitutional AI）**：Anthropic 与公开调查机构合作，让 1000+ 名不同文化背景的受访者投票决定 AI 该遵守哪些价值。结果发现全球受访者高度一致的几条：诚实、不协助暴力、尊重隐私。
- **减少过度回避**：增加原则 "拒绝请求应基于实际风险而非话题敏感度"。
- **多语言对齐**：Constitution 翻译成 20+ 语言，但保留**单一英文 master 版本**作为 ground truth，避免翻译引入的价值漂移。

工程上，Claude 3 用 **Constitutional Hindsight**（Bai et al. 2024 follow-up）：让模型对历史回答做"事后批评"，把这些批评作为额外的 SFT 数据。这相当于把部署数据闭环回训练。

### Claude 3.5（2024-2025） 与 CAI 与 RLVR 的融合

Claude 3.5 时代的关键变化：**CAI 不再是独立流程，而是和 RLVR 融合**。具体做法：

1. **Helpfulness 训练**：以 RLVR 为主，数学/代码用规则验证，写作/指令跟随仍用 RLAIF。
2. **Harmlessness 训练**：以 CAI 为主，因为"安全"无法用规则验证，只能靠 Constitution + AI judge。
3. **Honesty 训练**：混合——事实性问题用检索增强 + verifier 模型，开放性问题用 AI judge + RLVR。

这三条线在 PPO 中以加权 reward 形式组合：

$$
R(x, y) = w_{\text{task}} r_{\text{RLVR}}(x, y) + w_{\text{safe}} r_{\text{CAI}}(x, y) + w_{\text{hon}} r_{\text{verifier}}(x, y) - \beta D_{KL}
$$

这种 **multi-objective RL** 是 Claude 3.5 / 4 的核心训练范式，也是 [第 19 章 PRM 引导搜索](../chapter20_prm_search/inference-time-search) 的奖励组合方式之一。

### Claude 3.5 的几个工程经验

::: tip 工业界共识（截至 2025）

1. **纯 RLAIF 不可靠**：必须有少量人类 high-quality feedback 锚定。
2. **Constitution 越长越难调**：80 条已经是边际收益递减点，更多原则会导致相互冲突。
3. **Judge 模型必须比 generator 强**：否则 self-preference bias 严重。
4. **安全训练和能力训练必须解耦**：否则 KL 约束会拖慢能力提升。
   :::

## 本节总结

HHH（Helpful, Harmless, Honest）是 Anthropic 在 Claude 训练中实际使用的三原则。Helpful 要求模型尽力完成任务；Harmless 要求模型拒绝有害请求；Honest 要求模型不编造。这三者经常冲突——例如对一个敏感但合理的问题，过于 Harmless 会变成 evasiveness（回避），失去 Helpful 和 Honest。CAI 通过宪法让模型学会在冲突中找平衡。

下一节 [21.3 RLAIF 工程化与宪法扩展](./rlaif-engineering) 讲解 Anthropic 2026 年发布的 80 页 Constitution——这是目前工业界最详尽的 AI 宪法工程实践。
