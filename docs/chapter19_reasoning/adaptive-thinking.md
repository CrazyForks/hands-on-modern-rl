# 16.5 自适应思考

前面几节我们讨论了推理模型的兴起、test-time scaling、Hybrid Thinking、CoT 可见性。这一节聚焦一个具体案例：**Claude Opus 4.6**（2025.10 发布，2026.02 升级到 4.7）。

Anthropic 把 Opus 4.6 称为"reasoning model that thinks adaptively"——一个**自适应思考**的推理模型。它的设计与 o1、DeepSeek-R1 都有不同，代表了推理模型的另一条工业路线。

## 16.5.1 自适应思考的核心思想

**自适应思考（Adaptive Thinking）** 是 Hybrid Thinking 的进化版。Hybrid Thinking 是二选一——要么 think，要么 don't think；自适应思考是**连续的思考深度控制**——模型自己决定每道题该思考多深。

这个思想可以形式化为一个**思考深度参数** $\tau \in [0, 1]$：

- $\tau = 0$：完全不思考（直接给答案）
- $\tau = 0.5$：中等思考（几百 token CoT）
- $\tau = 1.0$：极致思考（数万 token CoT，可能反复修订）

模型在看到 prompt 后，**先决定 $\tau$**，然后基于 $\tau$ 生成相应深度的推理。

### 与 thinking_budget 的区别

Qwen3 的 thinking_budget 是**用户控制**的——用户在 API 调用里指定 budget。Claude Opus 4.6 的自适应思考是**模型自主决定**的——用户不指定 budget，模型自己判断。

这两种方式各有优势：

- **thinking_budget**：用户可控，适合**已知难度分布**的应用（如客服 API，简单题为主）
- **adaptive thinking**：模型自主，适合**难度未知**的应用（如科研助手，问题难度跨度大）

实际工业部署经常**两者结合**——模型自己决定一个 $\tau$，用户再加一个 budget 上限作为安全网。

## 16.5.2 Opus 4.6 的训练细节

Anthropic 没有公开 Opus 4.6 的完整训练细节（这是闭源策略），但从 [官方博客](https://www.anthropic.com/news/claude-opus-4-6) 和 [Extended Thinking 文档](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking) 可以推断出几个关键设计：

### 基于 prompt 难度的难度评估器

模型内部有一个"难度评估器"——看到 prompt 后，先快速估计难度，输出 $\tau$。这个评估器是 RL 训练中学到的，不是人工设计。

训练数据包含各种难度的 prompt，每个 prompt 的奖励不仅取决于答案正确性，还取决于**思考深度的合理性**：

- 简单题思考太多 → 浪费算力 → 惩罚
- 难题思考太少 → 答错 → 惩罚

通过这种"难度-深度匹配"的奖励信号，模型学会**自适应地分配思考深度**。

### Extended Thinking API

Opus 4.6 通过 **Extended Thinking** API 暴露给开发者。开发者可以：

```python
# 启用 extended thinking
response = client.messages.create(
    model="claude-opus-4-6",
    messages=[{"role": "user", "content": "..."}],
    thinking={
        "type": "enabled",
        "budget_tokens": 10000  # 可选 budget 上限
    }
)
```

`budget_tokens` 是可选的——不设的话模型自适应决定。这个 API 设计兼顾了**模型自主**和**用户可控**两种模式。

### thinking signatures

Opus 4.6 在输出 CoT 时，敏感内容会被替换为 `<thinking_signature>`——一个加密的"思考签名"。用户能看到 CoT 的结构和长度，但看不到具体的敏感 token。

这个机制是对 Hidden vs Visible CoT 折衷的尝试：

- Visible：用户能看到思考的结构（哪些是推理，哪些是答案）
- Hidden：敏感的具体内容被加密
- 防蒸馏：签名无法被反向工程，竞品无法直接用 CoT 训练

## 16.5.3 Opus 4.6 的旗舰能力 与 AI Research Eval Suite

Anthropic 在 Opus 4.6 发布时，重点宣传了一个内部 benchmark——**AI Research Eval Suite**。这个 suite 包含几个 RL 研究相关的子任务：

### LLM Training（LLM 训练）

让 Claude 自己设计一个小型 RL 训练实验——选数据、写训练代码、调超参、跑实验、分析结果。Opus 4.6 在这个任务上达到了**34× 人类研究员的速度**——一个研究员一周的工作，Claude 几小时就能完成。

### Text-RL（文本 RL）

让 Claude 设计和实现一个文本任务的 RL 算法（如对话对齐）。这个任务测试的是 Claude 对 RL 算法的理解和实现能力。

### Quadruped-RL（四足机器人 RL）

让 Claude 设计一个四足机器人的步态 RL 算法——这是经典控制 RL 任务。Opus 4.6 能写出可工作的 PPO + reward shaping 代码，在仿真环境里训练出能行走的策略。

这三个子任务展示了一个重要趋势：**推理模型不只是"会做题"，还能"做研究"**。Claude Opus 4.6 在这些任务上的表现，标志着推理模型从"考试高手"向"科研助手"的进化。

## 16.5.4 Anthropic 的 80 页 Constitution

2026 年 2 月，Anthropic 发布了一个 80 页的 Constitution 2.0——详细规定了 Claude 在推理时应该遵循的价值观。这个 Constitution 与推理模型的关系是：

### Constitution 作为推理约束

传统 RLHF 的宪法是**对最终答案的约束**——答案应该礼貌、有帮助、无害。Opus 4.6 的 Constitution 把约束扩展到**推理过程**：

- 推理时不应该表现出歧视
- 推理时应该诚实（不能为了讨好用户而扭曲事实）
- 推理时应该考虑多方利益（不只是用户，还有受影响的其他人）

### Constitution 与训练

Constitution 不是"prompt 注入"——它不是在推理时给模型读 80 页文档。而是**在训练阶段，让模型内化 Constitution**。具体做法：

1. 把 Constitution 拆解为可执行的判断准则
2. 用这些准则生成大量"宪法对齐"的偏好数据
3. 用这些数据做 RLHF / DPO 训练

这样训练出的模型，在推理时**自然遵守宪法**——不需要在 prompt 里提醒。

### Constitution 与可解释性

80 页 Constitution 也提供了**对齐可解释性**的新工具——研究者和用户可以用 Constitution 来检查模型的推理是否符合预期。如果模型在某个推理步骤中违反了 Constitution 的某条准则，就是潜在的对齐问题。

## 16.5.5 推理模型的安全挑战

Opus 4.6 的自适应思考也带来了新的安全挑战：

### 模型可能"假装思考"

如果模型在简单题上假装深度思考（生成看起来很长的 CoT 但内容是空的），用户难以发现。这是一种**思考欺骗**——用 token 数假装推理深度。

Anthropic 的应对是在训练时加入"思考质量"奖励——CoT 的内容必须对最终答案有贡献，纯粹凑长度的 CoT 被惩罚。

### 自适应思考的不可预测性

自适应思考让模型的行为更难预测——同一 prompt，模型可能这次思考 1000 token，下次思考 5000 token。这在工业部署中是个问题——延迟和成本难以预估。

应对策略是给自适应思考加 budget 上限，但这又削弱了"自适应"的特性。这是一个**算法能力 vs 工程可控性**的根本矛盾。

### 推理链中的攻击面

自适应思考让推理链更长、更复杂——攻击者有更多机会在 CoT 中注入恶意触发器。比如 prompt injection 可以诱导模型在 CoT 中泄露系统 prompt 内容。

这是 [OpenAI 的 Instruction Hierarchy](https://openai.com/index/the-instruction-hierarchy/)（2025）要解决的问题——明确系统 prompt、用户 prompt、工具返回结果的优先级，防止低优先级内容劫持高优先级行为。

## 16.5.6 自适应思考 vs 固定深度思考

最后做一个对比总结：

| 维度       | 固定深度思考（o1 早期） | Hybrid Thinking（Qwen3）  | 自适应思考（Claude Opus 4.6） |
| ---------- | ----------------------- | ------------------------- | ----------------------------- |
| 模式       | 总是深度思考            | 二选一（think / nothink） | 连续深度控制                  |
| 控制方     | 模型固定                | 用户指定 / 模型二选一     | 模型自主                      |
| 算力效率   | 低（简单题浪费）        | 中（粗粒度控制）          | 高（细粒度匹配）              |
| 工程复杂度 | 简单                    | 中等                      | 高                            |
| 典型应用   | 数学 / 代码竞赛         | 通用对话 + 推理           | 科研助手 / 复杂任务           |

可以看到，**自适应思考是 Hybrid Thinking 的精细化版本**——从二选一到连续控制。这是推理模型的演化方向，但也意味着工程复杂度的提升。

## 推理模型的对齐 与 未来展望

这一章我们看到了推理模型从 o1 到 Claude Opus 4.6 的演化。但推理模型的对齐问题才刚刚开始被研究：

### 过程对齐（Process Alignment）

未来对齐研究将更关注**推理过程的对齐**——不只是最终答案符合价值观，每个推理步骤都符合。这需要：

- 更精细的 PRM（过程奖励模型）
- 推理时的监控（CoT monitoring）
- 宪法式推理训练

### 推理模型的安全沙箱

推理模型在思考时可能"想到"危险内容（即使最终答案不输出）。未来需要"安全沙箱"——隔离模型的思考过程，防止敏感内容外泄。这是 [Hidden CoT 路线](./cot-visibility-alignment) 的进一步发展。

### 推理 scaling 的对齐

test-time compute scaling 让模型越来越强，但也让对齐越来越难——更长的 CoT 意味着更多潜在的攻击面、更多对齐失败的可能。未来需要**对齐 scaling law**——让对齐能力随着模型能力同步增长。

### 推理与 agentic 的融合

Opus 4.6 的 AI Research Eval Suite 已经展示了这个趋势——推理模型不只是"思考"，还能"执行"（写代码、跑实验、分析结果）。这与 [第 19 章 Agentic RL](../chapter22_agentic/overview) 深度融合。未来推理模型将变成"能思考 + 能行动"的完整 agent。

## 本章总结

这一章我们把推理模型的全貌梳理了一遍：

- **16.1 节**：推理模型的兴起——o1/o3/o4 的演化、Competitive Programming 论文的涌现证据、推理能力的本质
- **16.3 节**：Test-time Compute Scaling——Snell 的研究、并行 vs 顺序推理、Gemini Deep Think 的旗舰案例
- **16.4 节**：Hybrid Thinking 与思考预算——DeepSeek V3.1、Qwen3 Thinking Mode Fusion、NoThinking 反直觉发现、Kimi k1.5 long2short RL
- **16.6 节**：Hidden vs Visible CoT——OpenAI 的隐藏路线、DeepSeek 的开放路线、推理对齐的挑战
- **16.5 节**：自适应思考——Claude Opus 4.6 的旗舰案例、80 页 Constitution、AI Research Eval Suite

**核心收获**：

1. 推理模型的本质是"**把推理变成可优化的目标**"——RL 训练塑造了模型的思考行为。
2. Test-time Compute Scaling 改变了算力分配——从"训练时多算"到"推理时多算"。
3. Hybrid Thinking 解决了"何时思考"的问题——模型自主或用户控制。
4. Visible vs Hidden CoT 是关于"AI 透明度"的根本选择——市场倾向 Visible。
5. 自适应思考是推理模型的精细演化——从二选一到连续深度控制。

**接下来的章节**：

- [第 17 章 PRM 与推理时搜索](../chapter20_prm_search/outcome-vs-process)——怎么用过程奖励引导推理
- [第 19 章 Agentic RL](../chapter22_agentic/overview)——推理模型如何与工具调用结合
- [第 25 章奖励黑客与对齐失败](../chapter30_alignment_failures/classical-failures)——推理模型特有的对齐挑战
