# 25.1 从 RLVR 到 RLHF 的音频奖励设计

<!--@include: ./intro.md{3,}-->

前文介绍了 Step-Audio 系列的发展。本节聚焦核心工程问题：**音频奖励如何设计？** 文本奖励模型可以直接使用偏好对训练，音频还包含韵律、情感和口音等维度，单一奖励信号无法覆盖。

## RLVR → RLHF 演进

Step-Audio-R1 用 MGRD + RLVR 在客观 benchmark 上达到 SOTA。但部署到真实对话后，团队发现了一个反直觉问题：**benchmark 分数越高，对话越难听**。

### 30.3.1 可验证奖励陷阱

[Step-Audio-R1.5](https://arxiv.org/abs/2604.25719) 把这个问题命名为**可验证奖励陷阱（Verifiable Reward Trap）**。

::: warning 可验证奖励陷阱
当音频 benchmark 的 ground truth 只是一个离散标签（情感类别、ASR 文本、场景标签）时，RLVR 只能奖励"猜对标签"，**结构性无视**韵律自然度、情感连贯性、对话流畅度。
:::

陷阱的机制：

```text
RLVR 目标 = 答案正确性 → 模型学到 "最 token 高效" → 回答变简短、机械、扁平
                ↓
         benchmark ↑  真实对话体验 ↓
```

RLVR 优化的是"what to say"（说什么），用户关心的是"how to say it"（怎么说）。两者解耦时，模型退化成**答题机**——技术上准确，体验上空洞。

### 30.3.2 Step-Audio-R1.5 与 从 RLVR 到 RLHF

R1.5 的解法：**用 RLHF 补 RLHF**——训练一个 holistic 偏好奖励模型，把正确性、流畅度、情感共鸣蒸馏成统一监督信号。

#### Audio-Centric Mid-Training

RLHF 之前先做一轮中间训练，强化音频理解和推理基底：

$$\mathcal{L}_{\text{mid}} = \mathbb{E}_{(x,q,r,y) \sim \mathcal{D}_{\text{audio}}}\left[\log \pi_\theta(r, y \mid x, q)\right] + \mathbb{E}_{(q,r,y) \sim \mathcal{D}_{\text{text}}}\left[\log \pi_\theta(r, y \mid q)\right]$$

其中 $(x, q, r, y)$ 是音频输入 + 上下文 + 推理 + 回复。文本数据保留长 CoT 推理结构， facilitating transfer 到音频。

#### Cold-Start SFT

Cold-start SFT 不再扩领域知识，而是**对齐交互行为**：

1. **多轮对话连续性**：跨轮保持上下文和约束
2. **指令遵循**：按用户指定的内容、格式、风格响应
3. **回复自然度**：连贯、对话得当
4. **交互感知**：处理追问、澄清、打断、用户修正

这一步为后续 RLHF 提供更好的初始化——避免 preference optimization 浪费在纠正基本对话行为上。

#### RLHF with Rubric-based Reward Model

音频交互是多目标优化——内容正确、韵律自然、情感连贯、延迟可控。R1.5 用 **rubric-based 生成式奖励模型（Generated Reward Model, GRM）**替代标量 RM：

```python
def audio_rlhf_reward(response, context, rubric):
    """多维度打分而非标量"""
    scores = {}
    scores["correctness"] = grm.score(response, context, rubric="内容是否正确")
    scores["fluency"] = grm.score(response, context, rubric="表达是否流畅自然")
    scores["prosody"] = grm.score(response, context, rubric="韵律是否符合情感")
    scores["emotional_resonance"] = grm.score(response, context, rubric="情感共鸣")
    scores["latency"] = grm.score(response, context, rubric="响应延迟")
    # 加权聚合（权重由人类偏好回归学到）
    return sum(w[k] * scores[k] for k in scores)
```

GRM 的优势：**人类偏好多维度**，标量 RM 无法捕捉。用 LLM-as-judge 给每个维度打分（rubric prompting），再学一个权重聚合器，相当于把 [RLHF](../chapter15_rlhf/intro) 的 RM 从"打总分"升级成"打分卡"。

#### 多目标 RL 训练目标

R1.5 的 RL 损失综合 RLVR 和 RLHF：

$$\mathcal{L}_{\text{RL}} = \underbrace{\mathbb{E}_{\mathcal{D}_{\text{verified}}}\left[R_{\text{verify}}(r, a)\right]}_{\text{客观正确性（RLVR）}} + \lambda \cdot \underbrace{\mathbb{E}_{\mathcal{D}_{\text{pref}}}\left[\log\sigma\left(\beta \log\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]}_{\text{主观偏好（DPO 形式）}}$$

前项保住客观推理能力（不让 RLHF 把 RLVR 学到的东西遗忘），后项用 DPO 损失（见 [第 16 章 GRPO/DPO](../chapter18_grpo/grpo-family)）对齐主观体验。$\lambda$ 平衡两者——这是音频 RL 的核心超参。

### 30.3.3 韵律自然度的保留

RLVR 最大的破坏是**韵律扁平化**：模型为最大化答案正确性，把语音变成单调的"朗读"。R1.5 用三个机制保住韵律：

1. **偏好数据包含韵律维度**：标注者比较两个回复时，不仅看内容，还听"哪个更自然、情感更对、节奏更像人"
2. **Rubric 显式评分 prosody**：GRM 单独打韵律分，不与正确性混淆
3. **Codec token 层级监督**：RVQ 的 $c_2 \ldots c_K$（声学层）参与 preference，确保生成阶段就保留韵律信息

R1.5 在 AudioMultiChallenge（多轮对话基准，测 Inference Memory / Instruction Retention / Self Coherence / Voice Editing）上达到或超越 Gemini-2.5-Flash，**同时**在传统推理 benchmark 上不掉分。RLVR 的"陷阱"被 RLHF 解开。

## 音频奖励设计

音频 RL 的 reward 比文本复杂得多——文本主要看正确性，音频要看内容、韵律、实时性三层。本节系统讨论三类奖励的设计。

### 30.4.1 内容正确性奖励

最直接：最终答案与 ground truth 比对。

$$R_{\text{content}}(r, a) = \begin{cases}1, & \text{if } a = a^* \\ 0, & \text{else}\end{cases}$$

变体包括：

- **ASR 字错率**：WER 越低奖励越高，$R = 1 - \text{WER}$
- **语义匹配**：用 embedding cosine 相似度，$R = \cos(\text{emb}(a), \text{emb}(a^*))$
- **LLM-as-judge**：让大模型判答案是否等价，$R \in [0, 1]$

内容奖励适合客观任务（数学、知识问答、ASR），但对开放式对话失效——没有标准答案。

### 30.4.2 韵律自然度奖励

韵律（prosody）包括音高、节奏、强度、停顿。建模人类对自然度的偏好是音频 RL 的难点。

#### 标量 RM 的局限

传统做法：训练一个 RM $R_\phi(\text{audio}) \to \mathbb{R}$，用人类两两偏好数据：

$$\mathcal{L}_{\text{RM}} = -\log\sigma(R_\phi(y_w) - R_\phi(y_l))$$

问题：标量 RM 把多维偏好压成一维，丢失了"内容对但韵律怪"vs"内容错但韵律自然"的区别。

#### 多维偏好建模

R1.5 的 GRM 用 **rubric prompting** 让 LLM 分维度打分：

```text
请按以下 rubric 评估回复（0-10 分）：
1. 内容正确性：答案是否准确？
2. 流畅度：是否连贯无卡顿？
3. 韵律自然度：音高、节奏是否符合人类说话习惯？
4. 情感匹配：语气是否与上下文情感一致？
5. 沉浸感：是否像在与人对话？

回复：[音频]
```

每个维度独立打分，再学权重 $w_k$ 聚合：

$$R_{\text{prosody}}(y) = \sum_k w_k \cdot \text{GRM}_k(y), \quad w = \arg\min_w \|R_{\text{human}}(y) - \sum_k w_k \cdot \text{GRM}_k(y)\|^2$$

权重通过 Bradley-Terry 回归从人类偏好学到。

#### 直接韵律特征奖励

除了偏好建模，还可以用声学特征直接打分：

```python
def prosody_reward(audio):
    # 提取韵律特征
    f0 = extract_pitch(audio)          # 基频曲线
    energy = extract_energy(audio)     # 能量包络
    duration = extract_durations(audio)  # 音素时长

    # 与参考韵律分布对比
    f0_score = -wasserstein(f0_dist(audio), f0_dist_human)
    energy_score = -wasserstein(energy_dist(audio), energy_dist_human)

    # 抑制单调（避免 RLVR 导致的扁平化）
    f0_var = np.std(f0)
    monotonicity_penalty = -max(0, 0.2 - f0_var)  # f0 方差太低就罚

    return 0.5 * f0_score + 0.3 * energy_score + 0.2 * monotonicity_penalty
```

这种"基于人类韵律分布"的奖励，能在没有偏好标注时抑制 RLVR 的扁平化倾向。

### 30.4.3 实时性奖励

实时对话要求首 packet 延迟 < 1 s，整体响应时间合理。把延迟纳入 reward：

$$R_{\text{latency}}(y) = \begin{cases}1, & T_{\text{first-packet}} < 0.5\text{s} \\ 0.5, & 0.5\text{s} \leq T_{\text{first-packet}} < 1.0\text{s} \\ 0, & T_{\text{first-packet}} \geq 1.0\text{s}\end{cases}$$

或用连续形式：

$$R_{\text{latency}}(y) = \exp(-\alpha \cdot T_{\text{first-packet}})$$

实时性奖励会和深度推理冲突——想得越久首 packet 越晚。这是 [双脑架构](#_30-2-3-dual-brain-architecture-双脑架构) 的价值：表达脑可以在构思脑还在想时就开始合成，把延迟隐藏在生成流水线里。

### 综合奖励

最终音频 RL 的 reward 通常加权组合三类：

$$R_{\text{total}} = w_c \cdot R_{\text{content}} + w_p \cdot R_{\text{prosody}} + w_l \cdot R_{\text{latency}}$$

权重 $(w_c, w_p, w_l)$ 反映应用场景：客服侧重内容（$w_c$ 大），陪伴机器人侧重韵律（$w_p$ 大），实时翻译侧重延迟（$w_l$ 大）。R1.5 的核心贡献就是证明了**只在 $w_c$ 上优化会掉进 verifiable reward trap**——必须引入 $w_p$ 才能保住真实对话体验。

## 本节总结

音频奖励设计比文本复杂得多——除了内容正确性，还要考虑韵律、情感、口音、说话风格。多维度 reward 的工程化方案有两条路线：(1) 加权多个 RM；(2) 用 LLM-as-Judge 直接评估综合质量。Step-Audio-R1.5 采用后者，把音频理解 + 评估合二为一。

下一节 [25.2 多模态音频 Agent 与未来方向](./future) 走向更前沿——音频不再只是输入输出，而是 agent 调用的工具（语音搜索、语音翻译、实时对话）。
