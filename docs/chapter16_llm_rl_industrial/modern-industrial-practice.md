# 14.3 优化器与训练稳定性

[14.2 节工业实践](./industrial-post-training) 讨论了 2024-2025 年的主流工业方案——MiniMax、Qwen、Kimi、Seed、DeepSeek。这一节我们补充几个 2025-2026 年的最新工业实践：

- **GLM-4.5 / GLM-4.6**（智谱）：中国开源推理模型新秀
- **Llama 4**（Meta）：开源旗舰的进化
- **Seed-Thinking**（字节 Seed）：reasoning model 工业配方
- **MuonClip + QK-clip**（Kimi K2）：训练稳定性的新工具

这些工作代表了 LLM RL 工业训练的**最新 SOTA**——把 [前面章节](./intro) 的算法用到极致。

## 14.3.1 智谱 GLM-4.5 / GLM-4.6

[GLM-4.5](https://github.com/zai-org/GLM-4.5)（智谱 AI, 2025.07 发布）和 GLM-4.6（2025.10）是中国开源推理模型的重要进展。

### GLM 系列的特点

GLM（General Language Model）系列与 Qwen、DeepSeek 的差异：

- **Mixture of Experts (MoE)**：GLM-4.5 用 MoE 架构，355B 总参数 / 32B 激活
- **双模式**：thinking / non-thinking，与 Qwen3 类似
- **完全开源**：权重 + 训练方法 + 部分数据
- **代码能力**：特别强化了代码生成与 agentic 能力

### GLM-4.5 的训练流程

```text
┌──────────────────────────────────────────────────────────┐
│ Phase 1: Base 预训练（MoE 架构）                          │
│   - 15T tokens 高质量数据                                 │
│   - MoE: 355B total / 32B active                         │
│   - RoPE scaling 支持长 context                          │
├──────────────────────────────────────────────────────────┤
│ Phase 2: 通用 SFT                                         │
│   - 多语言对话数据                                        │
│   - 工具调用格式训练                                      │
├──────────────────────────────────────────────────────────┤
│ Phase 3: 推理 RL                                          │
│   - 数学、代码、推理任务                                  │
│   - GRPO + 规则奖励                                      │
│   - Self-validation 集成                                  │
├──────────────────────────────────────────────────────────┤
│ Phase 4: 通用 RLHF                                        │
│   - 对话质量、安全性                                      │
│   - Helpfulness / Harmlessness 双目标                    │
├──────────────────────────────────────────────────────────┤
│ Phase 5: Thinking / Non-Thinking 统一                    │
│   - 混合数据 SFT                                          │
│   - 让模型学会模式切换                                    │
└──────────────────────────────────────────────────────────┘
```

这个流程与 [DeepSeek-R1 的训练流程](../chapter18_grpo/deepseek-dapo) 高度相似——都是 SFT + 推理 RL + 通用 RLHF 的多阶段范式。

### GLM-4.6 的改进

2025.10 的 GLM-4.6 升级：

- **更长的 thinking**：支持 100K+ token 的长 CoT
- **更强的 agentic**：内部集成了更多工具（搜索、代码执行、文件操作）
- **更好的多模态**：与 GLM-4.5V 视觉模型协同
- **更细的 thinking budget**：用户可以指定 budget（与 Qwen3 类似）

### GLM-4.6 的 benchmark 表现

| Benchmark     | GLM-4.5 | GLM-4.6 |
| ------------- | ------- | ------- |
| AIME 2025     | 75.3    | 83.6    |
| MATH-500      | 92.1    | 95.4    |
| LiveCodeBench | 56.2    | 62.7    |
| GPQA Diamond  | 68.5    | 72.4    |

GLM-4.6 在多项 benchmark 上达到开源 SOTA，与 Claude Opus 4.5 / GPT-5 接近。

### GLM 的工业意义

GLM 系列的工业意义：

1. **开源推理模型的多样性**——不止 Qwen、DeepSeek，GLM 是第三个有影响力的中国开源模型
2. **MoE 架构在推理 RL 上的验证**——证明了 [GSPO](../chapter18_grpo/grpo-family) 在 MoE 上的有效性
3. **代码 + 推理的整合**——GLM-4.6 特别强调 agentic 能力，与 Claude Code 等竞争

## 14.3.2 Meta Llama 4

[Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)（Meta, 2025.04 发布）是 Meta 的开源旗舰。

### Llama 4 系列

Llama 4 包含三个变体：

- **Llama 4 Scout**：109B total / 17B active（MoE），10M context
- **Llama 4 Maverick**：400B total / 17B active，1M context
- **Llama 4 Behemoth**（未发布）：2T total / 288B active，Meta 内部训练

### Llama 4 的关键创新

**创新 1：原生多模态**

Llama 4 不是先训文本再加视觉，而是**从头开始多模态训练**——文本和图像 token 同等对待。这让 Llama 4 在视觉理解上优于"后接视觉"的方案。

**创新 2：Early Fusion**

不同于后期融合（先各自处理文本和图像再合并），Llama 4 用 **early fusion**——在模型早期就融合多模态信息。

**创新 3：MoE 架构**

Llama 4 全系采用 MoE——这是 Meta 第一次大规模 MoE。MoE 让模型在固定激活参数下，可以扩展总参数（提高能力，不增加推理成本）。

**创新 4：超长 Context**

Llama 4 Scout 支持 **10M token context**——这是当时开源模型中最长的。通过 iRoPE（interleaved RoPE）和 attention sparsity 实现。

### Llama 4 的训练方法

Meta 没有公开 Llama 4 的完整训练细节，但从论文和博客可以推断：

```text
┌─────────────────────────────────────────────────────────┐
│ Phase 1: 多模态预训练                                   │
│   - 文本 + 图像 + 视频联合训练                          │
│   - 22T tokens 级别（推测）                            │
│   - Early fusion 架构                                  │
├─────────────────────────────────────────────────────────┤
│ Phase 2: Mid-training（中等规模 SFT）                  │
│   - 通用指令跟随                                        │
│   - 工具调用格式                                        │
├─────────────────────────────────────────────────────────┤
│ Phase 3: 后训练 RL                                      │
│   - RLHF + RLVR 混合                                   │
│   - Helpfulness / Safety / Reasoning 多目标           │
└─────────────────────────────────────────────────────────┘
```

### Llama 4 的争议

Llama 4 发布后引发了一些争议：

**争议一：Benchmark 与实际表现差距**

Llama 4 Maverick 在多个 benchmark 上分数高，但用户实际使用时感觉不如 Claude 3.5 / GPT-5。Meta 后来承认 benchmark 评估与实际体验有差距。

**争议二：Maverick 的"特殊版本"**

LM Arena 上跑的 Maverick 是一个**经过专门优化的版本**——使用了 chat template 调整和 prompt engineering。开源的 Maverick 与 arena 版本有差异。

这个事件是 [Qwen3 数据污染](../chapter30_alignment_failures/modern-incidents) 类似的诚信问题——**benchmark 评估的脆弱性**。

### Llama 4 的工业意义

尽管有争议，Llama 4 仍然是开源 LLM 的重要进展：

1. **开源 MoE 的成熟**——证明 MoE 在开源生态的可行性
2. **多模态新范式**——early fusion 是后续工作的参考
3. **超长 context**——10M context 开启新应用场景

## 14.3.3 Seed-Thinking 与 字节的 Reasoning 配方

[Seed1.5-Thinking](https://arxiv.org/abs/2504.13914)（字节 Seed, 2025.04）是字节对 reasoning model 工业训练的系统总结。

### Seed-Thinking 的核心贡献

Seed-Thinking 不是一个新算法，而是**工业配方的系统化**——把多个组件组合起来，达到 SOTA：

**组件 1：数据 curation**

```text
数学数据：
  - 高质量数学题（AIME、Putnam 历年题）
  - 自动生成题（用强 LLM 生成新题）
  - 难度分级（按 base model 的通过率）

代码数据：
  - Codeforces 题目（带测试用例）
  - SWE-bench / SWE-smith（带 PR 数据）
  - 函数生成（HumanEval 扩展）
```

**组件 2：GRPO + DAPO 改进**

Seed-Thinking 用了 [DAPO](../chapter18_grpo/deepseek-dapo) 的四项工程改造 + 一些新改进：

- **动态 KL**：训练初期 KL 强，后期减弱
- **Adaptive clip**：根据训练进度调整 clip range
- **Group size 调度**：早期大 group，后期小 group

**组件 3：Self-Verification**

让模型在生成答案后做自我验证：

```python
def self_verification_reward(response, ground_truth):
    answer = extract_answer(response)

    # 让模型重新读题、验证答案
    verification_prompt = f"检查这个答案是否正确：{answer}"
    verification = model.generate(verification_prompt)

    if "正确" in verification and answer == ground_truth:
        return 1.0  # 答对 + 验证通过
    elif "错误" in verification and answer != ground_truth:
        return 0.5  # 答错但识别出错误
    else:
        return 0.0  # 答错且没识别
```

这种 self-verification reward 让模型学会**反思**——不只是答对，还要能识别错误。

**组件 4：Curriculum Learning**

按难度排序训练数据，先训简单的，再训复杂的。课程学习让训练更稳定，避免训练初期 reward 信号太稀疏。

### Seed-Thinking 的成绩

Seed-Thinking 1.5 在多个 benchmark 上：

| Benchmark         | 分数  |
| ----------------- | ----- |
| AIME 2024         | 86.4% |
| MATH-500          | 96.2% |
| GPQA Diamond      | 75.1% |
| Codeforces Rating | 1822  |

这是字节 Seed 内部 reasoning model 的核心配方——后来用于豆包 Pro 推理版等产品。

## 14.3.4 Kimi K2 的 MuonClip + QK-clip

[Kimi K2](https://arxiv.org/abs/2507.20534)（Moonshot, 2025.07）的工业贡献之一是 **MuonClip + QK-clip**——训练稳定性的新工具。

### Muon optimizer

[Muon](https://kellerjordan.github.io/posts/muon/)（2025.02）是新提出的 optimizer——Muon (Momentum + Orthogonalization)。它结合了：

- **Momentum**（来自 Adam）
- **Orthogonalization**（对梯度做正交化）

正交化让更新方向更稳定——避免 Adam 在某些方向震荡。

### MuonClip 的核心

MuonClip 在 Muon 基础上加入了 **clip**：

```python
def muon_clip_update(grad, momentum, clip_threshold=1.0):
    # Muon 主流程
    momentum = beta * momentum + (1 - beta) * grad
    orthogonalized = orthogonalize(momentum)

    # Clip 防止爆炸
    norm = torch.norm(orthogonalized)
    if norm > clip_threshold:
        orthogonalized = orthogonalized * (clip_threshold / norm)

    return -lr * orthogonalized
```

Clip 在大规模训练中非常重要——避免单个 outlier 梯度破坏整个训练。

### QK-clip 与 注意力稳定化

QK-clip 是 Kimi K2 的另一个创新——**裁剪 attention 的 Q·K 乘积**：

```python
def attention_with_qk_clip(Q, K, V, clip_value=30.0):
    # 标准 attention
    scores = Q @ K.T / sqrt(d)

    # QK-clip: 防止 attention scores 过大
    scores = torch.clamp(scores, min=-clip_value, max=clip_value)

    # Softmax + 加权
    attn = softmax(scores)
    output = attn @ V

    return output
```

**为什么需要 QK-clip？**

在长 context 训练中，attention scores 可能因为 **attention sink**（某些 token 吸引过多注意力）而变得极大——破坏 softmax 分布，导致梯度爆炸。

QK-clip 通过限制 scores 的范围，避免这种问题。

### Kimi K2 的训练效果

用 MuonClip + QK-clip，Kimi K2 在大规模训练中：

- **训练稳定性**：从平均每 1T tokens 一次 loss spike，提升到每 10T tokens 一次
- **训练速度**：比 Adam 快约 15%
- **最终性能**：Kimi K2 在多项 benchmark 上达到开源 SOTA

### MuonClip 的工业意义

MuonClip 是训练**超大规模 LLM**的关键工具：

- **万亿参数级训练**：传统 Adam 在万亿参数上不稳定，MuonClip 解决
- **超长 context**：QK-clip 让 1M+ context 训练可行
- **开源生态**：Muon 已经在开源社区普及（OpenLM、PyTorch 都支持）

## 14.3.5 中国工业实践的总结

到 2026 年中，中国 LLM RL 工业实践的格局：

| 厂商              | 旗舰模型           | RL 算法         | 工业贡献      |
| ----------------- | ------------------ | --------------- | ------------- |
| **DeepSeek**      | R1, V3.2           | GRPO + 改进     | 透明度、开源  |
| **阿里 Qwen**     | Qwen3 系列         | GSPO            | MoE 训练稳定  |
| **字节 Seed**     | 豆包 Pro, Seedance | DAPO + VAPO     | 多线并行      |
| **Moonshot Kimi** | K2, K2.5           | GRPO + MuonClip | 训练稳定工具  |
| **智谱 GLM**      | GLM-4.6            | GSPO-style      | 开源 MoE      |
| **MiniMax**       | M1, M2             | CISPO           | 低精度训练    |
| **StepFun**       | Step3              | 内部方法        | 推理 + 多模态 |

可以看到：

1. **每家厂商都有自己的"招牌 RL 算法"**——GRPO、GSPO、DAPO、CISPO、VAPO、MuonClip
2. **算法创新主要来自中国**——这与美国闭源派（OpenAI、Anthropic）形成对比
3. **工业贡献互补**——不是替代，是从不同角度解决 RL 训练问题

## 14.3.6 未来的工业方向

### 万亿参数 + 超长 context

- 万亿参数 base model（DeepSeek V3 是 671B，下一代会到 1T+）
- 10M+ token context（Llama 4 Scout 已经支持）
- 训练稳定性是核心挑战——MuonClip 是方向

### 多模态原生 RL

- 不再是"文本 RL + 视觉 SFT"，而是"多模态联合 RL"
- Llama 4 的 early fusion 是开始
- 未来会有更多原生多模态 RL 算法

### Agentic RL 工业化

- Agent 训练成为主流（不只是 SWE，还有客服、研究、操作）
- Agent trajectory 数据是关键
- Agent RL infra 投入巨大

### 训练成本下降

- 2024 年训练一个 SOTA 模型需要 $100M+
- 2026 年可能下降到 $10M（更优算法 + 更便宜算力）
- 这让小团队也能参与 SOTA 研究

## 训推不一致：LLM-RL 训练稳定性的隐藏杀手

前面讨论的 GLM-4.6、Llama 4、MuonClip 等方法都在解决"显式"的训练不稳定问题——loss spike、梯度爆炸、KV 崩塌。但 LLM-RL 中还有一个**被长期忽视的隐藏杀手**：**训推不一致（Training-Inference Mismatch）**。

严格来说，训推不一致本身并不是大模型独有的问题——任何 RL 系统中，只要采样策略和待优化策略之间存在漂移，就都会产生类似的分布偏差。AlphaGo、Atari DQN 时代就已经有了策略滞后（Policy Lag）导致训练不稳定的经验。但这个问题**在大模型 RL 的工程实现中被急剧放大了**，因为在 LLM-RL 系统中，采样和训练使用的是完全不同的引擎和精度，导致了一个根本性的撕裂。

### 问题根源：$\pi_{\text{rollout}}$ 与 $\pi_{\text{old}}$ 不是同一个策略

> **"When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch"**
> _(Liu et al., 2025)_

在绝大多数 LLM-RL 实现中，$\pi_{\text{rollout}}$（负责采样数据的推理策略）和 $\pi_{\text{old}}$（训练框架里记录的"旧策略"）**根本就不是同一个策略**：

- **推理侧**（生成 rollout 数据）：vLLM / SGLang，FP8/BF16 精度，KV Cache 优化
- **训练侧**（计算 log prob 和梯度）：FSDP/Megatron，BF16/FP32 精度，激活重计算

同一个模型参数在不同的精度、不同的计算图下，输出的 log-probability **天然就不一样**。你以为行为策略 $\mu$ 等于目标策略 $\pi_\theta$，实际上 $\mu \approx \pi_\theta$ 里的那个"约等于"可能已经偏离了几十个百分点。

### 精度是首要嫌疑人

> **"Defeating the Training-Inference Mismatch via FP16"**
> _(Qi et al., 2025)_

这篇论文把根因追到了浮点精度。BF16 的尾数位太少，在 token 级别的 log-probability 计算中引入了系统性舍入误差。而仅仅把精度切回 FP16，这个偏差就几乎消失了——几行代码解决了 LLM-RL 最令人头疼的训练崩溃。

> **"Taming the Tail: Stable LLM Reinforcement Learning via Dynamic Vocabulary Pruning"**
> _(arXiv 2512.23087, 2025)_

这篇论文进一步揭示训推不一致的**非对称性**：偏差与 $(1-p)$ 成正比——高频 token 误差微乎其微，但长尾低频 token 会产生系统性偏差，在梯度估计中持续累积，最终导致崩溃。

> **"Stabilizing Reinforcement Learning with LLMs: Formulation and Practices"**
> _(Zheng et al., Qwen Team, arXiv 2512.01374, 2025)_

阿里 Qwen 团队提出了统一的理论框架：token-level 的 REINFORCE 目标本质上是对序列级奖励的**一阶近似**，而这个近似成立需要两个前提——**(1) 训推一致**，**(2) 策略不过时**。一旦训推不一致成立，一阶近似就失效了。

### 与 PPO Clipping 的关系

读者可能会问：这跟 PPO 有什么关系？答案是：**PPO 的 Clipping 机制就是对训推不一致的一种"防御术"，但它只能防住一半**。

PPO 的核心公式是：

$$
\mathcal{L}^{\text{CLIP}} = \mathbb{E}\left[\min\left( r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right)\right]
$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ 是重要性采样比率。但 PPO 的 Clipping 有一个默认的前提假设——**分母 $\pi_{\text{old}}$ 确实是"采样时真正执行的那个策略"**。

在经典 RL 中，采样的进程和训练的进程是同一个 Python 进程，$\pi_{\text{old}}$ 就是采样那一瞬间保存下来的网络权重。但在 LLM-RL 中：

- $\pi_{\text{rollout}}$：vLLM 引擎在 FP8 下采样时**真实生效的策略**
- $\pi_{\text{old}}$：训练框架事后用 BF16/FP32 重新算出来的"你以为采样时用的策略"

这两个**本来就不是同一个策略**。重要性采样比率 $r_t$ 的**分母本身就是有偏的**——PPO 的 Clipping 在试图纠正优化导致的漂移，但它没有机制去纠正推理引擎和训练引擎之间的不一致。

打个比方：PPO 的 Clipping 保证了你**从旧策略出发不会走太远**，但它没保证"旧策略"那张地图本身是准的。训推不一致意味着**地图一开始就有偏差**，Clipping 发现不了这个问题。

### 工业界的主流修复路线

围绕训推不一致的修复方案，前沿工作大致沿着几条线展开：

- **精度修复**：FP16/BF16 替代 FP8 做 Rollout，减少 $\pi_{\text{rollout}}$ 和 $\pi_{\text{old}}$ 之间的数值偏差（Qi et al., 2025）；也有工作反过来压低训练端精度——FP8-RL 在 veRL 框架中实现了 W8A8 全栈低精度训练，配合重要性采样纠正，Rollout 吞吐提升 44% 同时匹配 BF16 基线（Qiu et al., arXiv 2601.18150）。
- **重要性采样（IS）纠正**：既然 $\pi_{\text{rollout}} \neq \pi_{\text{old}}$，那就显式引入重要性权重来纠正分布偏移。Truncated IS（TIS）是最直接的做法，剪掉极端的 IS 比率避免梯度爆炸（Yao et al., NeurIPS 2025）；更新的工作是 MinPRO（Lei et al., arXiv 2601.22718），用前缀内最小 token 级比率替代累积乘积，在 Off-policy 漂移较大时更稳定。
- **剪枝长尾 token**：训推不一致集中在低概率区域，直接剔除极端长尾 token 可以从源头消除最大偏差源（"Taming the Tail", arXiv 2512.23087）。
- **MoE 路由回放**：推理时的 Expert 路由与训练时天然不同，R3（Rollout Routing Replay）在训练时回放推理的路由分布，解决了 MoE-RL 独有的训推不一致放大效应（Zheng et al., arXiv 2512.01374）。
- **优化视角**：将训推不一致视为动态优化问题，通过响应长度激增等信号触发学习率调度（Zhang et al., arXiv 2602.01826）。
- **工程侧回滚纠正**：在训练前用当前训练引擎重新计算 Rollout 策略的 log-probability，暴力对齐 $\pi_{\text{rollout}}$ 和 $\pi_{\text{old}}$——成本高但最可靠。

### 与现实和解

这些论文共同指向一个结论：在 LLM-RL 的工程实践中，不存在"纯粹"的 On-policy。我们能做到的只是**把 $\mu$ 和 $\pi_\theta$ 的差距控制在可接受范围内**——PPO 的 Clipping 是一种控制，FP16 是一种控制，R3 路由回放也是一种控制。[第 4 章算法分类](../chapter03_mdp/algorithm-taxonomy)里讲的 On/Off-policy 理论是干净的二值分类，而工程现实是一个**连续的光谱**——理论上的 On-policy，实践中总是带着一点 Off-policy 的味道。

## 小结

现代 LLM RL 工业实践的核心趋势：

- **GLM-4.6 / Llama 4**：开源旗舰的进化，MoE + 多模态
- **Seed-Thinking**：reasoning model 的工业配方系统化
- **MuonClip + QK-clip**：训练稳定性的新工具
- **中国厂商主导算法创新**：每家都有自己的招牌方法

这些工作共同把 RL 在 LLM 训练中的地位**从可选变为核心**——没有 RL，就无法训出 SOTA 模型。

接下来：

- [第 17 章 Reasoning Models](../chapter19_reasoning/intro)——推理模型的详细讨论
- [第 18 章 PRM](../chapter20_prm_search/outcome-vs-process)——过程奖励的工业实践
- [第 21 章 RL-based SWE](../chapter23_rl_based_swe/swe-bench-and-rlvr)——代码 agent 的训练
