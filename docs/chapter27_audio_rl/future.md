# 25.2 多模态音频 Agent 未来方向

> [27.1](./reward-design) 讲了音频奖励设计。本节看音频 RL 的前沿——多模态音频 Agent（Step-Audio-Chat、Qwen2-Audio）、实时语音对话（GPT-4o Voice）、以及未来方向。

## 简单语音对话 RL

本节用一个最小可运行流程展示音频 RL 的核心机制。完整工业训练需要 8 卡 H100 + 数周时间，这里只演示**奖励设计与 PPO 更新的耦合**。

### 实验设置

```python
# requirements: torch, transformers, librosa, soundfile
import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioDialogueConfig:
    # 音频编码器（伪代码：实际用 Qwen2-Audio encoder）
    audio_encoder_dim = 1280
    audio_frame_rate = 12.5  # Hz，下采样后
    # LLM 解码器（实际用 Qwen2.5-32B，这里简化）
    llm_hidden = 4096
    vocab_size = 152000
    # RL 配置
    group_size = 16         # GRPO 每组采样数
    max_response_len = 1024
    clip_eps = 0.2          # PPO clip
    beta_kl = 0.0           # Step-Audio 设为 0，允许自由探索
```

### 模型结构

```python
class AudioDialoguePolicy(nn.Module):
    """音频对话策略：音频编码 → LLM 推理 → 文本 + codec 生成"""
    def __init__(self, config):
        super().__init__()
        # 音频编码器（frozen）
        self.audio_encoder = AudioEncoder(config.audio_encoder_dim)
        for p in self.audio_encoder.parameters():
            p.requires_grad = False
        # adaptor: 25 Hz → 12.5 Hz
        self.adaptor = nn.Conv1d(config.audio_encoder_dim, config.llm_hidden,
                                  kernel_size=2, stride=2)
        # LLM 解码器
        self.llm = TransformerDecoder(config.llm_hidden, config.vocab_size)

    def forward(self, audio, question, response_tokens):
        # 1. 编码音频
        audio_feat = self.audio_encoder(audio)         # (B, T, D)
        audio_feat = self.adaptor(audio_feat.transpose(1,2)).transpose(1,2)

        # 2. 拼接 [audio, question, response] 序列
        inputs = concat_modalities(audio_feat, question, response_tokens)

        # 3. 自回归预测 response 的 logits
        logits = self.llm(inputs)
        return logits
```

### 奖励函数

实现 30.4 节描述的三类奖励：

```python
class AudioReward:
    def __init__(self, grm_model, prosody_ref_dist):
        self.grm = grm_model                # 生成式奖励模型
        self.prosody_ref = prosody_ref_dist # 人类韵律分布

    def content_reward(self, response_text, ground_truth):
        """内容正确性"""
        # 用 LLM-as-judge 判断语义等价
        prompt = f"判断答案是否等价：\n参考：{ground_truth}\n答案：{response_text}\n等价返回1否则0"
        return float(self.grm(prompt))

    def prosody_reward(self, response_audio):
        """韵律自然度"""
        f0 = librosa.pyin(response_audio)         # 基频
        f0_var = np.std(f0)
        # 与人类分布的 Wasserstein 距离
        f0_w = wasserstein_distance(
            np.histogram(f0, bins=50)[0] / len(f0),
            self.prosody_ref['f0_hist']
        )
        # 抑制扁平化（RLVR 的常见失败模式）
        flat_penalty = -max(0, 0.3 - f0_var)
        return -f0_w + 0.5 * flat_penalty

    def format_reward(self, response_text):
        """检查 <think>...</think> 格式（MGRD 的关键 trick）"""
        has_think = '<think>' in response_text and '</think>' in response_text
        return 1.0 if has_think else 0.0

    def total(self, response_text, response_audio, ground_truth, weights=(0.7, 0.2, 0.1)):
        w_c, w_p, w_f = weights
        return (w_c * self.content_reward(response_text, ground_truth)
              + w_p * self.prosody_reward(response_audio)
              + w_f * self.format_reward(response_text))
```

::: tip 格式奖励的作用
Step-Audio-R1 论文发现：去掉 format reward（即 $w_f = 0$）后，推理 token 数从 2800 跌到 1500，MMAU 掉 1.2 个百分点。原因是 RL 优化器天然倾向最 token 高效策略——直接给答案，跳过 `<think>`。

把 format reward 设为 0.2（占总 reward 20%）就足以稳定推理行为。这是音频 RL 与文本 RL 的关键差异：文本 RL 的 reward 信号密度足够高，自然涌现 CoT；音频 RL 必须显式奖励推理过程。
:::

### GRPO 训练循环

用 [GRPO](../chapter18_grpo/grpo-family)（Group Relative Policy Optimization）训练——不需要 critic，更适合大模型：

```python
def grpo_train_step(policy, ref_policy, reward_fn, batch, config):
    """单步 GRPO 训练"""
    advantages = []
    log_probs_all = []

    for prompt, audio in batch:
        # 1. 每个提示采样 G 条响应
        responses = []
        for _ in range(config.group_size):
            with torch.no_grad():
                resp = policy.sample(audio, prompt, config.max_response_len)
            responses.append(resp)

        # 2. 计算每条响应的 reward
        rewards = torch.tensor([
            reward_fn.total(r.text, r.audio, r.gt) for r in responses
        ])

        # 3. 组内归一化得 advantage（GRPO 核心）
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        advantages.extend(adv.tolist())

        # 4. 计算新策略 log π(a|s)
        for resp, a in zip(responses, adv):
            log_probs = policy.log_prob(audio, prompt, resp.tokens)
            log_probs_all.append(log_probs)

    # 5. PPO clip 目标（Step-Audio 设 β_kl = 0）
    advantages = torch.tensor(advantages).unsqueeze(1)
    policy_loss = 0
    for logp_new, resp in zip(log_probs_all, [r for b in batch for r in [None]]):
        # 简化：实际实现要按 token 求 ratio
        pass

    # 完整 PPO clip（参考第 5 章）
    # ratio = exp(logp_new - logp_old)
    # clipped = clip(ratio, 1-eps, 1+eps)
    # loss = -min(ratio * adv, clipped * adv).mean()

    return policy_loss

# 主循环
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = grpo_train_step(policy, ref_policy, reward_fn, batch, config)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

::: details 为什么用 GRPO 而不是 PPO
工业音频 LLM 几乎都用 GRPO（[DeepSeek-R1](https://arxiv.org/abs/2501.12948)）或其变体，不用经典 PPO。原因：

1. **省 critic**：32B 模型训 critic 又要一份显存，GRPO 用组内归一替代 critic
2. **更适合离散奖励**：音频 reward 多为 0/1 二值，critic 难学
3. **训练稳定**：组内 baseline 自然适应不同难度

Step-Audio-R1 的 RL 实现就是 on-policy PPO 但每个 prompt 采样 16 条，组内归一——本质就是 GRPO 的工程表述。
:::

### 自认知修正

工业音频 RL 有个非显学但关键的问题：**模型会忘记自己是音频模型**。预训练数据以文本为主，模型经常回答"我听不到声音"或"我是文本模型"。Step-Audio-R1 的修正流程：

```python
def self_cognition_correction(policy):
    """三阶段修正自认知错误"""
    # 阶段 1：迭代自蒸馏 + LLM judge 过滤
    for t in range(T):
        responses = policy.sample(audio_perception_queries)
        # judge 只保留正确自认知的回复
        correct = [r for r in responses if judge_acknowledges_audio(r)]
        policy.sft(correct)

    # 阶段 2：DPO 精修
    # 8000 偏好对：正确认知(w) vs 错误认知(l)
    pref_pairs = build_preference_pairs(correct_cog=positive, text_only=negative)
    policy.dpo(pref_pairs, beta=0.1)
```

效果：

| 训练阶段         | 自认知错误率 |
| ---------------- | ------------ |
| 基础模型         | 6.76%        |
| 迭代自蒸馏       | 2.63%        |
| 迭代自蒸馏 + DPO | **0.02%**    |

DPO 的精准对齐把错误率压到接近零。这一步看似琐碎，但部署时至关重要——用户期待模型自信地处理音频输入，而不是道歉式地说"我听不了"。

## 本章总结

音频 RL 是 2025-2026 年 RL 在 LLM 时代的最后一块拼图。本章覆盖了三个核心进展：

1. **Step-Audio-R1 的 MGRD**：解决了音频域的 inverted scaling 问题——根因是文本替代推理，解法是迭代蒸馏把推理基底从文本迁移到声学。R1 首次让音频模型从 test-time compute scaling 中受益
2. **Step-Audio-R1.5 的 RLHF 范式迁移**：识别并破解了"可验证奖励陷阱"——RLVR 优化"说什么"，用户关心"怎么说"，必须用 RLHF 的多维偏好建模补全韵律、情感、连贯性
3. **音频奖励设计**：内容 + 韵律 + 实时性三层的加权组合，rubric-based 生成式 RM 替代标量 RM，是音频 RL 区别于文本 RL 的核心工程

方法论层面，本章揭示了三个普适教训：

- **模态接地决定推理质量**：推理能力可跨模态迁移，但必须显式锚定到正确模态的特征上
- **数据质量 >> 数据数量**：pass@8 ∈ [3, 6] 的精选 5K 样本优于 200K 无筛选
- **奖励设计是 RL 的灵魂**：单一可验证奖励会塌缩模型行为，多维 rubric 是对齐真实体验的关键

多智能体协作 RL 的训练方法见[第 20 章的多智能体协作与 Agent Swarm](../chapter22_agentic/multi-agent-swarm)：当多个 LLM agent 协同完成任务时，需要同时处理信用分配和奖励分配。

## 延伸阅读

- [Step-Audio-R1 Technical Report (StepFun, 2025.11, arXiv:2511.15848)](https://arxiv.org/abs/2511.15848) — MGRD 框架原文，音频推理的奠基工作
- [Step-Audio-R1.5 Technical Report (StepFun, 2026.04, arXiv:2604.25719)](https://arxiv.org/abs/2604.25719) — RLHF 范式迁移，破解 verifiable reward trap
- [Step-Audio 2 Technical Report](https://arxiv.org/abs/2507.16632) — Step-Audio 系列基础模型
- [EnCodec: High Fidelity Neural Audio Compression (Meta, 2022)](https://arxiv.org/abs/2210.13438) — RVQ 编解码器经典工作
- [SoundStream: An End-to-End Neural Audio Codec (Google, 2021)](https://arxiv.org/abs/2107.03312) — SoundStream 原始论文
- [SpeechTokenizer: Unified Speech Tokenizer for Speech LLMs (2023)](https://arxiv.org/abs/2308.16692) — 语义/声学分层 token 化
- [WavTokenizer: An Efficient Acoustic Discrete Codec Tokenizer (ICLR 2025)](https://arxiv.org/abs/2408.16532) — 极致压缩（40-75 token/s）
- [Moshi: A Speech-Text Foundation Model for Real-Time Dialogue (Kyutai, 2024)](https://arxiv.org/abs/2410.00037) — 全双工实时对话，Mimi 编解码器
- [GPT-4o System Card (OpenAI, 2024)](https://arxiv.org/abs/2410.21276) — 工业级实时语音交互的里程碑
- [DeepSeek-R1: Incentivizing Reasoning Capability via RL (2025)](https://arxiv.org/abs/2501.12948) — RLVR + GRPO 训练范式，Step-Audio-R1 的方法基础
