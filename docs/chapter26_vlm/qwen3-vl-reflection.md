# 24.2 视觉反思 RL

这一节讨论两个 2025 年的多模态 RL 进展：

- **Qwen3-VL 的反思机制**：让视觉语言模型在回答前显式反思视觉内容
- **音频 RL（MGRD）**：Step-Audio-R1 的多模态推理

这两个工作代表了多模态 RL 从"简单 alignment"到"复杂推理"的演化。

## 11.7.1 Qwen3-VL 的反思机制

[Qwen3-VL](https://arxiv.org/abs/2511.21631)（阿里, 2025.05，与 Qwen3 同期发布）是 Qwen3 系列的视觉语言版本。

### 视觉理解的反思

传统 VLM（视觉语言模型）的工作方式：

```text
图像 + 问题 → VLM → 答案
```

模型一次前向就给出答案，没有"思考"过程。这在简单视觉任务（图像分类、物体识别）上够用，但在复杂任务（图表推理、几何证明、视觉数学）上不够——模型容易看错图、漏掉关键细节。

Qwen3-VL 引入了**视觉反思机制**——让模型在回答前显式反思：

```text
图像 + 问题
  → VLM 看图："我看到图中有..."
  → VLM 反思："让我再仔细看一眼..."
  → VLM 推理："基于我看到的内容，问题是..."
  → VLM 给答案
```

### Qwen3-VL 的训练

Qwen3-VL 的训练流程与 Qwen3 文本版类似，但增加了视觉数据：

```text
Phase 1: 多模态预训练（文本 + 图像）
Phase 2: 多模态 SFT（视觉问答、图像描述、几何推理）
Phase 3: 视觉推理 RL
  - 数学几何题（带图）
  - 图表理解
  - 视觉推理题（找规律、空间想象）
Phase 4: 通用 RLHF（对话质量 + 安全性）
```

**Phase 3 的关键数据**：

- **几何题**：带几何图形的数学题，需要先看图再解题
- **图表题**：理解 bar chart、line graph、table
- **视觉推理**： Raven's Progressive Matrices、visual analogy

这些数据让模型学会**视觉-语言联合推理**。

### 反思机制的工程实现

Qwen3-VL 的反思通过 **CoT prompting** 实现：

```python
def qwen3_vl_inference(image, question):
    prompt = f"""
    Image: {image}
    Question: {question}

    Please think step by step:
    1. First, describe what you see in the image.
    2. Then, identify the key elements relevant to the question.
    3. Reason about the answer based on what you see.
    4. Verify your answer by re-checking the image.
    5. Provide the final answer.
    """

    response = model.generate(prompt)
    return response
```

这种 prompting 让模型**显式进行视觉反思**。RL 训练时，模型因为"反思后再回答"得到更高 reward，反思行为被强化。

### Qwen3-VL 的成绩

| Benchmark             | Qwen2.5-VL | Qwen3-VL |
| --------------------- | ---------- | -------- |
| MathVista（视觉数学） | 65.3%      | 78.2%    |
| MMMU（多模态理解）    | 50.2%      | 58.7%    |
| DocVQA（文档问答）    | 92.1%      | 95.4%    |
| ChartQA（图表理解）   | 80.5%      | 87.3%    |

Qwen3-VL 在多项视觉推理 benchmark 上显著超越 Qwen2.5-VL——反思机制带来了 10+ 个百分点的提升。

### 反思机制的意义

1. **视觉理解也需要思考**：与文本推理一样，视觉任务也受益于 CoT
2. **反思是 RL 学到的**：不是 prompt engineering，是 RL 训练让模型内化了反思行为
3. **多模态推理 RL 的成熟**：从"学会看图"到"学会反思看图"

## 11.7.2 音频 RL 与 Step-Audio-R1 的 MGRD

[Step-Audio-R1](https://arxiv.org/abs/2511.15848)（StepFun, 2025.11）是音频领域的 RL 突破——**Multimodal Generative Reasoning with Direct Preference Optimization (MGRD)**。

### 音频 RL 的挑战

音频是比图像更复杂的多模态：

- **时序长**：一段音频可能几十秒到几分钟
- **多信息层**：语音内容、说话人身份、情感、语速、口音
- **标注昂贵**：音频偏好标注需要听完整段，比看图慢

传统音频模型（如 Whisper、SpeechT5）只做单一任务——语音识别或语音合成。**音频理解 + 推理 + 生成**的联合训练是 Step-Audio-R1 的突破。

### 多模态生成推理 + DPO

MGRD 的核心思想：

```text
┌──────────────────────────────────────────────────────────┐
│ 1. 多模态输入                                            │
│    - 音频（用户说话）                                    │
│    - 文本（可选上下文）                                  │
│    - 图像（可选视觉上下文）                              │
├──────────────────────────────────────────────────────────┤
│ 2. 联合推理                                              │
│    - 理解音频内容                                        │
│    - 识别说话人、情感、意图                              │
│    - 生成回复内容                                        │
├──────────────────────────────────────────────────────────┤
│ 3. 多模态输出                                            │
│    - 文本回复                                            │
│    - 语音合成（带情感、语速匹配）                        │
├──────────────────────────────────────────────────────────┤
│ 4. RL 训练                                               │
│    - 用 DPO 优化多模态输出                               │
│    - 偏好数据：好的（音频+文本）vs 差的（音频+文本）      │
└──────────────────────────────────────────────────────────┘
```

### MGRD 的训练数据

Step-Audio-R1 的训练数据：

- **音频对话**：100 万 + 轮多模态对话
- **情感标注**：音频 + 情感标签（开心、悲伤、愤怒等）
- **多语言**：中文、英文、方言
- **专业领域**：客服、教育、医疗等场景

### MGRD 与 DPO 的关系

MGRD 是 [DPO](../chapter17_dpo/dpo-theory-and-family) 在多模态的扩展：

- DPO：用文本偏好数据训练文本生成
- MGRD：用多模态偏好数据训练多模态生成

MGRD 的损失函数与 DPO 类似：

$$\mathcal{L}_{\text{MGRD}} = -\log\sigma\left(\beta \log\frac{\pi_\theta(y_w^{\text{multi}} | x)}{\pi_{\text{ref}}(y_w^{\text{multi}} | x)} - \beta \log\frac{\pi_\theta(y_l^{\text{multi}} | x)}{\pi_{\text{ref}}(y_l^{\text{multi}} | x)}\right)$$

其中 $y_w^{\text{multi}}$ 和 $y_l^{\text{multi}}$ 是多模态（音频 + 文本）的偏好对。

### Step-Audio-R1 的能力

Step-Audio-R1 的工业能力：

- **多轮语音对话**：自然、流畅、有情感的语音交互
- **方言理解**：支持多种中文方言（粤语、四川话等）
- **情感反馈**：识别用户情感，匹配情感回复
- **专业场景**：客服、教育、医疗等垂直领域

### 音频 RL 的意义

1. **音频是下一个 RL 战场**：文本 RL 已经成熟，图像 RL 在 2025 年突破，音频 RL 是 2026 年的新方向
2. **多模态联合**：音频 RL 不只是音频——是音频 + 文本 + 视觉的联合
3. **中国厂商领先**：StepFun、字节、阿里都在音频 RL 上投入

## 11.7.3 多模态 RL 的工业格局

到 2026 年中，多模态 RL 的工业格局：

### 视觉理解 RL

| 厂商      | 代表模型        | 特点           |
| --------- | --------------- | -------------- |
| 阿里      | Qwen3-VL        | 反思机制       |
| 字节      | Doubao-Vision   | 视觉推理       |
| Google    | Gemini 3 Vision | 原生多模态     |
| OpenAI    | GPT-5 Vision    | 通用           |
| Anthropic | Claude Opus 4.6 | 视觉 + agentic |

### 视觉生成 RL

（参考 [11.6 节现代视频生成 RL](./video-generation-modern)）

### 音频 RL

| 厂商    | 代表模型              | 特点            |
| ------- | --------------------- | --------------- |
| StepFun | Step-Audio-R1         | MGRD 多模态推理 |
| 字节    | Doubao-Voice          | 情感语音        |
| 阿里    | Qwen2-Audio           | 音频理解        |
| OpenAI  | GPT-4o Advanced Voice | 实时语音        |
| Google  | Gemini Live           | 实时多模态      |

### VLA（Vision-Language-Action）RL

| 厂商                  | 代表模型            | 特点              |
| --------------------- | ------------------- | ----------------- |
| Google                | Gemini Robotics 1.5 | Embodied Thinking |
| Physical Intelligence | π0                  | 通用机器人        |
| 字节                  | RoboBrain           | 中国 SOTA         |
| Skild AI              | Skild Brain         | 重工业机器人      |

## 11.7.4 多模态 RL 的共同挑战

尽管具体任务不同，多模态 RL 有几个共同挑战：

### 数据稀缺

- 视觉 RL：高质量视觉推理题稀缺
- 音频 RL：音频偏好数据标注昂贵
- VLA RL：机器人 trajectory 数据采集困难

### Reward 设计

- 视觉 RL：怎么自动评估"图像理解"？
- 音频 RL：怎么评估"语音情感"？
- VLA RL：怎么评估"机器人动作"？

### 长 horizon

- 视觉 RL：视频生成（30+ 帧）
- 音频 RL：长对话（几十轮）
- VLA RL：长 trajectory（机器人 100+ 步动作）

这些挑战指向同一个方向——**需要更强的算法、更精细的 reward、更长的 context**。

## 11.7.5 未来的多模态 RL 方向

### 原生多模态 RL

不是"文本 RL + 多模态 SFT"，而是**从头开始多模态 RL**。Llama 4 的 early fusion 是开始。

### 实时多模态 RL

实时交互（语音 + 视觉 + 动作）是下一代 agentic RL 的核心。

### 跨模态对齐

让模型理解"音频中说的内容 = 图像中显示的内容 = 文本中描述的内容"——跨模态语义对齐。

### Embodied AI 的成熟

VLA + world model + RL = 真正的通用机器人。这是 [第 12 章 embodied intelligence](../chapter28_vla/embodied-intelligence/) 的核心议题。

## 小结

2025-2026 年多模态 RL 的进展：

- **Qwen3-VL**：视觉反思机制，把推理 RL 用到视觉
- **Step-Audio-R1 MGRD**：音频多模态推理 + DPO
- **Gemini Robotics 1.5**：VLA 的下一步（参考 [embodied intelligence](../chapter28_vla/embodied-intelligence/)）

多模态 RL 是 RL 在 LLM 时代的自然扩展——从文本到图像、视频、音频、动作。每个模态都有自己的挑战，但核心 RL 思想（policy optimization、reward design、credit assignment）是通用的。
