# 23.2 Visual Reflection in RL

This section discusses two 2025 advances in multi-modal RL:

- **Reflection Mechanism of Qwen3-VL**: Enables visual language models to explicitly reflect on visual content before answering
- **Audio RL (MGRD)**: Step-Audio-R1's multi-modal reasoning

These two works represent the evolution of multi-modal RL from "simple alignment" to "complex reasoning".

## 23.2.1 Reflection Mechanism of Qwen3-VL

[Qwen3-VL](https://arxiv.org/abs/2511.21631) (Alibaba, 2025.05, released concurrently with Qwen3) is the visual language version of the Qwen3 series.

### Reflection on Visual Understanding

Traditional VLM (visual language model) works as follows:

```text
Image + Question → VLM → Answer
```

The model provides an answer in a single forward pass without a "thinking" process. This is sufficient for simple visual tasks (image classification, object recognition), but inadequate for complex tasks (chart reasoning, geometric proofs, visual mathematics) — the model is prone to misinterpreting images and missing key details.

Qwen3-VL introduces a **reflection mechanism** — allowing the model to explicitly reflect before answering:

```text
Image + Question
  → VLM looks at the image: "I see..."
  → VLM reflects: "Let me look again..."
  → VLM reasons: "Based on what I see, the question is..."
  → VLM provides the answer
```

### Training of Qwen3-VL

The training process of Qwen3-VL is similar to that of the text version of Qwen3, but with the addition of visual data:

```text
Phase 1: Multimodal Pretraining (Text + Image)
Phase 2: Multimodal SFT (Visual Question Answering, Image Description, Geometric Reasoning)
Phase 3: Visual Reasoning RL
  - Math problems with figures
  analogies, spatial imagination)
Phase 4: General RLHF (Dialogue Quality + Safety)
```

**Key Data for Phase 3**:

- **Geometry Problems**: Math problems with geometric figures, requiring the model to first observe the image and then solve the problem.
- **Chart Problems**: Understanding bar charts, line graphs, and tables.
- **Visual Reasoning**: Raven's Progressive Matrices, visual analogies.

These data enable the model to learn **visual-linguistic joint reasoning**.

### Engineering Implementation of the Reflection Mechanism

The reflection mechanism in Qwen3-VL is implemented through **CoT prompting**:

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

This prompting strategy enables the model to **explicitly perform visual reflection**. During RL training, the model receives higher rewards for "answering after reflection," thereby reinforcing the reflective behavior.

### Achievements of Qwen3-VL

| Benchmark                            | Qwen2.5-VL | Qwen3-VL |
| ------------------------------------ | ---------- | -------- |
| MathVista (Visual Mathematics)       | 65.3%      | 78.2%    |
| MMMU (Multimodal Understanding)      | 50.2%      | 58.7%    |
| DocVQA (Document Question Answering) | 92.1%      | 95.4%    |
| ChartQA (Chart Understanding)        | 80.5%      | 87.3%    |

Qwen3-VL significantly outperforms Qwen2.5-VL on multiple visual reasoning benchmarks — the reflection mechanism brings an improvement of over 10 percentage points.

### The Significance of the Reflection Mechanism

1. **Visual Understanding Also Requires Thinking**: Like text reasoning, visual tasks also benefit from Chain-of-Thought (CoT).
2. **Reflection is Learned via RL**: It is not prompt engineering, but RL training that internalizes the reflection behavior into the model.
3. **Maturity of RL for Multimodal Reasoning**: From "learning to see" to "learning to reflect on seeing."

## 23.2.2 Audio RL and Step-Audio-R1's MGRD

[Step-Audio-R1](https://arxiv.org/abs/2511.15848) (StepFun, 2025.11) represents a breakthrough in the audio domain — **Multimodal Generative Reasoning with Direct Preference Optimization (MGRD)**.

### Challenges in Audio Reinforcement Learning

Audio is more complex than images in terms of multimodality:

- **Long temporal sequences**: An audio clip can range from tens of seconds to several minutes.
- **Multiple information layers**: Speech content, speaker identity, emotion, speaking rate, and accent.
- **Expensive annotation**: Audio preference annotation requires listening to the entire clip, which is slower than visual annotation.

Traditional audio models (e.g., Whisper, SpeechT5) are designed for single tasks—speech recognition or speech synthesis. The joint training of **audio understanding + reasoning + generation** is the breakthrough of Step-Audio-R1.

### Multimodal Generation and Reasoning + DPO

The core idea of MGRD is as follows:

```text
┌──────────────────────────────────────────────────────────┐
│ 1. Multimodal Input                                       │
│    - Audio (user speech)                                  │
│    - Text (optional context)                              │
│    - Image (optional visual context)                      │
├──────────────────────────────────────────────────────────┤
│ 2. Joint Reasoning                                        │
│    - Understand audio content                            │
│    - Identify speaker, emotion, and intent                │
│    - Generate response content                            │
├──────────────────────────────────────────────────────────┤
│ 3. Multimodal Output                                      │
│    - Text response                                        │
│    - Speech synthesis (with emotion and speaking rate)   │
├──────────────────────────────────────────────────────────┤
│ 4. RL Training                                            │
│    - Optimize multimodal output using DPO                │
│    - Preference data: good (audio + text) vs. bad (audio + text) │
└──────────────────────────────────────────────────────────┘
```

### Training Data for MGRD

Training data for Step-Audio-R1:

- **Audio Dialogues**: 1 million + rounds of multimodal dialogues
- **Emotion Annotations**: Audio + emotion labels (happy, sad, angry, etc.)
- **Multilingual Support**: Chinese, English, dialects
- **Professional Domains**: Customer service, education, healthcare, etc.

### Relationship Between MGRD and DPO

MGRD is an extension of [DPO](../chapter17_dpo/dpo-theory-and-family) to the multimodal setting:

- DPO: Trains text generation using text preference data
- MGRD: Trains multimodal generation using multimodal preference data

The loss function of MGRD is similar to that of DPO:

$$\mathcal{L}_{\text{MGRD}} = -\log\sigma\left(\beta \log\frac{\pi_\theta(y_w^{\text{multi}} | x)}{\pi_{\text{ref}}(y_w^{\text{multi}} | x)} - \beta \log\frac{\pi_\theta(y_l^{\text{multi}} | x)}{\pi_{\text{ref}}(y_l^{\text{multi}} | x)}\right)$$

where $y_w^{\text{multi}}$ and $y_l^{\text{multi}}$ are preference pairs in the multimodal (audio + text) setting.

### Capabilities of Step-Audio-R1

Industrial capabilities of Step-Audio-R1:

- **Multi-turn Voice Dialogue**: Natural, fluent, and emotionally expressive voice interaction
- **Dialect Understanding**: Support for multiple Chinese dialects (Cantonese, Sichuan dialect, etc.)
- **Emotional Feedback**: Recognize user emotions and match emotional responses
- **Professional Scenarios**: Customer service, education, healthcare, and other vertical domains

### Significance of Audio RL

1. **Audio is the Next Battlefield for RL**: Text RL has matured, image RL broke through in 2025, and audio RL is the new direction in 2026
2. analogies: Audio RL is not just audio — it is the combination of audio, text, and vision
3. **Chinese Companies Lead**: StepFun, ByteDance, and Alibaba are all investing heavily in audio RL

## 23.2.3 Industrial Landscape of Multimodal RL

By mid-2026, the industrial landscape of multimodal RL will be as follows:

### Visual Understanding RL

| Vendor    | Representative Model | Features             |
| --------- | -------------------- | -------------------- |
| Alibaba   | Qwen3-VL             | Reflection Mechanism |
| ByteDance | Doubao-Vision        | Visual Reasoning     |
| Google    | Gemini 3 Vision      | Native Multimodal    |
| OpenAI    | GPT-5 Vision         | General              |
| Anthropic | Claude Opus 4.6      | Vision + Agentic     |

### Visual Generation RL

(Reference [24.5 Modern Video Generation RL](../chapter29_visual_generation/video-generation-modern))

### Audio RL

| Vendor    | Representative Model  | Features                  |
| --------- | --------------------- | ------------------------- |
| StepFun   | Step-Audio-R1         | MGRD Multimodal Reasoning |
| ByteDance | Doubao-Voice          | Emotional Voice           |
| Alibaba   | Qwen2-Audio           | Audio Understanding       |
| OpenAI    | GPT-4o Advanced Voice | Real-time Voice           |
| Google    | Gemini Live           | Real-time Multimodal      |

### VLA (Vision-Language-Action) RL

| Vendor                | Representative Model | Features                 |
| --------------------- | -------------------- | ------------------------ |
| Google                | Gemini Robotics 1.5  | Embodied Thinking        |
| Physical Intelligence | π0                   | General-purpose Robotics |
| ByteDance             | RoboBrain            | Chinese SOTA             |
| Skild AI              | Skild Brain          | Heavy Industry Robotics  |

## 23.2.4 Common Challenges in Multimodal RL

Although the specific tasks differ, multimodal RL faces several common challenges:

### Data Scarcity

- **Visual RL**: High-quality visual reasoning tasks are scarce.
- **Audio RL**: Labeling audio preference data is expensive.
- **VLA RL**: Collecting robot trajectory data is difficult.

### Reward Design

- **Visual RL**: How to automatically evaluate "image understanding"?
- **Audio RL**: How to assess "speech emotion"?
- **VLA RL**: How to evaluate "robot actions"?

### Long Horizon

- **Visual RL**: Video generation (30+ frames)
- **Audio RL**: Long conversations (tens of rounds)
- **VLA RL**: Long trajectories (robot actions over 100 steps)

These challenges point in the same direction — **we need stronger algorithms, more refined rewards, and longer context**.

## 23.2.5 Future Directions in Multimodal RL

### Native Multimodal RL

Not "text RL + multimodal SFT", but rather **multimodal RL from scratch**. Early fusion in Llama 4 is the beginning.

### Real-Time Multimodal RL

Real-time interaction (speech + vision + action) is the core of the next-generation agentic RL.

### Cross-Modal Alignment

Making the model understand that "what is said in audio = what is shown in image = what is described in text" — cross-modal semantic alignment.

### Maturity of Embodied AI

VLA + world model + RL = True General-Purpose Robotics. This is the core topic of [Chapter 28: Embodied Intelligence](../chapter28_vla/embodied-intelligence/).

## Summary

Advancements in multimodal RL from 2025 to 2026:

- **Qwen3-VL**: Visual reflection mechanism, applying reasoning RL to vision
- **Step-Audio-R1 MGRD**: Audio multimodal reasoning + DPO
- **Gemini Robotics 1.5**: Next step of VLA (refer to [embodied intelligence](../chapter28_vla/embodied-intelligence/))

Multimodal RL is a natural extension of RL in the LLM era — from text to image, video, audio, and action. Each modality has its own challenges, but the core RL ideas (policy optimization, reward design, credit assignment) are universal.
