# 26.4 Evolutionary Search and Scientific Discovery

[The previous sections of this chapter](../self-play-outlook/) discussed the traditional frontiers of reinforcement learning — self-play, multi-agent systems, and scaling. This section turns to a new direction: **LLM-driven scientific discovery**.

The characteristics of this direction are:

- **LLMs are not just actors** — they are idea generators, code writers, and experiment designers
- **RL is not just policy optimization** — it is search, evolution, and self-improvement
- **The goal is not "playing games" or "conversing"** — it is discovering new algorithms, new mathematics, and new science

Representative works from 2024 to 2026:

- **AlphaEvolve** (DeepMind, 2024.05): LLM + evolutionary algorithms to discover new mathematics
- **Genie 3** (DeepMind, 2025.08): Generative world model
- **Titans** (Google, 2024.12): Long-term memory architecture
- **Multi-Agent Deep Research** (Byte Seed, 2025.11): Training multi-agent search systems with M-GRPO

These works represent the "next-generation paradigm" of the integration of RL and LLM — from "training a policy" to "training a research system."

## 26.4.1 AlphaEvolve and LLM + Evolution for Mathematical Discovery

[AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) (DeepMind, 2024.05 released, 2025 full version) is a flagship case of LLM-driven scientific discovery.

### Core Idea of AlphaEvolve

Model mathematical discovery as **evolutionary search + LLM code generation**:

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. Population Initialization: Existing algorithms / proofs   │
│    (e.g., Strassen's algorithm for matrix multiplication)    │
├─────────────────────────────────────────────────────────────┤
│ 2. LLM Mutation: Let Gemini generate branches                 │
│    "Please improve this algorithm" / "Try different approaches" │
│    Output: New code                                           │
├─────────────────────────────────────────────────────────────┤
│ 3. Automatic Evaluation: Run the code, measure performance    │
│    (e.g., the number of multiplications for matrix multiplication) │
├─────────────────────────────────────────────────────────────┤
│ 4. Selection: Retain the ones with good performance, eliminate the poor ones │
├─────────────────────────────────────────────────────────────┤
│ 5. Iteration: Return to step 2                                │
└─────────────────────────────────────────────────────────────┘
```

This process is almost identical to the [classic genetic algorithm (GA)](../../chapter03_mdp/dp-mc-td) — the only difference is that the mutation operation changes from "random modification" to "LLM intelligent generation."

### Key Innovations of AlphaEvolve

**Innovation 1: LLM as an Intelligent Mutation Operator**

Traditional GA mutation is random modification — with low success rate. LLM mutation is "understanding the current code + proposing meaningful improvements" — with high success rate.

**Innovation 2: Code as Genes**

Instead of using bit strings as genes, AlphaEvolve uses **executable code**. This allows fitness to be **automatically measured** — by running the code, we can evaluate its performance.

**Innovation 3: Gemini as the LLM Backend**

AlphaEvolve uses Gemini Pro/Ultra as the LLM backend — strong LLMs significantly improve the quality of mutations.

### Discoveries of AlphaEvolve

AlphaEvolve has made **real new discoveries** in multiple domains:

**Discovery 1: New Algorithm for Matrix Multiplication**

In 1969, Strassen discovered that $4 \times 4$ matrix multiplication can be done in 49 multiplications (previously thought to be 64). AlphaEvolve discovered a new algorithm that uses **48 multiplications** — **surpassing human research for over 50 years**.

**Discovery 2: New Bounds in Combinatorics**

In combinatorial problems such as [tensor decomposition](https://en.wikipedia.org/wiki/Tensor_decomposition) and [sorting networks](https://en.wikipedia.org/wiki/Sorting_network), AlphaEvolve has discovered multiple new bounds that surpass the known optimal results.

**Discovery 3: Optimization of Google Infrastructure**

Within DeepMind, AlphaEvolve has been used to optimize:

- Data center scheduling algorithms (saving 0.7% of global computing resources)
- TPU matrix multiplication hardware design
- Machine learning kernel optimization

### The Significance of AlphaEvolve

AlphaEvolve demonstrates:

1. **LLMs can perform real scientific research** — not just "answer questions," but "discover new knowledge"
2. **Evolution + LLM is a powerful combination** — LLM provides intelligence, evolution provides exploration
3. **Automatic evaluation is critical** — only domains that can be automatically evaluated are suitable for this paradigm

## 26.4.2 Genie 3 and Generative World Models

[Genie 3](https://deepmind.google/models/genie/) (DeepMind, 2025.08) is a representative work of generative world models.

### What is a World Model?

A world model is a model that can **predict environmental dynamics**:

```text
Input: current state s_t + action a_t
Output: next state s_{t+1}
```

In reinforcement learning, a world model can **replace the real environment** — policy is trained on the world model, avoiding the costly interaction with the real environment.

### Evolution of the Genie Series

**Genie 1** (2024.02): Learning a world model from videos

- Input: Internet videos
- Output: Can generate controllable "game" environments
- Key: No explicit action labels, the model learns "what is an action" on its own

**Genie 2** (2024.12): 3D world model

- Input: A single image
- Output: Can generate interactive 3D environments
- Key: The environment can maintain consistency for several minutes

**Genie 3** (2025.08): Large-scale, controllable, long-horizon

- Input: Natural language description
- Output: Fully controllable, long-horizon 3D environments
- Key: Can be used to train embodied agents

### Training Genie 3

```text
┌─────────────────────────────────────────────────────────┐
│ Phase 1: Video Pretraining                              │
│   - Large amounts of unlabeled videos                   │
│   - Learning "how the world works"                      │
├─────────────────────────────────────────────────────────┤
│ Phase 2: Action Labeling                                 │
│   - Letting LLM label actions in videos                  │
│   - Learning "what actions lead to what changes"        │
├─────────────────────────────────────────────────────────┤
│ Phase 3: World Model Training                            │
│   - (s_t, a_t, s_{t+1}) triplets                         │
│   - Training a model that can predict s_{t+1}           │
├─────────────────────────────────────────────────────────┤
│ Phase 4: RL Training                                     │
│   - Policy trained on the world model                    │
│   - Avoiding expensive real environment interactions     │
└─────────────────────────────────────────────────────────┘
```

### Applications of Genie 3

**Application 1: Training Embodied Agents**

Robots learn to walk, grasp, and manipulate objects within a world model, avoiding trial-and-error on real robots (which is costly and dangerous).

**Application 2: Game Generation**

Genie 3 can automatically generate playable games. Players describe the desired game, and Genie 3 generates the complete environment.

**Application 3: Simulation Training**

High-risk scenarios such as autonomous driving, industrial control, and surgical procedures can be trained within a world model and then deployed into real-world environments.

### Limitations of Genie 3

- **Accuracy**: The world model is not 100% accurate—long-term predictions may drift.
- **Generalization**: Simulating environments outside the training distribution is challenging.
- **Computational Cost**: High-quality world model inference is computationally expensive.

## 26.4.3 Titans and Long-Term Memory Architecture

[Titans](https://arxiv.org/abs/2501.00663) (Google, 2024.12 release, 2025 revised version) represent a new direction in LLM architecture—**long-term memory**.

### Motivation for Titans

Transformers have a fundamental limitation: the **context window**. Even when extended to 1M tokens, they cannot handle "infinite-length" inputs. Titans aim to address this issue.

### Design of Titans

Titans introduces a **neural long-term memory**:

```text
┌──────────────────────────────────────────────────────────┐
│ Short-term memory: attention (standard Transformer)      │
│   - Processes recent tokens                              │
│   - Limited capacity (context window)                    │
├──────────────────────────────────────────────────────────┤
│ Long-term memory: neural memory module (new)             │
│   - Continuously learns and stores                       │
│   - Infinite capacity                                     │
├──────────────────────────────────────────────────────────┤
│ Persistent memory: task-related knowledge (system prompt, knowledge base) │
│   - Constant                                           │
└──────────────────────────────────────────────────────────┘
```

The three-layer memory system enables Titans to handle **infinite-length inputs**—the long-term memory continuously stores historical information.

### The Relationship Between Titans and RL

The core of Titans is **learning how to remember**—and this itself is an RL problem:

- **State**: Current input + current memory
- **Action**: How to update memory (write / forget / update)
- **Reward**: Future task performance (if useful information is remembered, task performance improves)

Titans use **surprise** as an internal reward—when the input is "surprising," memory is strengthened; when the input is "repetitive," memory is weakened. This is a form of **self-supervised RL**—the model generates its own reward.

### Experimental Results of Titans

On long-horizon tasks, Titans significantly outperform Transformers:

| Task                            | Transformer | Titans                     |
| ------------------------------- | ----------- | -------------------------- |
| Language modeling (10M context) | OOM         | 67% perplexity improvement |
| Long-document QA                | 55%         | 78%                        |
| Temporal prediction             | 65%         | 82%                        |

Titans demonstrate that **long-term memory is the next direction for scaling**—not just "wider and deeper," but also "better at remembering."

## 26.4.4 M-GRPO and Multi-Agent Search Training

[Multi-Agent Deep Research: Training Multi-Agent Systems with M-GRPO](https://arxiv.org/abs/2511.13288) (Byte Seed, 2025.11) uses M-GRPO — a multi-agent extension of Group Relative Policy Optimization — to train multi-agent search systems.

### System Design of M-GRPO

A multi-agent system consists of a main agent and multiple sub-agents:

```text
┌─────────────────────────────────────────────────────────┐
│ Main Agent (planner): Overall planning                  │
│   - Receives a task                                     │
│   - Breaks it into subtasks                             │
│   - Schedules sub-agents                                │
├─────────────────────────────────────────────────────────┤
│ Sub Agents (tool executors): Tool execution             │
│   - Multi-round invocation of search, code, etc. tools  │
│   - Each has a different frequency and variable call count │
├─────────────────────────────────────────────────────────┤
│ Hierarchical Credit Assignment                          │
│   - The main agent and sub-agents compute group-relative │
│   - Exchange minimal statistics via a shared store      │
└─────────────────────────────────────────────────────────┘
```

### Training Method of M-GRPO

M-GRPO addresses three challenges in multi-agent reinforcement learning training:

- **Hierarchical credit assignment**: The main agent and sub-agents separately compute group-relative advantages, avoiding "contribution confusion"
- **Trajectory alignment**: Sub-agent call counts vary, and trajectory alignment schemes are used to generate fixed-size batches
- **Decoupled training**: Agents are distributed across independent servers, exchanging statistics through a shared store, without requiring cross-server backpropagation

On benchmarks such as GAIA, XBench-DeepSearch, and WebWalkerQA, M-GRPO consistently outperforms single-agent GRPO and multi-agent GRPO with "frozen sub-agents."

### Relationship Between M-GRPO and AlphaEvolve

Both approaches use LLM + RL/search, but from different perspectives:

- **AlphaEvolve**: Evolutionary search (gradient-free, population-based), focused on algorithm discovery
- **M-GRPO**: Multi-agent RL (based on GRPO), focused on tool-enhanced deep research

They represent two complementary paradigms in LLM-driven discovery.

## 26.4.5 Recursive Self-Improvement

**Recursive Self-Improvement (RSI)** is the ultimate form of LLM-driven discovery — **the model improves itself**.

This concept represents the next stage in the evolution of LLMs, where the model not only learns from data but also iteratively refines its own architecture, training procedures, and reasoning capabilities. RSI enables models to autonomously enhance their performance, leading to more sophisticated and adaptive systems.

### The Core Loop of RSI

```text
┌─────────────────────────────────────────────────────┐
│ 1. Current model M_t evaluates its own capabilities  │
│    - In which tasks does it perform well? In which   │
│      tasks does it perform poorly?                   │
├─────────────────────────────────────────────────────┤
│ 2. Generate improvement plans                        │
│    - Design new training data                        │
│    - Adjust training hyperparameters                 │
│    - Improve algorithms                              │
├─────────────────────────────────────────────────────┤
│ 3. Execute improvements                              │
│    - Train a new model M_{t+1} using the plan         │
├─────────────────────────────────────────────────────┤
│ 4. Evaluate the new model                           │
│    - Is M_{t+1} better than M_t?                     │
│    - If yes, retain M_{t+1}; if not, roll back       │
├─────────────────────────────────────────────────────┤
│ 5. Return to step 1                                  │
└─────────────────────────────────────────────────────┘
```

### Current State of RSI

By mid-2026, RSI remains a **research concept** with no industrial-level implementation. The reasons are as follows:

**Challenge 1: Inaccuracy in Self-Assessment**

Models find it difficult to accurately evaluate their own capabilities — they are prone to overestimation (Dunning-Kruger effect).

**Challenge 2: Explosion of Search Space for Improvement Strategies**

The possible combinations of training data, hyperparameters, and algorithms are astronomically large.

**Challenge 3: Safety Risks**

If a model can improve itself indefinitely, it may surpass human control — this is a core concern in [AI safety](../../chapter30_alignment_failures/classical-failures).

### Partial Implementations of RSI

Although there is no complete RSI system, there are several **partial implementations**:

- **AutoGPT** (2023): An early attempt with limited effectiveness
- **SRPO** ([arXiv:2406.01660](https://arxiv.org/abs/2406.01660)): Trains a preference model using self-improvement process (Cohere, 2024)
- **Voyager** ([arXiv:2305.16291](https://arxiv.org/abs/2305.16291)): A Minecraft agent that autonomously learns new skills
- **DeepMind's Self-Play System**: A model improves itself by playing against itself

These are early forms of RSI — demonstrating partial feasibility, but still far from the true "recursive self-improvement."

## 26.4.6 Common Pattern of LLM-driven Discovery

The common pattern of these works (AlphaEvolve, Genie 3, Titans, MIRAS, RSI):

### LLM as Intelligent Guidance for Search

Traditional search (MCTS, Beam Search) requires manually designed heuristics. LLMs can **automatically generate heuristics** — making the search more intelligent.

### Automatic Evaluation is Key

AlphaEvolve is able to discover new algorithms because **the performance of algorithms can be automatically measured** (by running the code). This is the prerequisite for LLM-driven discovery — **only domains that can be automatically evaluated are suitable**.

### Combination > Single Method

- AlphaEvolve = LLM + Evolution
- Genie 3 = LLM + World Model
- Titans = LLM + Long-term Memory
- Multi-Agent Deep Research = LLM + Multi-agent + RL

**Combining multiple methods** is stronger than using a single method — this is the new form of RL in the LLM era.

### From "Training Policy" to "Training a System"

Traditional RL trains a single policy. LLM-driven discovery trains a **complete research system** — multiple agents + memory + search + tools.

## 26.4.7 Future Directions

### Scientific Discovery

Extend the AlphaEvolve approach to:

- **Biology**: Protein design, drug discovery
- **Chemistry**: New molecular synthesis pathways
- **Physics**: New experimental design, new theory validation

### Education

Use LLM-driven discovery to personalize education — identifying the most suitable learning path for each student.

### AGI

Recursive self-improvement is one potential path toward AGI — if a model can continuously improve itself, it may surpass human capabilities at some point.

This comes with **serious safety risks** — this is also the core issue of [Alignment research](../../chapter30_alignment_failures/scaling-and-defenses).

## Summary

LLM-driven discovery is a new frontier of RL in 2025–2026:

- **AlphaEvolve**: LLM + evolution, discovering new mathematics
- **Genie 3**: Generative world model, for embodied agents
- **Titans**: Long-term memory architecture, extending context
- **M-GRPO**: Multi-agent RL training
- **RSI**: Recursive self-improvement (partially implemented)

These works collectively point toward a new paradigm — **from training policies to training research systems**. This is the next-generation direction of the integration between RL and LLMs, and also the most advanced frontier of AGI research.
