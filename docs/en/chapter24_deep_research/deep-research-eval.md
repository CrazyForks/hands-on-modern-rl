# 21.2 Evaluation Benchmarks and Open-Source Projects

> [21.1](./browser-rl-harness) sets up the training harness. But how good is the trained Deep Research Agent? This requires **evaluation benchmarks**. This section covers two things: (1) mainstream Deep Research evaluation benchmarks (BrowseComp, xbench-DeepSearch, GAIA) and their design philosophies and pitfalls; (2) reproducible open-source projects (GPT-Researcher, STORM, OpenResearcher), which allow you to avoid building wheels from scratch.

## Why Deep Research Evaluation Is Particularly Challenging

Traditional LLM evaluation (MMLU, GSM8K) has two characteristics: (1) **unique answers** (a math problem has only one correct answer); (2) **no tools required** (the model answers directly). Deep Research breaks both of these points:

- **Non-unique answers**: Asking "Compare React and Vue's state management" has countless correct answers.
- **Must use tools**: The model cannot answer "What is the price of Bitcoin in June 2026?" based solely on memory.
- **Process matters**: Does the model answer correctly in 5 steps or 50 steps? Did it cite reliable sources?
- **Data pollution**: Internet content changes constantly, so today's answer may be invalid tomorrow.

Therefore, Deep Research evaluation requires specialized benchmark design.

## Mainstream Evaluation Benchmarks

### BrowseComp (Meta, 2025)

**BrowseComp** is a browser agent benchmark released by Meta in 2025, specifically designed to test an agent's ability to find information on the open web.

**Design Philosophy**:

- **Difficult to Answer Without a Browser**: Each question is designed in such a way that "it cannot be answered by the model's parameter memory alone."
- **Unique and Verifiable Answer**: Each question has a clear and precise answer that can be matched by string comparison.
- **Anti-Google**: Direct Google search cannot find the answer, requiring multi-step navigation.

**Example**:

> Q: "In the 1998 French World Cup 1/4 final, the player who scored the only goal for Argentina, where did he work as a youth coach after retiring?"
>
> A: "Argentinos Juniors" (exact string match)

To solve this problem, the model must: (1) Find the player who scored the only goal for Argentina in the 1998 World Cup 1/4 final → Batistuta; (2) Find where Batistuta worked as a youth coach after retiring; (3) Find the specific team. At least 3-5 steps of browser navigation are required.

**Metrics**: Exact Match Accuracy (EMA).

**State-of-the-Art Performance** (as of June 2026):

| System              | BrowseComp | Notes                          |
| ------------------- | ---------- | ------------------------------ |
| GPT-5 + Browser     | 38.2%      | OpenAI Operator upgrade        |
| Claude Opus 4.6     | 35.7%      | Anthropic internal             |
| Kimi K2.5 Swarm     | 72.1%      | Multi-agent collaboration      |
| Tongyi DeepResearch | 51.4%      | Alibaba, March 2026            |
| Human Expert        | 87.5%      | Single person, 30-minute limit |

Note that Kimi K2.5 Swarm outperforms single-agent systems by over 30 percentage points — this is practical evidence of [19.7 Multi-Agent Collaboration](../chapter22_agentic/multi-agent-swarm).

### xbench-DeepSearch (Tsinghua University, 2025)

**xbench-DeepSearch** is a Chinese Deep Research benchmark released by Tsinghua University and the University of Hong Kong in 2025, targeting several shortcomings of BrowseComp:

- **Chinese-centric**: BrowseComp is English-focused, while xbench-DeepSearch covers both Chinese and English.
- **Diverse task types**: BrowseComp consists of single-entity questions, whereas xbench-Deep/Searh includes multi-document synthesis, comparative analysis, and temporal reasoning.
- **Controllable difficulty**: Each question is annotated with a difficulty level (1–5 stars), allowing for the selection of subsets based on model capabilities.

**Task Types**:

| Type                     | Percentage | Example                                                                                   |
| ------------------------ | ---------- | ----------------------------------------------------------------------------------------- |
| Single-entity QA         | 30%        | "Which university did the 2025 Turing Award winner graduate from?"                        |
| Multi-document Synthesis | 25%        | "Compare the training cost of DeepSeek V3 and Llama 4"                                    |
| Comparative Analysis     | 20%        | "What are the differences in SSR performance between React 19 and Vue 3.5?"               |
| Temporal Reasoning       | 15%        | "What is the release date of Vision Pro in mainland China, announced at Apple WWDC 2024?" |
| Implicit Reasoning       | 10%        | "What is the expected accuracy rate of Y dataset using the method proposed in X paper?"   |

**Metrics**: In addition to EM (Exact Match), xbench-DeepSearch also reports:

- **Process Score**: Accuracy of intermediate steps
- **Efficiency**: Average steps / Minimum steps
- **Citation Quality**: Whether reliable sources are cited

### GAIA (Meta + HuggingFace, 2024)

**GAIA** (General AI Assistants) is an earlier benchmark, but it remains one of the standard test sets for Deep Research. GAIA defines three difficulty levels:

| Level   | Task Complexity | Average Steps | Example                                   |
| ------- | --------------- | ------------- | ----------------------------------------- |
| Level 1 | Simple          | 5–10          | "Find an image under specific conditions" |
| Level 2 | Medium          | 10–30         | "Organize a table from a PDF"             |
| Level 3 | Hard            | 30–100        | "Plan a multi-city trip across Europe"    |

**Metrics**: Accuracy + Average Steps (the fewer, the better).

GAIA differs from BrowseComp in key ways: GAIA tasks are closer to "personal assistant" scenarios, while BrowseComp is more aligned with "research tasks."

## Four Pitfalls in Evaluation

Deep Research evaluations have several unique pitfalls. If not carefully addressed, the scores can appear artificially high:

### Data Contamination

LLM pre-training data may already contain the answers. For example, if the question is "Who won the 2024 Nobel Prize in Physics," the model may answer based on memory (without needing a browser).

**Solutions**:

- Use **time-sensitive questions** (answers published after the training cutoff)
- Use **counterfactual questions** ("What would happen if event X did not occur?" — the model must investigate the actual occurrence of X)
- BrowseComp mitigates this to some extent through its design of "must-multi-step navigation"

### Diversity in Answer Expression

When asked to "contrast React and Vue", an agent's response of "React uses JSX, Vue uses template" or "Vue uses template, React uses JSX" is both correct, but EM (Exact Match) will mark it as incorrect.

**Solutions**:

- Use **LLM-as-Judge** (GPT-4 / Claude) to evaluate semantic equivalence
- Use **structured answers** (e.g., JSON, Markdown tables) to reduce expression differences
- xbench-DeepSearch uses LLM Judge for calibration

### Process Cheating

An agent may not actually browse, but instead generate answers that appear reasonable (hallucinate citations).

**Solutions**:

- **References must be clickable**: During evaluation, check whether the URLs provided by the agent are real
- **Web snapshots**: Save snapshots of the pages accessed by the agent during evaluation for later review
- BrowseComp designs a "reverse verification" mechanism: deliberately ask questions whose answers are random strings, which the agent cannot guess

### Cost Contamination

The token cost of different agents can vary by 10–30 times (as mentioned in [19.7](../chapter22_agentic/multi-agent-swarm), Kimi K2.5 Swarm is 15× more expensive than a single-agent system). Simply comparing accuracy will favor more expensive systems.

**Solutions**:

- Report **accuracy / token cost** efficiency metrics
- Compare under a fixed budget (e.g., "maximum of 100K tokens per question")

## Open-Source Project Reproduction

You don't need to build a harness from scratch — the following open-source projects provide complete Deep Research training / inference pipelines.

### GPT-Researcher (assafelovic-gpt-researcher)

**The most popular open-source Deep Research framework.** With over 18K GitHub stars and active maintenance.

**Features**:

- **Python**, based on Playwright
- Built-in Planner / Researcher / Writer three-tier architecture (typical Orchestrator-Worker)
- Supports multiple search backends (Tavily, SerpAPI, Google CSE, Bing)
- Outputs Markdown reports + citations

**Use Cases**: Quickly build a product-level Deep Research service. **Not suitable for**: RL training (designed for inference use).

```bash
pip install gpt-researcher
```

```python
from gpt_researcher import GPTResearcher

async def research():
    researcher = GPTResearcher("Compare React 19 vs Vue 3.5 SSR performance")
    report = await researcher.conduct_research()
    print(report)
```

### Stanford STORM (stanford-omp-storm)

**An open-source research framework from Stanford Oval Group**, specifically designed for "long-form structured article generation."

**Features**:

- Based on the Wikipedia writing process: first conduct a "simulated dialogue" (multiple personas ask each other questions), then create an outline, and finally write the full text
- Built-in Wikipedia search + citation management
- Outputs long-form Wikipedia-style articles (5K–20K words)

**Use Cases**: Academic reviews, in-depth reports. **Advantages**: High-quality citations (Wikipedia standard).

```bash
pip install knowledge-storm
```

```python
from storm import STORMWikiRunner

runner = STORMWikiRunner(...)
runner.run("History of reinforcement learning")
```

### OpenResearcher (tjuloonkopen-researcher)

**A fully open-source Deep Research training pipeline**, including RL training code.

**Features**:

- **Reproducible training**: Includes a 100K trajectory dataset + GRPO training script
- Based on the Search-R1 architecture
- 7B model achieves 31.2% on BrowseComp
- Complete documentation (in English)

**Use Cases**: Training a Deep Research Agent from scratch. **Advantages**: Complete vLLM + veRL pipeline, highly extensible.

```bash
git clone https://github.com/OPPO-PersonalAI/O-Researcher
cd open-researcher
bash train.sh --model qwen2.5-7b --algo grpo
```

### Other Projects Worth Noting

| Project                              | Institution          | Features                                                     |
| ------------------------------------ | -------------------- | ------------------------------------------------------------ |
| **Search-R1**                        | UIUC                 | The earliest open-source Deep Research RL training code      |
| **R1-Searcher**                      | Renmin University    | Multi-stage training (SFT → RL)                              |
| **Tongyi DeepResearch Reproduction** | Alibaba DAMO Academy | Industrial-scale, requires H100 cluster                      |
| **PokeeResearch**                    | Peking University    | 7B model achieves performance comparable to 70B-scale models |
| **DeepResearcher**                   | Renmin University    | End-to-end RL training open source                           |

## End-to-End Experiment and Training a Deep Research Agent from Scratch

This section provides a complete experimental workflow, integrating all the steps required to train a research agent from scratch.

### Step 1: Selecting a Base Model

- **Beginner-friendly**: Qwen2.5-7B-Instruct (easy to run)
- **Intermediate**: Llama-3.1-8B-Instruct
- **Advanced**: Qwen3-14B / DeepSeek-V2-Lite

### Step 2: Selecting the Action Space

- **Simple (API)**: Use the 3-action space from Search-R1
- **Realistic (Playwright)**: Use the 7-action space from OpenResearcher

### Step 3: Selecting the Training Data

- **xbench-DeepSearch Training Set** (10K)
- **HotpotQA + Natural Questions** (requires modification)
- **Self-generated**: Use GPT-5 / Claude to generate questions and answers

### Step 4: Training

```bash
# Use the training script from OpenResearcher
bash train.sh \
    --model qwen2.5-7b \
    --algo grpo \
    --env api \
    --data xbench-train.jsonl \
    --batch-size 256 \
    --lr 5e-7 \
    --epochs 3
```

### Step 5 and Evaluation

```bash
# BrowseComp Evaluation
python eval.py \
    --model checkpoint-final \
    --benchmark browsecomp \
    --max-steps 30
```

Expected Result: After training Qwen2.5-7B for 3 epochs, the accuracy on BrowseComp increases from 8% (SFT baseline) to 25-30%. This is already close to the level of GPT-4 plus browser (35%).

## Summary of This Section

The four benchmarks in Deep Research evaluation (BrowseComp / xbench-DeepSearch / GAIA / self-built) each have their own focus — BrowseComp tests the ability to perform "multi-step navigation," xbench-DeepSearch tests Chinese language and diverse task types, and GAIA tests personal assistant scenarios. **There is no one-size-fits-all solution**, and it is recommended to run at least two benchmarks.

In terms of open-source reproduction, **GPT-Researcher is suitable for product development**, while **OpenResearcher is suitable for research**. The former has mature engineering, while the latter has transparent training. If you are pursuing research or learning purposes, start with OpenResearcher. If you are aiming for product deployment, start with G-Researcher.

Next chapter [Chapter 22: Computer Use and GUI Agent](../chapter25_computer_use/training) shifts from the browser to the entire desktop — the agent is no longer just browsing the web, but can operate any GUI application (Excel, PS, internal OA).
