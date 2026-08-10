---
title: A.1 Training Debugging Guide
---

# A.1 Training Debugging Guide

Once you have learned the core RL algorithms, a different reality quickly becomes obvious: **the real difficulty is rarely the algorithm, but the engineering**.

The model does not fit on a single GPU. Training runs for an entire day and the loss still goes up. Offline scores disagree with your intuition. The evaluation looks fine, but the product regresses. These are not questions that standard RL textbooks answer, but in real work you will face them every week.

This appendix is deliberately structured so that each section explains one thing clearly. Jump around as needed.

## Structure of This Appendix

| Section                                                                                              | Topic                                                 | What Problem It Solves                                                                                |
| ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| [A.2 RL Training Infrastructure: Sampling, Asynchrony, and Distributed Systems](./rl-infrastructure) | How an RL training system actually runs               | Sampling bottlenecks, rollout engines, async training, weight synchronization, DP/TP/PP/EP            |
| [A.3 Agentic RL Infrastructure](./agentic-rl-infra)                                                  | What infrastructure Agentic RL requires               | Sandboxes, trajectory storage, tool execution, multi-turn scheduling, a Relax case study              |
| [A.4 RL Post-Training and Agentic RL Benchmarks](./evaluation-badcase)                               | How to tell whether the model and agent are improving | Post-training evaluation, agentic benchmarks, training monitoring, badcase attribution, release gates |
| [C.3 A Glossary of RL Training Metrics for LLMs](./metrics-glossary)                                 | What the metrics in training logs actually mean       | PPO/GRPO/DPO/RM metrics grouped by function, abnormal signals, framework differences                  |
| [C.4 Industry Exercises](./industrial-exercises)                                                     | Practical skills for post-training and RL roles       | Real job tasks decomposed into stable capabilities, a skills map, and 8 industry-style exercises      |

## Reading Suggestions

- **If you are doing LLM post-training**: read A.2 → A.4 → C.3.
- **If you are doing Agentic RL**: read A.2 → A.3 → A.4.
- **If you are doing game or robotics RL**: focus on the non-LLM part of A.2 and the monitoring part of A.4.
- **If you are preparing for interviews**: start from the exercises in C.4, and then jump back when you find gaps.
- **If you only need the meaning of a metric**: go directly to [C.3 Metrics Glossary](./metrics-glossary) and look it up.
