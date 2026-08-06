# v5.2 路径迁移映射表

> 旧目录 → 新目录映射，作为 git mv 操作的依据。
> 处理顺序：先处理 1:1 重命名（无需拆分），再处理需要拆分的目录。

## 一、1:1 重命名（直接 git mv）

| 旧路径                                     | 新路径                                     | 对应章节                                   |
| ------------------------------------------ | ------------------------------------------ | ------------------------------------------ |
| `chapter02_dpo/`                           | `chapter17_dpo/`                           | 第 17 章 DPO                               |
| `chapter03_bandits/`                       | `chapter02_bandits/`                       | 第 2 章 老虎机                             |
| `chapter04_dqn/`                           | `chapter07_dqn/`                           | 第 7 章 DQN                                |
| `chapter05_policy_gradient/`               | `chapter08_policy_gradient/`               | 第 8 章 策略梯度                           |
| `chapter06_actor_critic/`                  | `chapter09_actor_critic/`                  | 第 9 章 Actor-Critic                       |
| `chapter07_ppo/`                           | `chapter10_ppo/`                           | 第 10 章 PPO                               |
| `chapter08_rlhf/`                          | `chapter15_rlhf/`                          | 第 15 章 RLHF                              |
| `chapter09_grpo_rlvr/`                     | `chapter18_grpo/`                          | 第 18 章 GRPO                              |
| `chapter12_continuous_control/`            | `chapter11_continuous_control/`            | 第 11 章 连续控制                          |
| `chapter13_offline_rl/`                    | `chapter12_offline_rl/`                    | 第 12 章 离线 RL                           |
| `chapter13_reasoning_models/`              | `chapter19_reasoning/`                     | 第 19 章 Reasoning                         |
| `chapter14_imitation_meta_rl/`             | `chapter13_imitation_meta_rl/`             | 第 13 章 模仿/IRL/元 RL                    |
| `chapter14_prm_search/`                    | `chapter20_prm_search/`                    | 第 20 章 PRM                               |
| `chapter15_exploration_marl_hierarchical/` | `chapter14_exploration_marl_hierarchical/` | 第 14 章 探索/MARL/分层                    |
| `chapter15_rl_based_swe/`                  | `chapter23_rl_based_swe/`                  | 第 23 章 代码智能体                        |
| `chapter16_alignment_failures/`            | `chapter30_alignment_failures/`            | 第 30 章 奖励黑客                          |
| `chapter17_llm_rl_industrial/`             | `chapter16_llm_rl_industrial/`             | 第 16 章 LLM RL 工业                       |
| `chapter22_cai_rlvr/`                      | `chapter21_cai_rlvr/`                      | 第 21 章 CAI/RLAIF                         |
| `chapter28_computer_use/`                  | `chapter25_computer_use/`                  | 第 25 章 Computer Use                      |
| `chapter30_audio_rl/`                      | `chapter27_audio_rl/`                      | 第 27 章 音频                              |
| `chapter11_vlm_rl/`                        | `chapter26_vlm/`                           | 第 26 章 VLM（除 visual-generation-rl.md） |

## 二、保留不改名

| 路径                  | 原因                                          |
| --------------------- | --------------------------------------------- |
| `chapter00_overview/` | 序章已合并，目录可保留作历史归档              |
| `chapter01_cartpole/` | 第 1 章编号一致                               |
| `chapter03_mdp/`      | 第 3-6 章共享此目录（MDP/价值/DP/Q-Learning） |
| `preface/`            | 序章目录                                      |

## 三、需要拆分的目录

### `chapter09_alignment/`

- `industrial-post-training.md` → `chapter16_llm_rl_industrial/industrial-post-training.md` (16.2)
- `modern-industrial-practice.md` → `chapter16_llm_rl_industrial/modern-industrial-practice.md` (16.4)
- `dpo-theory-and-family.md` → `chapter17_dpo/dpo-theory-and-family.md` (17.3)
- 其他文件按主题归入 16 或 17

### `chapter10_agentic_rl/`

- `multi-turn-rl.md`, `tool-use-*.md`, `industrial-*.md`, `trajectory-synthesis.md` → 保留在 `chapter22_agentic/`（第 22 章主体）
- `deep-research-agent.md` → 移到新 `chapter24_deep_research/intro.md`（第 24 章）
- 新增 `chapter22_agentic/multi-agent-swarm.md`（22.6）

### `chapter12_future_trends/`

- `embodied-intelligence/` → 移到 `chapter28_vla/`（第 28 章 VLA）
- `llm-driven-discovery.md` → 移到 `chapter31_alphaevolve/`（第 31 章）
- `self-play-outlook/`, `rl-scaling-outlook.md`, `llm-multi-agent-rl/` → 移到 `chapter32_selfplay/`（第 32 章）

### `chapter23_rl_environments/`

- 整个目录内容 → 并入 `chapter18_grpo/`（作为 18.5/18.6/18.7）

### `chapter35_rl_evaluation/`

- 整个目录内容 → 并入 `chapter30_alignment_failures/`（作为 30.6/30.7）

### `chapter36_distributed_rl_training/`

- 整个目录内容 → 并入 `chapter16_llm_rl_industrial/`（作为 16.5/16.6/16.7）

### `chapter34_scalable_oversight/`

- 已删除（v5.2 砍掉）。如果目录存在，归档到 `archive/`。

### `chapter11_vlm_rl/visual-generation-rl.md` 等

- 视觉生成相关 → 移到新 `chapter29_visual_generation/`（第 29 章）
- `video-generation-modern.md` → `chapter29_visual_generation/`

## 四、新增目录（v5.2 全新章节）

| 新路径                         | 内容                                                               |
| ------------------------------ | ------------------------------------------------------------------ |
| `chapter22_agentic/`           | 第 22 章（由 chapter10_agentic_rl 改名）                           |
| `chapter24_deep_research/`     | 第 24 章 Deep Research（新建，从 chapter10 拆出 + 新增 24.2/24.3） |
| `chapter25_computer_use/`      | 第 25 章（由 chapter28_computer_use 改名）+ 新增 25.2/25.3         |
| `chapter29_visual_generation/` | 第 29 章 视觉生成（新建，从 chapter11_vlm_rl 拆出）                |
| `chapter31_alphaevolve/`       | 第 31 章（新建，从 chapter12_future_trends 拆出）                  |
| `chapter32_selfplay/`          | 第 32 章（新建，从 chapter12_future_trends 拆出）                  |

## 五、操作顺序

1. **Phase 1**: 简单 1:1 git mv（20+ 个目录）
2. **Phase 2**: 拆分复合目录（chapter09_alignment, chapter10_agentic_rl, chapter12_future_trends, chapter23_rl_environments, chapter35_rl_evaluation, chapter36_distributed_rl_training）
3. **Phase 3**: 新建 chapter24/29/31/32 目录，移入对应文件
4. **Phase 4**: 补充缺失文件（22.6, 24.2, 24.3, 25.2, 25.3 等）
5. **Phase 5**: 重写 config.mjs zhSidebar
6. **Phase 6**: 全局更新跨章节链接
7. **Phase 7**: build 验证
