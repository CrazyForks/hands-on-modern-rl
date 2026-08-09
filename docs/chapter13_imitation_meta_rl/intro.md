# 第 11 章 · 模仿学习、反向 RL 与元 RL

> [第 10 章 离线强化学习](../chapter12_offline_rl/intro)处理"只有历史数据、不能交互"的场景，但仍假设数据带有显式奖励信号。本章处理两种更极端的情形：(1) **完全没有奖励函数**——只有专家示范轨迹，怎么办？(2) **环境本身在不断变化**——智能体必须学会"快速适应新任务"。前者引出**模仿学习（Imitation Learning, IL）**与**反向 RL（Inverse RL）**，后者引出**元 RL（Meta-RL）**。两者最终在 LLM 时代合流：SFT 本质是行为克隆，InstructGPT 三阶段可重写为 BC + RL + RL，而 In-Context RL 揭示了"RL 算法本身可被蒸馏进 transformer"。

## 章节地图

- [11.1 行为克隆与 DAgger](./bc-dagger)：监督学习的视角，以及 BC 的分布漂移问题与 DAgger 的解决方案
- [11.2 逆向 RL 与 GAIL](./irl-gail)：从专家反推 reward，以及 GAIL 用 GAN 框架绕开显式 reward
- [11.3 元 RL：MAML、RL²、PEARL、In-Context RL](./meta-rl)：学会快速适应新任务，以及 LLM 时代 Algorithm Distillation 揭示的"RL as in-context learning"

下一节 [11.1 行为克隆与 DAgger](./bc-dagger) 从最朴素的模仿学习开始。
