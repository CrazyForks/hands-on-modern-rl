---
title: Gymnasium CPU 训练游乐场
emoji: 🎮
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
license: apache-2.0
---

# Hands-On Modern RL · Gymnasium CPU Playground

《动手学现代强化学习》的在线训练合集。页面自动注册 Gymnasium 核心环境以及已安装的 Atari/ALE、MuJoCo、Box2D 和 Robotics 扩展环境，并保留 12 个经过调优的 CPU 快速训练配方。

首批实验：

- Multi-Armed Bandit：ε-greedy 探索与利用
- Blackjack：首次访问蒙特卡洛方法
- GridWorld、FrozenLake、Taxi：Q-Learning
- CliffWalking：SARSA 与安全路径
- CartPole：DQN 与 PPO 对照
- MountainCar：表格 Q-Learning
- Acrobot、Pendulum：PPO
- MountainCarContinuous：SAC 连续动作控制

完整目录中的 Auto 项会检查动作空间：离散动作使用 DQN，连续动作使用 SAC，其他兼容动作空间使用 PPO。缺少 ROM 或运行时依赖时，环境仍保留在目录中，错误与安装提示会显示在曲线下方的实时日志里，不会让页面整体退出。

课程项目：<https://github.com/walkinglabs/hands-on-modern-rl>

课程网站：<https://walkinglabs.github.io/hands-on-modern-rl/>
