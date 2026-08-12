---
title: 从 CartPole 开始
emoji: 🏋️
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
license: apache-2.0
---

# 从 CartPole 开始

这是《动手学现代强化学习》的浏览器一键训练演示。在页面中点击“开始训练”，即可使用 CPU 训练 PPO 智能体，并实时观察 CartPole-v1 的评估奖励曲线。训练结束后，页面会播放确定性策略的一回合动画。

## 训练入口

- [在当前创空间中训练](https://modelscope.cn/studios/walkinglab/cartpole-online-training)
- [在 Google Colab 中打开 Notebook](https://colab.research.google.com/github/walkinglabs/hands-on-modern-rl/blob/main/notebooks/cartpole-ppo.ipynb)
- [查看可直接运行的 `train.py`](https://modelscope.cn/studios/walkinglab/cartpole-online-training/file/view/master/train.py)

## 本地运行

```bash
pip install -r requirements.txt
python app.py

# 不启动网页，直接运行训练脚本
python train.py --timesteps 30000
```

完整课程与本地实验代码：<https://github.com/walkinglabs/hands-on-modern-rl>
