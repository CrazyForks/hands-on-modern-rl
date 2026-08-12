"""ModelScope Studio: train PPO on CartPole from the browser."""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
import gradio as gr
import imageio.v2 as imageio
import matplotlib
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)
SEED = 42
PROJECT_URL = "https://github.com/walkinglabs/hands-on-modern-rl"
COURSE_URL = "https://walkinglabs.github.io/hands-on-modern-rl/"
CHAPTER_URL = f"{COURSE_URL}chapter01_cartpole/training"
COLAB_URL = (
    "https://colab.research.google.com/github/walkinglabs/hands-on-modern-rl/"
    "blob/main/notebooks/cartpole-ppo.ipynb"
)
SCRIPT_URL = (
    "https://modelscope.cn/studios/walkinglab/modern-rl-experiment01-cartpole/"
    "file/view/master/train.py"
)
LOGO_URL = (
    "https://raw.githubusercontent.com/walkinglabs/hands-on-modern-rl/"
    "main/docs/public/readme/readmelogo.png"
)


def evaluate(model: PPO, episodes: int = 5) -> tuple[float, float]:
    """Evaluate the current deterministic policy without rendering."""
    env = gym.make("CartPole-v1")
    rewards, _ = evaluate_policy(
        model,
        env,
        n_eval_episodes=episodes,
        deterministic=True,
        return_episode_rewards=True,
        warn=False,
    )
    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


def reward_figure(steps: list[int], rewards: list[float]):
    """Build a compact reward chart suitable for Gradio streaming updates."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(steps, rewards, color="#4f46e5", marker="o", linewidth=2)
    ax.axhline(475, color="#16a34a", linestyle="--", linewidth=1.2, label="Solved threshold: 475")
    ax.set(xlabel="Training steps", ylabel="Mean reward", ylim=(0, 510))
    ax.set_title("PPO evaluation reward on CartPole-v1")
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def record_policy(model: PPO) -> tuple[str, float]:
    """Render one deterministic episode and save it as a browser-friendly GIF."""
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs, _ = env.reset(seed=SEED + 1)
    frames: list[np.ndarray] = []
    score = 0.0

    for _ in range(500):
        frames.append(env.render())
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        score += float(reward)
        if terminated or truncated:
            break

    env.close()
    gif_path = ARTIFACT_DIR / "cartpole-trained-policy.gif"
    imageio.mimsave(gif_path, frames, duration=1 / 30, loop=0)
    return str(gif_path), score


def train(total_timesteps: int):
    """Train in chunks so the browser receives live progress and reward updates."""
    total_timesteps = int(total_timesteps)
    chunk_size = 2_000
    env = gym.make("CartPole-v1")
    env.reset(seed=SEED)
    model = PPO(
        "MlpPolicy",
        env,
        seed=SEED,
        verbose=0,
        device="cpu",
        n_steps=1_024,
        batch_size=64,
        learning_rate=3e-4,
    )

    steps: list[int] = [0]
    mean_rewards: list[float] = []
    initial_mean, initial_std = evaluate(model)
    mean_rewards.append(initial_mean)
    started_at = time.perf_counter()

    yield (
        "训练已启动：先收集环境交互，再更新策略参数。",
        reward_figure(steps, mean_rewards),
        None,
        f"初始策略：{initial_mean:.1f} ± {initial_std:.1f}",
    )

    trained = 0
    while trained < total_timesteps:
        current_chunk = min(chunk_size, total_timesteps - trained)
        model.learn(
            total_timesteps=current_chunk,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        trained += current_chunk
        mean_reward, std_reward = evaluate(model)
        steps.append(trained)
        mean_rewards.append(mean_reward)
        elapsed = time.perf_counter() - started_at
        status = (
            f"训练中：{trained:,}/{total_timesteps:,} 步（{trained / total_timesteps:.0%}），"
            f"耗时 {elapsed:.1f} 秒"
        )
        yield (
            status,
            reward_figure(steps, mean_rewards),
            None,
            f"当前评估：{mean_reward:.1f} ± {std_reward:.1f}",
        )

    model_path = ARTIFACT_DIR / "ppo-cartpole"
    model.save(model_path)
    gif_path, demo_score = record_policy(model)
    elapsed = time.perf_counter() - started_at
    final_mean, final_std = evaluate(model, episodes=10)
    env.close()

    yield (
        f"训练完成：共 {total_timesteps:,} 步，耗时 {elapsed:.1f} 秒。",
        reward_figure(steps, mean_rewards),
        gif_path,
        (
            f"10 回合平均奖励：{final_mean:.1f} ± {final_std:.1f}；"
            f"右侧动画回合得分：{demo_score:.0f}/500。"
        ),
    )


CSS = """
.gradio-container {
  max-width: 1240px !important;
  margin: 0 auto !important;
  padding: 24px 22px 44px !important;
  background: #f7f8fc;
}
.app-shell { font-family: Inter, "PingFang SC", "Microsoft YaHei", sans-serif; }
.hero {
  position: relative;
  overflow: hidden;
  padding: 34px 38px;
  border: 1px solid rgba(129, 140, 248, .22);
  border-radius: 24px;
  color: #f8fafc;
  background:
    radial-gradient(circle at 82% 5%, rgba(129, 140, 248, .35), transparent 32%),
    radial-gradient(circle at 95% 90%, rgba(34, 211, 238, .16), transparent 28%),
    linear-gradient(135deg, #07162f 0%, #132450 55%, #273181 100%);
  box-shadow: 0 18px 50px rgba(15, 23, 42, .16);
}
.hero::after {
  content: "";
  position: absolute;
  width: 250px;
  height: 250px;
  right: -92px;
  top: -112px;
  border: 1px solid rgba(255,255,255,.14);
  border-radius: 50%;
}
.project-mark {
  width: 270px;
  max-width: 70%;
  height: auto;
  margin: 0 0 22px;
  filter: brightness(0) invert(1);
  opacity: .94;
}
.chapter-kicker {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 14px;
  padding: 7px 12px;
  border: 1px solid rgba(199, 210, 254, .32);
  border-radius: 999px;
  color: #c7d2fe;
  background: rgba(255, 255, 255, .08);
  font-size: 13px;
  font-weight: 700;
  letter-spacing: .04em;
}
.hero h1 {
  margin: 0 0 10px;
  color: #ffffff;
  font-size: clamp(30px, 5vw, 46px);
  line-height: 1.12;
  letter-spacing: -.03em;
}
.hero-copy {
  max-width: 760px;
  margin: 0;
  color: #cbd5e1;
  font-size: 16px;
  line-height: 1.75;
}
.hero-links { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 24px; }
.hero-link {
  display: inline-flex;
  align-items: center;
  min-height: 40px;
  padding: 0 15px;
  border: 1px solid rgba(255,255,255,.2);
  border-radius: 10px;
  color: #eef2ff !important;
  background: rgba(255,255,255,.08);
  font-size: 14px;
  font-weight: 650;
  text-decoration: none !important;
  transition: transform .16s ease, background .16s ease;
}
.hero-link:hover { transform: translateY(-1px); background: rgba(255,255,255,.15); }
.hero-link.primary { color: #172554 !important; background: #ffffff; border-color: #ffffff; }
.stat-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin: 16px 0 22px;
}
.stat-card {
  padding: 16px 18px;
  border: 1px solid #e4e7f0;
  border-radius: 14px;
  background: #ffffff;
  box-shadow: 0 6px 20px rgba(15, 23, 42, .045);
}
.stat-label { margin-bottom: 5px; color: #64748b; font-size: 12px; font-weight: 650; }
.stat-value { color: #0f172a; font-size: 20px; font-weight: 780; letter-spacing: -.02em; }
.section-title { margin: 3px 0 5px; color: #111827; font-size: 22px; font-weight: 780; }
.section-copy { margin: 0 0 14px; color: #64748b; font-size: 14px; line-height: 1.7; }
.control-card, .result-card, .lesson-card {
  border: 1px solid #e2e5ef !important;
  border-radius: 18px !important;
  background: #ffffff !important;
  box-shadow: 0 8px 28px rgba(15, 23, 42, .055) !important;
}
.control-card { padding: 22px !important; }
.result-card { padding: 14px 16px !important; }
.lesson-card { margin-top: 16px; padding: 20px 22px; }
.primary-btn {
  min-height: 50px !important;
  border: 0 !important;
  border-radius: 12px !important;
  background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
  box-shadow: 0 8px 22px rgba(79, 70, 229, .25) !important;
  font-size: 16px !important;
  font-weight: 750 !important;
}
.status-box, .metric-box {
  padding: 12px 14px !important;
  border-radius: 11px !important;
  background: #f7f8ff !important;
}
.status-box { border-left: 3px solid #6366f1 !important; }
.metric-box { border-left: 3px solid #22c55e !important; }
.flow-list { display: grid; gap: 10px; margin-top: 16px; }
.flow-item { display: flex; align-items: flex-start; gap: 10px; color: #475569; font-size: 13px; line-height: 1.55; }
.flow-index {
  display: inline-grid;
  flex: 0 0 24px;
  width: 24px;
  height: 24px;
  place-items: center;
  border-radius: 7px;
  color: #4338ca;
  background: #eef2ff;
  font-weight: 750;
}
.lesson-card h3 { margin: 0 0 7px; color: #0f172a; font-size: 17px; }
.lesson-card p { margin: 0; color: #64748b; font-size: 14px; line-height: 1.7; }
.lesson-card a { color: #4f46e5 !important; font-weight: 650; }
.footer-note { margin-top: 18px; text-align: center; color: #94a3b8; font-size: 12px; }
@media (max-width: 760px) {
  .gradio-container { padding: 12px 10px 30px !important; }
  .hero { padding: 26px 22px; border-radius: 18px; }
  .stat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
"""


with gr.Blocks(css=CSS, title="第 1 章配套实验 · CartPole 在线训练") as demo:
    gr.HTML(
        f"""
        <main class="app-shell">
          <section class="hero">
            <img class="project-mark" src="{LOGO_URL}" alt="Hands-On Modern RL" />
            <div class="chapter-kicker">配套实验 · 第 1 章 · CartPole 入门</div>
            <h1>1.3 PPO 训练可视化</h1>
            <p class="hero-copy">
              在浏览器中完成一次可观察的强化学习训练：让 PPO 智能体从随机控制开始，
              通过环境交互逐步学会保持倒立摆平衡。整个实验只使用 CPU，无需配置本地环境。
            </p>
            <nav class="hero-links" aria-label="项目入口">
              <a class="hero-link primary" href="{PROJECT_URL}" target="_blank" rel="noreferrer">GitHub 项目 ↗</a>
              <a class="hero-link" href="{COURSE_URL}" target="_blank" rel="noreferrer">课程网站 ↗</a>
              <a class="hero-link" href="{CHAPTER_URL}" target="_blank" rel="noreferrer">阅读第 1 章 ↗</a>
              <a class="hero-link" href="{COLAB_URL}" target="_blank" rel="noreferrer">在 Colab 运行 ↗</a>
              <a class="hero-link" href="{SCRIPT_URL}" target="_blank" rel="noreferrer">查看 train.py ↗</a>
            </nav>
          </section>
          <section class="stat-grid" aria-label="实验信息">
            <div class="stat-card"><div class="stat-label">环境</div><div class="stat-value">CartPole-v1</div></div>
            <div class="stat-card"><div class="stat-label">算法</div><div class="stat-value">PPO</div></div>
            <div class="stat-card"><div class="stat-label">动作空间</div><div class="stat-value">左 / 右</div></div>
            <div class="stat-card"><div class="stat-label">目标奖励</div><div class="stat-value">500</div></div>
          </section>
        </main>
        """
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=300, elem_classes="control-card"):
            gr.HTML(
                """
                <h2 class="section-title">运行实验</h2>
                <p class="section-copy">选择训练步数并启动。训练期间，右侧奖励曲线会每 2,000 步更新一次。</p>
                """
            )
            timesteps = gr.Slider(
                minimum=10_000,
                maximum=50_000,
                value=30_000,
                step=5_000,
                label="训练步数",
                info="默认设置通常可在普通 CPU 上较快完成。",
            )
            start = gr.Button("开始训练", variant="primary", elem_classes="primary-btn")
            status = gr.Markdown("**等待开始** · 点击按钮后持续更新训练进度。", elem_classes="status-box")
            metrics = gr.Markdown("**评估结果** · 尚未开始评估。", elem_classes="metric-box")
            gr.HTML(
                """
                <div class="flow-list">
                  <div class="flow-item"><span class="flow-index">1</span><span>智能体观察小车位置、速度、杆角度与角速度。</span></div>
                  <div class="flow-item"><span class="flow-index">2</span><span>策略选择向左或向右推动小车，并收集交互轨迹。</span></div>
                  <div class="flow-item"><span class="flow-index">3</span><span>PPO 更新策略；每 2,000 步独立评估 5 个回合。</span></div>
                  <div class="flow-item"><span class="flow-index">4</span><span>训练结束后生成奖励曲线、策略动画与模型文件。</span></div>
                </div>
                """
            )
        with gr.Column(scale=2, elem_classes="result-card"):
            gr.HTML(
                """
                <h2 class="section-title">观察学习过程</h2>
                <p class="section-copy">曲线展示确定性策略的平均奖励；虚线 475 是 CartPole-v1 的常用解决阈值。</p>
                """
            )
            curve = gr.Plot(label="奖励曲线")
            animation = gr.Image(label="训练后策略演示", type="filepath")

    gr.HTML(
        f"""
        <section class="lesson-card">
          <h3>这是《动手学现代强化学习》的第 1 章配套实验</h3>
          <p>
            实验对应课程的 <a href="{CHAPTER_URL}" target="_blank" rel="noreferrer">1.3 PPO 训练可视化</a>。
            完整源码、后续章节与实验代码收录在
            <a href="{PROJECT_URL}" target="_blank" rel="noreferrer">walkinglabs/hands-on-modern-rl</a>。
            如果当前环境排队，也可以直接使用 <a href="{COLAB_URL}" target="_blank" rel="noreferrer">Colab Notebook</a>。
          </p>
        </section>
        <div class="footer-note">Hands-On Modern RL · WalkingLabs · 开源强化学习课程</div>
        """
    )

    start.click(
        fn=train,
        inputs=timesteps,
        outputs=[status, curve, animation, metrics],
        concurrency_limit=1,
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
