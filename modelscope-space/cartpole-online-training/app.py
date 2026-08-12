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
    ax.axhline(475, color="#16a34a", linestyle="--", linewidth=1.2, label="解决阈值 475")
    ax.set(xlabel="训练步数", ylabel="平均奖励", ylim=(0, 510))
    ax.set_title("PPO 在 CartPole-v1 上的评估奖励")
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
.gradio-container { max-width: 1180px !important; }
.hero { padding: 22px 24px; border-radius: 18px; background: linear-gradient(135deg, #eef2ff, #f8fafc); }
.hero h1 { margin: 0 0 8px; font-size: 2rem; }
.hero p { color: #475569; margin-bottom: 0; }
.primary-btn { min-height: 48px; font-size: 16px; }
"""


with gr.Blocks(css=CSS, title="从 CartPole 开始 · 在线训练") as demo:
    gr.HTML(
        """
        <section class="hero">
          <h1>从 CartPole 开始</h1>
          <p>在浏览器中用 CPU 训练一个 PPO 智能体。观察奖励从随机策略的约 20 分逐步上升，最高为 500 分。</p>
        </section>
        """
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            timesteps = gr.Slider(
                minimum=10_000,
                maximum=50_000,
                value=30_000,
                step=5_000,
                label="训练步数",
                info="默认设置通常可在普通 CPU 上较快完成。",
            )
            start = gr.Button("开始训练", variant="primary", elem_classes="primary-btn")
            status = gr.Markdown("点击按钮后，页面会持续更新训练进度。")
            metrics = gr.Markdown("尚未开始评估。")
            gr.Markdown(
                """
                **你会看到什么**

                - 智能体每一步选择向左或向右推动小车；
                - PPO 根据采样到的轨迹更新策略；
                - 每 2,000 步独立评估 5 个回合；
                - 训练结束后生成策略动画和可下载模型。
                """
            )
        with gr.Column(scale=2):
            curve = gr.Plot(label="奖励曲线")
            animation = gr.Image(label="训练后策略演示", type="filepath")

    start.click(
        fn=train,
        inputs=timesteps,
        outputs=[status, curve, animation, metrics],
        concurrency_limit=1,
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
