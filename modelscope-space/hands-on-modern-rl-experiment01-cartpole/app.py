"""ModelScope Studio: train PPO on CartPole from the browser."""

from __future__ import annotations

import base64
import contextlib
import io
import os
import re
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
LOGO_PATH = Path(__file__).parent / "assets" / "readmelogo.png"
LOGO_DATA_URI = f"data:image/png;base64,{base64.b64encode(LOGO_PATH.read_bytes()).decode()}"
SEED = 42
PROJECT_URL = "https://github.com/walkinglabs/hands-on-modern-rl"
COURSE_URL = "https://walkinglabs.github.io/hands-on-modern-rl/"
CHAPTER_URL = f"{COURSE_URL}chapter01_cartpole/training"
MODELSCOPE_NOTEBOOK_URL = "https://modelscope.cn/my/mynotebook"
SCRIPT_URL = (
    "https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment01-cartpole/"
    "file/view/master/train.py"
)
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


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


def clean_output(text: str) -> str:
    """Remove terminal control characters before showing library output."""
    return ANSI_ESCAPE.sub("", text).strip()


def log_line(started_at: float, level: str, message: str) -> str:
    """Format a compact console line with elapsed time."""
    elapsed = time.perf_counter() - started_at
    return f"{elapsed:7.1f}s  {level:<7} {message}"


def status_card(state: str, title: str, detail: str) -> str:
    """Render a compact run-status summary without nested borders."""
    return f"""
    <div class="run-state run-state--{state}">
      <span class="run-state__dot" aria-hidden="true"></span>
      <div class="run-state__body">
        <span class="summary-label">训练状态</span>
        <strong>{title}</strong>
        <small>{detail}</small>
      </div>
    </div>
    """


def metric_card(label: str, value: str, detail: str) -> str:
    """Render the latest evaluation result as a compact summary."""
    return f"""
    <div class="live-metric">
      <span class="summary-label">{label}</span>
      <div class="metric-reading"><strong>{value}</strong><small>{detail}</small></div>
    </div>
    """


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
    started_at = time.perf_counter()
    logs = [
        "CartPole PPO training console",
        "=" * 72,
        log_line(started_at, "CONFIG", "environment=CartPole-v1  algorithm=PPO  device=CPU"),
        log_line(started_at, "CONFIG", f"timesteps={total_timesteps}  seed={SEED}  eval_episodes=5"),
    ]
    env = gym.make("CartPole-v1")
    env.reset(seed=SEED)
    library_output = io.StringIO()
    with contextlib.redirect_stdout(library_output), contextlib.redirect_stderr(library_output):
        model = PPO(
            "MlpPolicy",
            env,
            seed=SEED,
            verbose=1,
            device="cpu",
            n_steps=1_024,
            batch_size=64,
            learning_rate=3e-4,
        )
    initialization_output = clean_output(library_output.getvalue())
    if initialization_output:
        logs.extend(["", "PPO initialization", initialization_output])

    steps: list[int] = [0]
    mean_rewards: list[float] = []
    initial_mean, initial_std = evaluate(model)
    mean_rewards.append(initial_mean)
    logs.extend(
        [
            "",
            log_line(
                started_at,
                "EVAL",
                f"step=0  mean_reward={initial_mean:.1f}  std={initial_std:.1f}",
            ),
            log_line(started_at, "TRAIN", "collecting the first rollout"),
        ]
    )

    yield (
        status_card("running", "训练进行中", f"0 / {total_timesteps:,} 步"),
        metric_card("平均奖励", f"{initial_mean:.1f}", f"标准差 {initial_std:.1f}"),
        reward_figure(steps, mean_rewards),
        None,
        None,
        "\n".join(logs),
    )

    trained = 0
    while trained < total_timesteps:
        current_chunk = min(chunk_size, total_timesteps - trained)
        library_output = io.StringIO()
        with contextlib.redirect_stdout(library_output), contextlib.redirect_stderr(library_output):
            model.learn(
                total_timesteps=current_chunk,
                reset_num_timesteps=False,
                progress_bar=False,
            )
        trained += current_chunk
        ppo_output = clean_output(library_output.getvalue())
        if ppo_output:
            logs.extend(["", f"PPO update · step {trained:,}", ppo_output])
        mean_reward, std_reward = evaluate(model)
        steps.append(trained)
        mean_rewards.append(mean_reward)
        elapsed = time.perf_counter() - started_at
        logs.append(
            log_line(
                started_at,
                "EVAL",
                f"step={trained}  mean_reward={mean_reward:.1f}  std={std_reward:.1f}",
            )
        )
        yield (
            status_card(
                "running",
                "训练进行中",
                f"{trained:,} / {total_timesteps:,} 步 · {trained / total_timesteps:.0%} · {elapsed:.1f} 秒",
            ),
            metric_card("平均奖励", f"{mean_reward:.1f}", f"标准差 {std_reward:.1f}"),
            reward_figure(steps, mean_rewards),
            None,
            None,
            "\n".join(logs),
        )

    model_path = ARTIFACT_DIR / "ppo-cartpole"
    model.save(model_path)
    model_file = str(model_path.with_suffix(".zip"))
    logs.append(log_line(started_at, "SAVE", f"model={model_file}"))
    gif_path, demo_score = record_policy(model)
    logs.append(log_line(started_at, "RENDER", f"animation={gif_path}  episode_reward={demo_score:.0f}"))
    elapsed = time.perf_counter() - started_at
    final_mean, final_std = evaluate(model, episodes=10)
    env.close()
    logs.extend(
        [
            log_line(
                started_at,
                "FINAL",
                f"episodes=10  mean_reward={final_mean:.1f}  std={final_std:.1f}",
            ),
            log_line(started_at, "DONE", f"training completed in {elapsed:.1f}s"),
        ]
    )

    yield (
        status_card("complete", "训练完成", f"{total_timesteps:,} 步 · {elapsed:.1f} 秒"),
        metric_card("最终平均奖励", f"{final_mean:.1f}", f"10 回合 · 标准差 {final_std:.1f}"),
        reward_figure(steps, mean_rewards),
        gif_path,
        model_file,
        "\n".join(logs),
    )


CSS = """
:root {
  --ink: #172033;
  --muted: #68748a;
  --line: #e4e8f0;
  --paper: #ffffff;
  --canvas: #f4f6fa;
  --brand: #5b5ce2;
  --brand-dark: #4446be;
  --green: #13a36f;
}
.gradio-container {
  max-width: 1180px !important;
  margin: 0 auto !important;
  padding: 28px 22px 52px !important;
  background: var(--canvas);
}
.app-shell { font-family: Inter, "PingFang SC", "Microsoft YaHei", sans-serif; color: var(--ink); }
.hero {
  position: relative;
  overflow: hidden;
  padding: 38px 42px 34px;
  border: 1px solid rgba(129, 140, 248, .2);
  border-radius: 26px;
  color: #f8fafc;
  background:
    radial-gradient(circle at 88% 8%, rgba(125, 127, 255, .42), transparent 31%),
    radial-gradient(circle at 92% 92%, rgba(61, 207, 170, .18), transparent 30%),
    linear-gradient(132deg, #11182c 0%, #25265d 58%, #4546a4 100%);
  box-shadow: 0 22px 54px rgba(25, 32, 56, .16);
}
.hero::after {
  content: "";
  position: absolute;
  width: 290px;
  height: 290px;
  right: -104px;
  top: -136px;
  border: 1px solid rgba(255,255,255,.12);
  border-radius: 50%;
}
.project-mark {
  display: block;
  width: 290px;
  max-width: 72%;
  height: auto;
  margin: 0 0 22px;
  padding: 9px 13px;
  border-radius: 11px;
  background: #ffffff;
  box-shadow: 0 8px 24px rgba(8, 15, 35, .2);
}
.hero-topline { display: flex; align-items: center; gap: 11px; margin-bottom: 22px; }
.experiment-badge {
  display: inline-flex;
  align-items: center;
  padding: 6px 11px;
  border: 1px solid rgba(221, 224, 255, .3);
  border-radius: 999px;
  color: #eef0ff;
  background: rgba(255,255,255,.1);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: .06em;
}
.hero-course { color: #b9c0d4; font-size: 13px; font-weight: 650; }
.hero h1 {
  max-width: 760px;
  margin: 0 0 12px;
  color: #ffffff;
  font-size: clamp(32px, 5vw, 48px);
  line-height: 1.1;
  letter-spacing: -.035em;
}
.hero-copy {
  max-width: 700px;
  margin: 0;
  color: #cdd3e2;
  font-size: 15px;
  line-height: 1.7;
}
.hero-links { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 25px; }
.hero-link {
  display: inline-flex;
  align-items: center;
  min-height: 38px;
  padding: 0 14px;
  border: 1px solid rgba(255,255,255,.18);
  border-radius: 9px;
  color: #eef2ff !important;
  background: rgba(255,255,255,.08);
  font-size: 13px;
  font-weight: 650;
  text-decoration: none !important;
  transition: transform .16s ease, background .16s ease;
}
.hero-link:hover { transform: translateY(-1px); background: rgba(255,255,255,.15); }
.hero-link.primary { color: #172554 !important; background: #ffffff; border-color: #ffffff; }
.lab-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 22px;
  margin: 17px 0 22px;
  padding: 13px 18px;
  border: 1px solid var(--line);
  border-radius: 13px;
  background: rgba(255,255,255,.84);
  color: var(--muted);
  font-size: 13px;
  box-shadow: 0 6px 20px rgba(18, 25, 43, .035);
}
.lab-strip strong { margin-left: 5px; color: var(--ink); font-weight: 750; }
.panel-title { margin: 0 0 5px; color: var(--ink); font-size: 19px; font-weight: 780; letter-spacing: -.015em; }
.panel-copy { margin: 0 0 17px; color: var(--muted); font-size: 13px; line-height: 1.6; }
.control-card, .chart-card, .output-card, .console-card {
  border: 1px solid var(--line) !important;
  border-radius: 17px !important;
  background: #ffffff !important;
  box-shadow: 0 9px 26px rgba(20, 28, 48, .05) !important;
}
.control-card { padding: 22px !important; }
.chart-card { padding: 18px 18px 8px !important; }
.output-card { margin-top: 14px !important; padding: 18px !important; }
.console-card { margin-top: 14px !important; padding: 0 !important; overflow: hidden; }
.primary-btn {
  min-height: 48px !important;
  border: 0 !important;
  border-radius: 11px !important;
  background: linear-gradient(135deg, var(--brand-dark), #6969ec) !important;
  box-shadow: 0 8px 20px rgba(76, 77, 202, .22) !important;
  font-size: 15px !important;
  font-weight: 750 !important;
}
.status-output, .metric-output {
  min-width: 0 !important;
  margin: 10px 0 0 !important;
  padding: 0 !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
.status-output > div, .metric-output > div {
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
.run-state, .live-metric {
  box-sizing: border-box;
  min-height: 88px;
  padding: 15px 16px;
  border: 0;
  border-radius: 13px;
  background: #f5f6fa;
}
.run-state { display: flex; align-items: center; gap: 13px; }
.run-state__dot {
  flex: 0 0 auto;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #98a2b3;
}
.run-state--running .run-state__dot { background: var(--brand); }
.run-state--complete .run-state__dot { background: var(--green); }
.run-state__body { min-width: 0; }
.summary-label {
  display: block;
  margin-bottom: 4px;
  color: #8993a5;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: .055em;
}
.run-state strong { display: block; color: var(--ink); font-size: 15px; line-height: 1.35; }
.run-state small, .live-metric small { display: block; margin-top: 4px; color: var(--muted); font-size: 12px; line-height: 1.45; }
.live-metric { display: flex; flex-direction: column; justify-content: center; }
.metric-reading { display: flex; align-items: baseline; gap: 9px; }
.live-metric strong { color: var(--ink); font-size: 23px; line-height: 1; letter-spacing: -.025em; }
.live-metric small { margin: 0; }
.console-head {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 13px 16px;
  border-bottom: 1px solid #263044;
  color: #d9e0eb;
  background: #151c2b;
  font-size: 13px;
  font-weight: 700;
}
.console-dot { width: 8px; height: 8px; border-radius: 50%; background: #24c689; box-shadow: 0 0 0 4px rgba(36,198,137,.12); }
.training-console, .training-console > div { border: 0 !important; border-radius: 0 !important; background: #0f1623 !important; }
.training-console textarea {
  min-height: 350px !important;
  padding: 17px 18px !important;
  border: 0 !important;
  color: #cbd5e1 !important;
  background: #0f1623 !important;
  font: 12px/1.58 "SFMono-Regular", Consolas, "Liberation Mono", monospace !important;
  resize: vertical !important;
}
.artifact-note { margin: 0 0 12px; color: var(--muted); font-size: 13px; line-height: 1.55; }
.footer-note a { color: var(--brand) !important; font-weight: 650; text-decoration: none !important; }
.footer-note { margin-top: 18px; text-align: center; color: #94a3b8; font-size: 12px; }
@media (max-width: 760px) {
  .gradio-container { padding: 12px 10px 30px !important; }
  .hero { padding: 27px 22px 25px; border-radius: 19px; }
  .hero-topline { align-items: flex-start; flex-direction: column; gap: 8px; }
  .lab-strip { gap: 8px 16px; }
}
"""


with gr.Blocks(title="实验 01 · CartPole 在线训练") as demo:
    gr.HTML(
        f"""
        <main class="app-shell">
          <section class="hero">
            <img class="project-mark" src="{LOGO_DATA_URI}" alt="Hands-On Modern RL" />
            <div class="hero-topline">
              <span class="experiment-badge">EXPERIMENT 01</span>
              <span class="hero-course">《动手学现代强化学习》· 第 1 章配套</span>
            </div>
            <h1>CartPole 在线训练实验</h1>
            <p class="hero-copy">
              使用 PPO 从零训练倒立摆策略。启动后可以实时查看奖励曲线、PPO 输出和评估记录，
              训练结束后下载模型并播放策略动画。全程使用 CPU。
            </p>
            <nav class="hero-links" aria-label="项目入口">
              <a class="hero-link primary" href="{CHAPTER_URL}" target="_blank" rel="noreferrer">阅读配套章节</a>
              <a class="hero-link" href="{MODELSCOPE_NOTEBOOK_URL}" target="_blank" rel="noreferrer">魔搭 Notebook</a>
              <a class="hero-link" href="{SCRIPT_URL}" target="_blank" rel="noreferrer">训练脚本</a>
              <a class="hero-link" href="{PROJECT_URL}" target="_blank" rel="noreferrer">GitHub 项目</a>
            </nav>
          </section>
          <section class="lab-strip" aria-label="实验配置">
            <span>环境 <strong>CartPole-v1</strong></span>
            <span>算法 <strong>PPO</strong></span>
            <span>设备 <strong>CPU</strong></span>
            <span>解决阈值 <strong>475</strong></span>
            <span>满分 <strong>500</strong></span>
          </section>
        </main>
        """
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=300, elem_classes="control-card"):
            gr.HTML(
                """
                <h2 class="panel-title">训练设置</h2>
                <p class="panel-copy">选择总交互步数。系统每训练 2,000 步评估一次策略。</p>
                """
            )
            timesteps = gr.Slider(
                minimum=10_000,
                maximum=50_000,
                value=30_000,
                step=5_000,
                label="训练步数",
                info="建议首次使用 30,000 步",
            )
            start = gr.Button("开始训练", variant="primary", elem_classes="primary-btn")
            status = gr.HTML(
                status_card("idle", "等待开始", "设置训练步数后启动实验"),
                elem_classes="status-output",
            )
            metrics = gr.HTML(
                metric_card("平均奖励", "—", "训练开始后显示评估结果"),
                elem_classes="metric-output",
            )
        with gr.Column(scale=2, elem_classes="chart-card"):
            gr.HTML(
                """
                <h2 class="panel-title">奖励曲线</h2>
                <p class="panel-copy">纵轴为确定性策略的平均奖励，绿色虚线表示 475 分解决阈值。</p>
                """
            )
            curve = gr.Plot(show_label=False)

    with gr.Group(elem_classes="console-card"):
        gr.HTML('<div class="console-head"><span class="console-dot"></span>实时训练日志</div>')
        console = gr.Textbox(
            value="等待训练任务...",
            lines=18,
            max_lines=28,
            interactive=False,
            show_label=False,
            elem_classes="training-console",
        )

    with gr.Row(elem_classes="output-card"):
        with gr.Column(scale=2):
            gr.HTML(
                """
                <h2 class="panel-title">训练结果</h2>
                <p class="artifact-note">任务完成后，这里会显示策略动画并提供模型文件。</p>
                """
            )
            animation = gr.Image(label="策略动画", type="filepath")
        with gr.Column(scale=1):
            model_download = gr.File(label="下载 PPO 模型", interactive=False)

    gr.HTML(
        f'<div class="footer-note">实验 01 · <a href="{COURSE_URL}" target="_blank" rel="noreferrer">Hands-On Modern RL</a> · WalkingLabs</div>'
    )

    start.click(
        fn=train,
        inputs=timesteps,
        outputs=[status, metrics, curve, animation, model_download, console],
        concurrency_limit=1,
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(css=CSS)
