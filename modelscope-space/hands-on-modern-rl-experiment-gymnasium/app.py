"""A browser-based collection of small CPU Gymnasium training experiments."""

from __future__ import annotations

import base64
from collections import defaultdict
import html
import importlib
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gradio as gr
import gymnasium as gym
import imageio.v2 as imageio
import matplotlib
import numpy as np
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).parent
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
LOGO_PATH = ROOT / "assets" / "readmelogo.png"
LOGO_DATA_URI = f"data:image/png;base64,{base64.b64encode(LOGO_PATH.read_bytes()).decode()}"

PROJECT_URL = "https://github.com/walkinglabs/hands-on-modern-rl"
COURSE_URL = "https://walkinglabs.github.io/hands-on-modern-rl/"
BANDIT = "Bandit · ε-greedy"
BLACKJACK = "Blackjack · Monte Carlo"
GRIDWORLD = "GridWorld · Q-Learning"
FROZENLAKE = "FrozenLake · Q-Learning"
CLIFF = "CliffWalking · SARSA"
TAXI = "Taxi · Q-Learning"
CARTPOLE_DQN = "CartPole · DQN"
CARTPOLE_PPO = "CartPole · PPO"
MOUNTAINCAR = "MountainCar · Tabular Q"
ACROBOT = "Acrobot · PPO"
PENDULUM = "Pendulum · PPO"
MOUNTAINCAR_CONTINUOUS = "MountainCarContinuous · SAC"

CHAPTER_URLS = {
    BANDIT: f"{COURSE_URL}chapter03_mdp/bandit",
    BLACKJACK: f"{COURSE_URL}chapter04_tabular",
    GRIDWORLD: f"{COURSE_URL}chapter03_mdp/value-experiment",
    FROZENLAKE: f"{COURSE_URL}chapter04_tabular",
    CLIFF: f"{COURSE_URL}chapter04_tabular",
    TAXI: f"{COURSE_URL}chapter04_tabular",
    CARTPOLE_DQN: f"{COURSE_URL}chapter07_dqn/from-q-to-dqn",
    CARTPOLE_PPO: f"{COURSE_URL}chapter09_actor_critic",
    MOUNTAINCAR: f"{COURSE_URL}chapter07_dqn/from-q-to-dqn",
    ACROBOT: f"{COURSE_URL}chapter09_actor_critic",
    PENDULUM: f"{COURSE_URL}chapter09_actor_critic/pendulum",
    MOUNTAINCAR_CONTINUOUS: f"{COURSE_URL}chapter11_continuous_control",
}
SCRIPT_URL = (
    "https://modelscope.cn/studios/walkinglab/hands-on-modern-rl-experiment-gymnasium/"
    "file/view/master/app.py"
)


EXPERIMENTS = {
    BANDIT: {
        "environment": "4-armed Bernoulli bandit",
        "family": "Bandit",
        "algorithm": "ε-greedy",
        "budget": (200, 10000, 2000, 200),
        "alpha": (0.01, 1.0, 0.1, 0.01),
        "gamma": (0.0, 1.0, 0.0, 0.05),
        "epsilon": (0.0, 1.0, 0.1, 0.01),
        "gamma_visible": False,
    },
    BLACKJACK: {
        "environment": "Blackjack-v1",
        "family": "Toy Text",
        "algorithm": "First-visit Monte Carlo",
        "budget": (5000, 100000, 30000, 5000),
        "alpha": (0.001, 0.2, 0.02, 0.001),
        "gamma": (0.0, 1.0, 1.0, 0.01),
        "epsilon": (0.0, 1.0, 0.2, 0.01),
        "gamma_visible": True,
    },
    GRIDWORLD: {
        "environment": "Custom 4×4 GridWorld",
        "family": "Tabular",
        "algorithm": "Q-Learning",
        "budget": (100, 5000, 1000, 100),
        "alpha": (0.01, 1.0, 0.15, 0.01),
        "gamma": (0.0, 1.0, 0.95, 0.01),
        "epsilon": (0.0, 1.0, 0.2, 0.01),
        "gamma_visible": True,
    },
    FROZENLAKE: {
        "environment": "FrozenLake-v1",
        "family": "Toy Text",
        "algorithm": "Q-Learning",
        "budget": (1000, 30000, 10000, 1000),
        "alpha": (0.01, 1.0, 0.2, 0.01),
        "gamma": (0.0, 1.0, 0.99, 0.01),
        "epsilon": (0.0, 1.0, 1.0, 0.01),
        "gamma_visible": True,
    },
    CLIFF: {
        "environment": "CliffWalking-v1",
        "family": "Toy Text",
        "algorithm": "SARSA",
        "budget": (200, 5000, 1500, 100),
        "alpha": (0.01, 1.0, 0.5, 0.01),
        "gamma": (0.0, 1.0, 1.0, 0.01),
        "epsilon": (0.0, 1.0, 0.1, 0.01),
        "gamma_visible": True,
    },
    TAXI: {
        "environment": "Taxi-v4",
        "family": "Toy Text",
        "algorithm": "Q-Learning",
        "budget": (1000, 30000, 8000, 1000),
        "alpha": (0.01, 1.0, 0.2, 0.01),
        "gamma": (0.0, 1.0, 0.95, 0.01),
        "epsilon": (0.0, 1.0, 1.0, 0.01),
        "gamma_visible": True,
    },
    CARTPOLE_DQN: {
        "environment": "CartPole-v1",
        "family": "Classic Control",
        "algorithm": "DQN",
        "budget": (5000, 100000, 30000, 5000),
        "alpha": (0.00001, 0.001, 0.0001, 0.00001),
        "gamma": (0.8, 1.0, 0.99, 0.01),
        "epsilon": (0.0, 1.0, 1.0, 0.05),
        "gamma_visible": True,
    },
    CARTPOLE_PPO: {
        "environment": "CartPole-v1",
        "family": "Classic Control",
        "algorithm": "PPO",
        "budget": (5000, 100000, 30000, 5000),
        "alpha": (0.00001, 0.003, 0.0003, 0.00001),
        "gamma": (0.8, 1.0, 0.99, 0.01),
        "epsilon": (0.0, 0.05, 0.0, 0.005),
        "gamma_visible": True,
    },
    MOUNTAINCAR: {
        "environment": "MountainCar-v0",
        "family": "Classic Control",
        "algorithm": "Tabular Q-Learning",
        "budget": (1000, 20000, 6000, 1000),
        "alpha": (0.01, 1.0, 0.12, 0.01),
        "gamma": (0.0, 1.0, 0.99, 0.01),
        "epsilon": (0.0, 1.0, 1.0, 0.01),
        "gamma_visible": True,
    },
    ACROBOT: {
        "environment": "Acrobot-v1",
        "family": "Classic Control",
        "algorithm": "PPO",
        "budget": (5000, 100000, 30000, 5000),
        "alpha": (0.00001, 0.003, 0.0003, 0.00001),
        "gamma": (0.8, 1.0, 0.99, 0.01),
        "epsilon": (0.0, 0.05, 0.0, 0.005),
        "gamma_visible": True,
    },
    PENDULUM: {
        "environment": "Pendulum-v1",
        "family": "Classic Control",
        "algorithm": "PPO",
        "budget": (5000, 100000, 30000, 5000),
        "alpha": (0.0001, 0.003, 0.0003, 0.0001),
        "gamma": (0.8, 1.0, 0.99, 0.01),
        "epsilon": (0.0, 0.05, 0.0, 0.005),
        "gamma_visible": True,
    },
    MOUNTAINCAR_CONTINUOUS: {
        "environment": "MountainCarContinuous-v0",
        "family": "Classic Control",
        "algorithm": "SAC",
        "budget": (5000, 100000, 30000, 5000),
        "alpha": (0.00001, 0.003, 0.0003, 0.00001),
        "gamma": (0.8, 1.0, 0.99, 0.01),
        "epsilon": (0.0, 0.05, 0.0, 0.005),
        "gamma_visible": True,
    },
}


def load_optional_registries() -> list[str]:
    """Register optional suites when their packages are installed."""
    loaded = []
    for module_name in ("ale_py", "gymnasium_robotics"):
        try:
            module = importlib.import_module(module_name)
            if hasattr(gym, "register_envs"):
                gym.register_envs(module)
            loaded.append(module_name)
        except Exception:
            continue
    return loaded


OPTIONAL_REGISTRIES = load_optional_registries()
INTERNAL_ENV_PREFIXES = ("GymV21Environment", "GymV26Environment")


def env_family(env_id: str, entry_point) -> str:
    text = f"{env_id} {entry_point}".lower()
    if env_id.startswith("ALE/"):
        return "Atari / ALE"
    if "robotics" in text or any(name in env_id for name in ("Fetch", "Adroit", "Hand", "Franka")):
        return "Robotics"
    if "mujoco" in text:
        return "MuJoCo"
    if "box2d" in text:
        return "Box2D"
    if "toy_text" in text:
        return "Toy Text"
    if "classic_control" in text:
        return "Classic Control"
    if env_id.startswith("phys2d/"):
        return "JAX Phys2D"
    if env_id.startswith("tabular/"):
        return "JAX Tabular"
    return "Other"


def discover_environment_catalog() -> list[str]:
    choices = []
    tuned_envs = {cfg["environment"] for cfg in EXPERIMENTS.values()}
    for spec in sorted(gym.registry.values(), key=lambda item: item.id.lower()):
        env_id = spec.id
        if env_id.startswith(INTERNAL_ENV_PREFIXES):
            continue
        if env_id in tuned_envs:
            continue
        choices.append(f"{env_family(env_id, spec.entry_point)} · {env_id} · Auto")
    return choices


CATALOG_EXPERIMENTS = discover_environment_catalog()
EXPERIMENT_CHOICES = list(EXPERIMENTS) + CATALOG_EXPERIMENTS


def is_catalog_experiment(experiment: str) -> bool:
    return experiment not in EXPERIMENTS


def catalog_env_id(experiment: str) -> str:
    return experiment.split(" · ", 2)[1]


def catalog_config(experiment: str) -> dict:
    env_id = catalog_env_id(experiment)
    return {
        "environment": env_id,
        "family": experiment.split(" · ", 1)[0],
        "algorithm": "Auto: inspect action space",
        "budget": (200, 100000, 10000, 1000),
        "alpha": (0.00001, 0.01, 0.0003, 0.00001),
        "gamma": (0.8, 1.0, 0.99, 0.01),
        "epsilon": (0.0, 1.0, 1.0, 0.05),
        "gamma_visible": True,
    }


def experiment_config(experiment: str) -> dict:
    return catalog_config(experiment) if is_catalog_experiment(experiment) else EXPERIMENTS[experiment]


TEXT = {
    "English": {
        "course": "Hands-On Modern RL · CPU experiment collection",
        "title": "Gymnasium Training Playground",
        "description": "Browse every environment registered by Gymnasium and its installed suites. Twelve curated recipes remain ready for quick CPU training.",
        "chapter": "Companion chapter",
        "script": "Training source",
        "project": "GitHub project",
        "device": "Device",
        "experiments": "Experiments",
        "settings": "Experiment setup",
        "settings_copy": "Search the full registry or choose a curated recipe. Auto entries inspect the action space and select DQN, PPO, or SAC.",
        "experiment": "Experiment",
        "budget": "Training budget",
        "budget_info": "Episodes for tabular tasks; environment steps for DQN, PPO, and SAC",
        "alpha": "Learning rate",
        "gamma": "Discount factor γ",
        "epsilon": "Exploration ε",
        "seed": "Random seed",
        "start": "Start training",
        "ready": "Ready to train",
        "ready_detail": "Choose an experiment and start a CPU run",
        "running": "Training in progress",
        "complete": "Training complete",
        "status": "Run status",
        "metric": "Latest evaluation",
        "metric_waiting": "Results appear after training starts",
        "curve": "Learning curve",
        "curve_copy": "The chart updates at each checkpoint. All labels stay in English for readability.",
        "log": "Live training log",
        "log_waiting": "Waiting for a training run...",
        "preview": "Learned policy preview",
        "preview_copy": "Tabular tasks show the learned policy; control tasks produce a replay after training.",
        "artifact": "Download run summary",
        "seconds": "s",
    },
    "中文": {
        "course": "《动手学现代强化学习》· CPU 实验合集",
        "title": "Gymnasium 在线训练游乐场",
        "description": "浏览 Gymnasium 及已安装扩展套件注册的全部环境，同时保留 12 个可快速训练的调优配方。",
        "chapter": "阅读配套章节",
        "script": "训练源码",
        "project": "GitHub 项目",
        "device": "设备",
        "experiments": "实验数量",
        "settings": "实验设置",
        "settings_copy": "可搜索完整环境目录或选择调优配方。Auto 项会检查动作空间并自动选择 DQN、PPO 或 SAC。",
        "experiment": "实验",
        "budget": "训练预算",
        "budget_info": "表格任务使用回合数；DQN、PPO 与 SAC 使用环境步数",
        "alpha": "学习率",
        "gamma": "折扣因子 γ",
        "epsilon": "探索率 ε",
        "seed": "随机种子",
        "start": "开始训练",
        "ready": "等待训练",
        "ready_detail": "选择一个实验并启动 CPU 训练",
        "running": "训练进行中",
        "complete": "训练完成",
        "status": "训练状态",
        "metric": "最新评估",
        "metric_waiting": "训练开始后显示结果",
        "curve": "学习曲线",
        "curve_copy": "每个检查点更新一次曲线。图表标记统一保留英文。",
        "log": "实时训练日志",
        "log_waiting": "等待训练任务...",
        "preview": "训练策略预览",
        "preview_copy": "表格任务显示学习后的策略；控制任务训练结束后生成回放。",
        "artifact": "下载运行摘要",
        "seconds": "秒",
    },
}


def copy_for(language: str) -> dict[str, str]:
    return TEXT["English" if language == "English" else "中文"]


def elapsed_line(started: float, level: str, message: str) -> str:
    return f"{time.perf_counter() - started:7.1f}s  {level:<7} {message}"


def console_panel(logs: str, language: str) -> str:
    return f"""
    <section class="console-panel" aria-live="polite" aria-atomic="true">
      <div class="console-head"><span class="console-dot"></span>{copy_for(language)['log']}</div>
      <pre class="console-text">{html.escape(logs)}</pre>
    </section>
    """


def status_card(state: str, title: str, detail: str, language: str) -> str:
    return f"""
    <div class="run-state run-state--{state}">
      <span class="run-state__dot"></span>
      <div><span class="summary-label">{copy_for(language)['status']}</span><strong>{title}</strong><small>{detail}</small></div>
    </div>
    """


def metric_card(value: str, detail: str, language: str) -> str:
    return f"""
    <div class="live-metric">
      <span class="summary-label">{copy_for(language)['metric']}</span>
      <div class="metric-reading"><strong>{value}</strong><small>{detail}</small></div>
    </div>
    """


def panel_html(title: str, text: str, cls: str = "panel-copy") -> str:
    return f'<h2 class="panel-title">{title}</h2><p class="{cls}">{text}</p>'


def hero_html(language: str, experiment: str = BANDIT) -> str:
    copy = copy_for(language)
    cfg = experiment_config(experiment)
    chapter_url = CHAPTER_URLS.get(experiment, COURSE_URL)
    return f"""
    <main class="app-shell">
      <section class="hero">
        <img class="project-mark" src="{LOGO_DATA_URI}" alt="Hands-On Modern RL" />
        <div class="hero-topline"><span class="experiment-badge">CPU PLAYGROUND</span><span class="hero-course">{copy['course']}</span></div>
        <h1>{copy['title']}</h1>
        <p class="hero-copy">{copy['description']}</p>
        <nav class="hero-links">
          <a class="hero-link primary" href="{chapter_url}" target="_blank" rel="noreferrer">{copy['chapter']}</a>
          <a class="hero-link" href="{SCRIPT_URL}" target="_blank" rel="noreferrer">{copy['script']}</a>
          <a class="hero-link" href="{PROJECT_URL}" target="_blank" rel="noreferrer">{copy['project']}</a>
        </nav>
      </section>
      <section class="lab-strip">
        <span>{copy['experiments']} <strong>{len(EXPERIMENT_CHOICES)}</strong></span>
        <span>{copy['device']} <strong>CPU</strong></span>
        <span>Environment <strong>{cfg['environment']}</strong></span>
        <span>Algorithm <strong>{cfg['algorithm']}</strong></span>
      </section>
    </main>
    """


def footer_html() -> str:
    return f'<div class="footer-note">Gymnasium CPU Playground · <a href="{COURSE_URL}" target="_blank">Hands-On Modern RL</a> · WalkingLabs</div>'


def learning_figure(x: list[float], y: list[float], title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.plot(x, y, color="#5b5ce2", linewidth=2.2)
    if x:
        ax.scatter([x[-1]], [y[-1]], color="#15a873", s=34, zorder=3)
    ax.set_title(title)
    ax.set_xlabel("Training progress")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def policy_grid_image(
    grid: list[str], policy: dict[tuple[int, int], str], title: str, filename: str,
    values: dict[tuple[int, int], float] | None = None,
):
    rows, cols = len(grid), len(grid[0])
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_aspect("equal")
    colors = {"S": "#dbeafe", "G": "#bbf7d0", "H": "#fecaca", "T": "#fecaca", "F": "#f8fafc", ".": "#f8fafc"}
    for row in range(rows):
        for col in range(cols):
            cell = grid[row][col]
            rect = plt.Rectangle((col, row), 1, 1, facecolor=colors.get(cell, "#f8fafc"), edgecolor="#cbd5e1", linewidth=1.5)
            ax.add_patch(rect)
            label = {"S": "START", "G": "GOAL", "H": "HOLE", "T": "TRAP"}.get(cell, policy.get((row, col), "·"))
            ax.text(col + 0.5, row + 0.46, label, ha="center", va="center", fontsize=12, fontweight="bold", color="#27324a")
            if values and (row, col) in values:
                ax.text(col + 0.5, row + 0.78, f"{values[(row, col)]:.2f}", ha="center", fontsize=8, color="#64748b")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    path = ARTIFACT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def save_summary(experiment: str, payload: dict) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", experiment.lower()).strip("-")
    path = ARTIFACT_DIR / f"{slug}-run-summary.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def run_bandit(budget: int, alpha: float, epsilon: float, seed: int, language: str):
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    probabilities = np.array([0.35, 0.50, 0.72, 0.58])
    q = np.zeros(4)
    counts = np.zeros(4, dtype=int)
    rewards: list[float] = []
    logs = ["Multi-Armed Bandit training console", "=" * 72, elapsed_line(started, "CONFIG", f"arms=4  probabilities={probabilities.tolist()}"), elapsed_line(started, "CONFIG", f"steps={budget}  alpha={alpha:g}  epsilon={epsilon:g}  seed={seed}")]
    chunk = max(20, budget // 20)
    for step in range(1, budget + 1):
        action = int(rng.integers(4)) if rng.random() < epsilon else int(np.argmax(q))
        reward = float(rng.random() < probabilities[action])
        counts[action] += 1
        q[action] += alpha * (reward - q[action])
        rewards.append(reward)
        if step % chunk == 0 or step == budget:
            avg = float(np.mean(rewards))
            logs.append(elapsed_line(started, "TRAIN", f"step={step}/{budget}  avg_reward={avg:.3f}  best_estimate=arm-{int(np.argmax(q)) + 1}"))
            yield status_card("running", copy_for(language)["running"], f"{step:,}/{budget:,} steps", language), metric_card(f"{avg:.3f}", f"estimated best arm: {int(np.argmax(q)) + 1}", language), learning_figure(list(range(1, step + 1)), (np.cumsum(rewards) / np.arange(1, step + 1)).tolist(), "Bandit cumulative average reward", "Average reward"), None, None, console_panel("\n".join(logs), language)
    fig, ax = plt.subplots(figsize=(6, 4))
    positions = np.arange(1, 5)
    ax.bar(positions - 0.16, probabilities, 0.32, label="True probability", color="#93c5fd")
    ax.bar(positions + 0.16, q, 0.32, label="Learned estimate", color="#5b5ce2")
    ax.set(xticks=positions, xlabel="Arm", ylabel="Reward probability", ylim=(0, 1), title="True vs learned arm values")
    ax.legend(); ax.grid(axis="y", alpha=0.2); fig.tight_layout()
    preview = ARTIFACT_DIR / "bandit-arm-estimates.png"
    fig.savefig(preview, dpi=150, bbox_inches="tight")
    plt.close(fig)
    summary = save_summary("bandit", {"experiment": "Bandit", "q_values": q.tolist(), "counts": counts.tolist(), "average_reward": float(np.mean(rewards)), "parameters": {"budget": budget, "alpha": alpha, "epsilon": epsilon, "seed": seed}})
    logs.append(elapsed_line(started, "DONE", f"best_arm={int(np.argmax(q)) + 1}  artifact={summary}"))
    yield status_card("complete", copy_for(language)["complete"], f"{budget:,} steps · {time.perf_counter() - started:.1f}s", language), metric_card(f"{np.mean(rewards):.3f}", f"best arm: {int(np.argmax(q)) + 1} · selected {counts[int(np.argmax(q))]} times", language), learning_figure(list(range(1, budget + 1)), (np.cumsum(rewards) / np.arange(1, budget + 1)).tolist(), "Bandit cumulative average reward", "Average reward"), str(preview), summary, console_panel("\n".join(logs), language)


GRID_ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ARROWS = ["↑", "↓", "←", "→"]


def grid_step(state: tuple[int, int], action: int):
    dr, dc = GRID_ACTIONS[action]
    nxt = (min(3, max(0, state[0] + dr)), min(3, max(0, state[1] + dc)))
    if nxt == (1, 1):
        return nxt, -1.0, True
    if nxt == (3, 3):
        return nxt, 1.0, True
    return nxt, -0.01, False


def run_gridworld(budget: int, alpha: float, gamma: float, epsilon: float, seed: int, language: str):
    started = time.perf_counter(); rng = np.random.default_rng(seed); q = np.zeros((4, 4, 4)); rewards = []
    logs = ["GridWorld Q-Learning console", "=" * 72, elapsed_line(started, "CONFIG", f"episodes={budget} alpha={alpha:g} gamma={gamma:g} epsilon={epsilon:g} seed={seed}")]
    chunk = max(10, budget // 20)
    for episode in range(1, budget + 1):
        state = (0, 0); total = 0.0
        for _ in range(100):
            action = int(rng.integers(4)) if rng.random() < epsilon else int(rng.choice(np.flatnonzero(q[state] == q[state].max())))
            nxt, reward, done = grid_step(state, action)
            q[state][action] += alpha * (reward + (0 if done else gamma * q[nxt].max()) - q[state][action])
            total += reward; state = nxt
            if done: break
        rewards.append(total)
        if episode % chunk == 0 or episode == budget:
            recent = float(np.mean(rewards[-min(50, len(rewards)):]))
            logs.append(elapsed_line(started, "TRAIN", f"episode={episode}/{budget} recent_reward={recent:.3f}"))
            yield status_card("running", copy_for(language)["running"], f"{episode:,}/{budget:,} episodes", language), metric_card(f"{recent:.3f}", "mean reward over recent episodes", language), learning_figure(list(range(1, episode + 1)), rewards, "GridWorld episode reward", "Episode reward"), None, None, console_panel("\n".join(logs), language)
    policy = {(r, c): ARROWS[int(np.argmax(q[r, c]))] for r in range(4) for c in range(4) if (r, c) not in {(1, 1), (3, 3)}}
    values = {(r, c): float(q[r, c].max()) for r in range(4) for c in range(4)}
    preview = policy_grid_image(["S...", ".T..", "....", "...G"], policy, "Learned GridWorld policy", "gridworld-policy.png", values)
    summary = save_summary("gridworld", {"experiment": "GridWorld", "q_values": q.tolist(), "policy": {f"{r},{c}": arrow for (r, c), arrow in policy.items()}, "parameters": {"budget": budget, "alpha": alpha, "gamma": gamma, "epsilon": epsilon, "seed": seed}})
    logs.append(elapsed_line(started, "DONE", f"artifact={summary}"))
    yield status_card("complete", copy_for(language)["complete"], f"{budget:,} episodes · {time.perf_counter() - started:.1f}s", language), metric_card(f"{np.mean(rewards[-50:]):.3f}", "final 50-episode mean reward", language), learning_figure(list(range(1, budget + 1)), rewards, "GridWorld episode reward", "Episode reward"), preview, summary, console_panel("\n".join(logs), language)


def run_frozenlake(budget: int, alpha: float, gamma: float, epsilon: float, seed: int, language: str):
    started = time.perf_counter(); rng = np.random.default_rng(seed); env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True); q = np.zeros((16, 4)); successes = []
    logs = ["FrozenLake Q-Learning console", "=" * 72, elapsed_line(started, "CONFIG", f"episodes={budget} slippery=true alpha={alpha:g} gamma={gamma:g} epsilon_start={epsilon:g}")]
    chunk = max(50, budget // 20)
    for episode in range(1, budget + 1):
        state, _ = env.reset(seed=seed + episode); done = False; won = 0.0
        current_eps = max(0.02, epsilon * (1 - episode / budget))
        while not done:
            action = int(rng.integers(4)) if rng.random() < current_eps else int(rng.choice(np.flatnonzero(q[state] == q[state].max())))
            nxt, reward, terminated, truncated, _ = env.step(action); done = terminated or truncated
            q[state, action] += alpha * (reward + (0 if done else gamma * q[nxt].max()) - q[state, action]); state = nxt; won = max(won, float(reward))
        successes.append(won)
        if episode % chunk == 0 or episode == budget:
            rate = float(np.mean(successes[-min(500, len(successes)):]))
            logs.append(elapsed_line(started, "TRAIN", f"episode={episode}/{budget} epsilon={current_eps:.3f} recent_success={rate:.1%}"))
            curve = (np.cumsum(successes) / np.arange(1, len(successes) + 1)).tolist()
            yield status_card("running", copy_for(language)["running"], f"{episode:,}/{budget:,} episodes", language), metric_card(f"{rate:.1%}", "recent success rate", language), learning_figure(list(range(1, episode + 1)), curve, "FrozenLake cumulative success rate", "Success rate"), None, None, console_panel("\n".join(logs), language)
    env.close(); desc = ["SFFF", "FHFH", "FFFH", "HFFG"]; policy = {(s // 4, s % 4): ARROWS[int(np.argmax(q[s]))] for s in range(16) if desc[s // 4][s % 4] not in "HG"}
    preview = policy_grid_image(desc, policy, "Learned policy on slippery FrozenLake", "frozenlake-policy.png")
    summary = save_summary("frozenlake", {"experiment": "FrozenLake", "q_values": q.tolist(), "success_rate": float(np.mean(successes[-500:])), "parameters": {"budget": budget, "alpha": alpha, "gamma": gamma, "epsilon": epsilon, "seed": seed}})
    logs.append(elapsed_line(started, "DONE", f"final_success={np.mean(successes[-500:]):.1%} artifact={summary}"))
    curve = (np.cumsum(successes) / np.arange(1, len(successes) + 1)).tolist()
    yield status_card("complete", copy_for(language)["complete"], f"{budget:,} episodes · {time.perf_counter() - started:.1f}s", language), metric_card(f"{np.mean(successes[-500:]):.1%}", "final 500-episode success rate", language), learning_figure(list(range(1, budget + 1)), curve, "FrozenLake cumulative success rate", "Success rate"), preview, summary, console_panel("\n".join(logs), language)


def blackjack_policy_image(q: dict, filename: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3), sharex=True, sharey=True)
    for usable_ace, ax in enumerate(axes):
        policy = np.zeros((10, 10))
        for player_sum in range(12, 22):
            for dealer_card in range(1, 11):
                policy[player_sum - 12, dealer_card - 1] = int(np.argmax(q[(player_sum, dealer_card, bool(usable_ace))]))
        image = ax.imshow(policy, origin="lower", cmap="coolwarm", vmin=0, vmax=1, aspect="auto")
        ax.set_title("Usable ace" if usable_ace else "No usable ace")
        ax.set_xlabel("Dealer showing")
        ax.set_xticks(range(10), range(1, 11))
        ax.set_yticks(range(10), range(12, 22))
    axes[0].set_ylabel("Player sum")
    colorbar = fig.colorbar(image, ax=axes, ticks=[0, 1], shrink=.82)
    colorbar.ax.set_yticklabels(["Stick", "Hit"])
    fig.suptitle("Blackjack Monte Carlo policy", fontweight="bold")
    path = ARTIFACT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def run_blackjack(budget: int, alpha: float, gamma: float, epsilon: float, seed: int, language: str):
    started = time.perf_counter(); rng = np.random.default_rng(seed); env = gym.make("Blackjack-v1", sab=True)
    q = defaultdict(lambda: np.zeros(2, dtype=float)); rewards = []
    logs = ["Blackjack first-visit Monte Carlo console", "=" * 72, elapsed_line(started, "CONFIG", f"episodes={budget} alpha={alpha:g} gamma={gamma:g} epsilon_start={epsilon:g}")]
    chunk = max(500, budget // 20)
    for episode in range(1, budget + 1):
        state, _ = env.reset(seed=seed + episode); trajectory = []; done = False
        current_eps = max(0.02, epsilon * (1 - episode / budget))
        while not done:
            action = int(rng.integers(2)) if rng.random() < current_eps else int(rng.choice(np.flatnonzero(q[state] == q[state].max())))
            nxt, reward, terminated, truncated, _ = env.step(action)
            trajectory.append((state, action, float(reward))); state = nxt; done = terminated or truncated
        rewards.append(float(reward)); returns = 0.0; visited = set()
        for old_state, action, reward_step in reversed(trajectory):
            returns = reward_step + gamma * returns
            key = (old_state, action)
            if key not in visited:
                q[old_state][action] += alpha * (returns - q[old_state][action]); visited.add(key)
        if episode % chunk == 0 or episode == budget:
            win_rate = float(np.mean(np.asarray(rewards[-min(5000, len(rewards)):]) > 0))
            logs.append(elapsed_line(started, "TRAIN", f"episode={episode}/{budget} epsilon={current_eps:.3f} recent_win_rate={win_rate:.1%}"))
            cumulative = (np.cumsum(rewards) / np.arange(1, len(rewards) + 1)).tolist()
            yield status_card("running", copy_for(language)["running"], f"{episode:,}/{budget:,} episodes", language), metric_card(f"{win_rate:.1%}", "recent win rate", language), learning_figure(list(range(1, episode + 1)), cumulative, "Blackjack cumulative mean return", "Mean return"), None, None, console_panel("\n".join(logs), language)
    env.close(); preview = blackjack_policy_image(q, "blackjack-policy.png")
    serialized_q = {str(state): values.tolist() for state, values in q.items()}
    summary = save_summary("blackjack", {"experiment": BLACKJACK, "q_values": serialized_q, "win_rate": float(np.mean(np.asarray(rewards[-5000:]) > 0)), "parameters": {"budget": budget, "alpha": alpha, "gamma": gamma, "epsilon": epsilon, "seed": seed}})
    logs.append(elapsed_line(started, "DONE", f"states={len(q)} artifact={summary}")); cumulative = (np.cumsum(rewards) / np.arange(1, len(rewards) + 1)).tolist()
    yield status_card("complete", copy_for(language)["complete"], f"{budget:,} episodes · {time.perf_counter() - started:.1f}s", language), metric_card(f"{np.mean(np.asarray(rewards[-5000:]) > 0):.1%}", "final 5,000-episode win rate", language), learning_figure(list(range(1, budget + 1)), cumulative, "Blackjack cumulative mean return", "Mean return"), preview, summary, console_panel("\n".join(logs), language)


def record_discrete_policy(env_id: str, q: np.ndarray, seed: int, filename: str, max_steps: int) -> str:
    env = gym.make(env_id, render_mode="rgb_array"); state, _ = env.reset(seed=seed); frames = []
    for step in range(max_steps):
        if step % 2 == 0:
            frames.append(env.render())
        state, _, terminated, truncated, _ = env.step(int(np.argmax(q[int(state)])))
        if terminated or truncated:
            frames.append(env.render()); break
    env.close(); path = ARTIFACT_DIR / filename; imageio.mimsave(path, frames, duration=1 / 15, loop=0); return str(path)


def run_discrete_control(experiment: str, env_id: str, method: str, budget: int, alpha: float, gamma: float, epsilon: float, seed: int, language: str):
    started = time.perf_counter(); rng = np.random.default_rng(seed); env = gym.make(env_id)
    q = np.zeros((env.observation_space.n, env.action_space.n)); rewards = []
    logs = [f"{experiment} training console", "=" * 72, elapsed_line(started, "CONFIG", f"method={method} episodes={budget} alpha={alpha:g} gamma={gamma:g} epsilon_start={epsilon:g}")]
    chunk = max(50, budget // 20); max_steps = 1000 if env_id.startswith("Cliff") else 200
    for episode in range(1, budget + 1):
        state, _ = env.reset(seed=seed + episode); current_eps = max(0.02, epsilon * (1 - episode / budget))
        action = int(rng.integers(env.action_space.n)) if rng.random() < current_eps else int(rng.choice(np.flatnonzero(q[state] == q[state].max())))
        total = 0.0
        for _ in range(max_steps):
            nxt, reward, terminated, truncated, _ = env.step(action); done = terminated or truncated
            nxt_action = int(rng.integers(env.action_space.n)) if rng.random() < current_eps else int(rng.choice(np.flatnonzero(q[nxt] == q[nxt].max())))
            target = reward if done else reward + gamma * (q[nxt, nxt_action] if method == "SARSA" else q[nxt].max())
            q[state, action] += alpha * (target - q[state, action]); total += float(reward); state, action = nxt, nxt_action
            if done: break
        rewards.append(total)
        if episode % chunk == 0 or episode == budget:
            recent = float(np.mean(rewards[-min(100, len(rewards)):]))
            logs.append(elapsed_line(started, "TRAIN", f"episode={episode}/{budget} epsilon={current_eps:.3f} recent_reward={recent:.1f}"))
            yield status_card("running", copy_for(language)["running"], f"{episode:,}/{budget:,} episodes", language), metric_card(f"{recent:.1f}", "recent mean episode reward", language), learning_figure(list(range(1, episode + 1)), rewards, f"{experiment} episode reward", "Episode reward"), None, None, console_panel("\n".join(logs), language)
    env.close(); slug = "cliffwalking" if env_id.startswith("Cliff") else "taxi"
    gif = record_discrete_policy(env_id, q, seed + 10000, f"{slug}-trained.gif", max_steps)
    summary = save_summary(slug, {"experiment": experiment, "q_values": q.tolist(), "parameters": {"budget": budget, "alpha": alpha, "gamma": gamma, "epsilon": epsilon, "seed": seed}})
    logs.append(elapsed_line(started, "DONE", f"replay={gif} artifact={summary}"))
    yield status_card("complete", copy_for(language)["complete"], f"{budget:,} episodes · {time.perf_counter() - started:.1f}s", language), metric_card(f"{np.mean(rewards[-100:]):.1f}", "final 100-episode mean reward", language), learning_figure(list(range(1, budget + 1)), rewards, f"{experiment} episode reward", "Episode reward"), gif, summary, console_panel("\n".join(logs), language)


def mountain_state(obs: np.ndarray, bins=(24, 20)) -> tuple[int, int]:
    low = np.array([-1.2, -0.07]); high = np.array([0.6, 0.07]); scaled = (np.asarray(obs) - low) / (high - low)
    indices = np.floor(scaled * np.array(bins)).astype(int)
    return tuple(np.clip(indices, 0, np.array(bins) - 1))


def record_tabular_control(env_id: str, policy, seed: int, filename: str, max_steps: int = 500) -> str:
    env = gym.make(env_id, render_mode="rgb_array"); obs, _ = env.reset(seed=seed); frames = []
    for _ in range(max_steps):
        frames.append(env.render()); action = policy(obs); obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated: break
    env.close(); path = ARTIFACT_DIR / filename; imageio.mimsave(path, frames, duration=1 / 30, loop=0); return str(path)


def run_mountaincar(budget: int, alpha: float, gamma: float, epsilon: float, seed: int, language: str):
    started = time.perf_counter(); rng = np.random.default_rng(seed); env = gym.make("MountainCar-v0"); q = np.zeros((24, 20, 3)); rewards = []
    logs = ["MountainCar tabular Q-Learning console", "=" * 72, elapsed_line(started, "CONFIG", f"episodes={budget} bins=24x20 alpha={alpha:g} gamma={gamma:g} epsilon_start={epsilon:g}")]
    chunk = max(25, budget // 20)
    for episode in range(1, budget + 1):
        obs, _ = env.reset(seed=seed + episode); state = mountain_state(obs); total = 0.0
        current_eps = max(0.02, epsilon * (1 - episode / budget))
        for _ in range(200):
            action = int(rng.integers(3)) if rng.random() < current_eps else int(np.argmax(q[state]))
            nxt_obs, reward, terminated, truncated, _ = env.step(action); nxt = mountain_state(nxt_obs)
            shaped = reward + 25.0 * max(0.0, float(nxt_obs[0] - obs[0])) + (100.0 if terminated else 0.0)
            q[state][action] += alpha * (shaped + (0 if terminated else gamma * q[nxt].max()) - q[state][action])
            total += reward; obs = nxt_obs; state = nxt
            if terminated or truncated: break
        rewards.append(total)
        if episode % chunk == 0 or episode == budget:
            recent = float(np.mean(rewards[-min(100, len(rewards)):]))
            logs.append(elapsed_line(started, "TRAIN", f"episode={episode}/{budget} epsilon={current_eps:.3f} recent_reward={recent:.1f}"))
            yield status_card("running", copy_for(language)["running"], f"{episode:,}/{budget:,} episodes", language), metric_card(f"{recent:.1f}", "recent mean episode reward", language), learning_figure(list(range(1, episode + 1)), rewards, "MountainCar episode reward", "Episode reward"), None, None, console_panel("\n".join(logs), language)
    env.close(); gif = record_tabular_control("MountainCar-v0", lambda obs: int(np.argmax(q[mountain_state(obs)])), seed + 10000, "mountaincar-trained.gif", 200)
    summary = save_summary("mountaincar", {"experiment": "MountainCar", "q_values": q.tolist(), "parameters": {"budget": budget, "alpha": alpha, "gamma": gamma, "epsilon": epsilon, "seed": seed}})
    logs.append(elapsed_line(started, "DONE", f"replay={gif} artifact={summary}"))
    yield status_card("complete", copy_for(language)["complete"], f"{budget:,} episodes · {time.perf_counter() - started:.1f}s", language), metric_card(f"{np.mean(rewards[-100:]):.1f}", "final 100-episode mean reward", language), learning_figure(list(range(1, budget + 1)), rewards, "MountainCar episode reward", "Episode reward"), gif, summary, console_panel("\n".join(logs), language)


def record_model(model, env_id: str, seed: int, filename: str, max_steps: int) -> str:
    env = gym.make(env_id, render_mode="rgb_array"); obs, _ = env.reset(seed=seed); frames = []
    for step in range(max_steps):
        if step % 2 == 0:
            frames.append(env.render())
        action, _ = model.predict(obs, deterministic=True); obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated: break
    env.close(); path = ARTIFACT_DIR / filename; imageio.mimsave(path, frames, duration=1 / 15, loop=0); return str(path)


def run_deep_control(experiment: str, env_id: str, algorithm: str, budget: int, alpha: float, gamma: float, epsilon: float, seed: int, language: str):
    started = time.perf_counter(); env = gym.make(env_id)
    if isinstance(env.observation_space, gym.spaces.Dict):
        policy = "MultiInputPolicy"
    elif isinstance(env.observation_space, gym.spaces.Box) and len(env.observation_space.shape) == 3:
        policy = "CnnPolicy"
    else:
        policy = "MlpPolicy"
    if algorithm == "DQN":
        model = DQN(policy, env, learning_rate=alpha, gamma=gamma, learning_starts=min(1000, max(100, budget // 10)), buffer_size=max(10000, budget), exploration_initial_eps=epsilon, exploration_final_eps=0.05, seed=seed, device="cpu", verbose=0)
    elif algorithm == "SAC":
        model = SAC(policy, env, learning_rate=alpha, gamma=gamma, learning_starts=min(1000, max(100, budget // 10)), buffer_size=max(10000, budget), batch_size=64, seed=seed, device="cpu", verbose=0)
    else:
        model = PPO(policy, env, learning_rate=alpha, gamma=gamma, n_steps=min(1024, max(128, budget)), batch_size=64, seed=seed, device="cpu", verbose=0)
    logs = [f"{experiment} training console", "=" * 72, elapsed_line(started, "CONFIG", f"environment={env_id} algorithm={algorithm} policy={policy} timesteps={budget} learning_rate={alpha:g} gamma={gamma:g} seed={seed} device=cpu")]
    xs, rewards = [], []; chunk = 5000; trained = 0
    while trained < budget:
        step = min(chunk, budget - trained); model.learn(total_timesteps=step, reset_num_timesteps=False, progress_bar=False); trained += step
        eval_env = gym.make(env_id); values, _ = evaluate_policy(model, eval_env, n_eval_episodes=3, deterministic=True, return_episode_rewards=True, warn=False); eval_env.close(); mean = float(np.mean(values)); xs.append(trained); rewards.append(mean)
        logs.append(elapsed_line(started, "EVAL", f"step={trained}/{budget} mean_reward={mean:.1f}"))
        yield status_card("running", copy_for(language)["running"], f"{trained:,}/{budget:,} steps", language), metric_card(f"{mean:.1f}", "3-episode evaluation reward", language), learning_figure(xs, rewards, f"{experiment} evaluation reward", "Mean reward"), None, None, console_panel("\n".join(logs), language)
    slug = re.sub(r"[^a-z0-9]+", "-", f"{env_id}-{algorithm}".lower()).strip("-"); model_path = ARTIFACT_DIR / slug; model.save(model_path); env.close()
    try:
        gif = record_model(model, env_id, seed + 10000, f"{slug}-trained.gif", 500 if env_id in {"CartPole-v1", "Acrobot-v1"} else 999)
    except Exception as exc:
        gif = None; logs.append(elapsed_line(started, "WARN", f"replay_unavailable={type(exc).__name__}: {exc}"))
    summary = save_summary(slug, {"experiment": experiment, "evaluation_steps": xs, "evaluation_rewards": rewards, "model": str(model_path.with_suffix('.zip')), "parameters": {"budget": budget, "learning_rate": alpha, "gamma": gamma, "epsilon": epsilon, "seed": seed}})
    logs.append(elapsed_line(started, "DONE", f"replay={gif} model={model_path}.zip artifact={summary}"))
    yield status_card("complete", copy_for(language)["complete"], f"{budget:,} steps · {time.perf_counter() - started:.1f}s", language), metric_card(f"{rewards[-1]:.1f}", "final evaluation reward", language), learning_figure(xs, rewards, f"{experiment} evaluation reward", "Mean reward"), gif, summary, console_panel("\n".join(logs), language)


def error_figure(title: str, message: str):
    fig, ax = plt.subplots(figsize=(8.2, 4.0)); ax.axis("off")
    ax.text(.5, .62, "Environment registered", ha="center", va="center", fontsize=20, fontweight="bold", color="#27324a")
    ax.text(.5, .43, title, ha="center", va="center", fontsize=13, color="#5b5ce2")
    ax.text(.5, .25, message[:180], ha="center", va="center", fontsize=10, color="#68748a", wrap=True)
    fig.tight_layout(); return fig


def run_catalog_experiment(experiment: str, budget: int, alpha: float, gamma: float, epsilon: float, seed: int, language: str):
    env_id = catalog_env_id(experiment); started = time.perf_counter()
    logs = [f"{env_id} automatic training console", "=" * 72, elapsed_line(started, "REGISTER", f"environment={env_id} family={experiment.split(' · ', 1)[0]}"), elapsed_line(started, "CONFIG", f"budget={budget} learning_rate={alpha:g} gamma={gamma:g} epsilon={epsilon:g} seed={seed}")]
    yield status_card("running", copy_for(language)["running"], "Inspecting environment and action space", language), metric_card("AUTO", "selecting a compatible baseline", language), error_figure(env_id, "Inspecting environment..."), None, None, console_panel("\n".join(logs), language)
    env = None
    try:
        env = gym.make(env_id)
        action_space = env.action_space; observation_space = env.observation_space
        logs.append(elapsed_line(started, "SPACE", f"observation={observation_space} action={action_space}"))
        if isinstance(action_space, gym.spaces.Discrete):
            algorithm = "DQN"
        elif isinstance(action_space, gym.spaces.Box):
            algorithm = "SAC"
        elif isinstance(action_space, (gym.spaces.MultiDiscrete, gym.spaces.MultiBinary)):
            algorithm = "PPO"
        else:
            raise ValueError(f"Unsupported action space for the automatic baseline: {action_space}")
        logs.append(elapsed_line(started, "AUTO", f"selected_algorithm={algorithm}")); env.close(); env = None
        yield status_card("running", copy_for(language)["running"], f"Auto selected {algorithm}", language), metric_card(algorithm, f"action space: {action_space}", language), error_figure(env_id, f"Starting {algorithm} training..."), None, None, console_panel("\n".join(logs), language)
        for status, metric, curve, preview, artifact, console in run_deep_control(experiment, env_id, algorithm, budget, alpha, gamma, epsilon, seed, language):
            deep_text = re.search(r'<pre class="console-text">(.*?)</pre>', console, re.DOTALL)
            combined = "\n".join(logs) + ("\n\n" + html.unescape(deep_text.group(1)) if deep_text else "")
            yield status, metric, curve, preview, artifact, console_panel(combined, language)
    except Exception as exc:
        if env is not None:
            env.close()
        message = f"{type(exc).__name__}: {exc}"; logs.append(elapsed_line(started, "ERROR", message)); logs.append(elapsed_line(started, "HINT", "The environment remains registered. Install its optional package, ROM, or runtime dependency and try again."))
        summary = save_summary(env_id, {"experiment": experiment, "environment": env_id, "status": "registered-but-unavailable", "error": message, "parameters": {"budget": budget, "learning_rate": alpha, "gamma": gamma, "epsilon": epsilon, "seed": seed}})
        yield status_card("idle", "Environment registered", "Runtime dependency required", language), metric_card("SETUP", "see the latest log lines", language), error_figure(env_id, message), None, summary, console_panel("\n".join(logs), language)


def train(experiment: str, budget: float, alpha: float, gamma: float, epsilon: float, seed: float, language: str):
    budget, seed = int(budget), int(seed)
    if is_catalog_experiment(experiment):
        yield from run_catalog_experiment(experiment, budget, alpha, gamma, epsilon, seed, language)
    elif experiment == BANDIT:
        yield from run_bandit(budget, alpha, epsilon, seed, language)
    elif experiment == BLACKJACK:
        yield from run_blackjack(budget, alpha, gamma, epsilon, seed, language)
    elif experiment == GRIDWORLD:
        yield from run_gridworld(budget, alpha, gamma, epsilon, seed, language)
    elif experiment == FROZENLAKE:
        yield from run_frozenlake(budget, alpha, gamma, epsilon, seed, language)
    elif experiment == CLIFF:
        yield from run_discrete_control(CLIFF, "CliffWalking-v1", "SARSA", budget, alpha, gamma, epsilon, seed, language)
    elif experiment == TAXI:
        yield from run_discrete_control(TAXI, "Taxi-v4", "Q-Learning", budget, alpha, gamma, epsilon, seed, language)
    elif experiment == MOUNTAINCAR:
        yield from run_mountaincar(budget, alpha, gamma, epsilon, seed, language)
    else:
        env_id = EXPERIMENTS[experiment]["environment"]
        yield from run_deep_control(experiment, env_id, EXPERIMENTS[experiment]["algorithm"], budget, alpha, gamma, epsilon, seed, language)


def slider_update(label: str, spec: tuple[float, float, float, float], visible: bool = True):
    minimum, maximum, value, step = spec
    return gr.Slider(minimum=minimum, maximum=maximum, value=value, step=step, label=label, visible=visible)


def select_experiment(experiment: str, language: str):
    copy = copy_for(language); cfg = experiment_config(experiment)
    return (
        hero_html(language, experiment),
        slider_update(copy["budget"], cfg["budget"]),
        slider_update(copy["alpha"], cfg["alpha"]),
        slider_update(copy["gamma"], cfg["gamma"], cfg["gamma_visible"]),
        slider_update(copy["epsilon"], cfg["epsilon"], cfg["algorithm"] not in {"PPO", "SAC"}),
        status_card("idle", copy["ready"], copy["ready_detail"], language),
        metric_card("—", copy["metric_waiting"], language),
        console_panel(copy["log_waiting"], language),
        None,
        None,
    )


def switch_language(language: str, experiment: str, seed: float):
    copy = copy_for(language); cfg = experiment_config(experiment)
    return (
        hero_html(language, experiment), panel_html(copy["settings"], copy["settings_copy"]),
        gr.Dropdown(choices=EXPERIMENT_CHOICES, value=experiment, label=copy["experiment"]), slider_update(copy["budget"], cfg["budget"]), slider_update(copy["alpha"], cfg["alpha"]),
        slider_update(copy["gamma"], cfg["gamma"], cfg["gamma_visible"]), slider_update(copy["epsilon"], cfg["epsilon"], cfg["algorithm"] not in {"PPO", "SAC"}),
        gr.Number(value=seed, precision=0, label=copy["seed"]), gr.Button(value=copy["start"]), status_card("idle", copy["ready"], copy["ready_detail"], language),
        metric_card("—", copy["metric_waiting"], language), panel_html(copy["curve"], copy["curve_copy"]), console_panel(copy["log_waiting"], language),
        panel_html(copy["preview"], copy["preview_copy"], "artifact-note"), gr.File(label=copy["artifact"]),
    )


CSS = """
:root { --ink:#172033; --muted:#68748a; --line:#e4e8f0; --canvas:#f4f6fa; --brand:#5b5ce2; --green:#13a36f; }
.gradio-container { max-width:1180px!important; margin:0 auto!important; padding:28px 22px 52px!important; background:var(--canvas); }
.hero-stack { position:relative!important; margin:0!important; padding:0!important; border:0!important; background:transparent!important; }
.language-bar { position:absolute!important; z-index:5!important; top:18px!important; right:20px!important; width:auto!important; min-width:0!important; margin:0!important; padding:0!important; border:0!important; background:transparent!important; }
.language-switch { width:216px!important; min-width:216px!important; margin:0!important; padding:3px!important; border:1px solid rgba(255,255,255,.18)!important; border-radius:10px!important; background:rgba(14,20,46,.58)!important; box-shadow:0 7px 20px rgba(5,8,24,.22)!important; backdrop-filter:blur(12px)!important; }
.language-switch>div { display:grid!important; grid-template-columns:1fr 1fr!important; gap:3px!important; }
.language-switch label { display:flex!important; cursor:pointer!important; }.language-switch input{display:none!important}
.language-switch label span { width:100%!important; min-height:34px!important; justify-content:center!important; padding:7px 13px!important; border-radius:7px!important; border:0!important; color:rgba(255,255,255,.72)!important; background:transparent!important; font-size:13px!important; font-weight:700!important; }
.language-switch label:has(input:checked) span,.language-switch input:checked+span { color:#fff!important; background:linear-gradient(135deg,#6667e8,#7778f2)!important; box-shadow:0 3px 9px rgba(13,15,55,.28)!important; }
.hero { position:relative; overflow:hidden; padding:38px 42px 34px; border:1px solid rgba(129,140,248,.2); border-radius:26px; color:#f8fafc; background:radial-gradient(circle at 88% 8%,rgba(125,127,255,.42),transparent 31%),radial-gradient(circle at 92% 92%,rgba(61,207,170,.18),transparent 30%),linear-gradient(132deg,#11182c 0%,#25265d 58%,#4546a4 100%); box-shadow:0 22px 54px rgba(25,32,56,.16); }
.project-mark { display:block; width:290px; max-width:55%; height:auto; margin:0 0 22px; padding:9px 13px; border-radius:11px; background:#fff; box-shadow:0 8px 24px rgba(8,15,35,.2); }
.hero-topline{display:flex;align-items:center;gap:11px;margin-bottom:22px}.experiment-badge{padding:6px 11px;border:1px solid rgba(221,224,255,.3);border-radius:999px;background:rgba(255,255,255,.1);font-size:12px;font-weight:700;letter-spacing:.06em}.hero-course{color:#b9c0d4;font-size:13px;font-weight:650}
.hero h1{max-width:760px;margin:0 0 12px;color:#fff;font-size:clamp(32px,5vw,48px);line-height:1.1;letter-spacing:-.035em}.hero-copy{max-width:760px;margin:0;color:#cdd3e2;font-size:15px;line-height:1.7}.hero-links{display:flex;flex-wrap:wrap;gap:9px;margin-top:25px}.hero-link{display:inline-flex;align-items:center;min-height:38px;padding:0 14px;border:1px solid rgba(255,255,255,.18);border-radius:9px;color:#eef2ff!important;background:rgba(255,255,255,.08);font-size:13px;font-weight:650;text-decoration:none!important}.hero-link.primary{color:#172554!important;background:#fff;border-color:#fff}
.lab-strip{display:flex;flex-wrap:wrap;gap:8px 22px;margin:17px 0 22px;padding:13px 18px;border:1px solid var(--line);border-radius:13px;background:#fff;color:var(--muted);font-size:13px;box-shadow:0 6px 20px rgba(18,25,43,.035)}.lab-strip strong{margin-left:5px;color:var(--ink)}
.control-card,.chart-card,.output-card{border:1px solid var(--line)!important;border-radius:17px!important;background:#fff!important;box-shadow:0 10px 30px rgba(18,25,43,.045)!important}.control-card,.chart-card{padding:22px!important}.output-card{margin-top:16px!important;padding:22px!important}.panel-title{margin:0 0 5px;color:var(--ink);font-size:19px}.panel-copy,.artifact-note{margin:0 0 17px;color:var(--muted);font-size:13px;line-height:1.6}
.primary-btn{min-height:46px!important;border:0!important;border-radius:11px!important;background:linear-gradient(135deg,#5153d6,#6969ec)!important;font-weight:750!important}.run-state,.live-metric{display:flex;gap:12px;margin-top:14px;padding:14px 15px;border-radius:13px;background:#f8f9fc}.run-state__dot{width:9px;height:9px;margin-top:6px;border-radius:50%;background:#94a3b8}.run-state--running .run-state__dot{background:#5b5ce2;box-shadow:0 0 0 5px rgba(91,92,226,.13)}.run-state--complete .run-state__dot{background:#13a36f}.run-state strong,.run-state small,.summary-label{display:block}.summary-label{color:#8a94a8;font-size:10px;font-weight:800;letter-spacing:.1em;text-transform:uppercase}.run-state strong{margin-top:3px;color:var(--ink);font-size:14px}.run-state small,.live-metric small{margin-top:3px;color:var(--muted);font-size:12px}.metric-reading{display:flex;align-items:baseline;gap:9px;margin-top:4px}.metric-reading strong{color:var(--ink);font-size:24px}
.console-panel{overflow:hidden;margin-top:18px;border:1px solid #202b3d;border-radius:13px;background:#0f1623}.console-head{display:flex;align-items:center;gap:9px;padding:11px 15px;border-bottom:1px solid #263244;color:#e2e8f0;font-size:12px;font-weight:750}.console-dot{width:8px;height:8px;border-radius:50%;background:#22c55e;box-shadow:0 0 0 4px rgba(34,197,94,.12)}.console-text{box-sizing:border-box;height:300px;margin:0;padding:17px 18px;overflow:auto;white-space:pre;color:#cbd5e1!important;background:#0f1623!important;font:12px/1.58 "SFMono-Regular",Consolas,monospace!important;scrollbar-gutter:stable}.footer-note{margin-top:18px;text-align:center;color:#94a3b8;font-size:12px}.footer-note a{color:var(--brand)!important;text-decoration:none!important;font-weight:650}
@media(max-width:760px){.gradio-container{padding:12px 10px 30px!important}.language-bar{top:14px!important;right:14px!important}.language-switch{width:196px!important;min-width:196px!important}.hero{padding:70px 22px 25px;border-radius:19px}.hero-topline{align-items:flex-start;flex-direction:column}.project-mark{max-width:70%}}
"""


AUTO_SCROLL_JS = """
() => {
  const selector = "#live-training-console .console-text";
  let active = null, follow = true, saved = 0, internal = false, scheduled = false;
  const update = () => {
    scheduled = false;
    const element = document.querySelector(selector);
    if (!element) return;
    if (element !== active) {
      active = element;
      active.addEventListener("scroll", () => {
        if (internal) return;
        follow = active.scrollHeight - active.clientHeight - active.scrollTop <= 24;
        saved = active.scrollTop;
      }, {passive:true});
    }
    internal = true;
    if (follow) active.scrollTop = active.scrollHeight;
    else active.scrollTop = Math.min(saved, Math.max(0, active.scrollHeight - active.clientHeight));
    requestAnimationFrame(() => { internal = false; });
  };
  const schedule = () => { if (!scheduled) { scheduled = true; requestAnimationFrame(update); } };
  new MutationObserver(schedule).observe(document.body, {childList:true, subtree:true, characterData:true});
  schedule();
}
"""


DEFAULT_LANGUAGE = "English"
DEFAULT_EXPERIMENT = BANDIT
copy = copy_for(DEFAULT_LANGUAGE)
cfg = EXPERIMENTS[DEFAULT_EXPERIMENT]

with gr.Blocks(title="Hands-On Modern RL · Gymnasium CPU Playground") as demo:
    with gr.Column(elem_classes="hero-stack"):
        hero = gr.HTML(hero_html(DEFAULT_LANGUAGE, DEFAULT_EXPERIMENT))
        with gr.Row(elem_classes="language-bar"):
            language = gr.Radio(choices=[("English", "English"), ("中文", "中文")], value=DEFAULT_LANGUAGE, show_label=False, elem_classes="language-switch")

    with gr.Row():
        with gr.Column(scale=1, min_width=310, elem_classes="control-card"):
            settings_header = gr.HTML(panel_html(copy["settings"], copy["settings_copy"]))
            experiment = gr.Dropdown(choices=EXPERIMENT_CHOICES, value=DEFAULT_EXPERIMENT, label=copy["experiment"], interactive=True, filterable=True)
            budget = gr.Slider(minimum=200, maximum=100000, value=cfg["budget"][2], step=100, label=copy["budget"], info=copy["budget_info"])
            alpha = gr.Slider(minimum=.00001, maximum=1, value=cfg["alpha"][2], step=.00001, label=copy["alpha"])
            gamma = gr.Slider(minimum=0, maximum=1, value=0, step=.05, label=copy["gamma"], visible=False)
            epsilon = gr.Slider(minimum=0, maximum=1, value=.1, step=.01, label=copy["epsilon"])
            seed = gr.Number(value=42, precision=0, label=copy["seed"])
            start = gr.Button(copy["start"], variant="primary", elem_classes="primary-btn")
            status = gr.HTML(status_card("idle", copy["ready"], copy["ready_detail"], DEFAULT_LANGUAGE))
            metric = gr.HTML(metric_card("—", copy["metric_waiting"], DEFAULT_LANGUAGE))
        with gr.Column(scale=2, elem_classes="chart-card"):
            chart_header = gr.HTML(panel_html(copy["curve"], copy["curve_copy"]))
            curve = gr.Plot(show_label=False)
            console = gr.HTML(console_panel(copy["log_waiting"], DEFAULT_LANGUAGE), elem_id="live-training-console")

    with gr.Row(elem_classes="output-card"):
        with gr.Column(scale=2):
            preview_header = gr.HTML(panel_html(copy["preview"], copy["preview_copy"], "artifact-note"))
            preview = gr.Image(show_label=False, interactive=False)
        with gr.Column(scale=1):
            artifact = gr.File(label=copy["artifact"], interactive=False)

    gr.HTML(footer_html())

    experiment.change(select_experiment, inputs=[experiment, language], outputs=[hero, budget, alpha, gamma, epsilon, status, metric, console, preview, artifact], queue=False)
    language.change(switch_language, inputs=[language, experiment, seed], outputs=[hero, settings_header, experiment, budget, alpha, gamma, epsilon, seed, start, status, metric, chart_header, console, preview_header, artifact], queue=False)
    start.click(train, inputs=[experiment, budget, alpha, gamma, epsilon, seed, language], outputs=[status, metric, curve, preview, artifact, console], concurrency_limit=1)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(css=CSS, js=AUTO_SCROLL_JS)
