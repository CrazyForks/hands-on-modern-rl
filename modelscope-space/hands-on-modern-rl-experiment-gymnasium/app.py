"""A browser-based collection of small CPU Gymnasium training experiments."""

from __future__ import annotations

import base64
from collections import defaultdict
import ctypes.util
import html
import importlib
import json
import os
import re
import sys
import textwrap
import time
import warnings
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
if sys.platform.startswith("linux") and ctypes.util.find_library("OSMesa"):
    # Prefer Mesa's CPU renderer when the Studio image provides it. Guarding
    # the setting preserves Gymnasium-Robotics registration on base images
    # where the optional system library is unavailable.
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

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
PREVIEW_DIR = ROOT / "assets" / "previews"
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

CURATED_PREVIEWS = {
    BANDIT: "bandit-arm-estimates.png",
    BLACKJACK: "blackjack-policy.png",
    GRIDWORLD: "gridworld-policy.png",
    FROZENLAKE: "frozenlake-policy.png",
    CLIFF: "cliffwalking-trained.gif",
    TAXI: "taxi-trained.gif",
    CARTPOLE_DQN: "cartpole-dqn-trained.gif",
    CARTPOLE_PPO: "cartpole-ppo-trained.gif",
    MOUNTAINCAR: "mountaincar-trained.gif",
    ACROBOT: "acrobot-trained.gif",
    PENDULUM: "pendulum-trained.gif",
    MOUNTAINCAR_CONTINUOUS: "mountaincarcontinuous-trained.gif",
}

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

RUNTIME_PROBES = {
    "Toy Text": "FrozenLake-v1",
    "Classic Control": "CartPole-v1",
    "Box2D": "LunarLander-v3",
    "Atari / ALE": "ALE/Pong-v5",
    "MuJoCo": "Ant-v5",
    "Robotics": "FetchReach-v4",
    "JAX Phys2D": "phys2d/CartPole-v1",
    "JAX Tabular": "tabular/Blackjack-v0",
}


def preload_runtimes() -> dict[str, str]:
    """Load native engines and representative assets before the UI opens."""
    results = {}
    for family, env_id in RUNTIME_PROBES.items():
        env = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                env = gym.make(env_id)
                env.reset(seed=0)
            results[family] = "Ready · preinstalled"
        except Exception as exc:
            results[family] = f"Unavailable · {type(exc).__name__}"
        finally:
            if env is not None:
                env.close()
    return results


RUNTIME_STATUS = preload_runtimes()
RUNTIME_READY = sum(value.startswith("Ready") for value in RUNTIME_STATUS.values())


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
CARD_PAGE_SIZE = 36
FAMILY_ORDER = ["Curated", "Toy Text", "Classic Control", "Box2D", "Atari / ALE", "MuJoCo", "Robotics", "JAX Phys2D", "JAX Tabular", "Other"]


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


FAMILY_VISUALS = {
    "Curated": ("#5b5ce2", "◆", "TUNED"), "Bandit": ("#7c3aed", "▥", "EXPLORE"),
    "Toy Text": ("#0f9f74", "▦", "TABULAR"), "Tabular": ("#0f9f74", "▦", "VALUES"),
    "Classic Control": ("#2563eb", "⚖", "CONTROL"), "Box2D": ("#ea580c", "⌁", "PHYSICS"),
    "Atari / ALE": ("#db2777", "▦", "PIXELS"), "MuJoCo": ("#0891b2", "⌁", "LOCOMOTION"),
    "Robotics": ("#4f46e5", "⌁", "GOAL"), "JAX Phys2D": ("#16a34a", "⚡", "JAX"),
    "JAX Tabular": ("#16a34a", "▦", "JAX"), "Other": ("#64748b", "◇", "ENV"),
}


def semantic_scene(env_id: str, family: str) -> tuple[str, str]:
    """Return a small SVG scene and a human-readable scene label."""
    name = env_id.lower()
    if "bandit" in name:
        return '<rect x="105" y="185" width="55" height="70" rx="8"/><rect x="190" y="125" width="55" height="130" rx="8"/><rect x="275" y="72" width="55" height="183" rx="8"/><rect x="360" y="145" width="55" height="110" rx="8"/><path d="M98 265H430"/>', "EXPLORE"
    if "blackjack" in name:
        return '<g transform="rotate(-9 220 160)"><rect x="130" y="72" width="142" height="188" rx="14"/><text x="154" y="124">A</text><path d="M190 174l20-25 20 25-20 25z"/></g><g transform="rotate(9 340 160)"><rect x="270" y="72" width="142" height="188" rx="14"/><text x="294" y="124">10</text><circle cx="341" cy="176" r="25"/></g>', "CARDS"
    if "taxi" in name:
        return '<path d="M95 70V260M175 70V260M255 70V260M335 70V260M415 70V260M95 70H415M95 133H415M95 196H415M95 260H415" opacity=".35"/><rect x="174" y="145" width="92" height="48" rx="12"/><circle cx="194" cy="201" r="11"/><circle cx="246" cy="201" r="11"/><circle cx="368" cy="102" r="15"/><path d="M366 118v34"/>', "PICK UP"
    if "frozenlake" in name:
        return '<path d="M90 70H410V262H90zM170 70v192M250 70v192M330 70v192M90 118h320M90 166h320M90 214h320" opacity=".35"/><path d="M107 91l28 18-20 28zM265 178l26 20-23 25z"/><circle cx="370" cy="236" r="18"/><path d="M120 238C170 190 205 225 255 170S335 130 372 105" fill="none" stroke-dasharray="10 12"/>', "SLIPPERY"
    if "cliff" in name:
        return '<path d="M75 244H445M92 206H420M108 168H398M122 130H376" opacity=".3"/><path d="M75 245L165 245 205 180 270 245 445 245" fill="none"/><path d="M170 252l35 38 35-38M245 252l35 38 35-38"/><circle cx="102" cy="222" r="14"/><path d="M110 211C170 145 265 160 407 216" fill="none" stroke-dasharray="10 12"/>', "SAFE PATH"
    if "cartpole" in name:
        return '<path d="M76 252H454"/><rect x="205" y="194" width="118" height="48" rx="11"/><circle cx="231" cy="250" r="14"/><circle cx="297" cy="250" r="14"/><circle cx="264" cy="192" r="12"/><path d="M264 190L304 55" stroke-width="16"/><circle cx="304" cy="55" r="13"/><path d="M164 219h-55M364 219h55"/>', "BALANCE"
    if "mountaincar" in name:
        return '<path d="M55 240C130 115 205 270 285 205S395 80 470 225" fill="none"/><g transform="translate(245 188) rotate(-18)"><rect x="0" y="0" width="85" height="35" rx="10"/><circle cx="20" cy="39" r="12"/><circle cx="66" cy="39" r="12"/></g><path d="M365 105h54v52"/><path d="M419 105l-25 17 25 17" fill="none"/>', "MOMENTUM"
    if "pendulum" in name and "inverted" not in name:
        return '<circle cx="260" cy="82" r="18"/><path d="M260 82L345 215" stroke-width="17"/><circle cx="345" cy="215" r="27"/><path d="M176 216A105 105 0 0 1 338 104" fill="none" stroke-dasharray="10 12"/><path d="M337 104l-2 35-31-15" fill="none"/>', "SWING UP"
    if "acrobot" in name:
        return '<circle cx="260" cy="54" r="15"/><path d="M260 54L214 157L310 229" stroke-width="18" fill="none"/><circle cx="214" cy="157" r="17"/><circle cx="310" cy="229" r="18"/><path d="M130 102H394" stroke-dasharray="10 12"/><path d="M320 224c43-22 65-55 71-99" fill="none"/>', "SWING"
    if "lunarlander" in name:
        return '<circle cx="402" cy="72" r="29" opacity=".3"/><path d="M75 250l85-24 74 20 92-17 125 22" fill="none"/><path d="M224 128h72l22 69-30 24h-56l-30-24z"/><path d="M232 215l-24 31M288 215l24 31M222 246h-35M303 246h35"/><path d="M242 198l18 48 18-48" fill="none"/><path d="M177 188v50M343 188v50"/>', "LAND"
    if "carracing" in name:
        return '<path d="M90 245C40 150 145 70 240 112S435 60 455 170 340 274 260 225 130 285 90 245" fill="none" stroke-width="30" opacity=".45"/><g transform="translate(242 163) rotate(-18)"><rect width="90" height="40" rx="10"/><circle cx="20" cy="44" r="11"/><circle cx="70" cy="44" r="11"/></g>', "RACE"
    if "bipedal" in name or "walker" in name or "humanoid" in name:
        return '<circle cx="257" cy="68" r="23"/><path d="M257 92v83M257 120l-68 47M257 120l66 35M257 175l-54 76M257 175l65 72" fill="none" stroke-width="17"/><path d="M65 260h400"/>', "WALK"
    if "hopper" in name:
        return '<circle cx="257" cy="66" r="23"/><path d="M257 90v82l-50 47 61 35" fill="none" stroke-width="20"/><path d="M65 260h400M155 235l-30-28M365 227l30-32"/>', "HOP"
    if "swimmer" in name:
        return '<path d="M85 165C155 100 220 235 285 165S405 100 465 165" fill="none" stroke-width="25"/><circle cx="85" cy="165" r="17"/><path d="M80 247c90-30 160 25 250-5s110 12 145 4" fill="none" opacity=".3"/>', "SWIM"
    if "cheetah" in name or "ant" in name:
        return '<path d="M150 163h184l48 42M164 165l-58 75M215 168l-28 82M296 168l35 79M338 162l80 69" fill="none" stroke-width="17"/><circle cx="358" cy="137" r="25"/><path d="M85 260h380"/>', "LOCOMOTION"
    if "hammer" in name:
        return '<rect x="82" y="224" width="150" height="26" rx="8"/><circle cx="140" cy="211" r="19"/><path d="M140 204l64-62 62 36 42-40" fill="none" stroke-width="19"/><circle cx="204" cy="142" r="16"/><circle cx="266" cy="178" r="16"/><path d="M299 127l68-68M346 48l39 39M318 108l39 39"/><path d="M401 172v76M376 248h50"/><circle cx="401" cy="163" r="10"/>', "HAMMER"
    if any(word in name for word in ("reach", "push", "slide", "pick", "place", "door", "hammer", "hand", "robot", "franka")) or family == "Robotics":
        return '<rect x="92" y="226" width="105" height="27" rx="8"/><circle cx="145" cy="213" r="20"/><path d="M145 207l65-74 75 37 54-68" fill="none" stroke-width="20"/><circle cx="210" cy="133" r="17"/><circle cx="285" cy="170" r="17"/><path d="M330 91l24 24M350 84l24 24"/><circle cx="414" cy="205" r="31" stroke-dasharray="9 9" fill="none"/>', "REACH"
    if "pong" in name:
        return '<rect x="82" y="102" width="18" height="118" rx="8"/><rect x="420" y="80" width="18" height="118" rx="8"/><circle cx="274" cy="151" r="16"/><path d="M110 155l130-4M305 147l105-29" stroke-dasharray="9 11"/>', "PONG"
    if "breakout" in name:
        return '<rect x="90" y="58" width="75" height="26"/><rect x="175" y="58" width="75" height="26"/><rect x="260" y="58" width="75" height="26"/><rect x="345" y="58" width="75" height="26"/><rect x="205" y="248" width="110" height="18" rx="8"/><circle cx="300" cy="175" r="14"/><path d="M300 189l-42 52" stroke-dasharray="9 10"/>', "BREAKOUT"
    if any(word in name for word in ("space", "asteroid", "battle", "beam", "galax", "star")):
        return '<path d="M260 82l34 88-34-18-34 18z"/><circle cx="108" cy="87" r="6"/><circle cx="412" cy="141" r="7"/><circle cx="370" cy="66" r="5"/><path d="M255 173l-25 79M265 173l25 79"/><path d="M88 225l26-35 22 30 30-50 35 55" fill="none" opacity=".4"/>', "ARCADE"
    if family == "Atari / ALE":
        return '<rect x="82" y="72" width="356" height="188" rx="16"/><path d="M103 233l65-78 58 36 69-83 113 104" fill="none"/><circle cx="350" cy="118" r="19"/><rect x="120" y="98" width="52" height="34" rx="7"/>', "PIXEL CONTROL"
    if family == "Toy Text" or family == "JAX Tabular" or "grid" in name:
        return '<path d="M90 66H430V258H90zM175 66v192M260 66v192M345 66v192M90 114h340M90 162h340M90 210h340" opacity=".35"/><circle cx="130" cy="234" r="15"/><path d="M146 226C225 190 260 130 390 91" fill="none" stroke-dasharray="10 12"/><path d="M376 82h28v28" fill="none"/>', "GRID"
    return '<circle cx="145" cy="183" r="47"/><circle cx="260" cy="112" r="37"/><circle cx="385" cy="195" r="54"/><path d="M184 158l42-27M292 136l51 33M188 201l139 2" fill="none"/><circle cx="145" cy="183" r="9"/><circle cx="260" cy="112" r="9"/><circle cx="385" cy="195" r="9"/>', "ENVIRONMENT"


def experiment_visual(experiment: str) -> str:
    cfg = experiment_config(experiment); family = cfg["family"]; env_id = cfg["environment"]
    color, _, _ = FAMILY_VISUALS.get("Curated" if experiment in EXPERIMENTS else family, FAMILY_VISUALS["Other"])
    scene, scene_label = semantic_scene(env_id if env_id != "Custom 4×4 GridWorld" else "gridworld", family)
    safe = re.sub(r"[^a-z0-9]+", "-", experiment.lower()).strip("-")
    path = ARTIFACT_DIR / f"card-{safe}.svg"
    title = html.escape(env_id.replace("ALE/", "")); short_title = title if len(title) <= 29 else title[:27] + "…"
    algorithm = html.escape(cfg["algorithm"].replace("Auto: inspect action space", "AUTO"))
    path.write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" width="520" height="320" viewBox="0 0 520 320">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop stop-color="{color}"/><stop offset="1" stop-color="#111827"/></linearGradient></defs>
<rect width="520" height="320" rx="26" fill="url(#g)"/><circle cx="455" cy="36" r="95" fill="#fff" opacity=".08"/>
<g fill="none" stroke="#fff" stroke-width="11" stroke-linecap="round" stroke-linejoin="round">{scene}</g>
<text x="28" y="38" fill="#fff" font-size="19" font-family="Arial" font-weight="700">{short_title}</text>
<rect x="28" y="276" width="170" height="25" rx="12" fill="#fff" opacity=".15"/><text x="42" y="294" fill="#fff" font-size="13" font-family="Arial" font-weight="700" letter-spacing="2">{scene_label}</text>
<rect x="402" y="276" width="90" height="25" rx="12" fill="#fff" opacity=".92"/><text x="447" y="294" text-anchor="middle" fill="#172033" font-size="12" font-family="Arial" font-weight="800">{algorithm}</text></svg>''', encoding="utf-8")
    return str(path)


def visual_data_uri(experiment: str) -> str:
    payload = Path(experiment_visual(experiment)).read_bytes()
    return f"data:image/svg+xml;base64,{base64.b64encode(payload).decode()}"


def experiment_goal(experiment: str) -> str:
    env_id = experiment_config(experiment)["environment"]
    goals = {
        "Blackjack-v1": "Beat the dealer without exceeding 21", "FrozenLake-v1": "Reach the goal across slippery ice",
        "CliffWalking-v1": "Cross safely without falling from the cliff", "Taxi-v4": "Pick up and deliver the passenger",
        "CartPole-v1": "Keep the pole balanced", "MountainCar-v0": "Build momentum to climb the hill",
        "MountainCarContinuous-v0": "Climb the hill with continuous force", "Acrobot-v1": "Swing the end link above the target",
        "Pendulum-v1": "Swing up and stabilize the pendulum", "LunarLander-v3": "Land between the flags",
    }
    if env_id in goals: return goals[env_id]
    name = env_id.lower()
    if "pong" in name: return "Move the paddle to return the ball past the opponent"
    if "breakout" in name: return "Keep the ball in play and clear the brick wall"
    if "lunarlander" in name: return "Control the engines and land softly between the flags"
    if "carracing" in name: return "Steer a car around the track as quickly and smoothly as possible"
    if "bipedal" in name or "walker" in name: return "Coordinate the legs to walk forward without falling"
    if "humanoid" in name: return "Coordinate a humanoid body to move forward without falling"
    if "hopper" in name: return "Hop forward while keeping the body upright"
    if "cheetah" in name: return "Coordinate the joints to run forward quickly"
    if "ant" in name: return "Coordinate four legs to travel forward stably"
    if "swimmer" in name: return "Propel the articulated body through fluid"
    if "reach" in name: return "Move the robot end effector to the target position"
    if "push" in name: return "Push an object from its initial position to the target"
    if any(word in name for word in ("pick", "place")): return "Pick up an object and move it to the target"
    if "door" in name: return "Manipulate the robot hand to open the door"
    if "hammer" in name: return "Control the robot hand to drive the nail with a hammer"
    family = experiment_config(experiment)["family"]
    return {
        "Atari / ALE": "Learn control directly from game pixels", "MuJoCo": "Learn continuous physics control",
        "Robotics": "Reach or manipulate a goal", "Box2D": "Learn control in a 2D physics task",
        "Toy Text": "Learn a policy in a compact discrete world", "JAX Phys2D": "Run a JAX physics control task",
        "JAX Tabular": "Run a JAX tabular task", "Bandit": "Balance exploration and exploitation",
        "Tabular": "Propagate values through a small world",
    }.get(family, "Explore a registered Gymnasium task")


def localized_goal(experiment: str, language: str) -> str:
    if language != "中文": return experiment_goal(experiment)
    env_id = experiment_config(experiment)["environment"]
    goals = {
        "4-armed Bernoulli bandit": "在探索未知选项和利用当前最优选项之间取得平衡",
        "Custom 4×4 GridWorld": "学习从起点到终点的高回报路径",
        "Blackjack-v1": "在点数不超过 21 的前提下战胜庄家", "FrozenLake-v1": "穿过湿滑冰面到达终点",
        "CliffWalking-v1": "避开悬崖并安全抵达终点", "Taxi-v4": "接到乘客并送到指定位置",
        "CartPole-v1": "移动小车，使杆子尽可能长时间保持竖直", "MountainCar-v0": "积累动量并冲上山顶",
        "MountainCarContinuous-v0": "用连续推力控制小车爬上山顶", "Acrobot-v1": "摆动双连杆，使末端超过目标高度",
        "Pendulum-v1": "将摆杆甩起并稳定在竖直位置", "LunarLander-v3": "控制推进器，在两面旗帜之间平稳着陆",
    }
    if env_id in goals: return goals[env_id]
    name = env_id.lower()
    if "pong" in name: return "移动球拍，把球回击到对手无法接到的位置"
    if "breakout" in name: return "保持球不落下，并清除上方的砖块"
    if "lunarlander" in name: return "控制推进器，在两面旗帜之间平稳着陆"
    if "carracing" in name: return "控制赛车快速而平稳地沿赛道行驶"
    if "bipedal" in name or "walker" in name or "humanoid" in name: return "协调身体关节向前移动，并避免摔倒"
    if "hopper" in name: return "保持身体直立并连续向前跳跃"
    if "cheetah" in name or "ant" in name: return "协调多个关节，稳定而快速地向前移动"
    if "swimmer" in name: return "控制多节身体，在流体环境中向前游动"
    if "reach" in name: return "把机械臂末端移动到指定目标位置"
    if "push" in name: return "把物体从初始位置推到目标位置"
    family = experiment_config(experiment)["family"]
    return {"Atari / ALE": "直接根据游戏像素学习动作策略", "MuJoCo": "学习连续物理控制策略", "Robotics": "完成机械臂到达或物体操作任务", "Box2D": "在二维物理环境中学习控制策略", "Toy Text": "在小型离散环境中学习策略", "JAX Phys2D": "运行 JAX 二维物理控制任务", "JAX Tabular": "运行 JAX 表格型任务", "Bandit": "平衡探索与利用", "Tabular": "在小型环境中学习状态价值"}.get(family, "探索一个已注册的 Gymnasium 任务")


def card_caption(experiment: str) -> str:
    cfg = experiment_config(experiment); title = experiment.split(" · ")[0] if not is_catalog_experiment(experiment) else cfg["environment"]
    return f"{title}\n{experiment_goal(experiment)}\n{cfg['family']} · {cfg['algorithm']}"


def card_items(experiments: list[str]) -> list[tuple[str, str]]:
    return [(experiment_visual(item), card_caption(item)) for item in experiments]


def filter_choices(query: str, family: str) -> list[str]:
    query = (query or "").strip().lower()
    source = EXPERIMENT_CHOICES
    if family == "Curated": source = list(EXPERIMENTS)
    elif family != "All": source = [item for item in source if experiment_config(item)["family"] == family]
    if query:
        source = [item for item in source if query in (item + " " + experiment_goal(item) + " " + experiment_config(item)["algorithm"]).lower()]
    return source


def catalog_page(query: str, family: str, page: int):
    matches = filter_choices(query, family); pages = max(1, (len(matches) + CARD_PAGE_SIZE - 1) // CARD_PAGE_SIZE); page = max(0, min(int(page), pages - 1))
    visible = matches[page * CARD_PAGE_SIZE:(page + 1) * CARD_PAGE_SIZE]
    return card_items(visible), visible, page, f"{len(matches):,} experiments · Page {page + 1}/{pages}"


def reset_catalog(query: str, family: str):
    return catalog_page(query, family, 0)


def move_catalog(query: str, family: str, page: int, direction: int):
    return catalog_page(query, family, int(page) + direction)


def choose_card(visible: list[str], event: gr.SelectData):
    return visible[event.index]


def space_text(space) -> str:
    if isinstance(space, gym.spaces.Discrete): return f"Discrete({space.n})"
    if isinstance(space, gym.spaces.Box): return f"Box{space.shape}"
    if isinstance(space, gym.spaces.MultiDiscrete): return f"MultiDiscrete{space.shape}"
    if isinstance(space, gym.spaces.MultiBinary): return f"MultiBinary({space.n})"
    if isinstance(space, gym.spaces.Dict): return "Dict(" + ", ".join(space.spaces.keys()) + ")"
    if isinstance(space, gym.spaces.Tuple): return f"Tuple({len(space.spaces)} parts)"
    return type(space).__name__


def infer_algorithm(action_space, configured: str) -> str:
    if configured != "Auto: inspect action space": return configured
    if isinstance(action_space, gym.spaces.Discrete): return "DQN"
    if isinstance(action_space, gym.spaces.Box): return "SAC"
    if isinstance(action_space, (gym.spaces.MultiDiscrete, gym.spaces.MultiBinary)): return "PPO"
    return "Manual setup"


def task_brief(experiment: str, language: str) -> str:
    cfg = experiment_config(experiment); env_id = cfg["environment"]
    observation, action, algorithm, availability = "Custom", "Custom", cfg["algorithm"], "Ready · built in"
    if env_id == "4-armed Bernoulli bandit": observation, action = "Estimated arm values", "Choose one of 4 arms"
    elif env_id == "Custom 4×4 GridWorld": observation, action = "Grid cell", "Up / Down / Left / Right"
    else:
        env = None
        try:
            env = gym.make(env_id); observation = space_text(env.observation_space); action = space_text(env.action_space); algorithm = infer_algorithm(env.action_space, cfg["algorithm"]); availability = "Ready · preinstalled"
        except Exception as exc:
            availability = f"Legacy / unavailable · {type(exc).__name__}"
        finally:
            if env is not None: env.close()
    if language == "中文":
        return f'''<section class="task-brief"><div class="task-brief__visual"><img src="{visual_data_uri(experiment)}" alt="{html.escape(env_id)} task scene"></div><div class="task-brief__body"><span class="task-kicker">训练前先理解任务</span><h3>{html.escape(env_id)}</h3><p>{html.escape(localized_goal(experiment, language))}</p><div class="task-facts"><span><b>观察</b>{html.escape(observation)}</span><span><b>动作</b>{html.escape(action)}</span><span><b>算法</b>{html.escape(algorithm)}</span><span><b>状态</b>{html.escape(availability)}</span></div><p class="task-hint">调整下方参数后再点击“开始训练”。训练曲线和实时日志会持续更新。</p></div></section>'''
    return f'''<section class="task-brief"><div class="task-brief__visual"><img src="{visual_data_uri(experiment)}" alt="{html.escape(env_id)} task scene"></div><div class="task-brief__body"><span class="task-kicker">UNDERSTAND BEFORE TRAINING</span><h3>{html.escape(env_id)}</h3><p>{html.escape(localized_goal(experiment, language))}</p><div class="task-facts"><span><b>Observation</b>{html.escape(observation)}</span><span><b>Action</b>{html.escape(action)}</span><span><b>Algorithm</b>{html.escape(algorithm)}</span><span><b>Status</b>{html.escape(availability)}</span></div><p class="task-hint">Review the task, adjust the parameters below, then press Start training. The curve and live console will keep updating.</p></div></section>'''


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
        "start_running": "Running…",
        "wait_title": "Run active · please keep this page open",
        "wait_detail": "Initializing the environment and model, training the policy, then rendering the result. This indicator stays active until every stage finishes.",
        "ready": "Ready to train",
        "ready_detail": "Review the task brief, adjust parameters, then start the CPU run",
        "running": "Training in progress",
        "complete": "Training complete",
        "status": "Run status",
        "metric": "Latest evaluation",
        "metric_waiting": "Results appear after training starts",
        "curve": "Learning curve",
        "curve_copy": "The chart updates at each checkpoint. All labels stay in English for readability.",
        "log": "Live training log",
        "log_waiting": "Waiting for a training run...",
        "preview": "Task preview / trained result",
        "preview_copy": "The 12 curated tasks include real trained examples. Other registry tasks show a task illustration until your run produces a replay GIF, policy map, or result plot.",
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
        "start_running": "运行中…",
        "wait_title": "任务正在运行 · 请保持页面打开",
        "wait_detail": "正在初始化环境与模型、训练策略并生成结果回放。所有阶段完成前，这里会一直显示等待状态。",
        "ready": "等待训练",
        "ready_detail": "先阅读任务说明，调整参数，再启动 CPU 训练",
        "running": "训练进行中",
        "complete": "训练完成",
        "status": "训练状态",
        "metric": "最新评估",
        "metric_waiting": "训练开始后显示结果",
        "curve": "学习曲线",
        "curve_copy": "每个检查点更新一次曲线。图表标记统一保留英文。",
        "log": "实时训练日志",
        "log_waiting": "等待训练任务...",
        "preview": "任务预览 / 训练结果",
        "preview_copy": "12 个精选任务附带真实训练示例；其他注册表任务在运行前显示任务示意图，本次训练完成后替换为回放 GIF、策略图或结果曲线。",
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


def waiting_panel(language: str) -> str:
    copy = copy_for(language)
    elapsed_label = "elapsed" if language == "English" else "已等待"
    return f"""
    <section class="run-wait" role="status" aria-live="polite">
      <span class="run-wait__spinner" aria-hidden="true"></span>
      <div class="run-wait__copy">
        <strong>{copy['wait_title']}</strong>
        <small>{copy['wait_detail']}</small>
        <em class="run-wait__elapsed" data-start-ms="{int(time.time() * 1000)}" data-label="{elapsed_label}">0s {elapsed_label}</em>
      </div>
      <span class="run-wait__pulse" aria-hidden="true"><i></i></span>
    </section>
    """


def begin_run(language: str):
    copy = copy_for(language)
    return gr.HTML(value=waiting_panel(language), visible=True), gr.Button(value=copy["start_running"], interactive=False)


def finish_run(language: str):
    copy = copy_for(language)
    return gr.HTML(value="", visible=False), gr.Button(value=copy["start"], interactive=True)


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
        <span>Runtimes <strong>{RUNTIME_READY}/{len(RUNTIME_PROBES)} READY</strong></span>
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


def result_preview_image(
    experiment: str,
    status: str,
    metric_value: str,
    metric_label: str,
    filename: str | None = None,
    x: list[float] | None = None,
    y: list[float] | None = None,
    note: str = "",
    algorithm: str | None = None,
) -> str:
    """Create a durable result preview whenever an environment has no replay."""
    cfg = experiment_config(experiment)
    slug = re.sub(r"[^a-z0-9]+", "-", experiment.lower()).strip("-")
    path = ARTIFACT_DIR / (filename or f"{slug}-result.png")
    fig = plt.figure(figsize=(9.6, 5.0), facecolor="#f7f8fc")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.72], wspace=.34)
    info = fig.add_subplot(grid[0, 0]); plot = fig.add_subplot(grid[0, 1])
    info.set_facecolor("#20245b"); info.set_xticks([]); info.set_yticks([])
    for spine in info.spines.values(): spine.set_visible(False)
    info.text(.09, .88, status.upper(), color="#a5b4fc", fontsize=10, fontweight="bold", transform=info.transAxes)
    info.text(.09, .71, metric_value, color="white", fontsize=27, fontweight="bold", transform=info.transAxes, wrap=True)
    info.text(.09, .61, metric_label, color="#cbd5e1", fontsize=10, transform=info.transAxes, wrap=True)
    environment_label = textwrap.fill(cfg["environment"], width=21, break_long_words=False)
    algorithm_label = textwrap.fill(f"{algorithm or cfg['algorithm']} · CPU", width=24, break_long_words=False)
    note_label = textwrap.fill(note[:130], width=32, break_long_words=False)
    info.text(.09, .43, environment_label, color="white", fontsize=12, fontweight="bold", linespacing=1.25, transform=info.transAxes)
    info.text(.09, .27, algorithm_label, color="#cbd5e1", fontsize=9.5, linespacing=1.25, transform=info.transAxes)
    info.text(.09, .07, note_label, color="#aeb7ca", fontsize=8.5, linespacing=1.2, transform=info.transAxes)
    if x and y:
        plot.plot(x, y, color="#5b5ce2", linewidth=2.4)
        plot.scatter([x[-1]], [y[-1]], color="#13a36f", s=45, zorder=3)
        plot.set_xlabel("Training progress"); plot.set_ylabel(metric_label); plot.grid(alpha=.2)
        plot.set_title("Training result", loc="left", fontweight="bold", color="#172033")
    else:
        plot.axis("off")
        plot.text(.5, .59, "RESULT PREVIEW", ha="center", va="center", color="#5b5ce2", fontsize=19, fontweight="bold")
        plot.text(.5, .43, experiment, ha="center", va="center", color="#172033", fontsize=13, wrap=True)
        plot.text(.5, .30, "A policy map, replay GIF, or result image\nwill replace this panel after training.", ha="center", va="center", color="#68748a", fontsize=10, linespacing=1.5)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)


def example_preview(experiment: str, _run_state: str | None = None) -> str:
    """Show a real trained artifact before a new run replaces it."""
    filename = CURATED_PREVIEWS.get(experiment)
    if filename:
        path = PREVIEW_DIR / filename
        if path.exists():
            return str(path)
    # The full registry cannot ship a trained replay for every optional or
    # legacy environment. Use its task illustration without presenting it as
    # a learned result; a successful run replaces it with the generated GIF.
    return experiment_visual(experiment)


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
            yield status_card("running", copy_for(language)["running"], f"{step:,}/{budget:,} steps", language), metric_card(f"{avg:.3f}", f"estimated best arm: {int(np.argmax(q)) + 1}", language), learning_figure(list(range(1, step + 1)), (np.cumsum(rewards) / np.arange(1, step + 1)).tolist(), "Bandit cumulative average reward", "Average reward"), gr.skip(), None, console_panel("\n".join(logs), language)
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
            yield status_card("running", copy_for(language)["running"], f"{episode:,}/{budget:,} episodes", language), metric_card(f"{recent:.3f}", "mean reward over recent episodes", language), learning_figure(list(range(1, episode + 1)), rewards, "GridWorld episode reward", "Episode reward"), gr.skip(), None, console_panel("\n".join(logs), language)
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
            yield status_card("running", copy_for(language)["running"], f"{episode:,}/{budget:,} episodes", language), metric_card(f"{rate:.1%}", "recent success rate", language), learning_figure(list(range(1, episode + 1)), curve, "FrozenLake cumulative success rate", "Success rate"), gr.skip(), None, console_panel("\n".join(logs), language)
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
            yield status_card("running", copy_for(language)["running"], f"{episode:,}/{budget:,} episodes", language), metric_card(f"{win_rate:.1%}", "recent win rate", language), learning_figure(list(range(1, episode + 1)), cumulative, "Blackjack cumulative mean return", "Mean return"), gr.skip(), None, console_panel("\n".join(logs), language)
    env.close(); preview = blackjack_policy_image(q, "blackjack-policy.png")
    serialized_q = {str(state): values.tolist() for state, values in q.items()}
    summary = save_summary("blackjack", {"experiment": BLACKJACK, "q_values": serialized_q, "win_rate": float(np.mean(np.asarray(rewards[-5000:]) > 0)), "parameters": {"budget": budget, "alpha": alpha, "gamma": gamma, "epsilon": epsilon, "seed": seed}})
    logs.append(elapsed_line(started, "DONE", f"states={len(q)} artifact={summary}")); cumulative = (np.cumsum(rewards) / np.arange(1, len(rewards) + 1)).tolist()
    yield status_card("complete", copy_for(language)["complete"], f"{budget:,} episodes · {time.perf_counter() - started:.1f}s", language), metric_card(f"{np.mean(np.asarray(rewards[-5000:]) > 0):.1%}", "final 5,000-episode win rate", language), learning_figure(list(range(1, budget + 1)), cumulative, "Blackjack cumulative mean return", "Mean return"), preview, summary, console_panel("\n".join(logs), language)


def record_discrete_policy(env_id: str, q: np.ndarray, seed: int, filename: str, max_steps: int) -> str:
    env = gym.make(env_id, render_mode="rgb_array"); frames = []
    try:
        state, _ = env.reset(seed=seed)
        for step in range(max_steps):
            if step % 2 == 0:
                frame = env.render()
                if frame is not None: frames.append(frame)
            state, _, terminated, truncated, _ = env.step(int(np.argmax(q[int(state)])))
            if terminated or truncated:
                frame = env.render()
                if frame is not None: frames.append(frame)
                break
    finally:
        env.close()
    if not frames: raise RuntimeError("Environment returned no RGB frames")
    path = ARTIFACT_DIR / filename; imageio.mimsave(path, frames, duration=1 / 15, loop=0); return str(path)


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
            yield status_card("running", copy_for(language)["running"], f"{episode:,}/{budget:,} episodes", language), metric_card(f"{recent:.1f}", "recent mean episode reward", language), learning_figure(list(range(1, episode + 1)), rewards, f"{experiment} episode reward", "Episode reward"), gr.skip(), None, console_panel("\n".join(logs), language)
    env.close(); slug = "cliffwalking" if env_id.startswith("Cliff") else "taxi"
    try:
        gif = record_discrete_policy(env_id, q, seed + 10000, f"{slug}-trained.gif", max_steps)
        preview_kind = "replay GIF"
    except Exception as exc:
        gif = result_preview_image(experiment, "Training complete", f"{np.mean(rewards[-100:]):.1f}", "Final mean reward", x=list(range(1, budget + 1)), y=rewards, note=f"Replay unavailable: {type(exc).__name__}")
        preview_kind = "result image"
        logs.append(elapsed_line(started, "WARN", f"replay_unavailable={type(exc).__name__}: {exc}"))
    summary = save_summary(slug, {"experiment": experiment, "q_values": q.tolist(), "parameters": {"budget": budget, "alpha": alpha, "gamma": gamma, "epsilon": epsilon, "seed": seed}})
    logs.append(elapsed_line(started, "DONE", f"preview={preview_kind} path={gif} artifact={summary}"))
    yield status_card("complete", copy_for(language)["complete"], f"{budget:,} episodes · {time.perf_counter() - started:.1f}s", language), metric_card(f"{np.mean(rewards[-100:]):.1f}", "final 100-episode mean reward", language), learning_figure(list(range(1, budget + 1)), rewards, f"{experiment} episode reward", "Episode reward"), gif, summary, console_panel("\n".join(logs), language)


def mountain_state(obs: np.ndarray, bins=(24, 20)) -> tuple[int, int]:
    low = np.array([-1.2, -0.07]); high = np.array([0.6, 0.07]); scaled = (np.asarray(obs) - low) / (high - low)
    indices = np.floor(scaled * np.array(bins)).astype(int)
    return tuple(np.clip(indices, 0, np.array(bins) - 1))


def record_tabular_control(env_id: str, policy, seed: int, filename: str, max_steps: int = 500) -> str:
    env = gym.make(env_id, render_mode="rgb_array"); frames = []
    try:
        obs, _ = env.reset(seed=seed)
        for _ in range(max_steps):
            frame = env.render()
            if frame is not None: frames.append(frame)
            action = policy(obs); obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated: break
    finally:
        env.close()
    if not frames: raise RuntimeError("Environment returned no RGB frames")
    path = ARTIFACT_DIR / filename; imageio.mimsave(path, frames, duration=1 / 30, loop=0); return str(path)


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
            yield status_card("running", copy_for(language)["running"], f"{episode:,}/{budget:,} episodes", language), metric_card(f"{recent:.1f}", "recent mean episode reward", language), learning_figure(list(range(1, episode + 1)), rewards, "MountainCar episode reward", "Episode reward"), gr.skip(), None, console_panel("\n".join(logs), language)
    env.close()
    try:
        gif = record_tabular_control("MountainCar-v0", lambda obs: int(np.argmax(q[mountain_state(obs)])), seed + 10000, "mountaincar-trained.gif", 200)
        preview_kind = "replay GIF"
    except Exception as exc:
        gif = result_preview_image(MOUNTAINCAR, "Training complete", f"{np.mean(rewards[-100:]):.1f}", "Final mean reward", x=list(range(1, budget + 1)), y=rewards, note=f"Replay unavailable: {type(exc).__name__}")
        preview_kind = "result image"
        logs.append(elapsed_line(started, "WARN", f"replay_unavailable={type(exc).__name__}: {exc}"))
    summary = save_summary("mountaincar", {"experiment": "MountainCar", "q_values": q.tolist(), "parameters": {"budget": budget, "alpha": alpha, "gamma": gamma, "epsilon": epsilon, "seed": seed}})
    logs.append(elapsed_line(started, "DONE", f"preview={preview_kind} path={gif} artifact={summary}"))
    yield status_card("complete", copy_for(language)["complete"], f"{budget:,} episodes · {time.perf_counter() - started:.1f}s", language), metric_card(f"{np.mean(rewards[-100:]):.1f}", "final 100-episode mean reward", language), learning_figure(list(range(1, budget + 1)), rewards, "MountainCar episode reward", "Episode reward"), gif, summary, console_panel("\n".join(logs), language)


def record_model(model, env_id: str, seed: int, filename: str, max_steps: int) -> str:
    """Record a deterministic policy using the configured headless renderer."""
    env = gym.make(env_id, render_mode="rgb_array")
    frames = []
    try:
        obs, _ = env.reset(seed=seed)
        for step in range(max_steps):
            if step % 2 == 0:
                frame = env.render()
                if isinstance(frame, np.ndarray) and frame.ndim == 3:
                    frames.append(frame)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    finally:
        env.close()
    if not frames:
        raise RuntimeError("Environment returned no RGB frames")
    path = ARTIFACT_DIR / filename
    imageio.mimsave(path, frames, duration=1 / 15, loop=0)
    return str(path)


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
        yield status_card("running", copy_for(language)["running"], f"{trained:,}/{budget:,} steps", language), metric_card(f"{mean:.1f}", "3-episode evaluation reward", language), learning_figure(xs, rewards, f"{experiment} evaluation reward", "Mean reward"), gr.skip(), None, console_panel("\n".join(logs), language)
    slug = re.sub(r"[^a-z0-9]+", "-", f"{env_id}-{algorithm}".lower()).strip("-"); model_path = ARTIFACT_DIR / slug; model.save(model_path); env.close()
    try:
        gif = record_model(model, env_id, seed + 10000, f"{slug}-trained.gif", 500 if env_id in {"CartPole-v1", "Acrobot-v1"} else 999)
        preview_kind = "replay GIF"
    except Exception as exc:
        gif = result_preview_image(experiment, "Training complete", f"{rewards[-1]:.1f}", "Final evaluation reward", x=xs, y=rewards, note="Training succeeded. Replay is unavailable in this CPU container.", algorithm=algorithm)
        preview_kind = "result image"
        logs.append(elapsed_line(started, "WARN", "training_succeeded=true replay_unavailable=headless renderer is not available in this CPU container"))
    summary = save_summary(slug, {"experiment": experiment, "evaluation_steps": xs, "evaluation_rewards": rewards, "model": str(model_path.with_suffix('.zip')), "parameters": {"budget": budget, "learning_rate": alpha, "gamma": gamma, "epsilon": epsilon, "seed": seed}})
    logs.append(elapsed_line(started, "DONE", f"preview={preview_kind} path={gif} model={model_path}.zip artifact={summary}"))
    yield status_card("complete", copy_for(language)["complete"], f"{budget:,} steps · {time.perf_counter() - started:.1f}s", language), metric_card(f"{rewards[-1]:.1f}", "final evaluation reward", language), learning_figure(xs, rewards, f"{experiment} evaluation reward", "Mean reward"), gif, summary, console_panel("\n".join(logs), language)


def error_figure(title: str, message: str, heading: str = "Run stopped"):
    fig, ax = plt.subplots(figsize=(8.2, 4.0)); ax.axis("off")
    ax.text(.5, .62, heading, ha="center", va="center", fontsize=20, fontweight="bold", color="#27324a")
    ax.text(.5, .43, title, ha="center", va="center", fontsize=13, color="#5b5ce2")
    ax.text(.5, .25, message[:180], ha="center", va="center", fontsize=10, color="#68748a", wrap=True)
    fig.tight_layout(); return fig


def run_catalog_experiment(experiment: str, budget: int, alpha: float, gamma: float, epsilon: float, seed: int, language: str):
    env_id = catalog_env_id(experiment); started = time.perf_counter()
    logs = [f"{env_id} automatic training console", "=" * 72, elapsed_line(started, "REGISTER", f"environment={env_id} family={experiment.split(' · ', 1)[0]}"), elapsed_line(started, "CONFIG", f"budget={budget} learning_rate={alpha:g} gamma={gamma:g} epsilon={epsilon:g} seed={seed}")]
    yield status_card("running", copy_for(language)["running"], "Inspecting environment and action space", language), metric_card("AUTO", "selecting a compatible baseline", language), error_figure(env_id, "Inspecting environment and action space...", "Preparing environment"), gr.skip(), None, console_panel("\n".join(logs), language)
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
        yield status_card("running", copy_for(language)["running"], f"Auto selected {algorithm}", language), metric_card(algorithm, f"action space: {action_space}", language), error_figure(env_id, f"Initializing the {algorithm} model...", "Starting training"), gr.skip(), None, console_panel("\n".join(logs), language)
        for status, metric, curve, preview, artifact, console in run_deep_control(experiment, env_id, algorithm, budget, alpha, gamma, epsilon, seed, language):
            deep_text = re.search(r'<pre class="console-text">(.*?)</pre>', console, re.DOTALL)
            combined = "\n".join(logs) + ("\n\n" + html.unescape(deep_text.group(1)) if deep_text else "")
            yield status, metric, curve, preview, artifact, console_panel(combined, language)
    except Exception as exc:
        if env is not None:
            env.close()
        message = f"{type(exc).__name__}: {exc}"; logs.append(elapsed_line(started, "ERROR", message)); logs.append(elapsed_line(started, "HINT", "All maintained runtimes are preinstalled. This registered ID may require a retired legacy engine; choose its current environment version."))
        summary = save_summary(env_id, {"experiment": experiment, "environment": env_id, "status": "registered-but-unavailable", "error": message, "parameters": {"budget": budget, "learning_rate": alpha, "gamma": gamma, "epsilon": epsilon, "seed": seed}})
        diagnostic = result_preview_image(experiment, "Diagnostic", "LEGACY", "environment status", note=message)
        yield status_card("idle", "Legacy environment", "Choose the current maintained version", language), metric_card("LEGACY", "see the latest log lines", language), error_figure(env_id, message), diagnostic, summary, console_panel("\n".join(logs), language)


def train(experiment: str, budget: float, alpha: float, gamma: float, epsilon: float, seed: float, language: str):
    budget, seed = int(budget), int(seed)
    try:
        if experiment not in EXPERIMENT_CHOICES:
            raise ValueError("This environment is not registered in the current runtime. Refresh the page and choose an available task.")
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
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        diagnostic = result_preview_image(experiment, "Run stopped", "ERROR", "training result", note=message)
        summary = save_summary(experiment, {"experiment": experiment, "status": "failed", "error": message, "parameters": {"budget": budget, "learning_rate": alpha, "gamma": gamma, "epsilon": epsilon, "seed": seed}})
        logs = [f"{experiment} training console", "=" * 72, elapsed_line(time.perf_counter(), "ERROR", message), "RESULT  A diagnostic preview and JSON summary were produced."]
        yield status_card("idle", "Training stopped", "Diagnostic result produced", language), metric_card("ERROR", "see the latest log lines", language), error_figure(experiment, message), diagnostic, summary, console_panel("\n".join(logs), language)


def slider_update(label: str, spec: tuple[float, float, float, float], visible: bool = True):
    minimum, maximum, value, step = spec
    return gr.Slider(minimum=minimum, maximum=maximum, value=value, step=step, label=label, visible=visible)


def select_experiment(experiment: str, language: str):
    copy = copy_for(language); cfg = experiment_config(experiment)
    return (
        hero_html(language, experiment),
        task_brief(experiment, language),
        slider_update(copy["budget"], cfg["budget"]),
        slider_update(copy["alpha"], cfg["alpha"]),
        slider_update(copy["gamma"], cfg["gamma"], cfg["gamma_visible"]),
        slider_update(copy["epsilon"], cfg["epsilon"], cfg["algorithm"] not in {"PPO", "SAC"}),
        status_card("idle", copy["ready"], copy["ready_detail"], language),
        metric_card("—", copy["metric_waiting"], language),
        console_panel(copy["log_waiting"], language),
        example_preview(experiment),
        None,
    )


def switch_language(language: str, experiment: str, seed: float):
    copy = copy_for(language); cfg = experiment_config(experiment)
    return (
        hero_html(language, experiment), panel_html(copy["settings"], copy["settings_copy"]), task_brief(experiment, language),
        slider_update(copy["budget"], cfg["budget"]), slider_update(copy["alpha"], cfg["alpha"]), slider_update(copy["gamma"], cfg["gamma"], cfg["gamma_visible"]), slider_update(copy["epsilon"], cfg["epsilon"], cfg["algorithm"] not in {"PPO", "SAC"}),
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
.hero-topline{display:flex;align-items:center;gap:11px;margin-bottom:22px}.experiment-badge{padding:6px 11px;border:1px solid #fff;border-radius:999px;color:#25265d;background:#fff;box-shadow:0 4px 12px rgba(8,15,35,.16);font-size:12px;font-weight:800;letter-spacing:.06em}.hero-course{color:#b9c0d4;font-size:13px;font-weight:650}
.hero h1{max-width:760px;margin:0 0 12px;color:#fff;font-size:clamp(32px,5vw,48px);line-height:1.1;letter-spacing:-.035em}.hero-copy{max-width:760px;margin:0;color:#cdd3e2;font-size:15px;line-height:1.7}.hero-links{display:flex;flex-wrap:wrap;gap:9px;margin-top:25px}.hero-link{display:inline-flex;align-items:center;min-height:38px;padding:0 14px;border:1px solid rgba(255,255,255,.18);border-radius:9px;color:#eef2ff!important;background:rgba(255,255,255,.08);font-size:13px;font-weight:650;text-decoration:none!important}.hero-link.primary{color:#172554!important;background:#fff;border-color:#fff}
.lab-strip{display:flex;flex-wrap:wrap;gap:8px 22px;margin:17px 0 22px;padding:13px 18px;border:1px solid var(--line);border-radius:13px;background:#fff;color:var(--muted);font-size:13px;box-shadow:0 6px 20px rgba(18,25,43,.035)}.lab-strip strong{margin-left:5px;color:var(--ink)}
.catalog-card{margin:0 0 18px!important;padding:22px!important;border:1px solid var(--line)!important;border-radius:17px!important;background:#fff!important;box-shadow:0 10px 30px rgba(18,25,43,.045)!important}.catalog-tools{align-items:end!important}.catalog-family{min-width:420px!important}.catalog-search{min-width:280px!important}.catalog-meta{color:var(--muted);font-size:12px;font-weight:700}.catalog-pager{justify-content:flex-end!important;gap:8px!important}.catalog-pager button{max-width:110px!important;border-radius:9px!important}.experiment-gallery{max-height:720px;overflow:auto;padding:4px!important}.experiment-gallery .grid-wrap{gap:12px!important}.experiment-gallery button,.experiment-gallery .thumbnail-item{overflow:hidden!important;border:1px solid var(--line)!important;border-radius:14px!important;background:#fff!important;box-shadow:0 7px 18px rgba(18,25,43,.045)!important;transition:transform .16s ease,box-shadow .16s ease,border-color .16s ease!important}.experiment-gallery button:hover,.experiment-gallery .thumbnail-item:hover{transform:translateY(-2px);border-color:#a5b4fc!important;box-shadow:0 12px 25px rgba(50,55,120,.12)!important}.experiment-gallery img{aspect-ratio:2/1!important;object-fit:cover!important}.experiment-gallery .caption,.experiment-gallery .label{white-space:pre-line!important;color:var(--ink)!important;font-size:11px!important;line-height:1.45!important;text-align:left!important}.selected-experiment input{font-weight:750!important;color:var(--brand)!important;background:#f5f5ff!important}
.task-brief{display:grid;grid-template-columns:minmax(210px,34%) 1fr;gap:20px;margin:0 0 18px;padding:14px;border:1px solid #dfe3f5;border-radius:15px;background:linear-gradient(135deg,#fafaff,#f6fbff)}.task-brief__visual{display:flex;align-items:center;overflow:hidden;border-radius:11px;background:#171b3f}.task-brief__visual img{display:block;width:100%;height:auto;max-height:250px;min-height:190px;object-fit:contain;border-radius:11px}.task-brief__body{padding:9px 9px 7px}.task-kicker{color:var(--brand);font-size:10px;font-weight:850;letter-spacing:.12em}.task-brief h3{margin:6px 0;color:var(--ink);font-size:23px}.task-brief p{margin:0 0 13px;color:var(--muted);font-size:13px;line-height:1.6}.task-facts{display:grid;grid-template-columns:1fr 1fr;gap:8px}.task-facts span{padding:9px 11px;border:1px solid var(--line);border-radius:9px;background:#fff;color:var(--ink);font-size:11px;overflow-wrap:anywhere}.task-facts b{display:block;margin-bottom:3px;color:#8a94a8;font-size:9px;letter-spacing:.09em;text-transform:uppercase}.task-hint{margin-top:12px!important;margin-bottom:0!important;font-weight:650;color:#4b5563!important}
.control-card,.chart-card,.output-card{border:1px solid var(--line)!important;border-radius:17px!important;background:#fff!important;box-shadow:0 10px 30px rgba(18,25,43,.045)!important}.control-card,.chart-card{padding:22px!important}.output-card{margin-top:16px!important;padding:22px!important}.panel-title{margin:0 0 5px;color:var(--ink);font-size:19px}.panel-copy,.artifact-note{margin:0 0 17px;color:var(--muted);font-size:13px;line-height:1.6}.policy-preview{min-height:360px!important;border:1px solid var(--line)!important;border-radius:13px!important;background:#f8f9fc!important;overflow:hidden!important}.policy-preview .image-container,.policy-preview [data-testid="image"]{min-height:360px!important;background:#f8f9fc!important}.policy-preview img{display:block!important;width:100%!important;height:100%!important;min-height:360px!important;max-height:560px!important;object-fit:contain!important;background:#f8f9fc!important}
.primary-btn{min-height:46px!important;border:0!important;border-radius:11px!important;background:linear-gradient(135deg,#5153d6,#6969ec)!important;font-weight:750!important}.primary-btn:disabled{opacity:.8!important;cursor:wait!important}.run-wait{position:relative;display:grid;grid-template-columns:auto 1fr;gap:12px;align-items:center;overflow:hidden;margin:0 0 16px;padding:14px 16px 17px;border:1px solid #c7d2fe;border-radius:13px;background:linear-gradient(135deg,#f5f5ff,#f0f7ff);box-shadow:0 8px 24px rgba(79,70,229,.08)}.run-wait__spinner{width:24px;height:24px;border:3px solid #d9ddff;border-top-color:#5b5ce2;border-radius:50%;animation:run-spin .8s linear infinite}.run-wait__copy strong,.run-wait__copy small{display:block}.run-wait__copy strong{color:#292d65;font-size:13px}.run-wait__copy small{margin-top:4px;color:#68748a;font-size:11px;line-height:1.5}.run-wait__elapsed{display:inline-block;margin-top:7px;color:#5b5ce2;font-size:11px;font-style:normal;font-weight:750}.run-wait__pulse{position:absolute;right:0;bottom:0;left:0;height:3px;background:#e0e7ff}.run-wait__pulse i{display:block;width:38%;height:100%;border-radius:999px;background:linear-gradient(90deg,transparent,#6366f1,#22c55e,transparent);animation:run-pulse 1.4s ease-in-out infinite}@keyframes run-spin{to{transform:rotate(360deg)}}@keyframes run-pulse{0%{transform:translateX(-110%)}100%{transform:translateX(285%)}}.run-state,.live-metric{display:flex;gap:12px;margin-top:14px;padding:14px 15px;border-radius:13px;background:#f8f9fc}.run-state__dot{width:9px;height:9px;margin-top:6px;border-radius:50%;background:#94a3b8}.run-state--running .run-state__dot{background:#5b5ce2;box-shadow:0 0 0 5px rgba(91,92,226,.13);animation:run-dot 1.2s ease-in-out infinite}@keyframes run-dot{50%{box-shadow:0 0 0 9px rgba(91,92,226,.04)}}.run-state--complete .run-state__dot{background:#13a36f}.run-state strong,.run-state small,.summary-label{display:block}.summary-label{color:#8a94a8;font-size:10px;font-weight:800;letter-spacing:.1em;text-transform:uppercase}.run-state strong{margin-top:3px;color:var(--ink);font-size:14px}.run-state small,.live-metric small{margin-top:3px;color:var(--muted);font-size:12px}.metric-reading{display:flex;align-items:baseline;gap:9px;margin-top:4px}.metric-reading strong{color:var(--ink);font-size:24px}
.console-panel{overflow:hidden;margin-top:18px;border:1px solid #202b3d;border-radius:13px;background:#0f1623}.console-head{display:flex;align-items:center;gap:9px;padding:11px 15px;border-bottom:1px solid #263244;color:#e2e8f0;font-size:12px;font-weight:750}.console-dot{width:8px;height:8px;border-radius:50%;background:#22c55e;box-shadow:0 0 0 4px rgba(34,197,94,.12)}.console-text{box-sizing:border-box;height:300px;margin:0;padding:17px 18px;overflow:auto;white-space:pre;color:#cbd5e1!important;background:#0f1623!important;font:12px/1.58 "SFMono-Regular",Consolas,monospace!important;scrollbar-gutter:stable}.footer-note{margin-top:18px;text-align:center;color:#94a3b8;font-size:12px}.footer-note a{color:var(--brand)!important;text-decoration:none!important;font-weight:650}
@media(max-width:760px){.gradio-container{padding:12px 10px 30px!important}.language-bar{top:14px!important;right:14px!important}.language-switch{width:196px!important;min-width:196px!important}.hero{padding:70px 22px 25px;border-radius:19px}.hero-topline{align-items:flex-start;flex-direction:column}.project-mark{max-width:70%}.catalog-family,.catalog-search{min-width:100%!important}.experiment-gallery{max-height:580px}.task-brief{grid-template-columns:1fr}.task-brief__visual img{min-height:160px}.task-facts{grid-template-columns:1fr}.policy-preview,.policy-preview .image-container,.policy-preview [data-testid="image"],.policy-preview img{min-height:230px!important}.policy-preview img{max-height:420px!important}}
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
    const timer = document.querySelector(".run-wait__elapsed");
    if (timer) {
      const elapsed = Math.max(0, Math.floor((Date.now() - Number(timer.dataset.startMs)) / 1000));
      timer.textContent = `${elapsed}s ${timer.dataset.label}`;
    }
    requestAnimationFrame(() => { internal = false; });
  };
  const schedule = () => { if (!scheduled) { scheduled = true; requestAnimationFrame(update); } };
  new MutationObserver(schedule).observe(document.body, {childList:true, subtree:true, characterData:true});
  setInterval(schedule, 1000);
  schedule();
}
"""


DEFAULT_LANGUAGE = "English"
DEFAULT_EXPERIMENT = CARTPOLE_PPO
copy = copy_for(DEFAULT_LANGUAGE)
cfg = EXPERIMENTS[DEFAULT_EXPERIMENT]
initial_cards, initial_visible, initial_page, initial_meta = catalog_page("", "Curated", 0)

with gr.Blocks(title="Hands-On Modern RL · Gymnasium CPU Playground") as demo:
    with gr.Column(elem_classes="hero-stack"):
        hero = gr.HTML(hero_html(DEFAULT_LANGUAGE, DEFAULT_EXPERIMENT))
        with gr.Row(elem_classes="language-bar"):
            language = gr.Radio(choices=[("English", "English"), ("中文", "中文")], value=DEFAULT_LANGUAGE, show_label=False, elem_classes="language-switch")

    with gr.Column(elem_classes="catalog-card"):
        gr.HTML('<h2 class="panel-title">Choose an experiment</h2><p class="panel-copy">Click a visual card. Search or filter the full registry when you want to explore beyond the curated set.</p>')
        with gr.Row(elem_classes="catalog-tools"):
            search = gr.Textbox(label="Search environments", placeholder="Try Pong, robot, lander, continuous...", scale=2, elem_classes="catalog-search")
            family = gr.Radio(choices=["All"] + FAMILY_ORDER, value="Curated", label="Category", scale=3, elem_classes="catalog-family")
        gallery = gr.Gallery(value=initial_cards, label=None, columns=4, rows=3, object_fit="cover", height="auto", allow_preview=False, elem_classes="experiment-gallery")
        visible_experiments = gr.State(initial_visible)
        catalog_page_state = gr.State(initial_page)
        with gr.Row(elem_classes="catalog-pager"):
            catalog_meta = gr.Markdown(initial_meta, elem_classes="catalog-meta")
            previous_page = gr.Button("← Previous", size="sm")
            next_page = gr.Button("Next →", size="sm")

    task_info = gr.HTML(task_brief(DEFAULT_EXPERIMENT, DEFAULT_LANGUAGE))

    with gr.Row():
        with gr.Column(scale=1, min_width=310, elem_classes="control-card"):
            settings_header = gr.HTML(panel_html(copy["settings"], copy["settings_copy"]))
            experiment = gr.Textbox(value=DEFAULT_EXPERIMENT, label="Selected experiment", interactive=False, elem_classes="selected-experiment")
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
            wait_state = gr.HTML(value="", visible=False)
            curve = gr.Plot(show_label=False)
            console = gr.HTML(console_panel(copy["log_waiting"], DEFAULT_LANGUAGE), elem_id="live-training-console")

    with gr.Row(elem_classes="output-card"):
        with gr.Column(scale=2):
            preview_header = gr.HTML(panel_html(copy["preview"], copy["preview_copy"], "artifact-note"))
            preview = gr.Image(value=example_preview(DEFAULT_EXPERIMENT), show_label=False, interactive=False, elem_classes="policy-preview")
        with gr.Column(scale=1):
            artifact = gr.File(label=copy["artifact"], interactive=False)

    gr.HTML(footer_html())

    search.change(reset_catalog, inputs=[search, family], outputs=[gallery, visible_experiments, catalog_page_state, catalog_meta], queue=False)
    family.change(reset_catalog, inputs=[search, family], outputs=[gallery, visible_experiments, catalog_page_state, catalog_meta], queue=False)
    previous_page.click(lambda q, f, p: move_catalog(q, f, p, -1), inputs=[search, family, catalog_page_state], outputs=[gallery, visible_experiments, catalog_page_state, catalog_meta], queue=False)
    next_page.click(lambda q, f, p: move_catalog(q, f, p, 1), inputs=[search, family, catalog_page_state], outputs=[gallery, visible_experiments, catalog_page_state, catalog_meta], queue=False)
    gallery.select(choose_card, inputs=[visible_experiments], outputs=[experiment], queue=False)
    experiment.change(select_experiment, inputs=[experiment, language], outputs=[hero, task_info, budget, alpha, gamma, epsilon, status, metric, console, preview, artifact], queue=False)
    language.change(switch_language, inputs=[language, experiment, seed], outputs=[hero, settings_header, task_info, budget, alpha, gamma, epsilon, seed, start, status, metric, chart_header, console, preview_header, artifact], queue=False)
    run_event = start.click(begin_run, inputs=[language], outputs=[wait_state, start], queue=False)
    run_event = run_event.then(train, inputs=[experiment, budget, alpha, gamma, epsilon, seed, language], outputs=[status, metric, curve, preview, artifact, console], concurrency_limit=1)
    run_event.then(finish_run, inputs=[language], outputs=[wait_state, start], queue=False)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(css=CSS, js=AUTO_SCROLL_JS)
