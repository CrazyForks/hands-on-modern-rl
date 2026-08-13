# code_reward.py
# veRL 代码生成 RLVR 的 reward 函数：把生成的代码当独立程序跑 stdin/stdout 测试。
#
# 背景（issue #53）：
#   原文档假设 Eurus-2-RL-Data 里有 tests（Python assert 语句）字段，可以直接 exec。
#   实际数据集的 code 样本里没有 tests / entry_point，reward_model.ground_truth 是
#   JSON 字符串 {"inputs": [...], "outputs": [...]}（stdin/stdout 测试对）。
#   所以这里改成：提取代码 -> 写入临时文件 -> 用 subprocess 以真实子进程执行，
#   对每个 input 喂入 stdin，把 stdout 和期望 output 比对，返回通过率。
#
# veRL 接口（verl/workers/reward_manager/naive.py）：
#   score = self.compute_score(data_source=..., solution_str=..., ground_truth=..., extra_info=...)
#   ground_truth 来自数据集 reward_model["ground_truth"]。
#   返回 dict 时以 "score" 作为主奖励，其余 key 会作为日志附加信息。
#
# 用法：
#   python code_reward.py            # 自检：跑几个构造用例验证 reward 逻辑
#
# 训练时通过 verl 配置接入：
#   custom_reward_function.path=.../code_reward.py
#   custom_reward_function.name=compute_score

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)
_TIMEOUT_S = 10.0
_MAX_OUTPUT_BYTES = 100_000
_MAX_MEMORY_BYTES = 2 * 1024**3
_SANDBOX_OPT_IN = "HOMRL_ALLOW_UNSAFE_CODE_EXECUTION"
_RUNNER_SOURCE = r"""
import math
import resource
import runpy
import sys


def set_soft_limit(kind, requested):
    _, current_hard = resource.getrlimit(kind)
    limit = requested if current_hard == resource.RLIM_INFINITY else min(
        requested, current_hard
    )
    resource.setrlimit(kind, (limit, current_hard))


solution_path = sys.argv[1]
timeout_s = float(sys.argv[2])
max_output_bytes = int(sys.argv[3])
max_memory_bytes = int(sys.argv[4])
set_soft_limit(resource.RLIMIT_CPU, max(1, math.ceil(timeout_s)))
set_soft_limit(resource.RLIMIT_FSIZE, max_output_bytes)
set_soft_limit(resource.RLIMIT_CORE, 0)
set_soft_limit(resource.RLIMIT_NOFILE, 64)
if sys.platform.startswith("linux") and hasattr(resource, "RLIMIT_AS"):
    set_soft_limit(resource.RLIMIT_AS, max_memory_bytes)

sys.argv = [solution_path]
runpy.run_path(solution_path, run_name="__main__")
"""


def extract_code(response: str) -> str:
    """从模型输出中提取 Python 代码块。

    模型通常输出类似：
        "```python\ndef solve():\n    ...```"
    我们只取 ```python 和 ``` 之间的部分。
    如果模型没有用代码块格式输出，则把整个回答当作代码（兜底，通常会导致语法错误，reward=0）。
    """
    match = _CODE_BLOCK_RE.search(response)
    if match:
        return match.group(1).strip()
    return response.strip()


def _normalize(text: str) -> str:
    """去掉行尾空白后再比较，避免 \r 或多余空行导致的误判。"""
    return "\n".join(line.rstrip() for line in text.strip().splitlines()).strip()


def _require_execution_opt_in() -> None:
    """阻止用户误把本地子进程当作安全沙箱。"""
    if os.environ.get(_SANDBOX_OPT_IN) != "1":
        raise RuntimeError(
            "拒绝直接执行模型生成的代码。subprocess 不是安全沙箱；"
            f"请先在容器/虚拟机中隔离网络、凭据和训练文件，再设置 "
            f"{_SANDBOX_OPT_IN}=1。"
        )


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """终止测试进程及其派生的同组子进程。"""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def run_io_tests(code: str, ground_truth_json: str, timeout_s: float = _TIMEOUT_S):
    """把 code 作为独立程序运行，用 ground_truth 里的 inputs/outputs 测试。

    返回 (pass_rate, 前几个测试的详细结果)。任何异常（语法错误、运行崩溃、
    超时、输出不匹配）都只影响对应用例，不会中断整个打分。
    """
    try:
        tests = json.loads(ground_truth_json)
    except (TypeError, json.JSONDecodeError) as exc:
        return 0.0, f"ground_truth 解析失败: {exc!r}"

    inputs = tests.get("inputs", [])
    outputs = tests.get("outputs", [])
    if not inputs or len(inputs) != len(outputs):
        return 0.0, f"inputs/outputs 数量不匹配: {len(inputs)} vs {len(outputs)}"

    _require_execution_opt_in()

    # 子进程只隔离解释器状态，不隔离文件系统、网络、凭据或资源。
    # 外层必须先把训练放进最小权限的容器或虚拟机。
    tmp_dir = tempfile.mkdtemp(prefix="homrl-code-reward-")
    tmp_path = Path(tmp_dir) / "solution.py"
    runner_path = Path(tmp_dir) / "runner.py"
    tmp_path.write_text(code, encoding="utf-8")
    runner_path.write_text(_RUNNER_SOURCE, encoding="utf-8")

    try:
        passed = 0
        details = []
        for inp, expected in zip(inputs, outputs):
            try:
                stdout_path = Path(tmp_dir) / "stdout.txt"
                with stdout_path.open("wb") as stdout_file:
                    proc = subprocess.Popen(
                        [
                            sys.executable,
                            "-I",
                            str(runner_path),
                            str(tmp_path),
                            str(timeout_s),
                            str(_MAX_OUTPUT_BYTES),
                            str(_MAX_MEMORY_BYTES),
                        ],
                        stdin=subprocess.PIPE,
                        stdout=stdout_file,
                        stderr=subprocess.DEVNULL,
                        text=True,
                        cwd=tmp_dir,
                        start_new_session=True,
                    )
                    try:
                        proc.communicate(input=inp, timeout=timeout_s)
                    except subprocess.TimeoutExpired:
                        _terminate_process_group(proc)
                        proc.communicate()
                        details.append("FAIL(超时)")
                        continue
                if proc.returncode != 0:
                    details.append("FAIL(非零退出)")
                    continue
                stdout = stdout_path.read_text(
                    encoding="utf-8", errors="replace"
                )
                got = _normalize(stdout)
                want = _normalize(expected)
                if got == want:
                    passed += 1
                    details.append("PASS")
                else:
                    details.append("FAIL(输出不匹配)")
            except Exception as exc:  # noqa: BLE001
                details.append(f"FAIL({exc!r})")
        return passed / len(inputs), "; ".join(details[:5])
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """veRL reward 入口函数。

    Args:
        data_source: 数据集来源（本实验中是 codecontests/taco/apps/codeforces）
        solution_str: 模型生成的完整回答（markdown 文本）
        ground_truth: 数据集 reward_model["ground_truth"]，code 样本是 I/O 测试的 JSON 字符串
        extra_info: 数据集 extra_info 列（本数据集只有 index/split，未使用）

    Returns:
        dict: {"score": pass_rate, "pass_rate": pass_rate, "format": 是否提取到代码}
        veRL 以 "score" 作为 PPO 的主奖励。
    """
    # format 只表示是否用 ```python 代码块输出了代码。
    # 没按格式输出时，extract_code 会把整个回答兜底当代码跑（通常会语法错误，score=0），
    # 但 format 指标应该如实反映「模型是否学会了输出代码块」。
    match = _CODE_BLOCK_RE.search(solution_str)
    format_ok = 1.0 if match else 0.0
    code = extract_code(solution_str)
    if not code:
        return {"score": 0.0, "pass_rate": 0.0, "format": 0.0}

    pass_rate, detail = run_io_tests(code, ground_truth)
    return {"score": pass_rate, "pass_rate": pass_rate, "format": format_ok}


# ---------------------------------------------------------------------------
# 自检：不需要训练环境，直接验证 reward 逻辑
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _require_execution_opt_in()
    # 用一个简单的 A+B 问题构造 ground_truth（inputs/outputs）
    ab_gt = json.dumps({"inputs": ["1 2", "10 20", "-3 5"], "outputs": ["3", "30", "2"]})

    correct = "```python\nimport sys\n\n\nfor line in sys.stdin:\n    a, b = map(int, line.split())\n    print(a + b)\n```"
    wrong = "```python\nimport sys\n\n\nfor line in sys.stdin:\n    a, b = map(int, line.split())\n    print(a - b)\n```"
    no_code = "I don't know how to solve this."

    for name, resp in [("正确代码", correct), ("错误代码", wrong), ("无代码", no_code)]:
        result = compute_score("synthetic", resp, ab_gt, None)
        print(f"{name:8s} -> score={result['score']:.2f} pass_rate={result['pass_rate']:.2f} "
              f"format={result['format']:.0f}")
