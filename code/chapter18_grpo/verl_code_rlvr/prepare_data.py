# prepare_data.py
# 把 PRIME-RL/Eurus-2-RL-Data 处理成 veRL 可以直接用的 parquet 格式。
#
# 背景（issue #53）：
#   Eurus-2-RL-Data 本身没有文档里说的 entry_point / tests / ground_truth 顶层字段。
#   它的真实结构是：
#     - prompt      : 一个 chat 消息数组（[{"role","content"}, ...]），其中 system 消息
#                     是 PRIME 推理的动作模板（[ASSESS]/[ADVANCE]/...），对代码生成没用
#     - ability     : "math" 或 "code"
#     - reward_model: {"ground_truth": <答案/测试>, "style": "rule"}
#                     其中 code 样本的 ground_truth 是 JSON 字符串：
#                     {"inputs": [...], "outputs": [...]}（stdin/stdout 测试对）
#     - data_source / extra_info : 元信息
#   veRL 的 RewardManager 会读数据集里的 reward_model["ground_truth"] 传给 reward 函数，
#   所以这个格式本身就是 veRL 原生格式，训练阶段不需要做字段转换。
#
# 本脚本只做三件事：
#   1. 过滤出 ability == "code" 的样本（25K 条）
#   2. 把 prompt 重建成「只含题目 + 代码生成指令」的纯文本（去掉 PRIME 推理模板）
#   3. train 集按 max_prompt_length 过滤 + 随机采样 1000 条，保存为 veRL parquet
#
# 用法：
#   conda activate test
#   python prepare_data.py

import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATA_DIR = Path.home() / "data" / "eurus2"
TRAIN_OUT = DATA_DIR / "train1000.parquet"
VAL_OUT = DATA_DIR / "validation.parquet"

# 过滤 prompt 超过 512 token 的样本（1 token ≈ 4 字符）
MAX_PROMPT_CHARS = 512 * 4
N_TRAIN_SAMPLES = 1000

# veRL 的 RLHFDataset / AgentLoop 期望 prompt 是 chat 消息列表（list[dict]），
# 会在 rollout 时对它做 apply_chat_template。注意：**不能**用纯文本字符串——
# Qwen 的 apply_chat_template 收到字符串会直接丢弃内容，只生成 system + assistant
# 两个特殊 token（实测只有 24 个 token），导致模型看不到题目、reward 恒为 0。
CODE_GEN_SYSTEM = "You are a competitive programming assistant."
CODE_GEN_USER_TEMPLATE = (
    "Read the problem below and write a Python solution that reads from stdin "
    "and writes to stdout.\n"
    "Return only one Python code block, with no explanations.\n\n"
    "Problem:\n{problem}"
)


def extract_user_content(prompt) -> str:
    """从 chat 消息数组里取出 user 消息（真正的题目）。

    原始 prompt 是 [{"role":"system",...},{"role":"user",...}]，
    其中 system 消息是 PRIME 的动作模板，对代码生成没有意义，这里只保留题目。
    """
    if isinstance(prompt, str):
        return prompt
    # numpy.ndarray / list of dict
    for msg in prompt:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    if isinstance(prompt, (list, tuple)):
        return "\n".join(m.get("content", "") if isinstance(m, dict) else str(m) for m in prompt)
    return str(prompt)


def build_prompt(problem: str) -> list:
    """重建 chat 格式 prompt：system 指令 + user 题目。"""
    return [
        {"role": "system", "content": CODE_GEN_SYSTEM},
        {"role": "user", "content": CODE_GEN_USER_TEMPLATE.format(problem=problem)},
    ]


def prompt_len_chars(prompt) -> int:
    """chat 格式 prompt 的字符长度（用于超长过滤）。"""
    if isinstance(prompt, str):
        return len(prompt)
    return sum(len(m.get("content", "")) for m in prompt)


def process_split(split: str, ds) -> pd.DataFrame:
    df = ds[split].to_pandas()
    code = df[df["ability"] == "code"].copy()
    print(f"[{split}] 总条数={len(df)}，其中 code={len(code)}")

    # 重建 prompt：只保留题目 + 代码生成指令，格式为 chat 消息列表
    code["prompt"] = code["prompt"].map(lambda p: build_prompt(extract_user_content(p)))

    if split == "train":
        before = len(code)
        code = code[code["prompt"].map(prompt_len_chars) < MAX_PROMPT_CHARS]
        print(f"[{split}] 过滤超长 prompt 后: {before} -> {len(code)}")
        n = min(N_TRAIN_SAMPLES, len(code))
        code = code.sample(n=n, random_state=42)
        print(f"[{split}] 随机采样 {n} 条")

    # veRL 原生需要的列：prompt / reward_model / data_source / extra_info
    keep = ["prompt", "reward_model", "data_source", "ability", "extra_info"]
    code = code[keep].reset_index(drop=True)
    return code


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("加载 PRIME-RL/Eurus-2-RL-Data ...")
    ds = load_dataset("PRIME-RL/Eurus-2-RL-Data")

    train = process_split("train", ds)
    train.to_parquet(TRAIN_OUT, index=False)
    print(f"已保存 {len(train)} 条 -> {TRAIN_OUT}")

    val = process_split("validation", ds)
    val.to_parquet(VAL_OUT, index=False)
    print(f"已保存 {len(val)} 条 -> {VAL_OUT}")

    # 自检：确认 reward_model.ground_truth 是可解析的 I/O 测试，prompt 是 chat 格式
    row = val.iloc[0]
    gt = json.loads(row["reward_model"]["ground_truth"])
    assert set(gt) >= {"inputs", "outputs"}, f"ground_truth 缺 inputs/outputs: {list(gt)}"
    assert len(gt["inputs"]) == len(gt["outputs"])
    assert isinstance(row["prompt"], list) and row["prompt"][0]["role"] == "system", \
        f"prompt 应为 chat 消息列表，实际类型: {type(row['prompt'])}"
    print("\n自检通过。示例样本：")
    print("  data_source :", row["data_source"])
    print("  prompt 角色 :", [m["role"] for m in row["prompt"]])
    print("  user 内容开头:", row["prompt"][-1]["content"][:60].replace("\n", "\\n"), "...")
    print("  测试用例数 :", len(gt["inputs"]))


if __name__ == "__main__":
    main()
