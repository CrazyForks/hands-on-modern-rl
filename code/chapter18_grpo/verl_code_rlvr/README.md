# 16.8 动手：veRL 代码生成 RL 实验

本目录是《动手学现代强化学习》[16.8 节](../../../docs/chapter18_grpo/verl-code-sandbox.md)的配套代码：**用 veRL 在代码生成任务上跑通 PPO 训练**。

代码题和数学题一样有一个关键优势：答案不是靠人打分，而是**可以运行测试来验证**。能通过测试就给正奖励，不能就低奖励——这就是"硬反馈"。本节要让模型学会写真正能跑通的程序。

## 数据：Eurus-2-RL-Data 到底长什么样

这个数据集容易让人困惑（见 [issue #53](https://github.com/walkinglabs/hands-on-modern-rl/issues/53)）。它**不是** HumanEval 那种带 `entry_point` / `tests` 字段的格式，真实结构是：

| 字段           | 含义                                                                                                                                                     |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prompt`       | chat 消息列表。`system` 是 PRIME 推理动作模板（`[ASSESS]`/`[ADVANCE]`…），`user` 才是题目                                                                |
| `ability`      | `"math"` 或 `"code"`，本实验只取 `code`（train 25,276 条 / val 1,024 条）                                                                                |
| `reward_model` | `{"ground_truth": <答案>, "style": "rule"}`。code 样本的 `ground_truth` 是 JSON 字符串 `{"inputs": [...], "outputs": [...]}`，即 **stdin/stdout 测试对** |
| `data_source`  | 来源：`codecontests` / `taco` / `apps` / `codeforces`                                                                                                    |
| `extra_info`   | `{index, split}`                                                                                                                                         |

也就是说，这些是**"读 stdin、写 stdout"的竞赛编程题**，不是"实现某个函数签名"的题。验证信息藏在 `reward_model.ground_truth` 里，veRL 会在训练时把它传给 reward 函数。

## 三步把它跑起来

### 1. 准备数据

把数据集过滤出 code 样本、重建 prompt、采样 1000 条，生成 veRL 需要的 parquet：

```bash
conda activate test
pip install datasets pandas pyarrow
python prepare_data.py
# 输出: ~/data/eurus2/train1000.parquet（1000 条）和 validation.parquet（1024 条）
```

### 2. 验证 reward 函数（不需要 GPU）

`code_reward.py` 是核心逻辑，可以独立自检：

```bash
HOMRL_ALLOW_UNSAFE_CODE_EXECUTION=1 python code_reward.py
# 正确代码 -> score=1.00 pass_rate=1.00 format=1
# 错误代码 -> score=0.00 pass_rate=0.00 format=1
# 无代码   -> score=0.00 pass_rate=0.00 format=0
```

### 3. 开始训练（需要 GPU）

```bash
chmod +x run_qwen_coder_ppo_single_gpu.sh
./run_qwen_coder_ppo_single_gpu.sh
```

在 1 张 GPU 上用 48 条样本跑 8 步 PPO 的实测输出（验证集 acc 从 0 开始随训练上升）：

```
step:2   critic/score/mean:0.15        # 训练集出现通过测试的代码
step:8   val-core/apps/acc/mean@1:0.147
         val-core/codeforces/acc/mean@1:0.153
```

## 关键设计讲解

### reward：为什么是 I/O 测试，而不是 assert

HumanEval 风格的 reward 会把测试写成 `assert two_sum(...) == ...` 然后 `exec`。但 Eurus-2-RL-Data 的 code 样本没有这种测试，`ground_truth` 是一组 **stdin/stdout 输入输出对**。所以 reward 的做法是：

1. 从模型回答里提取 ```python 代码块
2. 把代码写进临时 `.py` 文件
3. 用 `subprocess` 起独立进程，对每个输入喂入 stdin，比对 stdout 和期望输出
4. 返回通过率作为 reward

`subprocess` 能隔离解释器状态，并模拟真实的 stdin/stdout 程序运行；它**不是安全沙箱**。子进程仍能访问当前用户可见的文件、网络和环境变量。训练前必须先把整个训练任务放进最小权限的容器或虚拟机，并清除凭据、挂载只读数据目录。确认外层隔离完成后，再设置 `HOMRL_ALLOW_UNSAFE_CODE_EXECUTION=1` 启用本地执行器。

### prompt：为什么必须是 chat 消息格式

veRL 的 RLHFDataset 会把 `prompt` 交给模型的 `apply_chat_template`。**如果 `prompt` 是纯字符串，Qwen 的模板会直接丢弃内容**，只生成 system + assistant 两个特殊 token（实测只有 24 个 token），模型根本看不到题目，reward 恒为 0。

所以 `prepare_data.py` 把 prompt 重建为 chat 格式：

```
[{"role": "system", "content": "You are a competitive programming assistant."},
 {"role": "user",   "content": "Read the problem below and write a Python solution...\n\nProblem:\n{problem}"}]
```

### verl 接口：compute_score 的签名

veRL 的 RewardManager 用固定签名调用 reward 函数（见 `verl/workers/reward_manager/naive.py`）：

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """返回 {"score": pass_rate, "pass_rate": pass_rate, "format": 是否提取到代码}"""
```

- `ground_truth` 来自数据集 `reward_model["ground_truth"]`（不需要你自己传）
- 返回 dict 时 veRL 以 `"score"` 作为 PPO 主奖励，其余 key 作为日志

训练脚本里必须用 `custom_reward_function` 把它接进 verl，否则 reward 不会生效：

```bash
REWARD=(
    reward_model.enable=False
    custom_reward_function.path="$(pwd)/code_reward.py"
    custom_reward_function.name=compute_score
)
```

## 文件说明

| 文件                               | 作用                                                                           |
| ---------------------------------- | ------------------------------------------------------------------------------ |
| `prepare_data.py`                  | 下载 Eurus-2-RL-Data → 过滤 code 样本 → 重建 chat 格式 prompt → 采样 → parquet |
| `code_reward.py`                   | I/O 型 reward：提取代码 → 子进程跑 stdin/stdout 测试 → 返回通过率              |
| `run_qwen_coder_ppo_single_gpu.sh` | 单卡 0.5B PPO 启动脚本（含 `custom_reward_function` 接线）                     |

## 环境提示

跑这个实验需要一台有 GPU 的机器，并装好 veRL（含 vLLM rollout 依赖）。两个容易踩的坑：

- **verl 0.9 需要 `transfer_queue`**：`pip install git+https://github.com/Ascend/TransferQueue.git`，否则启动时 `import transfer_queue` 会直接报错。
- **多卡机器上要指定 GPU**：通过 `CUDA_VISIBLE_DEVICES`（或 `HIP_VISIBLE_DEVICES`）选一张卡跑，避免和别的任务抢显存。
