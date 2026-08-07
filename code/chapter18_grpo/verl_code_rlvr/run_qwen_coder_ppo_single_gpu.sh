#!/bin/bash
# run_qwen_coder_ppo_single_gpu.sh
# PPO | Eurus-2 代码生成 | 单卡 | Qwen2.5-Coder-0.5B
#
# 前置步骤：
#   1. 安装 veRL（见文档 16.8「安装 veRL」）
#   2. 准备数据：python prepare_data.py（生成 ~/data/eurus2/train1000.parquet）
#
# 说明（issue #53）：
#   Eurus-2-RL-Data 没有 entry_point / tests 字段，code 样本的测试在
#   reward_model.ground_truth（JSON: {"inputs": [...], "outputs": [...]}）。
#   reward 用 code_reward.py 把模型输出当独立程序跑 stdin/stdout 测试。
#   所以这里通过 custom_reward_function 把 code_reward.py 接进 verl。

set -xeuo pipefail

# ==================== 可调参数 ====================
# Qwen2.5-Coder 是代码专用变体，比通用 Instruct 更适合代码任务
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-Coder-0.5B-Instruct}
CRITIC_MODEL_PATH=${CRITIC_MODEL_PATH:-$MODEL_PATH}  # Critic 从同一模型初始化

# 硬件设置
NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-1}

# 训练参数
# batch_size=128 表示每步从 128 个 prompt 中采样回答
# mini_batch=64 表示 PPO 更新时分 2 个 mini-batch（128/64）
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}

# 序列长度
# 代码任务的 max_response_length 需要比 GSM8K 大（512 vs 256）
# 因为一个函数实现通常比一道数学题的推理过程更长
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}

# 学习率
# Actor lr 比 Critic lr 小一个量级，这是 PPO 的常见实践
# Actor 需要保守更新，Critic 需要快速学会 value function
ACTOR_LR=${ACTOR_LR:-1e-6}
CRITIC_LR=${CRITIC_LR:-1e-5}

# 推理参数
# vLLM 张量并行度，单卡=1
ROLLOUT_TP=${ROLLOUT_TP:-1}
# vLLM 预分配显存比例，单卡需要和训练模型共享显存
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.4}
# 每个 prompt 生成几条回答（PPO 组大小）
ROLLOUT_N=${ROLLOUT_N:-1}

# 训练控制
TOTAL_EPOCHS=${TOTAL_EPOCHS:-20}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-5}

# 数据路径
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/eurus2/train1000.parquet}
VAL_FILE=${VAL_FILE:-$HOME/data/eurus2/validation.parquet}

# reward 函数（和本脚本同目录下的 code_reward.py）
REWARD_FILE=${REWARD_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/code_reward.py}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-coder_ppo_eurus2_$(date +%Y%m%d_%H%M)}
# ==================== 可调参数结束 ====================

# ---- 数据配置 ----
# filter_overlong_prompts=True: 过滤超过 max_prompt_length 的样本
# truncation='error': 超长样本直接报错而不是截断，防止训练数据被静默截断
DATA=(
    algorithm.adv_estimator=gae
    data.train_files="['$TRAIN_FILE']"
    data.val_files="['$VAL_FILE']"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

# ---- 模型配置 ----
# enable_gradient_checkpointing=True: 用时间换显存，单卡必备
# use_remove_padding=True: 去掉 padding token 的冗余计算
MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

# ---- Actor 配置 ----
# clip_ratio=0.2: PPO 标准裁剪范围，限制策略更新幅度
# param_offload=False: 单卡不开启参数卸载（卸载到 CPU 更慢）
ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384
    actor_rollout_ref.actor.clip_ratio=0.2
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

# ---- Rollout 配置 ----
# name=vllm: 使用 vLLM 做 continuous batching 推理
# gpu_memory_utilization=0.4: vLLM 只用 40% 显存，剩下的给训练
ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
)

# ---- Reference 配置 ----
# param_offload=True: Reference 是冻结的，可以卸载到 CPU 省显存
REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

# ---- Critic 配置 ----
# Critic 学习率比 Actor 高一个量级（1e-5 vs 1e-6）
# Critic 需要快速收敛才能给 Actor 提供准确的 advantage 估计
CRITIC=(
    critic.model.path="$CRITIC_MODEL_PATH"
    critic.model.use_remove_padding=True
    critic.model.enable_gradient_checkpointing=True
    critic.optim.lr=${CRITIC_LR}
    critic.use_dynamic_bsz=True
    critic.ppo_max_token_len_per_gpu=16384
    critic.fsdp.param_offload=False
    critic.fsdp.optimizer_offload=False
)

# ---- Reward 配置 ----
# 用 code_reward.py 做规则奖励（跑 stdin/stdout 测试），不训练 Reward Model
# 这是文档里原本漏掉的关键接线：不配 custom_reward_function，reward 根本不会生效
REWARD=(
    reward_model.enable=False
    custom_reward_function.path="$REWARD_FILE"
    custom_reward_function.name=compute_score
)

# ---- Trainer 配置 ----
TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=verl_ppo_code
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
)

# ---- 启动训练 ----
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${CRITIC[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@"
