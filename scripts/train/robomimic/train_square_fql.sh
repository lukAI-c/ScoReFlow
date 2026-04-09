#!/bin/bash

###################################################################
# FQL (Flow Q-Learning) Training Script for Square Environment
# 基于 Flow Matching 的 Q-Learning 微调方法
# 
# 核心特点:
# - 使用 ReFlow 作为 BC (Behavior Cloning) 基础策略
# - 一步 Actor 网络用于快速推理
# - Q-Learning 进行离线+在线强化学习
# - 离线预训练 + 在线微调两阶段训练
###################################################################

echo "=========================================="
echo "Starting FQL (Flow Q-Learning) Training"
echo "Environment: Square (Robomimic)"
echo "=========================================="

# 渲染后端设置
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"

# GPU 设置
DEVICE="cuda:0"
SEED=42

# 注意: FQL 需要离线数据集进行预训练
# 确保 ${REINFLOW_DATA_DIR}/robomimic/square/train.npz 存在
# 数据集应包含状态观测 (23维) 而不是图像观测

# 运行训练
python run.py \
    --config-dir=cfg/robomimic/finetune/square \
    --config-name=ft_fql_mlp \
    device=${DEVICE} \
    sim_device=${DEVICE} \
    seed=${SEED} \
    wandb.offline_mode=true \
    env.n_envs=50 \
    # train.offline_steps=200000 \
    # train.online_steps=100000 \
    # train.batch_size=256 \
    # train.actor_lr=1e-4 \
    # train.critic_lr=3e-4 \
    # train.val_freq=5000 \
    # train.n_eval_episode=10 \
    # train.alpha=3.0 \
    # train.target_ema_rate=0.005

echo "=========================================="
echo "FQL Training Finished!"
echo "=========================================="

