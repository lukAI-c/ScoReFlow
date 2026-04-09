#!/bin/bash

###################################################################
# DPPO (Diffusion PPO) Training Script for Square Environment
# 基于 DDPM 的 PPO 微调方法
#
# 核心特点:
# - 使用 DDPM (Denoising Diffusion Probabilistic Model) 作为策略
# - PPO 算法进行在线强化学习微调
# - 支持图像观测 (Vision-based)
###################################################################

echo "=========================================="
echo "Starting DPPO (Diffusion PPO) Training"
echo "Environment: Square (Robomimic)"
echo "=========================================="

# 渲染后端设置 (根据系统选择)
# Linux with GPU: export MUJOCO_GL="egl"
# Linux without GPU: export MUJOCO_GL="osmesa"
# Windows/Mac: 通常不需要设置
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"

# 环境变量设置 (如果未设置则使用默认值)
export DPPO_LOG_DIR="${DPPO_LOG_DIR:-./logs}"
export DPPO_DATA_DIR="${DPPO_DATA_DIR:-./data}"
export DPPO_WANDB_ENTITY="${DPPO_WANDB_ENTITY:-default}"

# GPU 设置
DEVICE="cuda:0"
SEED=42

# 预训练模型路径 (需要先完成预训练)
# BASE_POLICY_PATH="${DPPO_LOG_DIR}/robomimic/pretrain/square/square_pre_diffusion_mlp_img_ta4_td100/checkpoint/state_4000.pt"

# 运行训练
python run.py \
    --config-dir=cfg/robomimic/finetune/square \
    --config-name=ft_ppo_diffusion_mlp_img \
    device=${DEVICE} \
    seed=${SEED} \
    +wandb.offline_mode=true \
    env.n_envs=50 \
    train.n_train_itr=301 \
    train.n_critic_warmup_itr=2 \
    train.n_steps=400 \
    train.gamma=0.999 \
    train.actor_lr=1e-5 \
    train.critic_lr=1e-3 \
    train.update_epochs=10 \
    train.batch_size=500 \
    train.val_freq=10 \
    train.save_model_freq=100

echo "=========================================="
echo "DPPO Training Finished!"
echo "=========================================="

