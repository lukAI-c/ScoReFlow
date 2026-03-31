#!/bin/bash

# Robomimic Fine-tuning Script for ScoReFlow
# Transport task - Constant Alpha Ablation (PPOShortCutWithScoreConst)
#
# 消融对比:
#   alpha_constant=0.0  → α(t)≡0，无 score 引导，纯 NoisyShortCut
#   alpha_constant=1.0  → α(t)≡1，常数 score 强度，无 time_mask 衰减
#
# 用法:
#   bash train_robomimic_finetune-score-gammanet-const.sh            # 默认 alpha=0 (cfg 默认值)
#   ALPHA=0.0 bash train_robomimic_finetune-score-gammanet-const.sh  # 显式 alpha=0
#   ALPHA=1.0 bash train_robomimic_finetune-score-gammanet-const.sh  # alpha=1

echo "Starting Robomimic Fine-tuning (Transport, Constant Alpha Ablation)..."
# 隔离出物理 GPU 3 和 4。此时程序眼里的 cuda:0 就是物理卡 3，cuda:1 就是物理卡 4
export CUDA_VISIBLE_DEVICES=4,5
# 设置 EGL 渲染默认卡，防止 Mujoco 将渲染上下文强行挂载到物理 0 卡
export EGL_DEVICE_ID=5
export MUJOCO_EGL_DEVICE_ID=5

# alpha_constant 默认值：未设置则用 0.0
ALPHA=${ALPHA:-0.0}

python script/run.py \
    --config-dir=cfg/robomimic/finetune/transport \
    --config-name=ft_ppo_shortcut_mlp_img_with_score_gammanet_const \
    base_policy_path=${REINFLOW_LOG_DIR}/robomimic/pretrain/transport/ShortCut/state_750.pt \
    device=cuda:0 \
    sim_device=cuda:1 \
    env.n_envs=50 \
    wandb.offline_mode=true \
    alpha_constant=${ALPHA} \
    gamma_score=1.0 \
    denoising_steps=4 \
    seed=42 \

echo "Fine-tuning finished."
