#!/bin/bash

# Gym Fine-tuning Script for ScoReFlow - Constant Alpha Ablation
#
# 支持任务:
#   humanoid  → FlowMLP  + PPOFlowWithScoreConst  (ppoflow_with_score_alphanet_const)
#   kitchen   → ShortCut + PPOShortCutWithScoreConst (pposhortcut_with_score_alphanet_const)
#
# 消融对比:
#   ALPHA=0.0  → α(t)≡0，无 score 引导，纯 NoisyFlow/NoisyShortCut
#   ALPHA=1.0  → α(t)≡1，常数 score 强度，无 time_mask 衰减
#
# 用法:
#   TASK=humanoid ALPHA=0.0 bash train_gym_finetune-score-alphanet-const.sh
#   TASK=humanoid ALPHA=1.0 bash train_gym_finetune-score-alphanet-const.sh
#   TASK=kitchen  ALPHA=0.0 bash train_gym_finetune-score-alphanet-const.sh
#   TASK=kitchen  ALPHA=1.0 bash train_gym_finetune-score-alphanet-const.sh

# 隔离出物理 GPU 3 和 4。此时程序眼里的 cuda:0 就是物理卡 3，cuda:1 就是物理卡 4
# export CUDA_VISIBLE_DEVICES=2,3
# # 设置 EGL 渲染默认卡，防止 Mujoco 将渲染上下文强行挂载到物理 0 卡
# export EGL_DEVICE_ID=3
# export MUJOCO_EGL_DEVICE_ID=3

TASK=${TASK:-humanoid}
ALPHA=${ALPHA:-0.0}

echo "Starting Gym Fine-tuning (task=${TASK}, alpha_constant=${ALPHA})..."

# ── Humanoid ──────────────────────────────────────────────────────────
if [ "${TASK}" = "humanoid" ]; then
    MUJOCO_GL="egl" xvfb-run -a -s "-screen 0 1024x768x24" python run.py \
        --config-dir=cfg/gym/finetune/Humanoid-v3 \
        --config-name=ft_ppo_reflow_mlp_with_score_alphanet_const \
        base_policy_path=${REINFLOW_LOG_DIR}/gym/pretrain/Humanoid-medium-v3_pre_reflow_mlp_ta4_td20_seed42/2025-12-26_03-33-44_42/checkpoint/best.pt \
        device=cuda:0 \
        sim_device=cuda:0 \
        wandb.offline_mode=false \
        alpha_constant=${ALPHA} \
        gamma_score=1.0 \
        denoising_steps=4 \
        seed=3407

# ── Kitchen-complete ──────────────────────────────────────────────────
elif [ "${TASK}" = "kitchen" ]; then
    MUJOCO_GL="egl" xvfb-run -a -s "-screen 0 1024x768x24" python run.py \
        --config-dir=cfg/gym/finetune/kitchen-complete-v0 \
        --config-name=ft_ppo_shortcut_mlp_with_score_alphanet_const \
        base_policy_path=${REINFLOW_LOG_DIR}/gym/pretrain/kitchen-complete-v0_pre_shortcut_mish_mlp_ta4_td4/2025-05-05_19-59-19_42/checkpoint/state_1500.pt \
        device=cuda:0 \
        sim_device=cuda:0 \
        wandb.offline_mode=false \
        alpha_constant=${ALPHA} \
        gamma_score=1.0 \
        denoising_steps=4 \
        seed=3407

else
    echo "Unknown TASK=${TASK}. Use: humanoid | kitchen"
    exit 1
fi

echo "Fine-tuning finished."
