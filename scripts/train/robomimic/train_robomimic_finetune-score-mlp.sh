#!/bin/bash

# Robomimic Fine-tuning Script for ScoReFlow with Learned EpsNet
# SDE: dxt = [vt + α(t)·st] dt + √(2·α(t)) dWt
# α(t) learned by MLP, drift and diffusion coupled

echo "Starting Robomimic Fine-tuning with Learned EpsNet..."

# 隔离出物理 GPU 3 和 4。此时程序眼里的 cuda:0 就是物理卡 3，cuda:1 就是物理卡 4
export CUDA_VISIBLE_DEVICES=0,1
# 设置 EGL 渲染默认卡，防止 Mujoco 将渲染上下文强行挂载到物理 0 卡
export EGL_DEVICE_ID=1
export MUJOCO_EGL_DEVICE_ID=1

# 2. 运行 Python
python run.py \
    --config-dir=cfg/robomimic/finetune/transport \
    --config-name=ft_ppo_shortcut_mlp_img_score_mlp  \
    base_policy_path=${REINFLOW_LOG_DIR}/robomimic/pretrain/transport/ShortCut/state_750.pt \
    device=cuda:0 \
    sim_device=cuda:1 \
    wandb.offline_mode=true \
    env.n_envs=50 \
    denoising_steps=4 \
    gamma_score=1.0 \
    epsilon_t=0.01 \
    eps_mlp_hidden_dim=32 \
    eps_mlp_type=mlp \
    eps_max_clamp=2.0 \
    +train.ent_coef_schedule_on=false \
    +train.ent_coef_schedule='linear_decay' \
    +train.ent_coef_start=0.0001 \
    +train.ent_coef_end=0.00001 \
    +train.ent_decay_start_itr=5 \
    +train.ent_decay_end_itr=50 \
    seed=516 \



# export CUDA_VISIBLE_DEVICES=2,3
# export EGL_DEVICE_ID=2
# export MUJOCO_EGL_DEVICE_ID=2

# python run.py \
#     --config-dir=cfg/robomimic/finetune/square \
#     --config-name=ft_ppo_reflow_mlp_img_score_mlp \
#     base_policy_path=${REINFLOW_LOG_DIR}/robomimic/pretrain/square/square_pre_reflow_mlp_img_ta4_td100/2025-12-10_08-22-54_42/checkpoint/last.pt \
#     device=cuda:0 \
#     sim_device=cuda:1 \
#     wandb.offline_mode=true \
#     env.n_envs=50 \
#     denoising_steps=4 \
#     gamma_score=1.0 \
#     epsilon_t=0.1 \
#     eps_mlp_hidden_dim=32 \
#     eps_mlp_type=mlp \
#     eps_max_clamp=2.0 \
#     +train.ent_coef_schedule_on=false \
#     +train.ent_coef_schedule='linear_decay' \
#     +train.ent_coef_start=0.0001 \
#     +train.ent_coef_end=0.00001 \
#     +train.ent_decay_start_itr=5 \
#     +train.ent_decay_end_itr=50 \
#     seed=128 \

echo "Fine-tuning finished."

# 41 42 128 2025 2026