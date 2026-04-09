#!/bin/bash

# Franka Kitchen Fine-tuning Script for ReinFlow
# This script fine-tunes a ReinFlow policy on the Kitchen-Complete-v0 environment.

# Configuration:
# --config-dir: Path to the configuration directory.
# --config-name: 'ft_ppo_reflow_mlp' for PPO fine-tuning of ReinFlow.

# echo "Starting Franka Kitchen Fine-tuning..."

# MUJOCO_GL="egl" xvfb-run -a -s "-screen 0 1024x768x24" python run.py \
#     --config-dir=cfg/gym/finetune/kitchen-complete-v0 \
#     --config-name=ft_ppo_reflow_mlp_score_distribute \
#     base_policy_path=${REINFLOW_LOG_DIR}/gym/pretrain/kitchen-complete-v0_pre_reflow_mlp_ta4_td20_seed42/2025-12-12_03-46-29_42/checkpoint/best.pt \
#     device=cuda:0 \
#     sim_device=cuda:0 \
#     wandb.offline_mode=false \
#     gamma_score=1 \
#     # env.save_video=true \
#     # train.render.num=5 \
#     # train.render.freq=10

# complete epsilon_t=0.01
# mixed
# partial epsilon_t=
# echo "Fine-tuning finished."

echo "Starting Franka Kitchen Fine-tuning..."

MUJOCO_GL="egl" xvfb-run -a -s "-screen 0 1024x768x24" python run.py \
    --config-dir=cfg/gym/finetune/kitchen-complete-v0 \
    --config-name=ft_ppo_reflow_mlp_score \
    device=cuda:2 \
    sim_device=cuda:3 \
    wandb.offline_mode=true \
    gamma_score=1 \
    epsilon_t=0.1 \
    seed=3407 \
    # env.save_video=true \
    # train.render.num=5 \
    # train.render.freq=10


echo "Fine-tuning finished."