#!/bin/bash

# Franka Kitchen Fine-tuning Script for ReinFlow
# This script fine-tunes a ReinFlow policy on the Kitchen-Complete-v0 environment.

# Configuration:
# --config-dir: Path to the configuration directory.
# --config-name: 'ft_ppo_reflow_mlp' for PPO fine-tuning of ReinFlow.

# echo "Starting Franka Kitchen Fine-tuning..."

MUJOCO_GL="egl" xvfb-run -a -s "-screen 0 1024x768x24" python script/run.py \
    --config-dir=cfg/gym/finetune/kitchen-partial-v0 \
    --config-name=ft_ppo_shortcut_mlp_score \
    base_policy_path=${REINFLOW_LOG_DIR}/gym/pretrain/kitchen-partial-v0_pre_shortcut_mlp_ta4_td20/2025-05-08_03-15-13_42/state_2600.pt \
    device=cuda:0 \
    sim_device=cuda:0 \
    wandb.offline_mode=false \
    gamma_score=1 \
    denoising_steps=4 \
    epsilon_t=0.05 \
    seed=2026 \
    train.ent_coef=0.01 \
    # env.save_video=true \
    # train.render.num=5 \
    # train.render.freq=10



# complete epsilon_t=0.01-0.1
# mixed 0.1
# partial epsilon_t= 0.0
# echo "Fine-tuning finished."

echo "Starting Franka Kitchen Fine-tuning..."

# MUJOCO_GL="egl" xvfb-run -a -s "-screen 0 1024x768x24" python script/run.py \
#     --config-dir=cfg/gym/finetune/kitchen-mixed-v0 \
#     --config-name=ft_ppo_shortcut_mlp_score \
#     base_policy_path=${REINFLOW_LOG_DIR}/gym/pretrain/kitchen-mixed-v0_pre_shortcut_mlp_ta4_td20/2025-05-08_03-11-00_42/checkpoint/state_2400.pt \
#     device=cuda:0 \
#     sim_device=cuda:1 \
#     wandb.offline_mode=false \
#     gamma_score=1 \
#     denoising_steps=4 \
#     epsilon_t=0.01 \
#     seed=128 \
#     # train.ent_coef=0.001 \
#     # env.save_video=true \
#     # train.render.num=5 \
#     # train.render.freq=10


echo "Fine-tuning finished."


# MUJOCO_GL="egl" xvfb-run -a -s "-screen 0 1024x768x24" python script/run.py \
#     --config-dir=cfg/gym/finetune/kitchen-partial-v0 \
#     --config-name=ft_ppo_shortcut_mlp_score \
#     base_policy_path=${REINFLOW_LOG_DIR}/gym/pretrain/kitchen-partial-v0_pre_shortcut_mlp_ta4_td20/2025-05-08_03-15-13_42/state_2600.pt \
#     device=cuda:0 \
#     sim_device=cuda:0 \
#     wandb.offline_mode=false \
#     gamma_score=1 \
#     denoising_steps=4 \
#     epsilon_t=0.05 \
#     seed=128 \
#     # train.ent_coef=0.001 \
#     # env.save_video=true \
#     # train.render.num=5 \
#     # train.render.freq=10