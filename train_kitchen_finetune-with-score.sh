#!/bin/bash

# Franka Kitchen Fine-tuning Script for ReinFlow
# This script fine-tunes a ReinFlow policy on the Kitchen-Complete-v0 environment.

# Configuration:
# --config-dir: Path to the configuration directory.
# --config-name: 'ft_ppo_reflow_mlp' for PPO fine-tuning of ReinFlow.

# echo "Starting Franka Kitchen Fine-tuning..."

# MUJOCO_GL="egl" xvfb-run -a -s "-screen 0 1024x768x24" python script/run.py \
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


# echo "Fine-tuning finished."
# kitchen-complete: /gym/pretrain/kitchen-complete-v0_pre_reflow_mlp_ta4_td20_seed42/2025-12-12_03-46-29_42/checkpoint/best.pt
# and /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_clk/ReinFlow/log/gym/pretrain/kitchen-complete-v0_pre_shortcut_mish_mlp_ta4_td4/2025-05-05_19-59-19_42/checkpoint/state_1500.pt
# kitchen-mixed: base_policy_path=${REINFLOW_LOG_DIR}/gym/pretrain/kitchen-complete-v0_pre_shortcut_mish_mlp_ta4_td4/2025-05-05_19-59-19_42/checkpoint/state_1500.pt
# and gym/pretrain/kitchen-mixed-v0_pre_shortcut_mlp_ta4_td20/2025-05-08_03-11-00_42/checkpoint/state_2400.pt
# kitchen-partial: /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_clk/ReinFlow/log/gym/pretrain/kitchen-partial-v0_pre_shortcut_mlp_ta4_td20/2025-05-08_03-15-13_42/state_2600.pt
# and ${REINFLOW_LOG_DIR}/gym/pretrain/kitchen-partial-v0_pre_shortcut_mlp_ta4_td20/2025-05-08_03-15-13_42/state_2600.pt \
echo "Starting Franka Kitchen Fine-tuning..."

MUJOCO_GL="egl" xvfb-run -a -s "-screen 0 1024x768x24" python script/run.py \
    --config-dir=cfg/gym/finetune/kitchen-partial-v0 \
    --config-name=ft_ppo_shortcut_mlp_with_score_gammanet_obs \
    base_policy_path=${REINFLOW_LOG_DIR}/gym/pretrain/kitchen-partial-v0_pre_shortcut_mlp_ta4_td20/2025-05-08_03-15-13_42/state_2600.pt \
    device=cuda:0 \
    sim_device=cuda:0 \
    wandb.offline_mode=false \
    denoising_steps=8 \
    gamma_score=1 \
    seed=516 \
    train.n_train_itr=301 \
    train.ent_coef=0.01 \
    
    # env.save_video=true \
    # train.render.num=5 \
    # train.render.freq=10


echo "Fine-tuning finished."