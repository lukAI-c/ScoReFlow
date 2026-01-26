#!/bin/bash

# Robomimic Fine-tuning Script (Score-based)
# square：base_policy_path=${REINFLOW_LOG_DIR}/robomimic/pretrain/square/square_pre_reflow_mlp_img_ta4_td100/2025-12-10_08-22-54_42/checkpoint/last.pt \
# can：${REINFLOW_LOG_DIR}/robomimic/pretrain/can/can_pre_reflow_mlp_img_ta4_td100/2025-12-29_14-07-21_42/checkpoint/last.pt 
# transport: base_policy_path=${REINFLOW_LOG_DIR}/robomimic/pretrain/transport/ShortCut/state_750.pt \
echo "Starting Robomimic Fine-tuning (Score)..."

export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"

python script/run.py \
    --config-dir=cfg/robomimic/finetune/square \
    --config-name=ft_ppo_reflow_mlp_img_score \
    base_policy_path=${REINFLOW_LOG_DIR}/robomimic/pretrain/square/checkpoint/last.pt \
    device=cuda:0 \
    sim_device=null \
    wandb.offline_mode=true \
    env.n_envs=50 \
    denoising_steps=4 \
    gamma_score=1 \
    epsilon_t=0.01 \
    +train.ent_coef_schedule_on=false \
    seed=42

echo "Fine-tuning finished."

