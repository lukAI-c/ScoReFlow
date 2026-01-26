#!/bin/bash

# Robomimic Fine-tuning Script (Base)
# square：base_policy_path=${REINFLOW_LOG_DIR}/robomimic/pretrain/square/square_pre_reflow_mlp_img_ta4_td100/2025-12-10_08-22-54_42/checkpoint/last.pt \
# can：${REINFLOW_LOG_DIR}/robomimic/pretrain/can/can_pre_reflow_mlp_img_ta4_td100/2025-12-29_14-07-21_42/checkpoint/last.pt 
# transport: base_policy_path=${REINFLOW_LOG_DIR}/robomimic/pretrain/transport/ShortCut/state_750.pt \
echo "Starting Robomimic Fine-tuning..."

export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"

python script/run.py \
    --config-dir=cfg/robomimic/finetune/square \
    --config-name=ft_ppo_shortcut_mlp_img \
    device=cuda:0 \
    sim_device=null \
    wandb.offline_mode=true \
    env.n_envs=50

echo "Fine-tuning finished."

