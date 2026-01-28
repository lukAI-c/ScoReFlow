#!/bin/bash

# Robomimic Evaluation Script for ReinFlow (Image-based)
# This script evaluates a trained ReinFlow policy on the Robomimic Square task.
# eval list
# square：base_policy_path=${REINFLOW_LOG_DIR}/robomimic/pretrain/square/square_pre_reflow_mlp_img_ta4_td100/2025-12-10_08-22-54_42/checkpoint/last.pt \
# can：${REINFLOW_LOG_DIR}/robomimic/pretrain/can/can_pre_reflow_mlp_img_ta4_td100/2025-12-29_14-07-21_42/checkpoint/last.pt 
# transport: base_policy_path=${REINFLOW_LOG_DIR}/robomimic/pretrain/transport/ShortCut/state_750.pt \

echo "Starting Robomimic Evaluation..."
python script/run.py \
    --config-dir=cfg/robomimic/eval/square \
    --config-name=eval_reflow_mlp_img \
    device=cuda:0 \
    +render_onscreen=false \
    env.save_video=true \
    +record_video=true \
    render_num=1 \
    load_ema=true \
    denoising_step_list=[4] \
    env.n_envs=50 \
    base_policy_path=${REINFLOW_LOG_DIR}/robomimic/pretrain/square/square_pre_reflow_mlp_img_ta4_td100/2025-12-10_08-22-54_42/checkpoint/last.pt \
    seed=3407 \

echo "Evaluation finished."