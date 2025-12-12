#!/bin/bash

# Robomimic Evaluation Script for ReinFlow (Image-based)
# This script evaluates a trained ReinFlow policy on the Robomimic Square task.

echo "Starting Robomimic Evaluation..."
python script/run.py \
    --config-dir=cfg/robomimic/eval/square \
    --config-name=eval_reflow_mlp_img \
    device=cuda:0 \
    +render_onscreen=false \
    env.save_video=true \
    render_num=5 \
    load_ema=false \
    base_policy_path=${REINFLOW_LOG_DIR}/robomimic/finetune/square_ft_flow_mlp_img_ta4_td1_tdf1/2025-12-10_10-53-04_42/checkpoint/best.pt

echo "Evaluation finished."