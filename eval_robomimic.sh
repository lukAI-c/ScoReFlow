#!/bin/bash

# Robomimic Evaluation Script for ReinFlow (Image-based)
# This script evaluates a trained ReinFlow policy on the Robomimic Square task.

echo "Starting Robomimic Evaluation..."
python script/run.py \
    --config-dir=cfg/robomimic/eval/square \
    --config-name=eval_reflow_mlp_img_with_score_gammanet \
    device=cuda:0 \
    +render_onscreen=false \
    env.save_video=true \
    +record_video=true \
    render_num=5 \
    load_ema=false \
    denoising_step_list=[4] \
    env.n_envs=5 \
    base_policy_path=${REINFLOW_LOG_DIR}/robomimic/finetune/square_ft_flow_mlp_img_with_score_gammanet_ta4_td4_tdf1/2026-01-20_02-50-55_3407/checkpoint/best.pt \

echo "Evaluation finished."