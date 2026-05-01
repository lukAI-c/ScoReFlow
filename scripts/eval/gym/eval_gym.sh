#!/bin/bash

# Robomimic Evaluation Script for ReinFlow (Image-based)
# This script evaluates a trained ReinFlow policy on the Robomimic Square task.

echo "Starting Robomimic Evaluation..."
python run.py \
    --config-dir=cfg/gym/eval/ant-medium-expert-v0 \
    --config-name=eval_reflow_mlp_score \
    device=cuda:0 \
    +render_onscreen=false \
    env.save_video=true \
    +record_video=true \
    render_num=5 \
    load_ema=false \
    denoising_step_list=[4] \
    env.n_envs=5 \
    base_policy_path=${REINFLOW_LOG_DIR}/gym/finetune/ant-medium-expert-v2_ppo_reflow_mlp_score_ta4_td4/2026-01-19_02-54-31_seed3407/checkpoint/best.pt \

echo "Evaluation finished."