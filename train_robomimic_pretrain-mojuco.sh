#!/bin/bash

# Robomimic Pre-training Script for ReinFlow (Image-based)
# This script pre-trains a ReinFlow policy on the Robomimic Square dataset using image observations.

# Configuration:
# --config-dir: Path to the configuration directory for Robomimic Square task.
# --config-name: 'pre_reflow_mlp_img' for image-based ReinFlow pre-training.

echo "Starting Robomimic Pre-training..."

MUJOCO_GL="egl" xvfb-run -a -s "-screen 0 1024x768x24" python script/run.py \
    --config-dir=cfg/robomimic/pretrain/transport\
    --config-name=pre_shortcut_mlp_img \
    device=cuda:0 \
    wandb.offline_mode=true \
    

echo "Pre-training finished."
