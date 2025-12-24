#!/bin/bash

# Robomimic Pre-training Script for ReinFlow (Image-based)
# This script pre-trains a ReinFlow policy on the Robomimic Square dataset using image observations.

# Configuration:
# --config-dir: Path to the configuration directory for Robomimic Square task.
# --config-name: 'pre_reflow_mlp_img' for image-based ReinFlow pre-training.

echo "Starting Robomimic Pre-training..."

python script/run.py \
    --config-dir=cfg/robomimic/pretrain/square \
    --config-name=pre_reflow_mlp_img \
    device=cuda:0 \

echo "Pre-training finished."
