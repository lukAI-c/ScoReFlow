#!/bin/bash

# Franka Kitchen Pre-training Script for ReinFlow
# This script pre-trains a ReinFlow policy on the Kitchen-Complete-v0 dataset.

# Configuration:
# --config-dir: Path to the configuration directory for Franka Kitchen.
# --config-name: 'pre_reflow_mlp' for ReinFlow pre-training.

echo "Starting Franka Kitchen Pre-training..."

MUJOCO_GL="egl" xvfb-run -a -s "-screen 0 1024x768x24" python script/run.py \
    --config-dir=cfg/gym/pretrain/kitchen-complete-v0 \
    --config-name=pre_reflow_mlp \
    device=cuda:1 \
    +sim_device=cuda:1 \
    +wandb.offline_mode=true \

echo "Pre-training finished."