#!/bin/bash

# Franka Kitchen Pre-training Script for ScoRe-Flow

echo "Starting Franka Kitchen Pre-training..."

MUJOCO_GL="egl" xvfb-run -a -s "-screen 0 1024x768x24" python script/run.py \
    --config-dir=cfg/gym/pretrain/kitchen-partial-v0 \
    --config-name=pre_shortcut_mlp \
    device=cuda:0 \
    +sim_device=cuda:0 \
    +wandb.offline_mode=true

echo "Pre-training finished."

