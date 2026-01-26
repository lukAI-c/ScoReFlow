#!/bin/bash

# Robomimic Pre-training Script for ScoRe-Flow (Image-based)

echo "Starting Robomimic Pre-training..."

python script/run.py \
    --config-dir=cfg/robomimic/pretrain/square \
    --config-name=pre_reflow_mlp_img \
    device=cuda:0 \
    wandb.offline_mode=true

echo "Pre-training finished."

