#!/bin/bash

# OpenAI Gym Pre-training Script for ScoRe-Flow
# Pre-trains a ReinFlow policy on the D4RL dataset.

echo "Starting OpenAI Gym Pre-training..."

python script/run.py \
    --config-dir=cfg/gym/pretrain/hopper-medium-v2 \
    --config-name=pre_reflow_mlp \
    device=cuda:0 \
    +wandb.offline_mode=true

echo "Pre-training finished."

