#!/bin/bash

# OpenAI Gym Pre-training Script for ReinFlow
# This script pre-trains a ReinFlow policy on the Walker2d-Medium-v2 dataset.

# Configuration:
# --config-dir: Path to the configuration directory for the specific environment and dataset.
# --config-name: Name of the configuration file (without .yaml). 'pre_reflow_mlp' uses the ReinFlow algorithm.
# device: Computation device (e.g., cuda:0).
# sim_device: Simulation rendering device (e.g., cuda:0).

echo "Starting OpenAI Gym Pre-training..."

python run.py \
    --config-dir=cfg/gym/pretrain/hopper-medium-v2 \
    --config-name=pre_reflow_mlp \
    device=cuda:0 \
    +wandb.offline_mode=true \


echo "Pre-training finished."
