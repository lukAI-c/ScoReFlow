#!/bin/bash

# OpenAI Gym Fine-tuning Script for ReinFlow
# This script fine-tunes a pre-trained ReinFlow policy on the Walker2d-v2 environment using PPO.

# Configuration:
# --config-dir: Path to the configuration directory for the specific environment.
# --config-name: Name of the configuration file. 'ft_ppo_reflow_mlp' uses PPO to fine-tune ReinFlow.
# device: Computation device.
# sim_device: Simulation rendering device.

# Note: Ensure you have a pre-trained checkpoint or let the script download/use the default one if configured.

echo "Starting OpenAI Gym Fine-tuning..."

python script/run.py \
    --config-dir=cfg/gym/finetune/walker2d-v2 \
    --config-name=ft_ppo_reflow_mlp_score \
    base_policy_path=${REINFLOW_LOG_DIR}/gym/pretrain/walker2d-medium-v2_pre_reflow_mlp_ta4_td20_seed42/2025-12-10_13-16-15_42/checkpoint/last.pt\
    device=cuda:0 \

echo "Fine-tuning finished."
