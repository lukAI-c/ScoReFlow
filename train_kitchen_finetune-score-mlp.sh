#!/bin/bash

# Gym Fine-tuning Script for ScoReFlow with Learned EpsNet (ShortCut)
# SDE: dxt = [vt + α(t)·st] dt + √(2·α(t)) dWt
# α(t) learned by MLP, drift and diffusion coupled

echo "Starting Gym Fine-tuning (ShortCut + EpsNet)..."

MUJOCO_GL="egl" xvfb-run -a -s "-screen 0 1024x768x24" python script/run.py \
    --config-dir=cfg/gym/finetune/kitchen-complete-v0 \
    --config-name=ft_ppo_shortcut_mlp_score_mlp \
    base_policy_path=${REINFLOW_LOG_DIR}/gym/pretrain/kitchen-complete-v0_pre_shortcut_mish_mlp_ta4_td4/2025-05-05_19-59-19_42/checkpoint/state_1500.pt \
    device=cuda:0 \
    sim_device=cuda:0 \
    wandb.offline_mode=false \
    denoising_steps=4 \
    gamma_score=1.0 \
    epsilon_t=0.1 \
    eps_mlp_hidden_dim=32 \
    eps_mlp_type=mlp \
    eps_max_clamp=2.0 \
    seed=618 \

echo "Fine-tuning finished."

# seed 0 42 3407 2026 618