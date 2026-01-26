#!/bin/bash

# OpenAI Gym Fine-tuning Script (With Score GammaNet)
# Fine-tunes using score-based RL with learnable gamma network.

# Pre-trained model paths:
# walker2d: ${REINFLOW_LOG_DIR}/gym/pretrain/walker2d-medium-v2_pre_reflow_mlp_ta4_td20_seed42/2025-12-06_03-05-56_42/checkpoint/last.pt
# humanoid: ${REINFLOW_LOG_DIR}/gym/pretrain/Humanoid-medium-v3_pre_reflow_mlp_ta4_td20_seed42/2025-12-26_03-33-44_42/checkpoint/best.pt
# hopper ${REINFLOW_LOG_DIR}/gym/pretrain/hopper-v2/ReFlow/2025-02-06_01-35-03_D4RL_42/state_40.pt
# ant: ${REINFLOW_LOG_DIR}/gym/pretrain/ant-medium-expert-v2_pre_reflow_mlp_ta4_td20_seed42/2025-12-26_07-49-07_42/checkpoint/last.pt


echo "Starting OpenAI Gym Fine-tuning (With Score)..."

MUJOCO_GL="egl" xvfb-run -a -s "-screen 0 1024x768x24" python script/run.py \
    --config-dir=cfg/gym/finetune/hopper-v2 \
    --config-name=ft_ppo_reflow_mlp_with_score_gammanet \
    base_policy_path=${REINFLOW_LOG_DIR}/gym/pretrain/hopper-v2/ReFlow/2025-02-06_01-35-03_D4RL_42/state_40.pt \
    device=cuda:0 \
    sim_device=cuda:0 \
    wandb.offline_mode=true \
    gamma_score=1 \
    seed=3407

echo "Fine-tuning finished."

