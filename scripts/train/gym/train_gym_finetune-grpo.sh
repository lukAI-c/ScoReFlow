#!/bin/bash
# ScoReFlow GRPO Fine-tuning (Critic-Free)
#
# 用法:
#   TASK=kitchen bash train_gym_finetune-grpo.sh
#   TASK=kitchen SEED=100 KL_COEF=0.04 bash train_gym_finetune-grpo.sh

TASK=${TASK:-kitchen}
SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda:0}
SIM_DEVICE=${SIM_DEVICE:-cuda:0}
OFFLINE_WANDB=${OFFLINE_WANDB:-false}
KL_COEF=${KL_COEF:-0.04}

echo "=========================================================="
echo "  ScoReFlow GRPO Fine-tuning (Critic-Free)"
echo "  TASK=${TASK}  SEED=${SEED}  DEVICE=${DEVICE}"
echo "  kl_coef=${KL_COEF}  wandb.offline=${OFFLINE_WANDB}"
echo "=========================================================="

if [ "${TASK}" = "kitchen" ]; then
    MUJOCO_GL="egl" xvfb-run -a -s "-screen 0 1024x768x24" python run.py \
        --config-dir=cfg/gym/finetune/kitchen-complete-v0 \
        --config-name=ft_grpo_shortcut_mlp \
        base_policy_path=${REINFLOW_LOG_DIR}/gym/pretrain/kitchen-complete-v0_pre_shortcut_mish_mlp_ta4_td4/2025-05-05_19-59-19_42/checkpoint/state_1500.pt \
        device=${DEVICE} \
        sim_device=${SIM_DEVICE} \
        wandb.offline_mode=${OFFLINE_WANDB} \
        train.kl_coef=${KL_COEF} \
        seed=${SEED}
else
    echo "Unknown TASK=${TASK}. Currently supports: kitchen"
    exit 1
fi

echo "GRPO fine-tuning finished (TASK=${TASK})."
