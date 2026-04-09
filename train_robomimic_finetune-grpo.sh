#!/bin/bash
# ScoReFlow GRPO Robomimic Fine-tuning (Critic-Free)
#
# 支持任务:
#   square    → ShortCut + GRPOShortCutWithScore (image, 1-step)
#   transport → ShortCut + GRPOShortCutWithScore (image, 4-step)
#
# 用法:
#   TASK=square    bash train_robomimic_finetune-grpo.sh
#   TASK=transport bash train_robomimic_finetune-grpo.sh
#   TASK=square SEED=100 KL_COEF=0.1 bash train_robomimic_finetune-grpo.sh


# 隔离出物理 GPU 3 和 4。此时程序眼里的 cuda:0 就是物理卡 3，cuda:1 就是物理卡 4
export CUDA_VISIBLE_DEVICES=3,4
# 设置 EGL 渲染默认卡，防止 Mujoco 将渲染上下文强行挂载到物理 0 卡
export EGL_DEVICE_ID=4
export MUJOCO_EGL_DEVICE_ID=4

TASK=${TASK:-square}
SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda:0}
SIM_DEVICE=${SIM_DEVICE:-cuda:1}
OFFLINE_WANDB=${OFFLINE_WANDB:-true}
KL_COEF=${KL_COEF:-0.2}

echo "=========================================================="
echo "  ScoReFlow GRPO Robomimic Fine-tuning (Critic-Free)"
echo "  TASK=${TASK}  SEED=${SEED}  DEVICE=${DEVICE}  SIM_DEVICE=${SIM_DEVICE}"
echo "  kl_coef=${KL_COEF}  wandb.offline=${OFFLINE_WANDB}"
echo "=========================================================="

# ── Square (1-step denoising) ────────────────────────────────────
if [ "${TASK}" = "square" ]; then
    python script/run.py \
        --config-dir=cfg/robomimic/finetune/square \
        --config-name=ft_grpo_reflow_mlp_img_with_score_gammanet \
        base_policy_path=${REINFLOW_LOG_DIR}/robomimic/pretrain/square/square_pre_reflow_mlp_img_ta4_td100/2025-12-10_08-22-54_42/checkpoint/last.pt \
        device=${DEVICE} \
        sim_device=${SIM_DEVICE} \
        wandb.offline_mode=${OFFLINE_WANDB} \
        train.kl_coef=${KL_COEF} \
        train.ent_coef=0.01 \
        denoising_steps=2 \
        env.n_envs=50 \
        seed=${SEED}

# ── Transport (4-step denoising, dual camera) ────────────────────
# elif [ "${TASK}" = "transport" ]; then
#     python script/run.py \
#         --config-dir=cfg/robomimic/finetune/transport \
#         --config-name=ft_grpo_shortcut_mlp_img_with_score_gammanet \
#         base_policy_path=${REINFLOW_LOG_DIR}/robomimic/pretrain/transport/ShortCut/state_750.pt \
#         device=${DEVICE} \
#         sim_device=${SIM_DEVICE} \
#         wandb.offline_mode=${OFFLINE_WANDB} \
#         train.kl_coef=${KL_COEF} \
#         denoising_steps=4 \
#         env.n_envs=50 \
#         seed=${SEED}

elif [ "${TASK}" = "transport" ]; then
    python script/run.py \
        --config-dir=cfg/robomimic/finetune/transport \
        --config-name=ft_grpo_shortcut_mlp_img_with_score_gammanet \
        base_policy_path=${REINFLOW_LOG_DIR}/robomimic/pretrain/transport/ShortCut/state_750.pt \
        device=${DEVICE} \
        sim_device=${SIM_DEVICE} \
        wandb.offline_mode=${OFFLINE_WANDB} \
        train.kl_coef=${KL_COEF} \
        train.ent_coef=0.01 \
        denoising_steps=4 \
        env.n_envs=50 \
        seed=${SEED}

else
    echo "Unknown TASK=${TASK}. Choose from: square | transport"
    exit 1
fi

echo "GRPO fine-tuning finished (TASK=${TASK})."
