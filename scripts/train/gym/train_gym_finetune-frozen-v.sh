#!/bin/bash
# ScoReFlow Gym Fine-tuning  |  Frozen Velocity Field
# 冻结速度场，只训练 variance (noise_net) + score_scheduler (alpha_t)
#
# 支持任务:
#   humanoid  → ReFlow   + PPOFlowWithScoreFrozenV
#   kitchen   → ShortCut + PPOShortCutWithScoreFrozenV
#
# 用法:
#   TASK=humanoid bash train_gym_finetune-frozen-v.sh
#   TASK=kitchen  bash train_gym_finetune-frozen-v.sh
#   TASK=humanoid SEED=100 DEVICE=cuda:1 bash train_gym_finetune-frozen-v.sh

# walker2d: ${REINFLOW_LOG_DIR}/gym/pretrain/walker2d-medium-v2_pre_reflow_mlp_ta4_td20_seed42/2025-12-06_03-05-56_42/checkpoint/last.pt
# humanoid: ${REINFLOW_LOG_DIR}/gym/pretrain/Humanoid-medium-v3_pre_reflow_mlp_ta4_td20_seed42/2025-12-26_03-33-44_42/checkpoint/best.pt
# hopper ${REINFLOW_LOG_DIR}/gym/pretrain/hopper-v2/ReFlow/2025-02-06_01-35-03_D4RL_42/state_40.pt
# ant: ${REINFLOW_LOG_DIR}/gym/pretrain/ant-medium-expert-v2_pre_reflow_mlp_ta4_td20_seed42/2025-12-26_07-49-07_42/checkpoint/last.pt

# kitchen-complete: /gym/pretrain/kitchen-complete-v0_pre_reflow_mlp_ta4_td20_seed42/2025-12-12_03-46-29_42/checkpoint/best.pt
# and /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_clk/ReinFlow/log/gym/pretrain/kitchen-complete-v0_pre_shortcut_mish_mlp_ta4_td4/2025-05-05_19-59-19_42/checkpoint/state_1500.pt
# kitchen-mixed: base_policy_path=${REINFLOW_LOG_DIR}/gym/pretrain/kitchen-complete-v0_pre_shortcut_mish_mlp_ta4_td4/2025-05-05_19-59-19_42/checkpoint/state_1500.pt
# and gym/pretrain/kitchen-mixed-v0_pre_shortcut_mlp_ta4_td20/2025-05-08_03-11-00_42/checkpoint/state_2400.pt
# kitchen-partial: /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_clk/ReinFlow/log/gym/pretrain/kitchen-partial-v0_pre_shortcut_mlp_ta4_td20/2025-05-08_03-15-13_42/state_2600.pt
# and ${REINFLOW_LOG_DIR}/gym/pretrain/kitchen-partial-v0_pre_shortcut_mlp_ta4_td20/2025-05-08_03-15-13_42/state_2600.pt \


TASK=${TASK:-humanoid}
SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda:0}
SIM_DEVICE=${SIM_DEVICE:-cuda:0}
OFFLINE_WANDB=${OFFLINE_WANDB:-false}
GAMMA_SCORE=${GAMMA_SCORE:-1.0}
ALPHA=${ALPHA:-null}

echo "=========================================================="
echo "  ScoReFlow Gym Frozen-V Fine-tuning"
echo "  TASK=${TASK}  SEED=${SEED}  DEVICE=${DEVICE}"
echo "  gamma_score=${GAMMA_SCORE}  alpha_constant=${ALPHA}  wandb.offline=${OFFLINE_WANDB}"
echo "=========================================================="

# ── Humanoid (ReFlow) ─────────────────────────────────────────
if [ "${TASK}" = "humanoid" ]; then
    MUJOCO_GL="egl" xvfb-run -a -s "-screen 0 1024x768x24" python run.py \
        --config-dir=cfg/gym/finetune/Humanoid-v3 \
        --config-name=ft_ppo_reflow_mlp_with_score_gammanet_frozen_v \
        base_policy_path=${REINFLOW_LOG_DIR}/gym/pretrain/Humanoid-medium-v3_pre_reflow_mlp_ta4_td20_seed42/2025-12-26_03-33-44_42/checkpoint/best.pt \
        device=${DEVICE} \
        sim_device=${SIM_DEVICE} \
        wandb.offline_mode=${OFFLINE_WANDB} \
        gamma_score=${GAMMA_SCORE} \
        model.alpha_constant=${ALPHA} \
        denoising_steps=4 \
        seed=${SEED}

# ── Kitchen-complete (ShortCut) ───────────────────────────────
elif [ "${TASK}" = "kitchen" ]; then
    MUJOCO_GL="egl" xvfb-run -a -s "-screen 0 1024x768x24" python run.py \
        --config-dir=cfg/gym/finetune/kitchen-complete-v0 \
        --config-name=ft_ppo_shortcut_mlp_with_score_gammanet_frozen_v \
        base_policy_path=${REINFLOW_LOG_DIR}/gym/pretrain/kitchen-complete-v0_pre_shortcut_mish_mlp_ta4_td4/2025-05-05_19-59-19_42/checkpoint/state_1500.pt \
        device=${DEVICE} \
        sim_device=${SIM_DEVICE} \
        wandb.offline_mode=${OFFLINE_WANDB} \
        gamma_score=${GAMMA_SCORE} \
        model.alpha_constant=${ALPHA} \
        denoising_steps=4 \
        seed=${SEED}

else
    echo "Unknown TASK=${TASK}. Choose from: humanoid | kitchen"
    exit 1
fi

echo "Fine-tuning finished (TASK=${TASK})."
