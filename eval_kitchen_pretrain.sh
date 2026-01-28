#!/bin/bash

# ============================================================
# Dual-Stream Score Editing PPO Flow Evaluation Script
# 
# 用法:
#   ./eval_dual_stream.sh <checkpoint_path> [cfg_weight] [device]
#
# 示例:
#   ./eval_dual_stream.sh /path/to/checkpoint/best.pt 1.5 cuda:0
#
# 参数:
#   checkpoint_path  - 必需，微调后的模型 checkpoint 路径
#   cfg_weight       - 可选，CFG 引导权重 (默认: 1.5)
#   device           - 可选，使用的 GPU 设备 (默认: cuda:0)
# ============================================================

# CHECKPOINT_PATH=${1:?"Error: Please provide checkpoint path as first argument"}
# CFG_WEIGHT=${2:-1.5}
# DEVICE=${3:-cuda:0}

# kitchen-complete: /gym/pretrain/kitchen-complete-v0_pre_reflow_mlp_ta4_td20_seed42/2025-12-12_03-46-29_42/checkpoint/best.pt
# and /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_clk/ReinFlow/log/gym/pretrain/kitchen-complete-v0_pre_shortcut_mish_mlp_ta4_td4/2025-05-05_19-59-19_42/checkpoint/state_1500.pt
# kitchen-mixed: base_policy_path=${REINFLOW_LOG_DIR}/gym/pretrain/kitchen-complete-v0_pre_shortcut_mish_mlp_ta4_td4/2025-05-05_19-59-19_42/checkpoint/state_1500.pt
# and gym/pretrain/kitchen-mixed-v0_pre_shortcut_mlp_ta4_td20/2025-05-08_03-11-00_42/checkpoint/state_2400.pt
# kitchen-partial: /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_clk/ReinFlow/log/gym/pretrain/kitchen-partial-v0_pre_shortcut_mlp_ta4_td20/2025-05-08_03-15-13_42/state_2600.pt
# and ${REINFLOW_LOG_DIR}/gym/pretrain/kitchen-partial-v0_pre_shortcut_mlp_ta4_td20/2025-05-08_03-15-13_42/state_2600.pt \

echo "============================================================"
echo "Dual-Stream Score Editing Evaluation"
echo "============================================================"
# echo "Checkpoint: ${CHECKPOINT_PATH}"
# echo "CFG Weight: ${CFG_WEIGHT}"
# echo "Device: ${DEVICE}"
DENOISING_STEPS="[4]"
echo "============================================================"

python script/run.py \
    --config-dir=cfg/gym/eval/kitchen-complete-v0 \
    --config-name=eval_shortcut_mlp \
    base_policy_path=${REINFLOW_LOG_DIR}/gym/pretrain/kitchen-mixed-v0_pre_shortcut_mlp_ta4_td20/2025-05-08_03-11-00_42/checkpoint/state_2400.pt \
    device=cuda:0 \
    denoising_step_list=$DENOISING_STEPS \
    +render_onscreen=false \
    load_ema=true \
    seed=3407 \

echo "Evaluation finished."

