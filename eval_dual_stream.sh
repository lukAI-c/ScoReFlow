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

echo "============================================================"
echo "Dual-Stream Score Editing Evaluation"
echo "============================================================"
# echo "Checkpoint: ${CHECKPOINT_PATH}"
# echo "CFG Weight: ${CFG_WEIGHT}"
# echo "Device: ${DEVICE}"
echo "============================================================"

python script/run.py \
    --config-dir=cfg/gym/eval/kitchen-complete-v0 \
    --config-name=eval_dual_stream_mlp \
    base_policy_path=${REINFLOW_LOG_DIR}/gym/finetune/kitchen-complete-v0_ppo_reflow_mlp_dual_stream_ta4_td4_cfg2.0/2025-12-19_05-09-27_seed42/checkpoint/best.pt \
    device=cuda:0 \
    +render_onscreen=false \
    eval_mode=true \
    load_ema=false \


echo "Evaluation finished."

