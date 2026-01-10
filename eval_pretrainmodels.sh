#!/bin/bash

# --- 1. 设置路径变量 ---
PROJECT_ROOT="/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_clk/ReinFlow"
CHECKPOINT_PATH="${PROJECT_ROOT}/log/gym/pretrain/hopper-medium-v2_pre_reflow_mlp_ta4_td20_seed42/2025-12-26_03-30-46_42/checkpoint/best.pt"
GPU_ID=0

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

# --- 3. 运行命令 ---
# env.n_envs=1 : 强制单进程，防止 XIO error
# MESA_... : 伪装显卡驱动版本

python script/run.py \
    --config-dir="cfg/gym/eval/hopper-medium-v2" \
    --config-name="eval_reflow_mlp" \
    base_policy_path="$CHECKPOINT_PATH" \
    denoising_step_list="[1,2,4,5,8,16,32,64,128]" \
    load_ema=false \
    device="cuda:${GPU_ID}" \
    env.n_envs=1 \
