#!/bin/bash

##############################################
# 评估 Robomimic Square 任务 - Dual-Stream 版本
# 
# 用途：评估微调后的 Dual-Stream 策略在不同推理步数下的性能
##############################################

# ============== 配置参数 ==============

# 策略路径 - 修改为您的实际路径
POLICY_PATH="${REINFLOW_LOG_DIR}/robomimic/finetune/square_dual_stream_safe_cfg_lr_fix_42/2025-12-22_12-44-45/checkpoint/best.pt"

# 或者使用最后一个 checkpoint
# POLICY_PATH="/path/to/your/checkpoint/last.pt"

# 评估的推理步数列表
DENOISING_STEPS="[4]"

# 是否加载 EMA 权重
# True: 用于预训练策略
# False: 用于微调策略
LOAD_EMA=False

# 设备配置
DEVICE="cuda:0"

# 环境数量 (评估时建议设置为 10-50)
N_ENVS=10

# 随机种子
SEED=42

# ============== 运行评估 ==============

echo "=========================================="
echo "评估 Robomimic Square - Dual-Stream PPO Flow"
echo "=========================================="
echo "策略路径: $POLICY_PATH"
echo "推理步数: $DENOISING_STEPS"
echo "加载 EMA: $LOAD_EMA"
echo "设备: $DEVICE"
echo "环境数量: $N_ENVS"
echo "=========================================="

python script/run.py \
    --config-dir=cfg/robomimic/eval/square \
    --config-name=eval_reflow_mlp_img \
    base_policy_path=$POLICY_PATH \
    denoising_step_list=$DENOISING_STEPS \
    load_ema=$LOAD_EMA \
    device=$DEVICE \
    env.n_envs=$N_ENVS \
    seed=$SEED \
    +render_onscreen=false \
    env.save_video=true \
    render_num=5 \

echo "=========================================="
echo "评估完成！"
echo "结果保存在对应的日志目录中"
echo "=========================================="
