#!/bin/bash
# ============================================================
# Kitchen-Partial-v0 策略评估脚本
# 用于评估训练好的策略并生成可视化视频
# ============================================================

# ==================== 配置参数 ====================
# 模型路径 (必须修改为你的实际路径)

# 环境名称
ENV_NAME="kitchen-partial-v0"

# 去噪步数列表
DENOISING_STEPS="[1,2,4]"

# 并行环境数量 (录制视频时建议设为1)
N_ENVS=1

# 是否加载 EMA 权重 (Fine-tuned 模型用 False, 预训练模型用 True)
LOAD_EMA="False"

# ==================== 运行评估 ====================
echo "=============================================="
echo "开始评估 Kitchen-Partial-v0 策略"
echo "模型路径: ${MODEL_PATH}"
echo "去噪步数: ${DENOISING_STEPS}"
echo "并行环境数: ${N_ENVS}"
echo "=============================================="

python run.py \
    --config-dir=cfg/gym/eval/kitchen-complete-v0 \
    --config-name=eval_shortcut_mlp \
    env_name=${ENV_NAME} \
    env.n_envs=${N_ENVS} \
    base_policy_path=${REINFLOW_LOG_DIR}/gym/finetune/kitchen-complete-v0_ppo_shortcut_mlp_ta4_td4_tdf4/2026-01-14_09-08-37_seed42/checkpoint/best.pt \
    denoising_step_list=${DENOISING_STEPS} \
    load_ema=${LOAD_EMA} \
    +record_video=true \
    +frame_width=640 \
    +frame_height=480

echo "=============================================="
echo "评估完成!"
echo "结果保存在 logs/gym/eval/ 目录下"
echo "=============================================="

