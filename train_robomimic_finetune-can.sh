#!/bin/bash

# 隔离出物理 GPU 3 和 4。此时程序眼里的 cuda:0 就是物理卡 3，cuda:1 就是物理卡 4
export CUDA_VISIBLE_DEVICES=3,4
# 设置 EGL 渲染默认卡，防止 Mujoco 将渲染上下文强行挂载到物理 0 卡
export EGL_DEVICE_ID=4
export MUJOCO_EGL_DEVICE_ID=4

# 2. 运行 Python
python script/run.py \
    --config-dir=cfg/robomimic/finetune/can \
    --config-name=ft_ppo_reflow_mlp_img_with_score_gammanet \
    base_policy_path=${REINFLOW_LOG_DIR}/robomimic/pretrain/can/can_pre_reflow_mlp_img_ta4_td100/2025-12-29_14-07-21_42/checkpoint/last.pt \
    device=cuda:0 \
    sim_device=cuda:1 \
    wandb.offline_mode=true \
    env.n_envs=50 \
    denoising_steps=4 \
    gamma_score=1 \
    +train.ent_coef_schedule_on=false \
    +train.ent_coef_schedule='linear_decay' \
    +train.ent_coef_start=0.0001 \
    +train.ent_coef_end=0.00001 \
    +train.ent_decay_start_itr=5 \
    +train.ent_decay_end_itr=50 \
    seed=521

echo "Fine-tuning finished."