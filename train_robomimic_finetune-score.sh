# #!/bin/bash

# # Robomimic Fine-tuning Script for ReinFlow (Image-based)
# echo "Starting Robomimic Fine-tuning with EGL..."

# export MUJOCO_GL="egl"
# export PYOPENGL_PLATFORM="egl"

# 强制指定只让程序看到 0 号卡，避免 EGL 尝试初始化其他不可用的卡
# export CUDA_VISIBLE_DEVICES=0
# export EGL_DEVICE_ID=0

# --- 关键修复步骤 3: 运行命令 ---
# 注意：移除了 xvfb-run，EGL 不需要它
# python script/run.py \
#     --config-dir=cfg/robomimic/finetune/square \
#     --config-name=ft_ppo_reflow_mlp_img_score \
#     base_policy_path=${REINFLOW_LOG_DIR}/robomimic/pretrain/square/square_pre_reflow_mlp_img_ta4_td100/2025-12-10_08-22-54_42/checkpoint/last.pt \
#     device=cuda:0 \
#     sim_device=cuda:1 \
#     wandb.offline_mode=true \
#     env.n_envs=1 \

# echo "Fine-tuning finished."

#!/bin/bash

# Robomimic Fine-tuning Script for ReinFlow (Image-based)
echo "Starting Robomimic Fine-tuning with CPU rendering (OSMesa)..."

export CUDA_VISIBLE_DEVICES=5,6
# 设置 EGL 渲染默认卡，防止 Mujoco 将渲染上下文强行挂载到物理 0 卡
export EGL_DEVICE_ID=6
export MUJOCO_EGL_DEVICE_ID=6

# 2. 运行 Python
python script/run.py \
    --config-dir=cfg/robomimic/finetune/square \
    --config-name=ft_ppo_reflow_mlp_img_score \
    base_policy_path=${REINFLOW_LOG_DIR}/robomimic/pretrain/square/square_pre_reflow_mlp_img_ta4_td100/2025-12-10_08-22-54_42/checkpoint/last.pt \
    device=cuda:3 \
    sim_device=cuda:4 \
    wandb.offline_mode=true \
    env.n_envs=50 \
    denoising_steps=4 \
    gamma_score=1 \
    epsilon_t=0.01 \
    +train.ent_coef_schedule_on=false \
    +train.ent_coef_schedule='linear_decay' \
    +train.ent_coef_start=0.0001 \
    +train.ent_coef_end=0.00001 \
    +train.ent_decay_start_itr=5 \
    +train.ent_decay_end_itr=50 \
    seed=412 \
# +train.use_bc_loss=true \
# +train.bc_loss_type='W2' \
# +train.bc_loss_coeff=0.1                

echo "Fine-tuning finished."
