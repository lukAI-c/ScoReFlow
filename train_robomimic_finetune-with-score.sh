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
# square：base_policy_path=${REINFLOW_LOG_DIR}/robomimic/pretrain/square/square_pre_reflow_mlp_img_ta4_td100/2025-12-10_08-22-54_42/checkpoint/last.pt \
# can：${REINFLOW_LOG_DIR}/robomimic/pretrain/can/can_pre_reflow_mlp_img_ta4_td100/2025-12-29_14-07-21_42/checkpoint/last.pt 
# transport: base_policy_path=${REINFLOW_LOG_DIR}/robomimic/pretrain/transport/ShortCut/state_750.pt \
echo "Starting Robomimic Fine-tuning with CPU rendering (OSMesa)..."

# 1. 强制使用 CPU 渲染后端
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"

# 2. 运行 Python
python script/run.py \
    --config-dir=cfg/robomimic/finetune/square \
    --config-name=ft_ppo_reflow_mlp_img_with_score_gammanet_obs \
    base_policy_path=${REINFLOW_LOG_DIR}/robomimic/pretrain/square/square_pre_reflow_mlp_img_ta4_td100/2025-12-10_08-22-54_42/checkpoint/last.pt \
    device=cuda:5 \
    sim_device=cuda:6 \
    sim_device=null \
    wandb.offline_mode=true \
    env.n_envs=50 \
    denoising_steps=2 \
    gamma_score=1 \
    +train.ent_coef_schedule_on=false \
    +train.ent_coef_schedule='linear_decay' \
    +train.ent_coef_start=0.0001 \
    +train.ent_coef_end=0.00001 \
    +train.ent_decay_start_itr=5 \
    +train.ent_decay_end_itr=50 \
    seed=0 \
# +train.use_bc_loss=true \
# +train.bc_loss_type='W2' \
# +train.bc_loss_coeff=0.1                

echo "Fine-tuning finished."
