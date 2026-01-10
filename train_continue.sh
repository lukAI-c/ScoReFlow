#!/bin/bash

echo "Resuming Robomimic Fine-tuning..."

export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"

python script/run.py \
    --config-dir=cfg/robomimic/finetune/square \
    --config-name=ft_ppo_reflow_mlp_img_score_distribute_qrdqn \
    base_policy_path=${REINFLOW_LOG_DIR}/robomimic/finetune/square_ppo_reflow_mlp_score_qrdqn_ta4_td4/2026-01-06_11-43-45_gamma1_42/checkpoint/last.pt\
    device=cuda:0 \
    sim_device=null \
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
    resume_path=/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_clk/ReinFlow/log/robomimic/finetune/square_ppo_reflow_mlp_score_qrdqn_ta4_td4/2026-01-06_11-43-45_gamma1_42/checkpoint/last.pt \

echo "Resume finished."