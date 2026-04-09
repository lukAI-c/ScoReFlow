#!/bin/bash

# 1. 【关键】设置 MuJoCo 后端为 osmesa (纯 CPU 渲染，不依赖显卡驱动和窗口系统)
export LIBGL_ALWAYS_SOFTWARE=1
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

# 2. 【关键】清除 DISPLAY 变量，防止代码尝试连接 X Server
# unset DISPLAY

# 3. 运行命令
# 注意：必须添加 sim_device=cpu (或者 null)，对应文档建议
python run.py \
  --config-dir=cfg/gym/eval/walker2d-medium-v2 \
  --config-name=eval_reflow_mlp \
  base_policy_path=./log/gym/pretrain/walker2d-medium-v2_pre_reflow_mlp_ta4_td20_seed42/2025-12-06_03-05-56_42/checkpoint/last.pt \
  denoising_step_list=[1,2,4] \
  load_ema=True \
  device=cuda:0 \
  env.n_envs=10 \
  sim_device=null