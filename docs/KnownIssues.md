



## Known Issues and Fixes to Common Bugs

### `ValueError: XML Error: top-level default class 'main' cannot be renamed`
* `dm_control` 与 `mujoco` 版本不兼容。运行 `pip install dm_control==1.0.16 mujoco==3.1.6` 即可解决。

### `KeyError: 'REINFLOW_DIR' / 'REINFLOW_DATA_DIR' / 'REINFLOW_LOG_DIR'`
* `util/dirs.py` 在导入时强校验这三个环境变量,且 `REINFLOW_DIR` 必须**严格等于**项目根的真实路径。Windows 下务必使用反斜杠形式:
  ```bash
  export REINFLOW_DIR='D:\GitLoadWareHouse\ScoReFlow'
  export REINFLOW_DATA_DIR='D:\GitLoadWareHouse\ScoReFlow\data'
  export REINFLOW_LOG_DIR='D:\GitLoadWareHouse\ScoReFlow\logs'
  ```

### 脚本里 `python run.py` 提示找不到 `cfg/...` 或 `download_url`
* 所有 `scripts/` 下的 `.sh` **必须从项目根目录调用**。配置目录与 `run.py` 都用相对路径解析:
  ```bash
  cd $REINFLOW_DIR
  bash scripts/train/robomimic/train_robomimic_finetune-grpo.sh
  ```

### Robomimic 图像任务渲染报 EGL 错
* 显式隔离物理 GPU 并设置 EGL 设备号:
  ```bash
  export CUDA_VISIBLE_DEVICES=0,1
  export EGL_DEVICE_ID=1
  export MUJOCO_EGL_DEVICE_ID=1
  ```
  没有 EGL 支持时可退回 CPU 渲染:`export MUJOCO_GL=osmesa` + `sim_device=null`。

### Kitchen / Gym 任务卡在 `xvfb-run`
* 服务器没有 X server 时需要 `xvfb-run`。脚本里已带 `MUJOCO_GL="egl" xvfb-run -a -s "-screen 0 1024x768x24"`。本地有显示器时可去掉 `xvfb-run`。

### GRPO 微调时 `train.kl_coef` 未生效
* GRPO agent (`agent/finetune/reinflow/train_grpo_flow_agent.py`) 才有 KL 系数。如果用 PPO 配置 (`ft_ppo_*`) 加 `train.kl_coef=...` Hydra 会报 unknown key — 请确认用的是 `ft_grpo_*` 配置。
