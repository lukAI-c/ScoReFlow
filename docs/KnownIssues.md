## Known Issues and Fixes for Common Bugs

### `ValueError: XML Error: top-level default class 'main' cannot be renamed`

dm_control and mujoco version incompatibility. Run:
```bash
pip install dm_control==1.0.16 mujoco==3.1.6
```

### `KeyError: 'REINFLOW_DIR' / 'REINFLOW_DATA_DIR' / 'REINFLOW_LOG_DIR'`

`util/dirs.py` strictly validates these three environment variables at import time, and `REINFLOW_DIR` must **exactly match** the project root path. On Windows, must use backslash notation:
```bash
export REINFLOW_DIR='D:\GitLoadWareHouse\ScoReFlow'
export REINFLOW_DATA_DIR='D:\GitLoadWareHouse\ScoReFlow\data'
export REINFLOW_LOG_DIR='D:\GitLoadWareHouse\ScoReFlow\logs'
```

For setup details, see [docs/ReproduceExps.md](ReproduceExps.md#02-environment-variables).

### Script can't find `cfg/...` or `download_url`

All scripts under `scripts/` **must be run from the project root directory**. Configs and paths are resolved relatively:
```bash
cd $REINFLOW_DIR
bash scripts/train/robomimic/train_robomimic_finetune-grpo.sh
```

### Robomimic image rendering fails with EGL error

Explicitly isolate physical GPUs and set EGL device number:
```bash
export CUDA_VISIBLE_DEVICES=0,1
export EGL_DEVICE_ID=1
export MUJOCO_EGL_DEVICE_ID=1
```

Without EGL support, fall back to CPU rendering: `export MUJOCO_GL=osmesa` + `sim_device=null`.

### Gym / Kitchen task hangs at `xvfb-run`

Without an X server, `xvfb-run` is needed. Scripts already include `MUJOCO_GL="egl" xvfb-run -a -s "-screen 0 1024x768x24"`. On a machine with display, remove `xvfb-run`.

### GRPO fine-tuning: `train.kl_coef` not taking effect

GRPO agent (`agent/finetune/reinflow/train_grpo_flow_agent.py`) **only** has KL coefficient. If using PPO config (`ft_ppo_*`) with `train.kl_coef=...`, Hydra will report unknown key. **Use `ft_grpo_*` config instead.**
