# ScoReFlow 实验复现指南

本指南覆盖数据准备、checkpoint 下载、预训练 / 微调 / 评估的运行方式,以及若干常见诀窍。

---

## 0. 准备工作

### 0.1 工作目录

**所有命令必须从项目根目录执行**,否则相对路径(`cfg/...`、`run.py`、`scripts/...`)会全部失效。

```
ScoReFlow/
├── run.py                  # 统一 Python 入口
├── download_url.py         # 下载链接表
├── cfg/                    # Hydra 配置
├── scripts/                # 所有 Bash 启动脚本
│   ├── train/{robomimic,gym,kitchen}/
│   ├── eval/{robomimic,gym,kitchen}/
│   └── utils/
├── agent/  model/  env/  util/  data_process/
├── external_libs/{mjrl,d4rl}/
├── data/  logs/            # 数据 / checkpoint(默认 gitignored)
└── docs/
```

### 0.2 环境变量

```bash
export REINFLOW_DIR='D:\GitLoadWareHouse\ScoReFlow'              # 必填,严格匹配真实路径
export REINFLOW_DATA_DIR='D:\GitLoadWareHouse\ScoReFlow\data'    # 离线数据集根
export REINFLOW_LOG_DIR='D:\GitLoadWareHouse\ScoReFlow\logs'     # checkpoint / wandb 输出根
```

> Windows 必须用反斜杠形式,正斜杠会让 `util/dirs.py` 启动校验失败。

---

## 1. 数据集准备

### 1.1 D4RL Locomotion (Hopper / Walker2d / Ant / Humanoid)

**方式 A:自动下载并预处理(推荐)**

直接跑预训练命令并加 `use_d4rl_dataset=True`:

```bash
python run.py --config-dir=cfg/gym/pretrain/walker2d-medium-v2 \
              --config-name=pre_reflow_mlp use_d4rl_dataset=True
```

**方式 B:手动从 Hugging Face 下载**

```bash
wget https://huggingface.co/datasets/imone/D4RL/resolve/main/hopper_medium-v2.hdf5
wget https://huggingface.co/datasets/imone/D4RL/resolve/main/walker2d_medium-v2.hdf5
wget https://huggingface.co/datasets/imone/D4RL/resolve/main/ant_medium_expert-v2.hdf5

# 查看 hdf5 结构
python data_process/read_hdf5.py --file_path=<HDF5_PATH>

# 转 npz + 归一化
python data_process/hdf5_to_npz.py --data_path=<HDF5_PATH>

# 检查 npz 内容
python data_process/read_npz.py --data_path=<DIR>/train.npz
```

### 1.2 Franka Kitchen

复用 D4RL 的 kitchen 数据。运行预训练时自动下载,无需手动操作。

### 1.3 Robomimic (image-based)

`cfg/robomimic/pretrain/<task>/pre_*.yaml` 已经写好 Google Drive 链接,首次运行预训练时 `run.py` 会通过 `gdown` 自动拉取数据集与归一化统计。

---

## 2. 下载预训练 Checkpoint

跑微调脚本时如果 `base_policy_path` 指向的文件不存在,`run.py` 会从 `download_url.py` 表里找对应链接并自动下载。

也可以手动:
```bash
python download_checkpoints.py --path "pretrain/hopper-v2/ReFlow/2025-02-06_01-35-03_42/checkpoint/state_1500.pt"
```

---

## 3. 预训练

### Robomimic ShortCut(图像)

```bash
bash scripts/train/robomimic/train_robomimic_pretrain.sh
# 等价于:
python run.py --config-dir=cfg/robomimic/pretrain/transport \
              --config-name=pre_shortcut_mlp_img \
              device=cuda:0 wandb.offline_mode=true
```

### Gym 1-ReFlow

```bash
bash scripts/train/gym/train_gym_pretrain.sh
# 等价于:
python run.py --config-dir=cfg/gym/pretrain/hopper-medium-v2 \
              --config-name=pre_reflow_mlp \
              device=cuda:0 +wandb.offline_mode=true
```

### Kitchen ShortCut

```bash
bash scripts/train/kitchen/train_kitchen_pretrain.sh
```

> 如果你只想用我们提供的预训练 checkpoint 而不自己重新训,可以跳过本节,直接进入「微调」。

---

## 4. 微调

ScoReFlow 提供 4 类微调入口:

| 类型 | 说明 | 配置名约定 |
|---|---|---|
| **PPO 基线** | drift-only,无 score | `ft_ppo_*_mlp[*_img]` |
| **PPO + Score-SDE + GammaNet (本工作)** | 联合优化 drift + diffusion | `ft_ppo_*_with_score_gammanet[*_obs]` |
| **GRPO 基线** | critic-free,组相对优势 | `ft_grpo_*_mlp[*_img]` |
| **GRPO + Score-SDE + GammaNet (本工作)** | GRPO + GammaNet 组合 | `ft_grpo_*_with_score_gammanet` |

### 4.1 Robomimic (image)

```bash
# PPO + ScoReFlow GammaNet
bash scripts/train/robomimic/train_robomimic_finetune-with-score.sh

# GRPO + ScoReFlow GammaNet (critic-free)
TASK=square    SEED=42 KL_COEF=0.2 bash scripts/train/robomimic/train_robomimic_finetune-grpo.sh
TASK=transport SEED=42 KL_COEF=0.2 bash scripts/train/robomimic/train_robomimic_finetune-grpo.sh

# 直接调 run.py 也可以
python run.py --config-dir=cfg/robomimic/finetune/square \
              --config-name=ft_ppo_reflow_mlp_img_with_score_gammanet_obs \
              base_policy_path=${REINFLOW_LOG_DIR}/robomimic/pretrain/square/.../last.pt \
              device=cuda:0 sim_device=cuda:1 \
              wandb.offline_mode=true \
              env.n_envs=50 denoising_steps=2 gamma_score=1 seed=42
```

### 4.2 Gym Locomotion (state)

```bash
bash scripts/train/gym/train_gym_finetune-with-score.sh
# 或 GRPO:
TASK=kitchen SEED=42 KL_COEF=0.04 bash scripts/train/gym/train_gym_finetune-grpo.sh
```

### 4.3 Kitchen (state)

```bash
bash scripts/train/kitchen/train_kitchen_finetune-with-score.sh
```

### 4.4 微调常见 override

```bash
# 切换 base policy
python run.py ... base_policy_path=NEW.pt

# 切换 normalization(用了不同 expert 数据时务必同步换!)
python run.py ... normalization_path=NEW_DIR/normalization.npz

# 自定义 wandb run 名(参数扫描时建议)
python run.py ... name=my_run_seed42

# 后台跑 + 日志重定向(过夜训练)
nohup python run.py ... > my.log 2>&1 &
```

### 4.5 训练崩了如何 resume

```bash
python run.py --config-dir=cfg/gym/finetune/walker2d-v2 \
              --config-name=ft_ppo_reflow_mlp \
              resume_path=CHECKPOINT_THAT_FAILED.pt
```

> Resume 会基于 checkpoint 状态继续训 `train.n_train_itr` 轮。
> 如果配置里没 `resume_path` 字段,直接在命令行追加即可。
> 目前支持 resume 的来源:ScoReFlow / ReinFlow 的预训练或微调 checkpoint,
> 以及 `agent/finetune/reinflow/train_ppo_diffusion_*.py` 输出的 DPPO checkpoint。

---

## 5. 评估

```bash
# 跑 robomimic 微调后的策略
bash scripts/eval/robomimic/eval_robomimic_finetune.sh

# 等价于
python run.py --config-dir=cfg/robomimic/eval/can \
              --config-name=eval_reflow_mlp_img_with_score_gammanet \
              base_policy_path=${REINFLOW_LOG_DIR}/robomimic/finetune/.../best.pt \
              device=cuda:0 \
              denoising_step_list=[1,2,4,8] \
              load_ema=False \
              env.n_envs=50 \
              env.save_video=true +record_video=true render_num=5
```

输出包括:
- 6 张图(episode reward / success rate / episode length / inference freq / duration / best reward,带 std 阴影)
- `.png` 与对应的数据文件
- 若 `env.save_video=true`,还会输出 `.mp4`

**Tips:**
- `load_ema=True` 评估预训练策略,`False` 评估微调后的策略
- `denoising_step_list` 不在配置里时直接命令行追加即可
- 评估时建议关闭机器上其他 GPU 进程,以获得准确的推理时延

**视频录制:** 在评估脚本里设置 `self.record_video=True` 与 `self.record_env_index`,可指定要录制的环境编号、`self.frame_width`/`self.frame_height` 控制分辨率。

**警告:** ScoReFlow 在微调时会裁剪中间动作,所以评估时务必保持 `clip_intermediate_actions=True`,否则 reward 会下降。

---

## 6. 实用技巧

### 6.1 wandb 离线 / 关闭

```bash
# 离线模式,稍后批量同步
python run.py ... wandb.offline_mode=True

# 同步某天所有离线 run
for dir in ./wandb_offline/wandb/offline-run-20260409*; do wandb sync "${dir}"; done

# 完全关闭 wandb (训练数据仍会落到本地 .pkl)
python run.py ... wandb=null

# 之后想恢复:用 util/pkl2wandb.py 把 .pkl 推回 wandb
python util/pkl2wandb.py --pkl <PATH>
```

### 6.2 显存不够

- 减小 `env.n_envs`,同时增大 `train.n_steps` 保持总样本量(`n_envs × n_steps × act_steps`)不变
- `train.n_steps` 应是 `env.max_episode_steps / act_steps` 的整数倍

### 6.3 渲染加速

- EGL 可用时:`sim_device=<gpu_id>` 加速渲染
- 无 EGL:`sim_device=null` 走 osmesa CPU 渲染

### 6.4 Robomimic 图像任务的 GPU 隔离

```bash
export CUDA_VISIBLE_DEVICES=3,4
export EGL_DEVICE_ID=4
export MUJOCO_EGL_DEVICE_ID=4
```

### 6.5 微调流匹配策略的几条经验

- **让它有时候失败**:用足够长的 rollout 来混合成功与失败轨迹,过短的轨迹会让 critic 学到错误的判断。
- **调 critic warmup**:根据初始策略的成功率调整 warmup 迭代数,必要时还可以改 critic 的初始化偏置(`model.critic.out_bias_init`)。

---

## 7. ScoReFlow 关键超参速查

| 超参 | 适用 | 推荐值 | 含义 |
|---|---|---|---|
| `gamma_score` | PPO+Score / GRPO+Score | `1.0` | GammaNet $\alpha_\psi$ 初始尺度 |
| `denoising_steps` | 全部 | `2 (square) / 4 (transport,kitchen)` | 去噪步数 |
| `train.kl_coef` | GRPO | `0.04 (state) / 0.2 (image)` | GRPO KL 惩罚系数 |
| `train.ent_coef` | 全部 | `0.01 (image) / 0.03 (state)` | 熵正则 |
| `model.clip_ploss_coef` | PPO | `0.001 (image) / 0.01 (state)` | PPO 裁剪 $\epsilon$ |
| `env.n_envs` | 全部 | `50` | 并行环境数 |
| `seed` | 全部 | `42 / 128 / 2026` | 随机种子(论文中至少跑 3 个) |
