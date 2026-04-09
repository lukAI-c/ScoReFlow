# ScoReFlow Experiment Reproduction Guide

This guide covers dataset preparation, checkpoint downloads, pre-training / fine-tuning / evaluation, and practical tips.

---

## 0. Preparation

### 0.1 Working Directory

**All commands must be run from the project root directory**, otherwise relative paths (`cfg/...`, `run.py`, `scripts/...`) will fail.

```
ScoReFlow/
├── run.py                  # Unified Python entry point
├── download_url.py         # Download URL table
├── cfg/                    # Hydra configurations
├── scripts/                # All bash launch scripts
│   ├── train/{robomimic,gym,kitchen}/
│   ├── eval/{robomimic,gym,kitchen}/
│   └── utils/
├── agent/  model/  env/  util/  data_process/
├── external_libs/{mjrl,d4rl}/
├── data/  logs/            # Offline datasets / checkpoints (gitignored by default)
└── docs/
```

### 0.2 Environment Variables

At startup, `util/dirs.py` strictly validates the following environment variables. **Failure to set any or path mismatch will cause an error.**

| Variable | Purpose |
|---|---|
| `REINFLOW_DIR` | Project root directory; must exactly match code location |
| `REINFLOW_DATA_DIR` | Offline dataset root directory |
| `REINFLOW_LOG_DIR` | Checkpoint / wandb output root; all yaml `logdir:` and `base_policy_path:` are resolved from this |
| `REINFLOW_WANDB_ENTITY` | WandB username or team name (optional; can skip by using `wandb=null`) |

#### One-time initialization (recommended)

```bash
cd <project-root>
source scripts/utils/set_path.sh
```

The script will interactively prompt for each path, save to `~/.bashrc`, and enable debug switches (`HYDRA_FULL_ERROR=1`, etc.). **Run once**, then open a new terminal and it takes effect.

Example input:
```
Enter the place where your reinflow script lies:
→ /your/path/to/ScoReFlow

Enter the desired data directory:
→ /your/path/to/ScoReFlow/data

Enter the desired logging directory:
→ /your/path/to/ScoReFlow/logs

Enter your WandB entity (press ENTER to skip):
→ your_wandb_username
```

#### Manual Setup

If you prefer not to use the interactive script, directly append to `~/.bashrc`:

```bash
export REINFLOW_DIR=/your/path/to/ScoReFlow
export REINFLOW_DATA_DIR=/your/path/to/ScoReFlow/data
export REINFLOW_LOG_DIR=/your/path/to/ScoReFlow/logs
export REINFLOW_WANDB_ENTITY=your_wandb_username   # Optional

# Debug switches (optional but recommended)
export D4RL_SUPPRESS_IMPORT_ERROR=1
export HYDRA_FULL_ERROR=1
```

Then `source ~/.bashrc`.

#### About the logs/ Directory

If you have pre-trained checkpoints from another project, use a symlink to avoid duplicating storage:

```bash
# Symlink old project's log directory to this project's logs/
ln -s /old/project/log /your/path/to/ScoReFlow/logs
```

This way `REINFLOW_LOG_DIR` points to the new path but data stays at the original location.

---

## 1. Dataset Preparation

### 1.1 D4RL Locomotion (Hopper / Walker2d / Ant / Humanoid)

**Method A: Automatic download and preprocessing (recommended)**

Run pre-training with `use_d4rl_dataset=True`:

```bash
python run.py --config-dir=cfg/gym/pretrain/walker2d-medium-v2 \
              --config-name=pre_reflow_mlp use_d4rl_dataset=True
```

**Method B: Manual download from Hugging Face**

```bash
wget https://huggingface.co/datasets/imone/D4RL/resolve/main/hopper_medium-v2.hdf5
wget https://huggingface.co/datasets/imone/D4RL/resolve/main/walker2d_medium-v2.hdf5
wget https://huggingface.co/datasets/imone/D4RL/resolve/main/ant_medium_expert-v2.hdf5

# Inspect hdf5 structure
python data_process/read_hdf5.py --file_path=<HDF5_PATH>

# Convert to npz + normalize
python data_process/hdf5_to_npz.py --data_path=<HDF5_PATH>

# Inspect npz contents
python data_process/read_npz.py --data_path=<DIR>/train.npz
```

### 1.2 Franka Kitchen

Reuses D4RL's kitchen data. Downloads automatically when running pre-training; no manual action needed.

### 1.3 Robomimic (image-based)

`cfg/robomimic/pretrain/<task>/pre_*.yaml` already contains Google Drive links. When running pre-training for the first time, `run.py` automatically downloads via `gdown`.

---

## 2. Download Pre-trained Checkpoints

When running fine-tuning, if `base_policy_path` points to a missing file, `run.py` will look up the download URL and fetch it automatically.

Or download manually:
```bash
python download_checkpoints.py --path "pretrain/hopper-v2/ReFlow/2025-02-06_01-35-03_42/checkpoint/state_1500.pt"
```

---

## 3. Pre-training

### Robomimic ShortCut (image)

```bash
bash scripts/train/robomimic/train_robomimic_pretrain.sh
# Equivalent to:
python run.py --config-dir=cfg/robomimic/pretrain/transport \
              --config-name=pre_shortcut_mlp_img \
              device=cuda:0 wandb.offline_mode=true
```

### Gym 1-ReFlow

```bash
bash scripts/train/gym/train_gym_pretrain.sh
# Equivalent to:
python run.py --config-dir=cfg/gym/pretrain/hopper-medium-v2 \
              --config-name=pre_reflow_mlp \
              device=cuda:0 +wandb.offline_mode=true
```

### Kitchen ShortCut

```bash
bash scripts/train/kitchen/train_kitchen_pretrain.sh
```

> If you only want to use our pre-trained checkpoints without retraining, skip this section and go directly to fine-tuning.

---

## 4. Fine-tuning

ScoReFlow offers 4 fine-tuning categories:

| Type | Description | Config Name Convention |
|---|---|---|
| **PPO Baseline** | Drift-only, no score | `ft_ppo_*_mlp[*_img]` |
| **PPO + Score-SDE + AlphaNet (ours)** | Joint drift + diffusion optimization | `ft_ppo_*_with_score_alphanet[*_obs]` |
| **GRPO Baseline** | Critic-free, group-relative advantage | `ft_grpo_*_mlp[*_img]` |
| **GRPO + Score-SDE + AlphaNet (ours)** | GRPO + AlphaNet combination | `ft_grpo_*_with_score_alphanet` |

### 4.1 Robomimic (image)

```bash
# PPO + ScoReFlow AlphaNet
bash scripts/train/robomimic/train_robomimic_finetune-with-score.sh

# GRPO + ScoReFlow AlphaNet (critic-free)
TASK=square    SEED=42 KL_COEF=0.2 bash scripts/train/robomimic/train_robomimic_finetune-grpo.sh
TASK=transport SEED=42 KL_COEF=0.2 bash scripts/train/robomimic/train_robomimic_finetune-grpo.sh

# Direct run.py invocation
python run.py --config-dir=cfg/robomimic/finetune/square \
              --config-name=ft_ppo_reflow_mlp_img_with_score_alphanet_obs \
              base_policy_path=${REINFLOW_LOG_DIR}/robomimic/pretrain/square/.../last.pt \
              device=cuda:0 sim_device=cuda:1 \
              wandb.offline_mode=true \
              env.n_envs=50 denoising_steps=2 gamma_score=1 seed=42
```

### 4.2 Gym Locomotion (state)

```bash
bash scripts/train/gym/train_gym_finetune-with-score.sh
# Or GRPO:
TASK=kitchen SEED=42 KL_COEF=0.04 bash scripts/train/gym/train_gym_finetune-grpo.sh
```

### 4.3 Kitchen (state)

```bash
bash scripts/train/kitchen/train_kitchen_finetune-with-score.sh
```

### 4.4 Common fine-tuning overrides

```bash
# Switch base policy
python run.py ... base_policy_path=NEW.pt

# Switch normalization (critical if using different expert data!)
python run.py ... normalization_path=NEW_DIR/normalization.npz

# Custom wandb run name (for parameter sweeps)
python run.py ... name=my_run_seed42

# Background run with log redirection (overnight training)
nohup python run.py ... > my.log 2>&1 &
```

### 4.5 Resume after training crash

```bash
python run.py --config-dir=cfg/gym/finetune/walker2d-v2 \
              --config-name=ft_ppo_reflow_mlp \
              resume_path=CHECKPOINT_THAT_FAILED.pt
```

> Resume will continue training for another `train.n_train_itr` rounds from the checkpoint state.
> If `resume_path` is not in the config, add it on the command line.
> Currently supports resuming from pre-training or fine-tuning checkpoints from ScoReFlow / ReinFlow,
> and DPPO checkpoints from `agent/finetune/reinflow/train_ppo_diffusion_*.py`.

---

## 5. Evaluation

```bash
# Evaluate robomimic fine-tuned policy
bash scripts/eval/robomimic/eval_robomimic_finetune.sh

# Equivalent to
python run.py --config-dir=cfg/robomimic/eval/can \
              --config-name=eval_reflow_mlp_img_with_score_alphanet \
              base_policy_path=${REINFLOW_LOG_DIR}/robomimic/finetune/.../best.pt \
              device=cuda:0 \
              denoising_step_list=[1,2,4,8] \
              load_ema=False \
              env.n_envs=50 \
              env.save_video=true +record_video=true render_num=5
```

Output includes:
- 6 plots (episode reward / success rate / episode length / inference frequency / duration / best reward, with std shading)
- `.png` and corresponding data files
- `.mp4` videos if `env.save_video=true`

**Tips:**
- `load_ema=True` for pre-trained policies, `False` for fine-tuned ones
- Add `denoising_step_list` to command line if not in config
- Close other GPU processes for accurate inference timing

**Video recording:** Set `self.record_video=True` in the evaluation script and specify `self.record_env_index` to choose which environment to record. Control resolution with `self.frame_width` / `self.frame_height`.

**Warning:** ScoReFlow clips intermediate actions during fine-tuning, so keep `clip_intermediate_actions=True` during evaluation, otherwise reward will drop.

---

## 6. Practical Tips

### 6.1 Offline / Disable wandb

```bash
# Offline mode, sync later
python run.py ... wandb.offline_mode=True

# Batch sync all offline runs from a date
for dir in ./wandb_offline/wandb/offline-run-20260409*; do wandb sync "${dir}"; done

# Completely disable wandb (training data still saved to local .pkl)
python run.py ... wandb=null

# Later recover: use util/pkl2wandb.py to push .pkl back to wandb
python util/pkl2wandb.py --pkl <PATH>
```

### 6.2 Out of GPU memory

- Reduce `env.n_envs`, increase `train.n_steps` to keep total samples constant (`n_envs × n_steps × act_steps`)
- `train.n_steps` should be a multiple of `env.max_episode_steps / act_steps`

### 6.3 Accelerate rendering

- With EGL: `sim_device=<gpu_id>` for fast rendering
- Without EGL: `sim_device=null` for CPU osmesa (slower)

### 6.4 Robomimic image task GPU isolation

```bash
export CUDA_VISIBLE_DEVICES=3,4
export EGL_DEVICE_ID=4
export MUJOCO_EGL_DEVICE_ID=4
```

### 6.5 Flow matching fine-tuning tips

- **Let it fail sometimes**: Use long rollouts to mix successes and failures; short trajectories trick the critic.
- **Tune critic warmup**: Adjust warmup iterations or initialization based on initial policy success rate.

---

## 7. ScoReFlow Quick Reference

| Hyperparameter | Applies to | Recommended | Meaning |
|---|---|---|---|
| `gamma_score` | PPO+Score / GRPO+Score | `1.0` | AlphaNet $\alpha_\psi$ initial scale |
| `denoising_steps` | All | `2 (square) / 4 (transport,kitchen)` | Denoising steps |
| `train.kl_coef` | GRPO | `0.04 (state) / 0.2 (image)` | GRPO KL penalty coefficient |
| `train.ent_coef` | All | `0.01 (image) / 0.03 (state)` | Entropy regularization |
| `model.clip_ploss_coef` | PPO | `0.001 (image) / 0.01 (state)` | PPO clipping $\epsilon$ |
| `env.n_envs` | All | `50` | Parallel environments |
| `seed` | All | `42 / 128 / 2026` | Random seed (run 3+ for papers) |
