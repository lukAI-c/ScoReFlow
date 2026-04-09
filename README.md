# ScoRe-Flow

> **ScoRe-Flow: Complete Distributional Control via Score-Based Reinforcement Learning for Flow Matching**

<p align="center">
  <img src="sample_figs/overview.png" alt="ScoRe-Flow Overview" width="80%">
</p>

---

## Table of Contents

- [Overview](#overview)
- [Method](#method)
- [Demo](#demo)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Experiments](#experiments)
- [Configuration](#configuration)
- [License](#license)

---

## Overview

**ScoRe-Flow** is a novel framework that achieves **complete distributional control** for flow matching policies through score-based reinforcement learning. Unlike prior methods that only control the drift term, ScoRe-Flow introduces a principled Score-SDE formulation that jointly optimizes both the **drift** and **diffusion** of the sampling process via learned score functions.

### The Problem

Flow matching policies generate actions through an iterative denoising process. Standard RL fine-tuning (e.g., DPPO, ReinFlow) only modifies the drift (velocity field), leaving the exploration noise as a fixed hyperparameter. This limits distributional expressiveness and sample efficiency.

### Our Solution: Score-SDE Formulation

ScoRe-Flow reformulates the flow matching sampling process as a **Score-SDE**:

$$dx_t = \underbrace{[v_\theta(x_t, t) + \alpha_\psi(t) \cdot s(x_t, t)]}_{\text{drift: velocity + score guidance}} dt + \underbrace{\sqrt{2 \cdot \alpha_\psi(t)}}_{\text{diffusion}} \, dW_t$$

where:
- $v_\theta(x_t, t)$ is the pre-trained velocity field (flow matching)
- $s(x_t, t) = \frac{t \cdot v_\theta - x_t}{1 - t}$ is the analytically derived score function
- $\alpha_\psi(t)$ is a **learnable time-dependent schedule** (GammaNet) that controls the strength of score guidance
- The same $\alpha_\psi(t)$ governs both drift correction and diffusion magnitude, ensuring theoretical consistency

### Core Contributions

1. **Score-Based RL Framework**: A principled approach that integrates analytically derived score functions with PPO for flow matching policy optimization
2. **Learnable Distributional Control**: A lightweight neural network ($\alpha_\psi(t)$) that learns to modulate score guidance strength across different denoising timesteps
3. **Decoupled Architecture**: Separate learnable networks for drift control and diffusion control, enabling more flexible optimization than coupled approaches
4. **Complete Distributional Control**: Unlike drift-only methods, ScoRe-Flow controls the full transition distribution (both mean and variance) at every denoising step

---

## Method

### Architecture

ScoRe-Flow builds on flow matching pre-trained policies and adds three learnable components during RL fine-tuning:

| Component | Role | Architecture |
|-----------|------|-------------|
| **Velocity Field** $v_\theta$ | Base policy (pre-trained, frozen) | FlowMLP / ShortCutMLP / ViT |
| **Actor** $v_\phi$ | Fine-tuned velocity field | Copy of $v_\theta$, trainable |
| **GammaNet** $\alpha_\psi(t)$ | Score guidance schedule | Lightweight MLP: $t \to \text{SiLU} \to \text{SiLU} \to \text{Softplus}$ |
| **Noise Network** $n_\omega(t)$ | Exploration noise | Time-conditioned MLP |

### Key Design Choices

- **Physical Constraint**: $\alpha_\psi(t)$ includes a hard time mask $(1-t)$ to ensure $\alpha \to 0$ as $t \to 1$, preventing score explosion at the boundary
- **Decoupled Training**: The score guidance ($\alpha_\psi$) and exploration noise ($n_\omega$) are trained independently, allowing each to specialize
- **Warm Initialization**: GammaNet is initialized to output values close to the baseline hyperparameter $\epsilon_t$, ensuring stable training from the start

---

## Demo

### Robomimic Tasks (Image-based Control)

ScoRe-Flow achieves state-of-the-art performance on challenging image-based robotic manipulation tasks:

<table>
  <tr>
    <td align="center"><b>Can</b></td>
    <td align="center"><b>Square</b></td>
    <td align="center"><b>Transport</b></td>
  </tr>
  <tr>
    <td align="center"><img src="sample_figs/demo_can.gif" width="280"></td>
    <td align="center"><img src="sample_figs/demo_square.gif" width="280"></td>
    <td align="center"><img src="sample_figs/demo_transport.gif" width="280"></td>
  </tr>
  <tr>
    <td align="center">Pick-and-place a can<br>into a target bin</td>
    <td align="center">Insert a square nut<br>onto a peg</td>
    <td align="center">Two-arm coordination<br>to transport an object</td>
  </tr>
</table>

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Flow Matching** | Support for 1-Rectified Flow and Shortcut Models |
| **Score-Based RL** | Analytically derived score functions for principled policy control |
| **Learnable Schedule** | GammaNet adaptively modulates score strength per timestep |
| **Decoupled Control** | Independent learnable networks for drift and diffusion |
| **Multi-Environment** | Tested on D4RL Gym, Franka Kitchen, and Robomimic benchmarks |
| **Flexible Architecture** | MLP and ViT-based policy networks |
| **Visual Observations** | Full support for image-based control tasks with dual cameras |

---

## Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 2.4.0
- CUDA (recommended for GPU acceleration)

### Setup

```bash
# Clone the repository
cd ScoRe-Flow

# Install as editable package (REQUIRED - enables running scripts from any directory)
pip install -e .

# For specific environments, add optional dependencies:
pip install -e ".[gym]"        # OpenAI Gym (Hopper, Walker2d, Ant, Humanoid)
pip install -e ".[kitchen]"    # Franka Kitchen
pip install -e ".[robomimic]"  # Robomimic (Square, Can, Transport)
```

> **Note**: The `pip install -e .` step is essential. It registers `util`, `agent`, `model` etc. as importable modules.

For detailed installation instructions, see [installation/reinflow-setup.md](./installation/reinflow-setup.md).

---

## Quick Start

### Training Example (Robomimic)

```bash
# Image-based task with Score-SDE (GammaNet)
export CUDA_VISIBLE_DEVICES=0,1
export EGL_DEVICE_ID=1
export MUJOCO_EGL_DEVICE_ID=1

python run.py \
    --config-dir=cfg/robomimic/finetune/square \
    --config-name=ft_ppo_shortcut_mlp_img_with_score_gammanet \
    device=cuda:0 \
    sim_device=cuda:1 \
    gamma_score=1.0 \
    seed=42
```

### Training Example (Gym / Kitchen)

```bash
# State-based task with Score-SDE
MUJOCO_GL="egl" xvfb-run -a -s "-screen 0 1024x768x24" python run.py \
    --config-dir=cfg/gym/finetune/kitchen-complete-v0 \
    --config-name=ft_ppo_shortcut_mlp_with_score_gammanet \
    device=cuda:0 \
    gamma_score=1.0 \
    seed=42
```

### Evaluation

```bash
# 所有 .sh 必须从项目根目录调用 (它们内部使用相对路径 cfg/... 和 run.py)
bash scripts/eval/robomimic/eval_robomimic_finetune.sh
```

---

## Project Structure

> 重构说明 (2026-04):根目录下大量历史 `train_*.sh` / `eval_*.sh` 已按任务分类搬迁到 `scripts/{train,eval}/{robomimic,gym,kitchen}/`;
> 原 `script/` 目录已移除,Python 入口 `run.py` 提升至项目根;`mjrl-master` / `d4rl` 收纳至 `external_libs/`。
> **所有 .sh 仍需从项目根目录运行** (例如 `bash scripts/train/robomimic/train_robomimic_finetune-grpo.sh`)。

```
ScoRe-Flow/
├── run.py                      # 🚀 统一 Python 入口 (Hydra 启动)
├── download_url.py             # 数据/checkpoint 下载 URL 表 (run.py 依赖)
├── download_checkpoints.py     # 单独下载预训练权重的 CLI
│
├── scripts/                    # 🌟 所有 Bash 启动/调度脚本
│   ├── train/
│   │   ├── robomimic/          # 替代根目录所有 train_robomimic_*.sh / train_square_*.sh
│   │   ├── gym/                # 替代根目录所有 train_gym_*.sh
│   │   └── kitchen/            # 替代根目录所有 train_kitchen_*.sh
│   ├── eval/
│   │   ├── robomimic/          # 替代根目录所有 eval_robomimic_*.sh
│   │   ├── gym/                # eval_gym.sh
│   │   └── kitchen/            # eval_kitchen*.sh
│   └── utils/                  # activate.sh, set_path.sh, 渲染测试, 数据预处理脚本
│       └── dataset/            # filter/process_d3il, get_d4rl, process_robomimic
│
├── cfg/                        # ⚙️ Hydra 配置 (保持不变)
│   ├── robomimic/{pretrain,finetune,eval}/
│   ├── gym/{pretrain,finetune,eval}/
│   └── kitchen/...
│
├── agent/                      # 📦 RL 训练 agent (保持在根,import 路径不变)
│   ├── pretrain/
│   ├── finetune/
│   │   └── reinflow/           # PPO / GRPO 微调 agent
│   └── eval/visualize/
├── model/                      # 神经网络架构
│   └── flow/ft_ppo/            # PPO/GRPO + Score-SDE + GammaNet 各变体
├── env/                        # 环境封装与注册
├── data_process/               # 数据集预处理逻辑
├── visualize/                  # 实验数据 / 出图代码
├── util/                       # 工具类 (clear_pycache, dirs, ...)
│
├── external_libs/              # 📚 第三方依赖源码
│   ├── mjrl/                   # 原 mjrl-master/
│   └── d4rl/                   # 原 d4rl/
│
├── data/                       # 🗂️ 离线数据集 (gitignored, 仅留 .gitkeep)
├── logs/                       # 📊 运行日志 / checkpoint (gitignored, REINFLOW_LOG_DIR 默认指向)
│
├── docs/                       # 文档
├── installation/               # 安装指南
├── pyproject.toml
└── README.md
```

### 必需的环境变量

```bash
export REINFLOW_DIR='D:\GitLoadWareHouse\ScoReFlow'        # 项目根 (Windows 路径分隔)
export REINFLOW_DATA_DIR='D:\GitLoadWareHouse\ScoReFlow\data'
export REINFLOW_LOG_DIR='D:\GitLoadWareHouse\ScoReFlow\logs'
```

### 脚本调用约定

所有 `scripts/` 下的 `.sh` 都假设 **当前工作目录是项目根**,并以相对路径调用 `python run.py --config-dir=cfg/...`。**正确用法:**

```bash
cd D:/GitLoadWareHouse/ScoReFlow
bash scripts/train/robomimic/train_robomimic_finetune-grpo.sh
bash scripts/train/gym/train_gym_finetune-with-score.sh
bash scripts/eval/kitchen/eval_kitchen_finetune.sh
```

---

## Experiments

### Supported Environments

| Environment | Type | Observation | Policy Architecture |
|------------|------|-------------|-------------------|
| D4RL Gym (Hopper, Walker2d, Ant, Humanoid) | Locomotion | State | FlowMLP |
| Franka Kitchen (Complete, Partial, Mixed) | Manipulation | State | ShortCutFlowMLP |
| Robomimic (Square, Can, Transport) | Manipulation | Image | ShortCutFlowViT |

### Compared Methods

| Method | Description |
|--------|-------------|
| **Diffusion (DPPO)** | Diffusion policy + PPO fine-tuning |
| **Flow (ReinFlow)** | Flow matching + PPO (drift-only control) |
| **ShortCut (ReinFlow-S)** | Shortcut flow + PPO (drift-only control) |
| **Score-SDE (fixed)** | Fixed hyperparameter $\epsilon_t$ for score guidance |
| **ScoRe-Flow (ours)** | Learnable decoupled GammaNet for score guidance |

### Ablation Studies

We provide comprehensive ablation experiments in `visualize/Final_experiments/data/ablation/`:

- **Alpha schedule ablation** (`alpha/`): Comparing $\alpha=0$ (no score), $\alpha=1$ (constant), and $\alpha_\psi(t)$ learned (ours)
- **Lambda coupling ablation** (`lamda(mlp)/`): Comparing hyperparameter-coupled, learned-coupled (EpsNet), and learned-decoupled (GammaNet, ours)
- **Alpha analysis** (`alpha/alpha_visual/`): Visualization of how the learned $\alpha_\psi(t)$ converges during training

### Reproducing Results

```bash
# Pre-training (flow matching)
bash script/train/robomimic/pretrain.sh

# Fine-tuning with ScoRe-Flow
bash train_robomimic_finetune-with-score.sh        # GammaNet (ours)
bash train_gym_finetune-with-score.sh               # Gym tasks

# Ablation experiments
TASK=humanoid ALPHA=0.0 bash train_gym_finetune-score-gammanet-const.sh
TASK=humanoid ALPHA=1.0 bash train_gym_finetune-score-gammanet-const.sh
```

For complete experiment reproduction, see [docs/ReproduceExps.md](docs/ReproduceExps.md).

---

## Configuration

We use [Hydra](https://hydra.cc/) for configuration management.

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `gamma_score` | Maximum score guidance coefficient | 1.0 |
| `score_clip_value` | Score function clipping threshold | 10.0 |
| `score_scheduler_type` | GammaNet type: `mlp`, `linear`, `fixed` | `mlp` |
| `train.n_train_itr` | Training iterations | 301 |
| `train.n_steps` | Rollout steps per iteration | 400 |
| `train.batch_size` | Batch size | 500 |
| `train.actor_lr` | Actor learning rate | 1e-5 |
| `train.critic_lr` | Critic learning rate | 1e-3 |
| `env.n_envs` | Parallel environments | 50 |

### Example Configuration Override

```bash
python run.py \
    --config-dir=cfg/robomimic/finetune/transport \
    --config-name=ft_ppo_shortcut_mlp_img_with_score_gammanet \
    gamma_score=1.0 \
    score_scheduler_type=mlp \
    train.n_train_itr=201 \
    device=cuda:0
```

---

## Documentation

- [Installation Guide](installation/reinflow-setup.md)
- [Experiment Reproduction](docs/ReproduceExps.md)
- [Implementation Details](docs/Implement.md)
- [Known Issues](docs/KnownIssues.md)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
