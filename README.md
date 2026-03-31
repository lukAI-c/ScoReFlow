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
git clone https://github.com/ScoRe-Flow/ScoRe-Flow.git
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
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"

python script/run.py \
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
MUJOCO_GL="egl" xvfb-run -a -s "-screen 0 1024x768x24" python script/run.py \
    --config-dir=cfg/gym/finetune/kitchen-complete-v0 \
    --config-name=ft_ppo_shortcut_mlp_with_score_gammanet \
    device=cuda:0 \
    gamma_score=1.0 \
    seed=42
```

### Evaluation

```bash
bash script/eval/robomimic.sh
```

---

## Project Structure

```
ScoRe-Flow/
├── agent/                      # Training and evaluation agents
│   ├── pretrain/               # Pre-training agents
│   ├── finetune/               # Fine-tuning agents
│   │   └── reinflow/           # PPO-based RL fine-tuning
│   │       ├── train_ppo_flow_agent.py           # Gym state agent
│   │       ├── train_ppo_flow_img_agent.py       # Robomimic image agent
│   │       └── train_ppo_shortcut_gammanet_agent.py  # GammaNet agent
│   └── eval/                   # Evaluation & visualization
│       └── visualize/          # Plotting scripts
├── model/                      # Neural network architectures
│   └── flow/
│       ├── ft_ppo/             # RL fine-tuning models
│       │   ├── ppoflow_score.py                  # Score-SDE (fixed schedule)
│       │   ├── ppoflow_with_score_gammanet.py    # Score-SDE + GammaNet (ours)
│       │   ├── pposhortcut_with_score_gammanet.py # ShortCut + GammaNet (ours)
│       │   ├── ppoflow_score_mlp.py              # Score-SDE + EpsNet (coupled)
│       │   └── pposhortcut_score_mlp.py          # ShortCut + EpsNet (coupled)
│       ├── mlp_flow.py         # FlowMLP, NoisyFlowMLP, VisionFlowMLP
│       ├── mlp_shortcut.py     # ShortCutFlowMLP, NoisyShortCutFlowMLP
│       └── score_utils.py      # ScoreFunctionMixin (score computation)
├── cfg/                        # Hydra configuration files
│   ├── gym/                    # D4RL Gym & Kitchen configs
│   └── robomimic/              # Robomimic configs
├── env/                        # Environment wrappers
├── script/                     # Training and utility scripts
├── visualize/                  # Experimental data and output figures
│   └── Final_experiments/
│       ├── data/               # CSV data from wandb
│       └── outs/               # Generated plots (PDF + PNG)
└── sample_figs/                # Demo videos and overview figure
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
python script/run.py \
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
