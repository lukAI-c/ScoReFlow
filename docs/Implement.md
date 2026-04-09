# Implementation Details

This document details key hyperparameters and code entry points for various fine-tuning methods in ScoReFlow.
All RL fine-tuning agents inherit from [`agent/finetune/train_agent.py`](../agent/finetune/train_agent.py),
with models concentrated in [`model/flow/ft_ppo/`](../model/flow/ft_ppo/).

---

## 1. ScoReFlow (Score-SDE + AlphaNet) — Core Method

Jointly optimizes drift (velocity field) and diffusion (noise) during RL fine-tuning, achieving complete distributional control via analytically-derived score functions + a lightweight learnable time-dependent schedule **AlphaNet** $\alpha_\psi(t)$.

$$
dx_t = \big[v_\theta(x_t,t) + \alpha_\psi(t)\cdot s(x_t,t)\big]\,dt + \sqrt{2\,\alpha_\psi(t)}\,dW_t
$$

where $s(x_t,t)=\dfrac{t\cdot v_\theta-x_t}{1-t}$ is the analytical score.

### Key Code

| Component | Path |
|---|---|
| 1-ReFlow + Score-SDE + AlphaNet | [`ppoflow_with_score_gammanet.py`](../model/flow/ft_ppo/ppoflow_with_score_gammanet.py) |
| ShortCut + Score-SDE + AlphaNet | [`pposhortcut_with_score_gammanet.py`](../model/flow/ft_ppo/pposhortcut_with_score_gammanet.py) |
| Generic score computation | [`score_utils.py`](../model/flow/score_utils.py) |
| Train agent (state) | [`train_ppo_flow_agent_score.py`](../agent/finetune/reinflow/train_ppo_flow_agent_score.py) |
| Train agent (image) | [`train_ppo_flow_img_agent_score.py`](../agent/finetune/reinflow/train_ppo_flow_img_agent_score.py) |
| ShortCut AlphaNet agent | [`train_ppo_shortcut_gammanet_agent.py`](../agent/finetune/reinflow/train_ppo_shortcut_gammanet_agent.py) |

> Note: Python filenames/class names still use `gammanet`; config files (yaml / sh) unified to `alphanet`.

### ScoReFlow-Specific Hyperparameters

- `gamma_score`: Overall scale of score guidance strength $\alpha_\psi$. Default `1.0`, tunable in `[0.1, 2.0]`.
- `score_scheduler_type`: AlphaNet schedule type (default `learn` in code).
- `model.gamma_net.hidden_dims`: AlphaNet hidden dimensions, default `[32, 32]`.
- `model.gamma_net.activation`: Activation function, default `silu` with final `softplus` ensuring $\alpha\geq 0$.
- Boundary handling: Hard time mask `(1-t)` prevents score singularity at $t=1$.

### Relationship to Existing Methods

| Method | Has score | $\alpha$ learnable | Config example |
|---|---|---|---|
| ReinFlow / DPPO | No (drift-only) | — | `ft_ppo_reflow_mlp.yaml` |
| Fixed Score-SDE | Yes | No (constant) | `ft_ppo_*_with_score_alphanet_const.yaml` |
| ScoReFlow | Yes | Yes (AlphaNet) | `ft_ppo_*_with_score_alphanet.yaml` |

### Common Ablation Entry Points

- **Freeze $v_\theta$, learn AlphaNet only**: [`pposhortcut_with_score_gammanet_frozen_v.py`](../model/flow/ft_ppo/pposhortcut_with_score_gammanet_frozen_v.py)
- **AlphaNet conditioned on observations**: [`ppoflow_with_score_gammanet_obs.py`](../model/flow/ft_ppo/ppoflow_with_score_gammanet_obs.py)
- **Decoupled dual-loss AlphaNet**: [`pposhortcut_with_score_gammanet_dloss.py`](../model/flow/ft_ppo/pposhortcut_with_score_gammanet_dloss.py)

---

## 2. GRPO (Group-Relative Policy Optimization) — Critic-Free Fine-tuning

Critic-free alternative that removes the critic network entirely, using episode-level group-relative advantage:
$$
A_i = \dfrac{R_i - \mathrm{mean}(R)}{\mathrm{std}(R)}
$$
with explicit KL penalty replacing value loss.

### Key Code

| Component | Path |
|---|---|
| GRPO replay buffer | [`buffer_grpo.py`](../agent/finetune/reinflow/buffer_grpo.py) |
| GRPO agent (state) | [`train_grpo_flow_agent.py`](../agent/finetune/reinflow/train_grpo_flow_agent.py) |
| GRPO agent (image) | [`train_grpo_flow_img_agent.py`](../agent/finetune/reinflow/train_grpo_flow_img_agent.py) |
| GRPO + Score-SDE model | [`grpoflow_with_score_gammanet.py`](../model/flow/ft_ppo/grpoflow_with_score_gammanet.py) |
| GRPO baseline model | [`grpoflow.py`](../model/flow/ft_ppo/grpoflow.py) |

### GRPO-Specific Hyperparameters

- `train.kl_coef`: KL penalty coefficient. Recommended `0.04` (state), `0.1~0.2` (image). Too high stops exploration; too low drifts from base policy.
- `train.ent_coef`: Entropy regularization. Recommended `0.01` (image) / `0.03` (state) for GRPO.
- Key differences from PPO: **no** `train.gamma`, `train.gae_lambda`, `train.value_loss_coef`, `model.critic.*`.

### Invocation

```bash
# Robomimic GRPO + Score-SDE
TASK=square SEED=42 KL_COEF=0.2 \
    bash scripts/train/robomimic/train_robomimic_finetune-grpo.sh

# Kitchen GRPO
TASK=kitchen SEED=42 KL_COEF=0.04 \
    bash scripts/train/gym/train_gym_finetune-grpo.sh
```

---

## 3. ReinFlow / Flow Matching PPO Baseline (Drift-Only)

PPO fine-tuning without score correction. Code entry points:

| Component | Path |
|---|---|
| FlowMLP / NoisyFlowMLP / VisionFlowMLP | [`mlp_flow.py`](../model/flow/mlp_flow.py) |
| ShortCutFlowMLP / NoisyShortCutFlowMLP | [`mlp_shortcut.py`](../model/flow/mlp_shortcut.py) |
| State agent | [`train_ppo_flow_agent.py`](../agent/finetune/reinflow/train_ppo_flow_agent.py) |
| Image agent | [`train_ppo_flow_img_agent.py`](../agent/finetune/reinflow/train_ppo_flow_img_agent.py) |

### Key Hyperparameters

- `min_std` / `max_std`: Noise std bounds at each denoising step.
- `denoising_steps`: Number of denoising steps (consistent across pre-training and fine-tuning).
- `ft_denoising_steps`: Actual fine-tuned steps counted backward from last step (default = `denoising_steps`).
- `train.clip_intermediate_actions`: Enable during fine-tuning and evaluation to prevent action overflow.
- `model.denoised_clip_value`: Max absolute denoised action, default `1`.
- `model.randn_clip_value`: Max std multiplier per sample, default `3`.
- `model.clip_ploss_coef`: PPO clipping $\epsilon$. State: `0.01`, visual: `0.001`.
- `model.logprob_min` / `model.logprob_max`: Log-probability clipping range. Policy collapse appears as very negative logprob.
- `model.noise_scheduler_type`: Noise schedule. Recommend `learn_decay` for state locomotion, `constant` for fixed noise.
- `model.use_time_independent_noise`: Whether noise depends on observations and time.
- `model.critic.out_bias_init`: Critic output layer bias. Set positive (e.g., 4.0) if critic outputs negative values despite high policy success.
- `train.use_bc_loss` / `train.bc_loss_type` / `train.bc_loss_coeff`: Behavior cloning regularization (usually only for hopper).
- `train.ent_coef`: Entropy coefficient. State: `0.03`, visual: `0`.

---

## 4. DPPO Baseline (Diffusion Policy + PPO)

| Component | Path |
|---|---|
| Official DPPO agent | [`agent/finetune/dppo/train_ppo_diffusion_agent.py`](../agent/finetune/dppo/train_ppo_diffusion_agent.py) |
| ReinFlow improved version (resume + verbose logging) | [`agent/finetune/reinflow/train_ppo_diffusion_agent.py`](../agent/finetune/reinflow/train_ppo_diffusion_agent.py) |
| Image version | [`agent/finetune/reinflow/train_ppo_diffusion_img_agent.py`](../agent/finetune/reinflow/train_ppo_diffusion_img_agent.py) |

### Key Hyperparameters

- `denoising_steps`: Number of denoising steps (pre-train and fine-tune consistent).
- `ft_denoising_steps`: Actual fine-tuned steps.
- `horizon_steps`: Action chunk size (should equal `act_steps` for MLP).
- `model.gamma_denoising`: Denoising discount factor.
- `model.min_sampling_denoising_std`, `model.min_logprob_denoising_std`.
- `model.clip_ploss_coef`: PPO clipping ratio.
- `train.batch_size`: DPPO uses large batches (takes expectation over both env and denoising steps).

### DDIM Fine-tuning

Pre-train with `denoising_steps=100`; fine-tune with `model.use_ddim=True`, `model.ddim_steps=<target>`, `ft_denoising_steps=<target>`.

---

## 5. FQL (Offline-to-Online Flow Q-Learning) Baseline

| Hyperparameter | Meaning |
|---|---|
| `offline_steps` | Offline fine-tuning iterations |
| `online_steps` | Online fine-tuning iterations |
| `eval_base_model` | Debug: periodically eval base policy evolution |

Entry example: `bash scripts/train/robomimic/train_square_fql.sh`

---

## 6. Offline RL Baselines (Cal-QL / IBRL / RLPD)

Inherited from DPPO, located in [`agent/finetune/offlinerl_baselines/`](../agent/finetune/offlinerl_baselines/):

- [`train_calql_agent.py`](../agent/finetune/offlinerl_baselines/train_calql_agent.py)
- [`train_ibrl_agent.py`](../agent/finetune/offlinerl_baselines/train_ibrl_agent.py)
- [`train_rlpd_agent.py`](../agent/finetune/offlinerl_baselines/train_rlpd_agent.py)

---

## 7. Diffusion x RL Baselines (RWR / DAWR / DIPO / DQL / IDQL / QSM)

Inherited from DPPO, located in [`agent/finetune/diffusion_baselines/`](../agent/finetune/diffusion_baselines/), used for fair comparison with ScoReFlow.

---

## 8. Adding a Custom RL Algorithm

Inherit from [`agent/finetune/train_agent.py`](../agent/finetune/train_agent.py)'s `TrainAgent` base class,
write a Hydra config in `cfg/.../ft_<your_algo>_*.yaml`,
and set `_target_:` to point to your class. `run.py` will auto-load via `hydra.utils.get_class(cfg._target_)`.
