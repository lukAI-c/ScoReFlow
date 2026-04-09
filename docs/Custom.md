## Custom Datasets / Environments

### Pre-training Data

Pre-training entry point: [`agent/pretrain/train_diffusion_agent.py`](../agent/pretrain/train_diffusion_agent.py).
Data loader: [`agent/dataset/sequence.py`](../agent/dataset/sequence.py).
Dataset must be a `.npz` file containing the following numpy arrays:

| Field | Shape | Description |
|---|---|---|
| `states` | `num_total_steps × obs_dim` | State sequence |
| `actions` | `num_total_steps × act_dim` | Action sequence |
| `images` | `num_total_steps × C × H × W` | Image observations (optional; multi-camera: channels concatenated; `H == W` and multiple of 8) |
| `traj_lengths` | `1-D` | Trajectory boundary indices for slicing |

#### Using Existing D4RL Data (Gym / Kitchen)

Download raw `.hdf5` from [Hugging Face mirror](https://huggingface.co/datasets/imone/D4RL/tree/main), then convert:

```bash
# 1. Inspect hdf5 structure
python data_process/read_hdf5.py --file_path=<PATH_TO_YOUR_HDF5>

# 2. Convert to npz + normalize to [-1, 1], output train.npz / normalization.npz
python data_process/hdf5_to_npz.py --data_path=<PATH_TO_YOUR_HDF5>

# 3. Inspect conversion result
python data_process/read_npz.py --data_path=<DIR>/train.npz
```

> Alternatively, pre-training commands with `use_d4rl_dataset=True` auto-download and process.

#### Robomimic Image Data

Pre-training config in `cfg/robomimic/pretrain/<task>/pre_*.yaml` already has Google Drive links.
First run of `python run.py --config-dir=cfg/robomimic/pretrain/square --config-name=pre_shortcut_mlp_img`
automatically downloads dataset and normalization statistics.

### Observation History

ScoReFlow defaults to current-timestep observations only, but code supports stacking history. In config set:

```yaml
cond_steps: 4         # State history steps
img_cond_steps: 2     # Image history steps (must be ≤ cond_steps)
```

Keep consistent between pre-training and fine-tuning.

### Fine-tuning Environments

Environments follow the Gym interface. Vectorized env init entry: [`env/gym_utils/__init__.py:make_async`](../env/gym_utils/__init__.py).
Parent class: [`agent/finetune/train_agent.py`](../agent/finetune/train_agent.py).

ScoReFlow uses these wrappers:
- [`env/gym_utils/wrapper/multi_step.py`](../env/gym_utils/wrapper/multi_step.py) — History stacking + multi-step action execution
- [`env/gym_utils/wrapper/robomimic_lowdim.py`](../env/gym_utils/wrapper/robomimic_lowdim.py) — Robomimic state normalization
- [`env/gym_utils/wrapper/mujoco_locomotion_lowdim.py`](../env/gym_utils/wrapper/mujoco_locomotion_lowdim.py) — D4RL Gym/Kitchen normalization

To integrate a new environment, implement a wrapper and add a branch in `make_async`.
