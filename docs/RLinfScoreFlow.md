# RLinf Original Score-Flow Port

This branch contains a clean RLinf/OpenPI port of the original Score-Flow idea.

## Method Boundary

Included:

- RLinf/OpenPI action sampling patch.
- Original Score-Flow score drift:

```text
score_t = (t * v_theta(x_t, t, s) - x_t) / (1 - t)
x_t_mean <- x_t_mean + dt * alpha_psi(t) * score_t
```

- Learned time coefficient `alpha_psi(t)` implemented as a small MLP.
- `flow_noise` stochastic rollout as the intended original Score-Flow noise path.
- Minimal LIBERO runner for `flow_noise_baseline` versus `scoreflow_original`.

Excluded:

- Terminal action critic guidance.
- Direct `x_t` critic guidance.
- FMTT / flow-map tilting.
- `sigma^2` score-coefficient ablation.
- Spec-Flow / verifier / draft-action tests.
- pIRL critic diagnostics.

## Files

- `scripts/rlinf_scoreflow/patch_openpi_score_flow.py`
  - Patches RLinf's `rlinf/models/embodiment/openpi/openpi_action_model.py`.
- `scripts/rlinf_scoreflow/run_libero_score_flow.sh`
  - Runs matched LIBERO jobs after applying the patch.
- `scripts/rlinf_scoreflow/collect_libero_score_flow.py`
  - Collects TensorBoard scalars and produces CSV summaries.

## Usage

Patch an RLinf checkout:

```bash
python scripts/rlinf_scoreflow/patch_openpi_score_flow.py \
  --rlinf-root /path/to/RLinf
python -m py_compile \
  /path/to/RLinf/rlinf/models/embodiment/openpi/openpi_action_model.py
```

Run LIBERO:

```bash
RLINF_ROOT=/path/to/RLinf \
PYTHON_BIN=/path/to/RLinf/.venv/bin/python \
MODEL_DIR=/path/to/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT \
SUITES="libero_spatial" \
SEEDS="42 43 44" \
bash scripts/rlinf_scoreflow/run_libero_score_flow.sh
```

Collect:

```bash
python scripts/rlinf_scoreflow/collect_libero_score_flow.py \
  --exp-root /path/to/RLinf/logs/scoreflow_original_libero \
  --output-dir /path/to/RLinf/logs/scoreflow_original_libero/artifacts
```

Only runs with at least three seeds should be treated as main-table evidence.
