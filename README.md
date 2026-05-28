# ScoReFlow RLinf Score-Flow

Clean RLinf/OpenPI port of the original Score-Flow mechanism.

This private repository is intentionally scoped to the original Score-Flow method only. It does not include terminal action guidance, direct `x_t` critic guidance, FMTT, `sigma^2` score-coefficient ablations, Spec-Flow, or pIRL diagnostic experiments.

## Method

The port adds a learned time-dependent score drift to RLinf/OpenPI action sampling:

```text
score_t = (t * v_theta(x_t, t, s) - x_t) / (1 - t)
x_t_mean <- x_t_mean + dt * alpha_psi(t) * score_t
```

The intended sampling path uses RLinf/OpenPI `flow_noise` stochastic rollout plus the learned coefficient `alpha_psi(t)`.

Supported modes:

```text
score_flow_mode=none
score_flow_mode=learned_alpha
```

## Files

```text
scripts/rlinf_scoreflow/patch_openpi_score_flow.py
scripts/rlinf_scoreflow/run_score_flow_benchmark.sh
scripts/rlinf_scoreflow/run_libero_score_flow.sh
scripts/rlinf_scoreflow/collect_libero_score_flow.py
docs/RLinfScoreFlow.md
docs/PiRLComparisonPlan.md
```

## Patch RLinf

```bash
python scripts/rlinf_scoreflow/patch_openpi_score_flow.py \
  --rlinf-root /path/to/RLinf

python -m py_compile \
  /path/to/RLinf/rlinf/models/embodiment/openpi/openpi_action_model.py
```

## Run LIBERO

```bash
RLINF_ROOT=/path/to/RLinf \
PYTHON_BIN=/path/to/RLinf/.venv/bin/python \
MODEL_DIR=/path/to/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT \
LIBERO_EXTRA_PYTHONPATH=/path/to/LIBERO \
OSMESA_LIBRARY_DIR=/path/to/osmesa_runtime/lib \
SUITES="libero_spatial libero_object libero_goal libero_10 maniskill metaworld_mt50 calvin_d_d" \
SEEDS="42 43 44" \
bash scripts/rlinf_scoreflow/run_score_flow_benchmark.sh
```

Default comparison:

```text
flow_noise_baseline
scoreflow_original
```

## Collect Results

```bash
python scripts/rlinf_scoreflow/collect_libero_score_flow.py \
  --exp-root /path/to/RLinf/logs/scoreflow_original_libero \
  --output-dir /path/to/RLinf/logs/scoreflow_original_libero/artifacts
```

Use at least three seeds before treating a result as main-table evidence.

## Notes

This repository is a clean method-transfer branch. For implementation details and method boundary, see [docs/RLinfScoreFlow.md](docs/RLinfScoreFlow.md). For the pi_RL comparison matrix, see [docs/PiRLComparisonPlan.md](docs/PiRLComparisonPlan.md).
