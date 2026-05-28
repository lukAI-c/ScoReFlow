# pi_RL Comparison Plan

Goal: compare the clean RLinf/OpenPI original Score-Flow port against pi_RL baselines on the benchmark families reported by RLinf/pi_RL.

## Benchmarks

| Benchmark | Suite in runner | Main metric |
| --- | --- | --- |
| LIBERO | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` | `env/success_once` |
| ManiSkill3 | `maniskill` | `env/success_once` |
| MetaWorld | `metaworld_mt50` | `env/success_once` |
| CALVIN | `calvin_d_d` | average completed subtasks and sequence success |

## Methods

| Method | Meaning |
| --- | --- |
| `flow_noise_baseline` | RLinf/OpenPI `flow_noise` without Score-Flow score drift |
| `scoreflow_original` | RLinf/OpenPI `flow_noise` plus learned-alpha Score-Flow drift |

Both methods use the same RLinf config, model checkpoint, seed, rollout settings, and training budget. The only intended difference is `score_flow_mode`.

## pi_RL Reference Points

RLinf/pi_RL reports the following reference metrics for the relevant flow baselines:

| Model | Benchmark | Task | Flow-SDE | Flow-Noise |
| --- | --- | --- | --- | --- |
| pi0 | ManiSkill3 | Multi-task | 78.8% | 77.8% |
| pi0 | MetaWorld | MT50 | 78.1% | 85.8% |
| pi0 | CALVIN | ABC-D | 61.7% | 59.9% |
| pi0.5 | ManiSkill3 | Multi-task | 90.9% | 89.7% |
| pi0.5 | MetaWorld | MT50 | 70.7% | 66.1% |
| pi0.5 | CALVIN | ABC-D | 87.0% | 84.5% |

For LIBERO, pi_RL reports suite-level success rates. The comparison should keep Spatial, Object, Goal, and Long separate before computing averages, because Long is the long-horizon stress case.

## Execution Matrix

Recommended first matrix:

```bash
POLICY_VARIANT=pi0 \
SUITES="libero_spatial libero_object libero_goal libero_10 maniskill metaworld_mt50 calvin_d_d" \
SEEDS="42 43 44" \
METHODS="flow_noise_baseline scoreflow_original" \
bash scripts/rlinf_scoreflow/run_score_flow_benchmark.sh
```

Then repeat with:

```bash
POLICY_VARIANT=pi05
```

## Decision Rule

Treat Score-Flow as outperforming pi_RL only when all of the following hold:

- At least three seeds are completed.
- The method beats the matched `flow_noise_baseline` under the same config and budget.
- The method beats the relevant pi_RL reference metric or improves the weakest suite without regressing average success.
- CALVIN reports both average completed subtasks and sequence length success, not only one scalar.

Single-seed runs are launch/readiness evidence only.

## Reference Sources

- RLinf pi_RL publication page: `https://rlinf.readthedocs.io/en/latest/rst_source/publications/pi_rl.html`
- RLinf pi0/pi0.5 embodied guide: `https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0.html`
