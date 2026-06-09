# pi_RL pi0.5 LIBERO Protocol Audit

## Decision

An official comparison in this repository means a pi0.5 LIBERO run that
passes `scripts/rlinf_scoreflow/pirl_protocol.py`. Training-time TensorBoard
scalars and reduced-budget runs are development evidence only.

The published Flow-Noise reference to exceed is 99.6% on LIBERO-Spatial and
98.3% averaged across Spatial, Object, Goal, and Long. The paper computes each
suite result over 500 initial states: 10 tasks times 50 states.

## Published Contract

Source: arXiv:2510.25889v3, Table 3 and Table 11.

Shared pi0.5 LIBERO settings:

| Field | Published value |
| --- | ---: |
| Train epochs | 500 |
| Global batch size | 2048 |
| Parallel environments | 64 |
| Rollout epochs | 8 |
| Actor learning rate | 5e-6 |
| Critic learning rate | 1e-4 |
| Reward discount | 0.99 |
| GAE lambda | 0.95 |
| PPO clip ratio | 0.2 |
| Action prediction horizon | 10 |
| Flow-Noise min/max log-var | 0.04 / 0.10 |
| Flow-Noise entropy bonus | 0.005 |
| Official evaluation states | 500 |

Suite-specific settings:

| Suite | Interaction steps | Update epochs | Replan horizon | Denoise steps | Scheduler |
| --- | ---: | ---: | ---: | ---: | --- |
| Spatial | 240 | 1 | 5 | 3 | false |
| Object | 320 | 1 | 5 | 5 | false |
| Goal | 320 | 3 | 5 | 5 | false |
| Long | 480 | 4 | 10 | 5 | true |

## Previous Runner Mismatch

The previous real-config preset used 15 epochs, global batch size 8, four
training environments, one rollout epoch, and eight evaluation environments.
Those runs do not match the published training or evaluation protocol and
cannot answer whether CR-Reflow exceeds pi_RL.

The previous collector summarized final training TensorBoard scalars. It did
not prove a standalone 500-state evaluation and now marks all such rows with
`official_comparison_eligible=false`.

## Current Upstream Config Audit

The current RLinf `main` Spatial and Long pi0.5 configs agree with the paper on
the core PPO batch, environment, rollout, learning-rate, and suite-specific
settings. They currently set `runner.max_epochs=1000`, while Table 11 reports
500 train epochs. The comparison contract therefore explicitly overrides the
published 500-epoch value instead of silently inheriting the current upstream
default.

The current Long config explicitly enables `use_fixed_reset_state_ids`. The
current Spatial config does not expose that flag in the top-level config.
Therefore, 500 parallel evaluation environments alone are not accepted as
proof that the exact ten-by-fifty official initial-state set was evaluated.
The standalone evaluator must emit raw per-episode state/task identifiers.

Primary upstream config references:

- `https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/libero_spatial_ppo_openpi_pi05.yaml`
- `https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/libero_10_ppo_openpi_pi05.yaml`
- `https://arxiv.org/pdf/2510.25889`

## Executable Enforcement

`PIRL_OFFICIAL_PROTOCOL=1` in
`scripts/rlinf_scoreflow/run_score_flow_benchmark.sh`:

- rejects non-LIBERO suites and non-immutable model provenance;
- applies the published pi0.5 settings, including suite-specific replan and
  denoise values;
- records the exact generated command and its SHA-256;
- emits and validates a machine-readable training protocol artifact before
  launch.

`scripts/rlinf_scoreflow/pirl_protocol.py` rejects:

- the previous reduced training budget;
- missing or mutable checkpoint provenance;
- evaluation artifacts that do not contain exactly ten unique tasks, 50
  evaluated states per task, 500 total states, raw episode evidence, and
  internally consistent success counts.

## Live Inspire Audit State

On June 9, 2026, `scoreflow-h100b-0603` was requested for live remote audit but
remained `PENDING` for 600 seconds because no H100 resource was allocated. It
was stopped and confirmed `STOPPED`. No GPU training or evaluation ran, and
the live remote checkout remains unaudited. This blocks a real launch, not the
local executable protocol implementation.
