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

The current Object, Goal, and Long configs explicitly enable
`use_fixed_reset_state_ids`. Spatial inherits ordered state selection during
evaluation through `is_eval`, but its top-level config does not explicitly
lock fixed reset IDs. Therefore, 500 parallel evaluation environments alone
are not accepted as proof that the exact ten-by-fifty official initial-state
set was evaluated. The standalone evaluator explicitly enables fixed and
ordered reset IDs and emits raw per-episode state/task identifiers.

Primary upstream config references:

- `https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/libero_spatial_ppo_openpi_pi05.yaml`
- `https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/libero_10_ppo_openpi_pi05.yaml`
- `https://arxiv.org/pdf/2510.25889`

## Executable Enforcement

`PIRL_OFFICIAL_PROTOCOL=1` in
`scripts/rlinf_scoreflow/run_score_flow_benchmark.sh`:

- rejects non-LIBERO suites, missing model directories, and all trailing
  `EXTRA_OVERRIDES` that could replace locked protocol values;
- applies the published pi0.5 settings, including suite-specific replan and
  denoise values;
- writes denoise steps through RLinf's live `actor.model.num_steps` key and
  saves resumable checkpoints every 40 steps during the 500-step run;
- locks the OpenPI dataconfig to `pi05_libero`, whose model config defines the
  published action prediction horizon of 10, and uses strict-Hydra additions
  only for fields absent from the upstream suite config;
- exposes only CR-Reflow-specific KL, eta, weight-clip, and anchor parameters
  as explicit optimization variables while keeping the matched protocol
  fields locked;
- records the exact generated command and its SHA-256;
- derives model provenance from a deterministic SHA-256 of the actual model
  tree rather than trusting a caller-supplied label;
- emits and validates a machine-readable training protocol artifact before
  launch, and records command, the referenced RLinf base config and its hash,
  code revisions, environment, status, and exit code in a reproducibility
  bundle.

`scripts/rlinf_scoreflow/pirl_protocol.py` rejects:

- the previous reduced training budget;
- duplicate or replaced locked command overrides;
- missing, tampered, or caller-asserted model/checkpoint provenance;
- evaluation artifacts that do not contain exactly ten unique tasks, 50
  evaluated states per task, 500 total states, raw episode evidence, and
  internally consistent success counts;
- missing, tampered, malformed, or unhashed command, receipt, raw episode,
  terminal-status, training-artifact, and reproducibility evidence.

`scripts/rlinf_scoreflow/run_pirl_official_evaluation.sh` runs the standalone
evaluation entry point with the suite-specific interaction, replan, and
denoise settings. `patch_rlinf_pirl_evaluator.py` preserves
`reset_state_id`, `task_id`, and `trial_id` before RLinf aggregates the
results. The evaluator keeps the base SFT model path for policy configuration
and normalization stats, while requiring a separate trained RLinf checkpoint.
RLinf's native `runner.ckpt_path` loads
`actor/model_state_dict/full_weights.pt` into the rollout policy after base
initialization. The patched rollout worker writes a load receipt only after
strict `load_state_dict` succeeds; the receipt includes the actual loaded
path, state-dict key count, and SHA-256. `pirl_official_eval.py` rejects a
missing or mismatched receipt, duplicate or incomplete states, and writes
aggregate and per-task success with Wilson intervals. The independent
collector revalidates every official evaluation artifact and all referenced
hashes before including it in the official comparison table.

## Live Inspire Audit State

On June 9, 2026, the live RLinf checkout at commit
`0f2e3813be5d393381b4b690d5f75781fe4a70c8` was audited from the existing CPU
notebook. Its evaluation runner discards raw state-level outcomes after
aggregation, and its LIBERO environment emits success but not state/task IDs.
The standalone evaluator patch was checked against those exact remote files;
all three environment, evaluation-runner, and rollout-worker anchors matched.
The audit also confirmed that denoise steps must be overridden through
`actor.model.num_steps`. A previous reduced-budget checkpoint was used only to
verify the checkpoint-file layout and hashing path; it is not eligible as
official evidence.

`scoreflow-h100b-0603` was requested for execution but remained `PENDING` for
600 seconds because no H100 resource was allocated. It was stopped and
confirmed `STOPPED`. No GPU training or evaluation ran. The remaining blocker
is GPU execution of the matched Flow-Noise baseline and CR-Reflow candidates,
not the live config/evaluator audit.

As of June 9, 2026, the official RLinf Hugging Face organization exposes the
pi0.5 LIBERO SFT checkpoint, but no public pi0.5 LIBERO Flow-Noise RL
checkpoint was found. The aligned baseline therefore cannot be replaced with
a downloaded official RL checkpoint and must be reproduced by training.
