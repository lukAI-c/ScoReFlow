# CR-Reflow Evidence and Edit Map

Date: 2026-06-05

## Live H100 Status

Command:

```bash
MPLCONFIGDIR=/private/tmp/mpl-inspire inspire notebook status scoreflow-h100b-0603 --workspace '分布式训练空间'
```

Observed output:

```text
Notebook: scoreflow-h100b-0603
Status         : STOPPED
Project        : 面向未来智能终端的端侧大模型推理芯片
Priority       : 10
Compute Group  : cuda12.9版本H100
Image          : pytorch:25.06-py3:25.06
GPU            : 1x H100
CPU            : 10
Memory         : 200 GiB
SHM            : 64 GiB
Workspace      : 分布式训练空间
```

Because the H100 notebook was stopped, Round 1 preserves the existing evidence
baseline and does not restart the notebook for a costly benchmark or remote file
inspection.

## Negative Gate Evidence

Source note: `/Users/qiuxiaotian/Documents/New project/plan/terminal_kl_flow_ppo_proposal_0603.md`.

The completed `tr_matched_0604` beta=0.03 manifest rows were:

| seed | exit | success trajectory summary | final approx_kl | pullback_disp | scalar_l2_disp |
| --- | ---: | --- | ---: | ---: | ---: |
| 42 | 0 | early 1.0, then 0.25, 0.125, terminal zeros | 0.107 | 0.681 | 0.108 |
| 43 | 0 | stable high final 0.875 | 0.138 | 0.604 | 0.177 |
| 44 | 0 | early 1.0, then 0.625, terminal zeros | 0.097 | 0.573 | 0.114 |

Measured decision: matched terminal pullback did not match the scalar L2 KL
control quality, and seeds 42/44 showed terminal-zero collapse. This is a
negative gate for pullback-as-core-novelty and motivates the CR-Reflow pivot.

The earlier `tr_compare_0603` table remains historical evidence and is not
rerun in this round.

## Local-to-Remote Edit Map

Remote root used by the handoff:

```text
/inspire/ssd/project/inference-chip/qiuxiaotian-253114010249
```

| Capability | Local file | Remote file or artifact |
| --- | --- | --- |
| OpenPI policy implementation | `openpi_action_model.LIVE.py` | `$R/RLinf/rlinf/models/embodiment/openpi/openpi_action_model.py` |
| Install policy into RLinf | `scripts/rlinf_scoreflow/patch_openpi_score_flow.py` | Copies local LIVE file into the remote OpenPI model with a backup |
| Benchmark launch routing | `scripts/rlinf_scoreflow/run_score_flow_benchmark.sh` | `$R/ScoReFlow-RLinf-ScoreFlow-clean/scripts/rlinf_scoreflow/run_score_flow_benchmark.sh` |
| Result collection | `scripts/rlinf_scoreflow/collect_libero_score_flow.py` | Parses remote log roots under `$R/logs/*` |
| Actor loss consumption | Not present in this repo checkout | `$R/RLinf/rlinf/workers/actor/fsdp_actor_worker.py` |
| Historical TR logs | Existing remote logs | `$R/logs/{tr_compare_0603,tr_matched_0604}` |

The 2026-06-05 handoff states that remote
`$R/RLinf/rlinf/models/embodiment/openpi/openpi_action_model.py` mirrors local
`openpi_action_model.LIVE.py`. This round edits the local LIVE file and updates
the installer so remote deployment preserves the full Score-Flow/TR/CR-Reflow
surface instead of replaying only the old narrow Score-Flow patch.

The local checkout does not contain `fsdp_actor_worker.py`, so this round exposes
`cr_reflow_loss` and `cr_reflow_diag` from the policy and records the remote
actor file as the remaining integration point for adding the loss to the PPO
objective, analogous to the existing remote `terminal_pullback_loss` hook.

## Round 2 Remote Actor Integration

Remote roots validated on the H100 notebook:

```text
ScoreFlow checkout: /inspire/ssd/project/inference-chip/qiuxiaotian-253114010249/ScoReFlow-RLinf-ScoreFlow-clean
RLinf checkout: /inspire/ssd/project/inference-chip/qiuxiaotian-253114010249/RLinf
Pi0.5 LIBERO model: /inspire/ssd/project/inference-chip/qiuxiaotian-253114010249/models/RLinf-Pi05-LIBERO-SFT
```

The real actor worker branch was inspected at:

```text
$R/RLinf/rlinf/workers/actor/fsdp_actor_worker.py
```

The embodied actor update path computes `loss, metrics_data = policy_loss(**kwargs)`,
then applies entropy and the existing `terminal_pullback_loss` hook. Round 2 adds
a backup-preserving installer for this real file:

```text
scripts/rlinf_scoreflow/patch_rlinf_cr_reflow_actor.py
```

Validated remote patch commands:

```bash
/usr/bin/python3 scripts/rlinf_scoreflow/patch_rlinf_cr_reflow_actor.py \
  --rlinf-root /inspire/ssd/project/inference-chip/qiuxiaotian-253114010249/RLinf \
  --check
```

Observed output:

```text
patched=False would_patch=True check=True path=/inspire/ssd/project/inference-chip/qiuxiaotian-253114010249/RLinf/rlinf/workers/actor/fsdp_actor_worker.py
```

The actual remote install then patched both OpenPI and the actor worker and
compiled both files:

```text
patched=True source=.../ScoReFlow-RLinf-ScoreFlow-clean/openpi_action_model.LIVE.py path=.../RLinf/rlinf/models/embodiment/openpi/openpi_action_model.py
patched=True would_patch=False check=False path=.../RLinf/rlinf/workers/actor/fsdp_actor_worker.py
```

The remote `py_compile` command completed with exit code 0 for:

```text
.../RLinf/rlinf/models/embodiment/openpi/openpi_action_model.py
.../RLinf/rlinf/workers/actor/fsdp_actor_worker.py
```

## Round 2 Launch-Readiness Manifest

The runner was executed in `PREPARE_ONLY=1` mode. This generated real pi0.5
LIBERO Spatial commands and manifest rows without starting training:

```bash
RLINF_ROOT=/inspire/ssd/project/inference-chip/qiuxiaotian-253114010249/RLinf \
PYTHON_BIN=/usr/bin/python3 \
SCOREFLOW_ROOT=/inspire/ssd/project/inference-chip/qiuxiaotian-253114010249/ScoReFlow-RLinf-ScoreFlow-clean \
MODEL_DIR=/inspire/ssd/project/inference-chip/qiuxiaotian-253114010249/models/RLinf-Pi05-LIBERO-SFT \
EXP_ROOT=/inspire/ssd/project/inference-chip/qiuxiaotian-253114010249/RLinf/logs/cr_reflow_readiness_0605 \
REAL_CONFIG_PRESET=1 \
PREPARE_ONLY=1 \
bash scripts/rlinf_scoreflow/run_score_flow_benchmark.sh
```

Observed output:

```text
PREPARED libero_spatial_cr_reflow_no_anchor_seed42
PREPARED libero_spatial_cr_reflow_seed42
```

Prepared manifest rows:

| suite | config | method | seed | status | exit |
| --- | --- | --- | ---: | --- | ---: |
| libero_spatial | libero_spatial_ppo_openpi_pi05 | cr_reflow_no_anchor | 42 | prepared | 0 |
| libero_spatial | libero_spatial_ppo_openpi_pi05 | cr_reflow | 42 | prepared | 0 |

The generated commands use `runner.max_epochs=15`, `env.train.total_num_envs=4`,
`env.eval.total_num_envs=8`, `actor.model.openpi.noise_method=flow_noise`,
`++actor.model.openpi.joint_logprob=true`, and either
`++actor.model.openpi.cr_reflow_mode=cr_reflow_no_anchor` with
`++actor.model.openpi.cr_reflow_anchor_beta=0.0`, or
`++actor.model.openpi.cr_reflow_mode=cr_reflow` with
`++actor.model.openpi.cr_reflow_anchor_beta=0.1`.

This is launch-readiness evidence only. No CR-Reflow benchmark step was started,
and these prepared rows must not be reported as method performance.

## Round 3 Live CR-Reflow Readiness

Round 3 fixed three launch blockers before the live check:

- `patch_rlinf_cr_reflow_actor.py` now encodes `cr_reflow_mode` as numeric
  `actor/cr_reflow_mode_code` and skips non-scalar diagnostics in actor scalar
  logging.
- `patch_openpi_score_flow.py` now installs only onto an identical target, the
  recorded pre-CR backup hash, or a marker-compatible Score-Flow/TR target; an
  unexpected remote target aborts and writes a `.unexpected_scoreflow_target.diff`
  evidence file.
- `collect_libero_score_flow.py` now prefers exact CR scalar tags, including the
  remote TensorBoard `train/actor/...` prefix, and handles bounded-stop readiness
  logs whose manifest has no completed `status` row.

Remote deployment on the H100 notebook used:

```text
ScoreFlow checkout: /inspire/ssd/project/inference-chip/qiuxiaotian-253114010249/ScoReFlow-RLinf-ScoreFlow-clean
RLinf checkout: /inspire/ssd/project/inference-chip/qiuxiaotian-253114010249/RLinf
Pi0.5 LIBERO model: /inspire/ssd/project/inference-chip/qiuxiaotian-253114010249/models/RLinf-Pi05-LIBERO-SFT
Readiness EXP_ROOT: /inspire/ssd/project/inference-chip/qiuxiaotian-253114010249/RLinf/logs/cr_reflow_round3_readiness_live_20260605_203440
```

Remote patch and compile evidence:

```text
patched=False compatibility=identical .../RLinf/rlinf/models/embodiment/openpi/openpi_action_model.py
patched=True would_patch=False check=False .../RLinf/rlinf/workers/actor/fsdp_actor_worker.py
remote py_compile passed for openpi_action_model.py, fsdp_actor_worker.py, and collect_libero_score_flow.py
```

Bounded real launches were run with the real pi0.5 LIBERO Spatial preset:

```text
--config-name libero_spatial_ppo_openpi_pi05
runner.max_epochs=15
env.train.total_num_envs=4
env.eval.total_num_envs=8
actor.seed=42
actor.model.model_path=$R/models/RLinf-Pi05-LIBERO-SFT
rollout.model.model_path=$R/models/RLinf-Pi05-LIBERO-SFT
```

Both CR methods reached real rollout/eval/actor-update execution before being
bounded-stopped:

| method | seed | event evidence | stop evidence |
| --- | ---: | --- | --- |
| `cr_reflow_no_anchor` | 42 | `.../libero_spatial_cr_reflow_no_anchor_seed42/tensorboard/events.out.tfevents.*` | `outer_exit_code_cr_reflow_no_anchor.txt` recorded `143` after manual bounded stop |
| `cr_reflow` | 42 | `.../libero_spatial_cr_reflow_seed42/tensorboard/events.out.tfevents.*` | `outer_exit_code_cr_reflow.txt` recorded `143` after manual bounded stop |

The collector output was written to:

```text
$EXP_ROOT/collected/scoreflow_benchmark_summary.csv
$EXP_ROOT/collected/scoreflow_benchmark_raw_scalars.csv
$EXP_ROOT/collected/scoreflow_benchmark_manifest.csv
```

Readiness scalar summary:

| method | seed | approx_kl | cr_loss | actor_loss | anchor_loss | eta | ESS | valid_fraction | legacy target MSE proxy | legacy target displacement | mode_code |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cr_reflow_no_anchor` | 42 | 0.004954 | 0.988706 | 0.988706 | 0.000000 | 3.088232 | 0.909466 | 1.000000 | 0.494353 | 5.840227 | 1.0 |
| `cr_reflow` | 42 | 0.026305 | 0.999127 | 0.999257 | 0.001298 | 2.888686 | 0.895039 | 0.723958 | 0.499564 | 5.813925 | 2.0 |

The selected CR tags were exact preferred matches against remote TensorBoard
tags such as `train/actor/cr_reflow_loss`,
`train/actor/cr_reflow_actor_loss`, and
`train/actor/cr_reflow_mode_code`.

Round 5 review established that the two legacy columns above were computed from
the new transition means versus sampled teacher-forcing targets. They are
reflow-target residuals, not new-vs-old policy movement. The measured values are
retained as historical readiness evidence, but must not be interpreted as a
chain KL or policy displacement.

This is launch-readiness and diagnostic evidence only. It is not a completed
method-performance comparison because both readiness jobs were intentionally
terminated after the first actor-update evidence was available.

## Round 3 Real Comparison Launch

After the bounded readiness checks passed, the planned real comparison was
started detached on the same H100 notebook:

```text
Comparison EXP_ROOT: /inspire/ssd/project/inference-chip/qiuxiaotian-253114010249/RLinf/logs/cr_reflow_round3_comparison_20260605_140344
Launcher PID: 3056367
Initial run: libero_spatial_flow_noise_baseline_seed42
```

The launcher uses:

```text
REAL_CONFIG_PRESET=1
PREPARE_ONLY=0
SEEDS="42 43 44"
METHODS="flow_noise_baseline tr_scalar_l2 cr_reflow_no_anchor cr_reflow"
```

Initial launcher log:

```text
patched=False compatibility=identical .../openpi_action_model.py
patched=False would_patch=False check=False .../fsdp_actor_worker.py
[Fri Jun  5 14:03:44 UTC 2026] START libero_spatial_flow_noise_baseline_seed42
```

At launch time, the H100 notebook was intentionally left running while this
approved comparison job was active.

## Round 4 Real Comparison Completion

The detached launcher exited after all planned real pi0.5 LIBERO Spatial runs
completed. Terminal verification against:

```text
/inspire/ssd/project/inference-chip/qiuxiaotian-253114010249/RLinf/logs/cr_reflow_round3_comparison_20260605_140344
```

showed:

```text
run_manifest.csv lines including header: 13
planned terminal rows: 12
status=done and exit_code=0 rows: 12
command.txt files: 12
run.log files: 12
TensorBoard event files: 12
launcher PID 3056367 alive: no
```

The completed method/seed matrix is:

```text
flow_noise_baseline: 42, 43, 44
tr_scalar_l2: 42, 43, 44
cr_reflow_no_anchor: 42, 43, 44
cr_reflow: 42, 43, 44
```

The collector ran only after terminal verification and wrote:

```text
$EXP_ROOT/collected/scoreflow_benchmark_summary.csv
$EXP_ROOT/collected/scoreflow_benchmark_raw_scalars.csv
$EXP_ROOT/collected/scoreflow_benchmark_manifest.csv
```

Measured final aggregates:

| method | seeds | final success mean | final success std | terminal collapse rate | approx_kl mean | approx_kl std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `flow_noise_baseline` | 3 | 0.500000 | 0.330719 | 0.333333 | 0.013045 | 0.007135 |
| `tr_scalar_l2` | 3 | 0.333333 | 0.381881 | 0.333333 | 0.013046 | 0.010650 |
| `cr_reflow_no_anchor` | 3 | 0.250000 | 0.000000 | 0.000000 | 0.009678 | 0.001252 |
| `cr_reflow` | 3 | 0.375000 | 0.216506 | 0.000000 | 0.005289 | 0.008758 |

Measured legacy CR target-residual aggregates:

| method | eta mean | ESS mean | valid fraction mean | legacy target MSE proxy mean | legacy target displacement mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cr_reflow_no_anchor` | 2.841646 | 0.910189 | 0.690972 | 0.500538 | 5.855503 |
| `cr_reflow` | 2.533030 | 0.905094 | 0.692708 | 0.489130 | 5.753311 |

The last two columns were originally logged under `chain_*` names. Round 5
review found that they compare new transition means to sampled targets rather
than old transition means. They remain valid measured target-residual values,
but are not policy-change evidence and do not support a conservative-policy
movement claim.

Measured comparison against `tr_scalar_l2`:

- `cr_reflow` final success mean is higher by `0.041667` (`0.375000` versus
  `0.333333`).
- `cr_reflow` mean approx_kl is `0.405446` times the `tr_scalar_l2` value
  (`0.005289` versus `0.013046`).
- `cr_reflow` terminal collapse rate is lower by `0.333333` (`0.000000`
  versus `0.333333`).

Interpretation: this 3-seed comparison gives a directional result in favor of
full CR-Reflow over `tr_scalar_l2` at a lower measured KL/collapse budget, but
the small sample and high seed variance do not support a statistical
significance or definitive superiority claim. `flow_noise_baseline` retains
the highest measured final success mean in this comparison.

After verifying that no training or GPU compute process remained, the H100
notebook `scoreflow-h100b-0603` was stopped. Live Inspire status returned
`STOPPED`.

## Round 5 Policy Diagnostics and FSDP Code Map

The policy diagnostics were corrected to separate teacher-forced fit from
new-vs-old policy movement:

- `cr_reflow_target_displacement` retains the new-mean-to-target residual.
- `cr_reflow_policy_displacement` is the weighted normalized norm between new
  and selected old transition means.
- `cr_reflow_policy_kl_proxy` is the weighted fixed-variance Gaussian
  mean-shift proxy, using selected old standard deviations.

The remote FSDP model manager was inspected at:

```text
$R/RLinf/rlinf/hybrid_engines/fsdp/fsdp_model_manager.py
```

The exact coverage path is:

1. `setup_model_and_optimizer()` creates the policy module and passes it through
   `self._strategy.wrap_model(model=module, device_mesh=self._device_mesh)`.
2. The wrapped `self.model` is passed into `build_optimizer(...)`.
3. `build_optimizer(...)` iterates `model.named_parameters()`, routes every
   trainable tensor into actor, critic, or score-flow parameter groups, and
   constructs `torch.optim.AdamW`.

CR-Reflow changes the loss over existing policy outputs and adds no standalone
trainable module or parameter. Its gradients therefore use the existing
FSDP-wrapped policy and actor optimizer group; no CR-specific optimizer or FSDP
registration is required.

The corrected policy and collector were deployed and compiled on the H100, then
verified with two completed real pi0.5 LIBERO Spatial one-epoch runs:

```text
Readiness EXP_ROOT: $R/RLinf/logs/cr_reflow_round5_policy_diag_readiness_20260606_062830
manifest terminal rows: 2
status=done and exit_code=0 rows: 2
```

The collector selected all corrected metrics from exact preferred
`train/actor/...` TensorBoard tags:

| method | target displacement | policy KL proxy | policy displacement | anchor loss | exact-tag source |
| --- | ---: | ---: | ---: | ---: | --- |
| `cr_reflow_no_anchor` | 5.863093 | 0.000635 | 0.175418 | 0.000000 | preferred |
| `cr_reflow` | 5.735723 | 0.000873 | 0.209690 | 0.001746 | preferred |

For full CR-Reflow, the observed `anchor_loss=0.001746008` equals twice the
observed `policy_kl_proxy=0.000873004`, as expected because the anchor is the
weighted normalized mean-shift squared and the Gaussian proxy applies the
`0.5` factor. The much smaller policy-change values are also clearly distinct
from the teacher-forced target displacement.

Collector artifacts:

```text
$EXP_ROOT/collected/scoreflow_benchmark_summary.csv
$EXP_ROOT/collected/scoreflow_benchmark_raw_scalars.csv
$EXP_ROOT/collected/scoreflow_benchmark_manifest.csv
```

After readiness and collection completed, live Inspire status confirmed H100
notebook `scoreflow-h100b-0603` was `STOPPED`.
