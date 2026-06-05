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
