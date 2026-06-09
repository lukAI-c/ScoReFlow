# Codex-Only Humanize Plan - Beat pi_RL Under the Official Protocol

## Goal Description

Develop, optimize, and evaluate CR-Reflow so that it exceeds the published
pi_RL pi0.5 result under an aligned LIBERO protocol. The previous 15-epoch,
8-evaluation-environment comparison is development evidence only and must not
be used to claim performance against pi_RL.

The first hard gate is reproduction: before method superiority is evaluated,
the repository must reproduce the published pi_RL Flow-Noise baseline using
the same model family, benchmark suite, training scale, and official
500-initial-state evaluation. Only after that gate passes may CR-Reflow
optimization results be compared with pi_RL.

Published reference points from arXiv:2510.25889v3, Table 3:

| pi0.5 method | Spatial | Object | Goal | Long | Average |
| --- | ---: | ---: | ---: | ---: | ---: |
| Few-shot SFT | 84.6 | 95.4 | 84.6 | 43.9 | 77.1 |
| Flow-SDE | 99.6 | 100.0 | 98.8 | 93.0 | 97.9 |
| Flow-Noise | 99.6 | 100.0 | 99.6 | 94.0 | 98.3 |

Published pi0.5 LIBERO training protocol from Table 11 includes 500 train
epochs, global batch size 2048, 64 parallel environments, 8 rollout epochs,
and suite-specific interaction steps. Published LIBERO evaluation uses 500
initial states per suite: 10 subtasks times 50 states.

## Acceptance Criteria

- AC-1: The pi_RL comparison contract is exact, executable, and enforced.
  - Positive Tests:
    - A machine-readable protocol records the pi0.5 checkpoint identity,
      suite, 500 training epochs, global batch size 2048, 64 parallel
      environments, 8 rollout epochs, suite-specific interaction steps,
      action horizon, update epochs, learning rates, scheduler, and evaluation
      state count.
    - Commands and manifests record every effective Hydra override and the
      exact checkpoint hash or immutable identifier.
    - The collector rejects official-comparison labeling when a run or
      evaluation does not satisfy the protocol.
  - Negative Tests:
    - The old 15-epoch or 8-evaluation-environment runs must never appear in an
      official pi_RL comparison table.
    - A final TensorBoard scalar from a training-time mini-evaluation must not
      be treated as the official 500-state result.

- AC-2: The published pi0.5 Flow-Noise baseline is reproduced before method
  optimization claims.
  - Positive Tests:
    - Flow-Noise is trained or restored from a protocol-aligned official
      checkpoint with provenance recorded.
    - A standalone evaluator runs all 500 initial states for LIBERO Spatial and
      emits per-task and aggregate success counts.
    - The Spatial reproduction reaches at least 99.0%, with the gap to the
      published 99.6% explicitly reported. A lower result blocks superiority
      claims and triggers root-cause diagnosis.
    - Before a four-suite superiority claim, aligned Flow-Noise evaluation is
      completed for Spatial, Object, Goal, and Long.
  - Negative Tests:
    - Do not tune CR-Reflow around a broken or materially underperforming
      Flow-Noise baseline.
    - Do not use fewer than 500 evaluation states as final benchmark evidence.

- AC-3: CR-Reflow is optimized against the aligned baseline without protocol
  drift.
  - Positive Tests:
    - CR-Reflow and Flow-Noise use the same starting checkpoint, training
      budget, environment distribution, evaluation states, and model-freezing
      policy unless an ablation explicitly documents a difference.
    - Optimization may cover anchor schedule, KL epsilon, eta bounds, weight
      clipping, update epochs, and interaction/replay choices while preserving
      the core reward-weighted reflow method.
    - Every optimization decision cites measured aligned runs and retains
      failed configurations.
    - Numerical invariants from the completed Round 9 implementation remain
      tested: valid weights are finite and strictly positive, returned KL is
      bounded, and metric-aware evidence counts are correct.
  - Negative Tests:
    - No toy, readiness, reduced-budget, or unmatched run may select the final
      method or support a superiority claim.
    - Do not silently increase CR-Reflow resources relative to the matched
      Flow-Noise control.

- AC-4: The final result exceeds pi_RL under a predeclared gate.
  - Positive Tests:
    - Primary Spatial gate: CR-Reflow records 500/500 successes, exceeding the
      published pi_RL Spatial result of 99.6%.
    - Four-suite gate: CR-Reflow is evaluated on 500 states for each of
      Spatial, Object, Goal, and Long and exceeds the published Flow-Noise
      average of 98.3%, without lowering any suite by more than 0.4 percentage
      points relative to the corresponding published result.
    - Results include exact success counts, confidence intervals, checkpoint
      identifiers, manifests, commands, logs, and raw per-episode outcomes.
    - Claims clearly distinguish exceeding the published reference from
      exceeding a locally reproduced matched baseline.
  - Negative Tests:
    - If neither superiority gate passes, report that the method has not beaten
      pi_RL and continue optimization while budget remains.
    - A tie, rounded tie, or lower-confidence mini-evaluation is not a win.

- AC-5: Real execution, reproducibility, and cost state are controlled.
  - Positive Tests:
    - All benchmark evidence comes from real pi0.5 LIBERO execution on Inspire.
    - Each run has a manifest, exact command, environment/config snapshot,
      checkpoint provenance, per-episode evaluation artifact, and terminal
      status.
    - GPU workload status is checked during execution and explicitly stopped
      when no approved experiment is active.
    - Local and remote code revisions are recorded and kept synchronized.
  - Negative Tests:
    - Do not report smoke tests, dry runs, launch-readiness, or environment
      checks as benchmark results.
    - Do not leave paid GPU resources idle.

## Path Boundaries

### Upper Bound

Reproduce the official pi_RL pi0.5 Flow-Noise LIBERO protocol, implement a
standalone 500-state evaluator and protocol validator, optimize CR-Reflow with
matched real runs, and complete the four-suite official comparison.

### Lower Bound

Complete the aligned Flow-Noise Spatial reproduction gate, prove the official
500-state evaluator, and identify with measured evidence any blocker that
prevents CR-Reflow optimization from proceeding.

### Allowed Choices

- Can use existing RLinf pi0.5 configs and released RLinf/pi_RL checkpoints
  after recording immutable provenance.
- Can use staged optimization: Spatial reproduction, Spatial method search,
  then four-suite confirmation.
- Can resume valid protocol-aligned checkpoints.
- Can improve runner, evaluator, collector, and remote patch tooling.
- Cannot treat the previous Round 8/9 reduced-budget results as pi_RL
  comparison evidence.
- Cannot weaken the official comparison gate to make the method appear better.
- Cannot fabricate, interpolate, or extrapolate missing benchmark results.

## Dependencies and Sequence

### Milestones

1. Protocol lock and audit
   - Inspect the live RLinf pi0.5 LIBERO configs and published implementation.
   - Record exact differences between the current runner and pi_RL protocol.
   - Add executable protocol validation and official-result labeling.

2. Official evaluator
   - Implement or expose deterministic 500-state LIBERO evaluation.
   - Emit raw per-episode and per-task outcomes plus aggregate metrics.
   - Verify that mini-evaluation and official evaluation cannot be confused.

3. Flow-Noise reproduction gate
   - Verify checkpoint provenance and model-freezing policy.
   - Run protocol-aligned Spatial Flow-Noise.
   - Diagnose and fix the pipeline until Spatial reaches at least 99.0%.

4. CR-Reflow aligned optimization
   - Run matched CR-Reflow configurations.
   - Use measured results to optimize method-specific parameters.
   - Preserve matched Flow-Noise controls and failed-run evidence.

5. Superiority confirmation
   - Confirm the best CR-Reflow checkpoint on the full Spatial 500-state gate.
   - Run all four LIBERO suites and compute the official four-suite result.
   - Accept success only if a predeclared AC-4 superiority gate passes.

## Implementation Notes

- Treat `/Users/qiuxiaotian/Documents/New project/ScoReFlow-rlinf-scoreflow-clean`
  as the Humanize project root.
- This is a Codex-only Humanize RLCR.
- The completed Round 9 branch is implementation history and numerical
  groundwork, not final performance evidence.
- Start from the current committed state and preserve the backup branch
  `codex/backup-cr-reflow-rlcr-round9-20260609`.
- Do not manually edit Humanize state files.
