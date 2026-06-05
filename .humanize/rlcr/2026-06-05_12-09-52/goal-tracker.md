# Goal Tracker

<!--
This file tracks the ultimate goal, acceptance criteria, and plan evolution.
It prevents goal drift by maintaining a persistent anchor across all rounds.

RULES:
- IMMUTABLE SECTION: Do not modify after initialization
- MUTABLE SECTION: Update each round, but document all changes
- Every task must be in one of: Active, Completed, or Deferred
- Deferred items require explicit justification
-->

## IMMUTABLE SECTION
<!-- Do not modify after initialization -->

### Ultimate Goal
Continue the ScoRe-Flow revision line from the Claude handoff using Codex-only Humanize. The
Terminal-Pullback fair test did not beat the scalar trust-region baseline, so the implementation goal
is to pivot to Conservative Reward-Weighted Reflow PI (CR-Reflow) as the next real algorithmic method
for RLinf/OpenPI pi0.5 on LIBERO.

### Acceptance Criteria
<!-- Each criterion must be independently verifiable -->
<!-- Claude must extract or define these in Round 0 -->

1. Evidence baseline remains accurate: `tr_matched_0604` is recorded as three completed seeds with
   exit code 0, Pullback is not claimed to beat scalar_l2, and measured facts are separated from
   interpretation.
2. CR-Reflow is added without breaking existing methods: baseline, score-flow, tr_scalar_l2,
   tr_pullback, and tr_pullback_matched keep their existing routing, and a new CR-Reflow method flag
   selects the new actor loss.
3. CR-Reflow E-step and M-step are implemented: KL-constrained advantage weights, weighted
   teacher-forced reflow loss, conservative anchor, and diagnostics for eta, ESS, losses, and
   chain-KL/displacement are available.
4. Real-run verification hooks are ready: the runner can launch CR-Reflow on the real pi0.5 LIBERO
   Spatial config, including no-anchor/full ablations and result collection.
5. Reproducibility and cost control are maintained: H100 cost state is explicit, new runs have
   manifests/logs/commands/seeds, and no toy or dry-run results are reported as method evidence.

---

## MUTABLE SECTION
<!-- Update each round with justification for changes -->

### Plan Version: 1 (Updated: Round 0)

#### Plan Evolution Log
<!-- Document any changes to the plan with justification -->
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initial plan | - | - |
| 0 | Pivot locked from Terminal-Pullback to CR-Reflow | Existing `tr_matched_0604` logs show matched Pullback did not beat scalar_l2 at the decision gate | AC-1 establishes the evidence baseline; AC-2 through AC-5 target the CR-Reflow path |

#### Active Tasks
<!-- Mainline tasks only: each task must directly advance the current round objective and carry routing metadata -->
| Task | Target AC | Status | Tag | Owner | Notes |
|------|-----------|--------|-----|-------|-------|
| Evidence lock-in for negative Pullback gate | AC-1, AC-5 | completed pending verification | coding | claude | `scoreflow-h100b-0603` is stopped; `tr_matched_0604` all seeds done; top-level proposal §11 updated before loop start |
| Map CR-Reflow edit points in local and remote RLinf files | AC-2, AC-3 | pending | coding | claude | Inspect local `openpi_action_model.LIVE.py` and remote RLinf live files before implementation |
| Implement CR-Reflow routing and rollout fields | AC-2 | pending | coding | claude | Preserve existing method branches |
| Implement CR-Reflow E-step/M-step losses and diagnostics | AC-3 | pending | coding | claude | No Pullback JVP dependency in MVP |
| Add runner and collection support for CR-Reflow ablations | AC-4, AC-5 | pending | coding | claude | Real config only for evidence |
| Run launch-readiness and prepare real comparison | AC-4, AC-5 | pending | coding | claude | Do not start costly multi-seed experiment without explicit cost state |

### Blocking Side Issues
<!-- Only issues that directly block current mainline progress belong here -->
| Issue | Discovered Round | Blocking AC | Resolution Path |
|-------|-----------------|-------------|-----------------|

### Queued Side Issues
<!-- Non-blocking issues stay queued and must NOT replace the round objective -->
| Issue | Discovered Round | Why Not Blocking | Revisit Trigger |
|-------|-----------------|------------------|-----------------|
| BitLesson selector subprocess hung in Codex read-only mode during Round 0 setup | 0 | `.humanize/bitlesson.md` contains no real lessons, so the effective selection is `NONE` | Revisit if real BitLesson entries are added or selector failure blocks review |
| Top-level `/Users/qiuxiaotian/Documents/New project` is not a committed git repo | 0 | RLCR is correctly rooted in the nested committed repo `ScoReFlow-rlinf-scoreflow-clean` | Revisit only if Humanize must manage top-level docs directly |

### Completed and Verified
<!-- Only move tasks here after Codex verification -->
| AC | Task | Completed Round | Verified Round | Evidence |
|----|------|-----------------|----------------|----------|
| AC-1, AC-5 | Evidence lock-in for negative Pullback gate | 0 | pending Codex review | `run_manifest.csv` showed seed42/43/44 done exit_code=0; H100 notebook status STOPPED; proposal §11 records pivot |

### Explicitly Deferred
<!-- Items here require strong justification -->
| Task | Original AC | Deferred Since | Justification | When to Reconsider |
|------|-------------|----------------|---------------|-------------------|
