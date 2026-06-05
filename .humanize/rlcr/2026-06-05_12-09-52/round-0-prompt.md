Read and execute below with ultrathink

## Goal Tracker Setup (REQUIRED FIRST STEP)

Before starting implementation, you MUST initialize the Goal Tracker:

1. Read @/Users/qiuxiaotian/Documents/New project/ScoReFlow-rlinf-scoreflow-clean/.humanize/rlcr/2026-06-05_12-09-52/goal-tracker.md
2. If the "Ultimate Goal" section says "[To be extracted...]", extract a clear goal statement from the plan
3. If the "Acceptance Criteria" section says "[To be defined...]", define 3-7 specific, testable criteria
4. Populate the "Active Tasks" table with MAINLINE tasks from the plan, mapping each to an AC and filling Tag/Owner
5. Record any already-known side issues in either "Blocking Side Issues" or "Queued Side Issues"
6. Write the updated goal-tracker.md

## Round Contract Setup (REQUIRED BEFORE CODING)

Before starting implementation, create @/Users/qiuxiaotian/Documents/New project/ScoReFlow-rlinf-scoreflow-clean/.humanize/rlcr/2026-06-05_12-09-52/round-0-contract.md with:

1. **One mainline objective** for this round
2. **Target ACs** (1-2 ACs only)
3. **Blocking side issues in scope** for this round
4. **Queued side issues out of scope** for this round
5. **Round success criteria**

Use this contract to keep the round focused. Do NOT let non-blocking bugs or cleanup work replace the mainline objective.

**IMPORTANT**: The IMMUTABLE SECTION can only be modified in Round 0. After this round, it becomes read-only.

---

## Implementation Plan

For all tasks that need to be completed, please use the Task system (TaskCreate, TaskUpdate, TaskList).

Every task MUST start with exactly one lane tag:
- `[mainline]` for plan-derived work that directly advances the round objective
- `[blocking]` for issues that prevent the mainline objective from succeeding safely
- `[queued]` for non-blocking bugs, cleanup, or follow-up work

Rules:
- `[mainline]` tasks are the primary success condition for the round
- `[blocking]` tasks may be resolved in the round only if they truly block mainline progress
- `[queued]` tasks must NOT become the round objective and do NOT need to be cleared before moving on
- If a new issue is not blocking the current objective, tag it `[queued]` and keep moving on the mainline

## Task Tag Routing (MUST FOLLOW)

Each task must have one routing tag from the plan: `coding` or `analyze`.

- Tag `coding`: Claude executes the task directly.
- Tag `analyze`: Claude must execute via `/humanize:ask-codex`, then integrate Codex output.
- Keep Goal Tracker "Active Tasks" columns **Tag** and **Owner** aligned with execution (`coding -> claude`, `analyze -> codex`).
- If a task has no explicit tag, default to `coding` (Claude executes directly).

# Codex-Only Humanize Plan — CR-Reflow Continuation

## Goal Description

Continue the ScoRe-Flow revision line from the Claude handoff using Codex-only Humanize. The
Terminal-Pullback fair test did not beat the scalar trust-region baseline, so the implementation goal
is to pivot to Conservative Reward-Weighted Reflow PI (CR-Reflow) as the next real algorithmic method
for RLinf/OpenPI pi0.5 on LIBERO.

The method principle is separation: reward search creates a constrained improved target over stored
denoising chains, while the flow mean is adapted by conservative supervised reflow instead of direct
PPO policy-gradient updates to the denoising mean.

## Acceptance Criteria

- AC-1: Evidence baseline is preserved.
  - Positive Tests:
    - The 2026-06-05 tr_matched_0604 evidence says all three matched seeds completed with exit code 0.
    - Pullback is not claimed to beat scalar_l2.
    - Measured facts are separated from interpretation.
  - Negative Tests:
    - Do not rerun tr_compare_0603.
    - Do not frame Terminal-Pullback as the main novelty after the negative gate.

- AC-2: CR-Reflow is added without breaking existing methods.
  - Positive Tests:
    - Existing flow_noise_baseline, score-flow, tr_scalar_l2, tr_pullback, and tr_pullback_matched
      branches keep their current routing.
    - A new method flag, for example rw_reflow or cr_reflow, selects the CR-Reflow actor loss.
    - The rollout-to-actor path carries chains, denoise indices, old transition means, old transition
      std/logvar if available, normalized advantages, masks, and diagnostics for weights/ESS.
  - Negative Tests:
    - Existing methods must not silently switch to CR-Reflow.
    - The MVP must not depend on the Pullback JVP/VJP code.

- AC-3: CR-Reflow E-step and M-step are implemented.
  - Positive Tests:
    - E-step computes KL-constrained reward/advantage weights `q_j ∝ p_old exp(A_j / eta)` with a
      bounded eta solve or clear fallback.
    - M-step computes weighted teacher-forced reflow loss to stored next-chain states plus a
      conservative anchor to old means.
    - Negative-advantage chains receive low positive weight, not a push-away policy-gradient update.
    - Diagnostics include eta, weight ESS, reflow loss, anchor loss, approximate chain KL or
      displacement proxy, and actor loss components.
  - Negative Tests:
    - CR-Reflow mean updates must not use PPO ratio/clipped policy-gradient as the primary mean loss.
    - The method must not require test-time steering or a permanent latent proposal teacher.

- AC-4: Real-run verification hooks are ready.
  - Positive Tests:
    - The runner can launch CR-Reflow on the same real config: pi0.5, LIBERO Spatial, 15 epochs,
      4 train envs, 8 eval envs.
    - Runner exposes at least cr_reflow_no_anchor and cr_reflow or equivalent ablation names.
    - Collection captures success, collapse-rate, approx_kl, chain-KL/displacement, and CR-Reflow
      diagnostics.
  - Negative Tests:
    - Do not report toy, dry-run, or reduced-config results as method evidence.
    - Do not start a costly multi-seed experiment before launch-readiness passes and GPU cost state
      is explicit.

- AC-5: Reproducibility and cost control are maintained.
  - Positive Tests:
    - H100 notebook scoreflow-h100b-0603 is confirmed stopped before new launches.
    - Any new run has a manifest, per-seed logs, exact command file, and seed list.
    - Claims cite log directories and distinguish measured facts from interpretation.
  - Negative Tests:
    - Do not leave a GPU notebook running after verification unless it is actively running an approved
      experiment.

## Path Boundaries

### Upper Bound

Implement CR-Reflow MVP in the RLinf/OpenPI pipeline, add real-config launch support, run
launch-readiness, and prepare the first 3-seed real comparison plan.

### Lower Bound

Produce a code-ready CR-Reflow implementation path, preserve the negative Pullback gate evidence, and
identify exact files/hooks to edit before code changes.

### Allowed Choices

- Can use existing old_means/chains/denoise_inds plumbing from Terminal-Pullback.
- Can add rollout forward-input fields and actor diagnostics when needed.
- Can freeze flow_noise for the MVP and defer learned variance or replay.
- Can keep scalar_l2 as a guard/ablation baseline.
- Cannot use Pullback JVP as the main method.
- Cannot make scalar_l2 alone the paper contribution.
- Cannot use toy experiments as final evidence.

## Dependencies and Sequence

### Milestones

1. Evidence lock-in
   - Record tr_matched_0604 completion and metrics.
   - Confirm H100 notebook is stopped.

2. Code map
   - Inspect local `openpi_action_model.LIVE.py`.
   - Inspect remote RLinf live files before editing:
     - `$R/RLinf/rlinf/models/embodiment/openpi/openpi_action_model.py`
     - `$R/RLinf/rlinf/workers/actor/fsdp_actor_worker.py`
     - `$R/RLinf/rlinf/hybrid_engines/fsdp/fsdp_model_manager.py`
     - `$R/ScoReFlow-RLinf-ScoreFlow-clean/scripts/rlinf_scoreflow/run_score_flow_benchmark.sh`

3. CR-Reflow implementation
   - Add CR-Reflow config/method routing.
   - Add teacher-forced new mean/std return path if not already sufficient.
   - Add E-step weight calculation and diagnostics.
   - Add M-step weighted reflow loss and conservative anchor.
   - Preserve all existing method branches.

4. Readiness verification
   - Run static/import or single-launch readiness only after code changes are complete.
   - Use real config for evidence; use short readiness only to catch launch/runtime errors.

5. Real comparison
   - After readiness, run at least 3 seeds on real pi0.5 LIBERO Spatial:
     baseline, scalar_l2, cr_reflow_no_anchor, cr_reflow.
   - Decide whether CR-Reflow beats scalar_l2 at comparable KL/collapse budget.

## Implementation Notes

- Treat `/Users/qiuxiaotian/Documents/New project/plan/HANDOFF_codex_humanize_0605.md` as the
  migration source, not `.humanize` state.
- Current provider mode is codex-only.
- Avoid editing Humanize state files manually. Start RLCR from this repo root with this relative plan
  path.

---

## BitLesson Selection (REQUIRED FOR EACH TASK)

Before executing each task or sub-task, you MUST:

1. Read @/Users/qiuxiaotian/Documents/New project/ScoReFlow-rlinf-scoreflow-clean/.humanize/bitlesson.md
2. Run `bitlesson-selector` for each task/sub-task to select relevant lesson IDs
3. Follow the selected lesson IDs (or `NONE`) during implementation

Include a `## BitLesson Delta` section in your summary with:
- Action: none|add|update
- Lesson ID(s): NONE or comma-separated IDs
- Notes: what changed and why (required if action is add or update)

Reference: @/Users/qiuxiaotian/Documents/New project/ScoReFlow-rlinf-scoreflow-clean/.humanize/bitlesson.md

---

## Goal Tracker Rules

Throughout your work, you MUST maintain the Goal Tracker:

1. **Before starting a round**: Re-anchor on the original plan and current round contract
2. **Before starting a task**: Mark the relevant mainline task as "in_progress" in Active Tasks
   - Confirm Tag/Owner routing is correct before execution
3. **Active Tasks** are MAINLINE tasks only - side issues do not belong there
4. **Blocking Side Issues** are reserved for issues that truly stop mainline progress
5. **Queued Side Issues** are non-blocking and must not take over the round
6. **After completing a mainline task**: Move it to "Completed and Verified" with evidence (but mark as "pending verification")
7. **If you discover the plan has errors**:
   - Do NOT silently change direction
   - Add entry to "Plan Evolution Log" with justification
   - Explain how the change still serves the Ultimate Goal
8. **If you need to defer a task**:
   - Move it to "Explicitly Deferred" section
   - Provide strong justification
   - Explain impact on Acceptance Criteria
9. **If you discover new issues**:
   - Add to "Blocking Side Issues" only if mainline progress is blocked
   - Otherwise add to "Queued Side Issues" or keep them as `[queued]` tasks/backlog

---

Note: You MUST NOT try to exit `start-rlcr-loop` loop by lying or edit loop state file or try to execute `cancel-rlcr-loop`

After completing the work, please:
0. If you have access to the `code-simplifier` agent, use it to review and optimize the code you just wrote
1. Finalize @/Users/qiuxiaotian/Documents/New project/ScoReFlow-rlinf-scoreflow-clean/.humanize/rlcr/2026-06-05_12-09-52/goal-tracker.md (this is Round 0, so you are initializing it - see "Goal Tracker Setup" above)
2. Write your round contract into @/Users/qiuxiaotian/Documents/New project/ScoReFlow-rlinf-scoreflow-clean/.humanize/rlcr/2026-06-05_12-09-52/round-0-contract.md
3. Commit your changes with a descriptive commit message
4. Write your work summary into @/Users/qiuxiaotian/Documents/New project/ScoReFlow-rlinf-scoreflow-clean/.humanize/rlcr/2026-06-05_12-09-52/round-0-summary.md
