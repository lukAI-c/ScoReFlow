# Round 0 Contract

## One Mainline Objective

Initialize the Codex-only Humanize RLCR loop for the CR-Reflow continuation and lock the factual
decision baseline: Terminal-Pullback did not pass the fair gate, so subsequent implementation work
should target CR-Reflow.

## Target ACs

- AC-1: Evidence baseline remains accurate.
- AC-5: Reproducibility and cost control are maintained.

## Blocking Side Issues In Scope

None. The current round does not require GPU execution or remote code edits.

## Queued Side Issues Out Of Scope

- BitLesson selector subprocess hung in Codex read-only mode. The BitLesson file has no real lessons,
  so Round 0 proceeds with `LESSON_IDS: NONE`.
- Top-level project directory is not a committed git repo. RLCR is rooted in the nested
  `ScoReFlow-rlinf-scoreflow-clean` git repo.
- CR-Reflow code implementation is out of scope for Round 0 and starts after this setup round is
  accepted.

## Round Success Criteria

- `.humanize/rlcr/2026-06-05_12-09-52/goal-tracker.md` has a concrete Ultimate Goal, Acceptance
  Criteria, Active Tasks, and side-issue queues.
- `.humanize/rlcr/2026-06-05_12-09-52/round-0-contract.md` exists and defines the focused setup
  scope.
- Round 0 summary records the checkpoint commit, H100 stopped state, `tr_matched_0604` completion,
  and BitLesson Delta.
- No new experiment, model review, or costly GPU launch is started in Round 0.
