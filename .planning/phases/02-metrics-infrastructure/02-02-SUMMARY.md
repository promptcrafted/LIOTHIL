---
phase: 02-metrics-infrastructure
plan: 02
subsystem: training
tags: [wandb, weight-verification, checksum, sampling, visual-monitoring, moe-integrity]

# Dependency graph
requires:
  - phase: 02-metrics-infrastructure
    plan: 01
    provides: TrainingLogger with W&B init, VRAMTracker, RunTimer, log_run_summary
provides:
  - WeightVerifier utility for frozen-expert checksum verification
  - W&B media logging for sample videos (wandb.Video) and keyframe grids (wandb.Image)
  - log_frozen_check() for console PASS/FAIL and W&B summary reporting
  - Default T2V sampling prompts covering 4 quality dimensions
  - Orchestrator wiring for weight verification at phase boundaries
affects: [phase-08, phase-09]

# Tech tracking
tech-stack:
  added: [hashlib (SHA-256 checksumming)]
  patterns: [file-based checksum primary + sentinel fallback, snapshot-before/verify-after phase boundary pattern]

key-files:
  created:
    - dimljus/training/verification.py
    - tests/test_training_verification.py
  modified:
    - dimljus/training/logger.py
    - dimljus/training/loop.py
    - dimljus/config/wan22_training_master.py
    - tests/test_training_logger.py
    - tests/test_training_loop.py
    - tests/test_training_config.py

key-decisions:
  - "File-based checksum is primary strategy (no GPU memory needed), sentinel checksum (3 tensors) is fallback for in-memory models"
  - "Frozen expert snapshot/verify happens around the entire phase, not per-epoch, because the frozen expert's checkpoint stays on disk unchanged"
  - "Default T2V prompts cover 4 quality dimensions: static person (identity), animal motion (temporal), scenic (composition), close-up (fine detail)"

patterns-established:
  - "WeightVerifier snapshot-before/verify-after pattern: snapshot frozen expert before phase start, verify after phase end"
  - "W&B media logging pattern: wandb.Video for .mp4 + wandb.Image for .grid.png at same step"
  - "_get_frozen_expert_name pattern: during high_noise training, low_noise is frozen (and vice versa); unified has no frozen expert"

requirements-completed: [METR-05, METR-06]

# Metrics
duration: 6min
completed: 2026-02-27
---

# Phase 02 Plan 02: Visual Samples & Weight Verification Summary

**WeightVerifier with file-based and sentinel checksumming for frozen-expert integrity, plus W&B media logging (wandb.Video + wandb.Image) for visual sample monitoring during training**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-27T21:00:07Z
- **Completed:** 2026-02-27T21:05:39Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- WeightVerifier utility with SHA-256 file-based checksum (primary) and sentinel tensor checksum (fallback) for verifying frozen expert integrity during MoE training (METR-06)
- W&B media logging: sample videos logged as wandb.Video, keyframe grids logged as wandb.Image, wired into orchestrator's _generate_samples flow (METR-05)
- Console PASS/FAIL reporting for frozen expert verification with W&B summary integration for cross-run comparison
- 4 default T2V sampling prompts covering key quality dimensions: static person, animal motion, scenic landscape, close-up detail
- Full orchestrator wiring: snapshot frozen expert before each expert phase, verify after, pass results to end-of-run summary
- 28 new tests added (15 verification, 6 logger, 7 orchestrator), all 592 training tests passing

## Task Commits

Each task was committed atomically:

1. **Task 1: Create WeightVerifier utility and add W&B sample logging to logger** - `a5dfc5d` (feat)
2. **Task 2: Wire weight verification and sample W&B logging into orchestrator** - `c6e2aaa` (feat)

## Files Created/Modified
- `dimljus/training/verification.py` - NEW: WeightVerifier class with file-based and sentinel checksum strategies, VerificationResult dataclass
- `dimljus/training/logger.py` - MODIFIED: Added log_samples_to_wandb() for video+grid W&B media, log_frozen_check() for console+W&B PASS/FAIL
- `dimljus/training/loop.py` - MODIFIED: WeightVerifier wiring (snapshot before phase, verify after), _get_frozen_expert_name(), log_samples_to_wandb in _generate_samples, frozen_checks in run summary
- `dimljus/config/wan22_training_master.py` - MODIFIED: T2V_SAMPLING_PROMPTS populated with 4 default prompts
- `tests/test_training_verification.py` - NEW: 15 tests for WeightVerifier (file checksum, sentinel checksum, snapshot/verify workflow, edge cases)
- `tests/test_training_logger.py` - MODIFIED: 6 new tests for log_samples_to_wandb and log_frozen_check
- `tests/test_training_loop.py` - MODIFIED: 7 new tests for WeightVerifier wiring, _get_frozen_expert_name, frozen checks in summary
- `tests/test_training_config.py` - MODIFIED: Updated test_default_prompts to validate 4 new default prompts

## Decisions Made
- File-based checksum is the primary verification strategy because the frozen expert stays on disk during the other expert's training -- fast SHA-256 over the file, no GPU memory needed
- Sentinel checksum uses first/middle/last tensors from sorted keys as a fallback -- avoids 30+ second full state dict checksum on 14B models
- Frozen expert verification happens at phase boundaries (before/after entire phase), not per-epoch, because the checkpoint file should be immutable during another expert's training
- Default prompts chosen to cover orthogonal quality dimensions that Minta evaluates during convergence

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated existing test expecting empty default prompts**
- **Found during:** Task 2 (full test suite verification)
- **Issue:** `tests/test_training_config.py::TestSamplingConfig::test_default_prompts_empty` asserted that default prompts were an empty list, which conflicted with the new default prompts added in Task 1
- **Fix:** Updated test to validate that 4 default prompts exist and cover the expected scenarios (person, cat, ocean, hands)
- **Files modified:** tests/test_training_config.py
- **Verification:** All 592 training tests pass
- **Committed in:** c6e2aaa (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Test expectation needed updating to match the planned new default prompts. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 02 (Metrics Infrastructure) is now complete: all 6 METR requirements delivered across Plans 01 and 02
- The training loop has full observability: per-phase W&B panels, VRAM tracking, wall-clock timing, visual sample logging, and frozen-expert integrity verification
- Ready to proceed to Phase 03 and beyond with confidence that training runs are fully instrumented

## Self-Check: PASSED

All 8 created/modified files verified present. Both task commits (a5dfc5d, c6e2aaa) verified in git log.
