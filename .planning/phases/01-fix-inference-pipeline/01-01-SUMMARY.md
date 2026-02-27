---
phase: 01-fix-inference-pipeline
plan: 01
subsystem: inference
tags: [wan-2.2, diffusers, from_single_file, config-detection, keyframe-grid, sampling]

# Dependency graph
requires: []
provides:
  - "Fixed WanTransformer3DModel.from_single_file() calls with explicit config= parameter"
  - "Keyframe grid PNG output alongside every sample video"
  - "_resolve_config_subfolder() helper method on WanModelBackend"
affects: [training-pipeline, inference-validation, runpod-scripts]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Always pass config='Wan-AI/Wan2.2-T2V-A14B-Diffusers' and subfolder= to WanTransformer3DModel.from_single_file()"
    - "Keyframe grid PNG saved alongside every sample MP4 for quick visual review"

key-files:
  created: []
  modified:
    - dimljus/training/wan/backend.py
    - dimljus/training/sampler.py
    - runpod/test_base_inference.py
    - runpod/test_inference.py

key-decisions:
  - "config= parameter added to from_single_file() only -- from_pretrained() reads config.json correctly and does not need it"
  - "VAE from_single_file() NOT modified -- AutoencoderKLWan config detection works correctly, bug is transformer-only"
  - "Keyframe grid uses frames at indices 0,4,8,12,16 -- evenly spaced across 17-frame Wan output"

patterns-established:
  - "from_single_file config pattern: every WanTransformer3DModel.from_single_file() call must include config= and subfolder="
  - "Keyframe grid companion pattern: every sample MP4 gets a .grid.png alongside it"

requirements-completed: [INFER-01, INFER-02, INFER-03]

# Metrics
duration: 4min
completed: 2026-02-27
---

# Phase 1 Plan 1: Fix from_single_file Config Detection and Add Keyframe Grids Summary

**Explicit config= parameter on all WanTransformer3DModel.from_single_file() calls to prevent diffusers#12329 silent Wan 2.1 misdetection, plus keyframe grid PNG output for quick visual sample review**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-27T05:42:13Z
- **Completed:** 2026-02-27T05:46:17Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Fixed root cause of noisy inference: diffusers#12329 auto-detects Wan 2.1 config instead of Wan 2.2 when using from_single_file() without explicit config=
- Added _resolve_config_subfolder() helper method to WanModelBackend that maps expert names to the correct subfolder strings
- Added _save_keyframe_grid() function that creates horizontal PNG grids from video frames for instant visual review
- Updated all test scripts to match the proven test3-inference.py loading pattern

## Task Commits

Each task was committed atomically:

1. **Task 1: Add config= parameter to all from_single_file() calls** - `3c1dca4` (fix)
2. **Task 2: Add keyframe grid output to sample saving** - `f39d9da` (feat)

## Files Created/Modified
- `dimljus/training/wan/backend.py` - Added _resolve_config_subfolder() helper, config= and subfolder= to from_single_file() call
- `dimljus/training/sampler.py` - Added _save_keyframe_grid() function, integrated into _save_frames_to_video() on both MP4 and PNG fallback paths
- `runpod/test_base_inference.py` - Updated both WanTransformer3DModel.from_single_file() calls with config= and subfolder=
- `runpod/test_inference.py` - Updated from_single_file() call with config= and subfolder= (deviation fix)

## Decisions Made
- config= parameter only needed for from_single_file(), not from_pretrained() -- from_pretrained reads the correct config.json from the model directory
- VAE from_single_file() does NOT need the fix -- AutoencoderKLWan has correct config detection, the bug is transformer-only
- Keyframe grid defaults to frame indices (0, 4, 8, 12, 16) -- evenly spaced across 17-frame Wan output for representative coverage

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added config= to test_inference.py**
- **Found during:** Task 2 verification (checking all from_single_file calls in codebase)
- **Issue:** runpod/test_inference.py also had a bare WanTransformer3DModel.from_single_file() call without config= parameter, same silent garbage output bug
- **Fix:** Added config="Wan-AI/Wan2.2-T2V-A14B-Diffusers" and subfolder="transformer" to the from_single_file() call
- **Files modified:** runpod/test_inference.py
- **Verification:** Confirmed all WanTransformer3DModel.from_single_file() calls in codebase now have config= parameter
- **Committed in:** `13dae14`

---

**Total deviations:** 1 auto-fixed (1 missing critical)
**Impact on plan:** Essential for correctness -- same bug that causes silent garbage output. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All from_single_file() calls now have explicit config= parameter
- Keyframe grids will be generated alongside every future sample video
- Ready for Plan 2 (RunPod validation) to verify inference produces recognizable video output on GPU
- 1872 tests still passing with no regressions

## Self-Check: PASSED

- All 5 files verified present on disk
- All 3 commit hashes verified in git log (3c1dca4, f39d9da, 13dae14)

---
*Phase: 01-fix-inference-pipeline*
*Completed: 2026-02-27*
