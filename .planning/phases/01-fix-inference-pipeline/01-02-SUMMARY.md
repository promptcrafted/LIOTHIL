---
phase: 01-fix-inference-pipeline
plan: 02
subsystem: inference
tags: [wan-2.2, t5, embed-tokens, scheduler, boundary-ratio, gpu-validation]

# Dependency graph
requires:
  - "01-01"
provides:
  - "GPU-validated base model inference producing coherent video"
  - "T5 embed_tokens weight tying fix across all code paths"
  - "FlowMatchEuler shift=5.0 + boundary=0.6 as validated inference settings"
affects: [inference-pipeline, training-sampling, runpod-scripts]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Always fix T5 embed_tokens after loading: copy shared.weight to encoder.embed_tokens.weight"
    - "Inference scheduler: FlowMatchEuler shift=5.0 with boundary_ratio=0.6 for T2V"
    - "from_pretrained pipeline approach with local from_single_file transformer swap"

key-files:
  created:
    - runpod/test_scheduler_comparison.py
  modified:
    - dimljus/training/wan/inference.py
    - dimljus/training/wan/backend.py
    - dimljus/encoding/text_encoder.py
    - runpod/test_base_inference.py

key-decisions:
  - "FlowMatchEuler shift=5.0 + boundary=0.6 chosen over UniPC + 0.875 — best temporal coherence per Minta's visual review"
  - "T5 must be loaded from HF repo, NOT standalone .pth file (wrong/uninitialized weights)"
  - "embed_tokens fix required for both from_pretrained and from_single_file T5 loading paths"
  - "Inference boundary (0.6) differs from training boundary (0.875) — scheduler-dependent SNR mapping"

patterns-established:
  - "T5 embed_tokens fix: always copy shared.weight to encoder.embed_tokens.weight after loading"
  - "Inference defaults: FlowMatchEuler shift=5.0, boundary=0.6, CFG=4.0, 30 steps"
  - "from_pretrained pipeline approach: load full pipeline from HF, swap transformers to local copies"

requirements-completed: [INFER-01, INFER-02, INFER-03]

# Metrics
duration: multi-session
completed: 2026-02-27
---

# Phase 1 Plan 2: GPU Validation of Inference Fix Summary

**Validated inference pipeline on RunPod GPU — T5 embed_tokens fix, scheduler comparison, FlowMatchEuler + boundary=0.6 selected as best temporal quality**

## Performance

- **Duration:** Multi-session (T5 fix + scheduler investigation + comparison test)
- **Completed:** 2026-02-27
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

### Bug Fixes (previous session, commits already landed)
1. **T5 loading fix**: Switched from standalone .pth file to HF repo loading — the .pth contained wrong/uninitialized weights producing noise
2. **T5 embed_tokens weight tying**: HF checkpoint stores token embedding as `shared.weight` but `encoder.embed_tokens.weight` is MISSING and gets zero-initialized. Fix copies `shared.weight` to `embed_tokens` after loading. Applied to all 3 loading paths (text_encoder.py: file, diffusers, HF) and both inference paths.

### Scheduler Validation (this session)
3. **Scheduler comparison test**: Created `test_scheduler_comparison.py` — loads pipeline once, runs 3 configs with same seed:
   - Test 1: UniPC + boundary=0.875 (HF default) — coherent, good
   - Test 2: FlowMatchEuler shift=5.0 + boundary=0.6 (ComfyUI-aligned) — **best temporal quality** (Minta's pick)
   - Test 3: FlowMatchEuler shift=5.0 + boundary=0.875 — coherent but less temporal quality than Test 2
4. **Updated inference.py**: boundary_ratio changed from 0.5 to 0.6 for T2V
5. **Updated backend.py**: default boundary_ratio changed from 0.5 to 0.6

### Key Finding: Boundary Ratio is Scheduler-Dependent
The training boundary (0.875, SNR-based) and inference boundary (0.6, step-fraction-based) are different because different schedulers map steps to noise levels differently. FlowMatchEuler at step 18/30 (0.6) hits the same SNR crossover as UniPC at step 26/30 (0.875). Previous dark output at boundary=0.5 was the expert handoff happening too early in the noise schedule.

## Task Commits

1. **Task 1: Push code to remotes** — `6fcf104` (plus earlier: `5af3092`, `d47dadf`)
2. **Task 2: GPU validation** — Visual review by Minta, scheduler settings updated in codebase

## Files Created/Modified
- `runpod/test_scheduler_comparison.py` — New: 3-way scheduler comparison test
- `dimljus/training/wan/inference.py` — boundary_ratio 0.5 → 0.6, updated comments
- `dimljus/training/wan/backend.py` — default boundary_ratio 0.5 → 0.6
- `dimljus/encoding/text_encoder.py` — T5 embed_tokens fix (previous session)
- `runpod/test_base_inference.py` — from_pretrained approach + embed_tokens fix (previous session)

## Decisions Made
- **FlowMatchEuler + boundary=0.6** is the Dimljus T2V inference default — best temporal coherence, matches ComfyUI production workflow
- **Training boundary stays at 0.875** (registry) — this is the Wan-AI training regime and is correct for noise-level-based expert routing during training
- **T5 loaded from HF repo** — standalone .pth files are unreliable, HF cached weights are the source of truth

## Deviations from Plan
None — escalation path (from_pretrained fallback) was successfully used in previous session.

## Issues Encountered
- **Dark output at boundary=0.5**: Previous session discovered that FlowMatchEuler + boundary=0.5 produces nearly black output. Root cause: expert handoff too early in noise schedule. Fixed by moving to 0.6 (ComfyUI-aligned).

## User Setup Required
None — scheduler settings are hardcoded in inference.py.

## Self-Check: PASSED

- Base model inference produces coherent video matching prompt (INFER-01) ✓
- Output grids at `runpod/outputs/scheduler_test/` confirm all 3 scheduler configs work ✓
- 165 related tests pass, no regressions ✓

---
*Phase: 01-fix-inference-pipeline*
*Completed: 2026-02-27*
