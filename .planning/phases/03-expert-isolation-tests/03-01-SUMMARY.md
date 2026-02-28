---
phase: 03-expert-isolation-tests
plan: 01
subsystem: training
tags: [wan-2.2, moe, expert-isolation, lora, runpod, wandb]

# Dependency graph
requires:
  - phase: 01-fix-inference-pipeline
    provides: "Working dual-expert inference pipeline with WanPipeline"
  - phase: 02-metrics-infrastructure
    provides: "W&B logging, visual samples, frozen-expert verification"
provides:
  - "Low-noise expert isolation LoRA checkpoint (5 epochs, rank 16)"
  - "High-noise expert isolation LoRA checkpoint (5 epochs, rank 16)"
  - "Validated dual-expert sampling pipeline (partner model loading)"
  - "Isolation YAML config templates (test4/test5)"
affects: [03-02, 04-checkpoint-resume, 06-base-strategy, 07-production-training]

# Tech tracking
tech-stack:
  added: [ftfy]
  patterns: [dual-expert-sampling, partner-model-loading, isolation-config]

key-files:
  created:
    - runpod/test4-low-only.yaml
    - runpod/test5-high-only.yaml
    - runpod/validate_isolation_samples.py
  modified:
    - dimljus/training/loop.py
    - dimljus/training/wan/inference.py
    - dimljus/training/sampler.py
    - pyproject.toml

key-decisions:
  - "Wan 2.2 experts are specialists -- MUST load both for inference (single expert produces noise)"
  - "diffusers 0.36.0 WanPipeline regression -- pinned <0.36 until upstream fix"
  - "Flat lr 5e-5 for isolation tests, no per-expert overrides"
  - "LoRA rank 16 for both isolation tests"
  - "Isolation checkpoints use transformer_2 prefix (low-noise) and transformer prefix (high-noise)"

patterns-established:
  - "Partner model loading: during expert phase sampling, always load the partner base model from disk"
  - "Isolation config pattern: unified_epochs=0, disable the other expert"
  - "Dual-expert pipeline: WanPipeline(transformer=high, transformer_2=low, boundary_ratio=0.6)"

requirements-completed: [TRAIN-01, TRAIN-02]

# Metrics
duration: multi-session (~3 hours total: config creation, 2x 5-epoch training runs, sampling validation)
completed: 2026-02-28
---

# Phase 3 Plan 01: Expert Isolation Tests Summary

**Both MoE experts (low-noise and high-noise) trained independently to completion with decreasing loss, and dual-expert sampling validated to produce recognizable Holly Golightly video output**

## Performance

- **Duration:** Multi-session (~3 hours total across config creation, training, and validation)
- **Training time:** ~90 min per expert (5 epochs each on RTX PRO 6000 Blackwell 98GB)
- **Validation time:** ~5 min (model loading + 20-step inference)
- **Started:** 2026-02-27 (config creation)
- **Completed:** 2026-02-28
- **Tasks:** 2 planned + 1 continuation task (sampling validation)
- **Files modified:** 8

## Accomplishments

- Low-noise isolation training ran 5 epochs, saved checkpoints at every epoch, loss trending down
- High-noise isolation training ran 5 epochs, saved checkpoints at every epoch, loss trending down
- Both runs logged to W&B project `dimljus-isolation` with distinct run names
- Discovered and fixed two critical sampling bugs (partner model loading + diffusers regression)
- Validated dual-expert sampling produces recognizable video (Holly on staircase, motion detected)
- Keyframe grid confirms temporal coherence and subject fidelity

## Task Commits

Each task was committed atomically:

1. **Task 1: Create isolation YAML configs** - `5b4f91f` (feat)
2. **Task 2: Execute training runs** - completed on RunPod (no local commit -- pod execution only)
3. **Sampling validation** - `681d205` (feat)

Bug fixes committed during execution:
- `387155c` - fix(sampling): always load partner expert for dual-expert sampling
- `e05f18b` - fix(deps): pin diffusers<0.36 to avoid WanPipeline regression

## Files Created/Modified

- `runpod/test4-low-only.yaml` - Low-noise isolation config (high_noise.enabled: false)
- `runpod/test5-high-only.yaml` - High-noise isolation config (low_noise.enabled: false)
- `runpod/validate_isolation_samples.py` - Standalone validation script for dual-expert sampling
- `dimljus/training/loop.py` - Added _load_partner_model() for dual-expert sampling during training
- `dimljus/training/wan/inference.py` - Added partner_model/active_expert params to generate()
- `dimljus/training/sampler.py` - Added partner_model/active_expert params to generate_samples()
- `pyproject.toml` - Pinned diffusers<0.36

## Decisions Made

1. **Wan 2.2 experts are specialists, not generalists.** Each expert only handles its portion of the noise schedule. Using one expert alone for full 30-step denoising produces pure noise because it can't handle timesteps outside its specialty. This is the most important finding of Phase 3 -- it means ALL sampling (during training and inference) must use both experts together.

2. **diffusers 0.36.0 has a WanPipeline regression.** Version 0.36.0 produces washed-out white output (pixel mean 0.91 vs expected ~0.4). Pinned to <0.36 in pyproject.toml. Pod uses 0.35.2.

3. **Flat lr 5e-5 for isolation tests.** No per-expert learning rate overrides. This establishes a baseline before differential hyperparameters are tested in later phases.

4. **Pod deployment via tarball/SCP, not git.** The pod dimljus installation is not a git repo. Code updates are deployed by SCP'ing changed files directly.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Single expert produces noise during sampling**
- **Found during:** Task 2 (training runs -- samples were pure noise)
- **Issue:** Training loop generated samples using only the active expert. Wan 2.2 experts are specialists that can't handle the full denoising trajectory alone.
- **Fix:** Added `_load_partner_model()` to loop.py that loads the partner base model from disk during sampling. Both experts work together for coherent output.
- **Files modified:** dimljus/training/loop.py, dimljus/training/wan/inference.py, dimljus/training/sampler.py
- **Verification:** Standalone validation produced recognizable video with mean=0.41, std=0.30, motion=0.25
- **Committed in:** 387155c

**2. [Rule 1 - Bug] diffusers 0.36.0 WanPipeline output regression**
- **Found during:** Task 2 (training runs -- white/washed-out output)
- **Issue:** diffusers 0.36.0 WanPipeline produced pixel mean 0.91 (expected ~0.25-0.4). Known upstream regression.
- **Fix:** Downgraded pod to diffusers 0.35.2, pinned <0.36 in pyproject.toml.
- **Files modified:** pyproject.toml
- **Verification:** Validation script confirmed healthy pixel statistics with 0.35.2
- **Committed in:** e05f18b

**3. [Rule 3 - Blocking] Missing ftfy dependency for WanPipeline prompt encoding**
- **Found during:** Validation task (sampling validation script)
- **Issue:** diffusers WanPipeline.encode_prompt() calls ftfy.fix_text() which requires the ftfy package, not installed on pod.
- **Fix:** Installed ftfy on pod via pip.
- **Files modified:** None (pip install on pod only)
- **Verification:** Validation script ran successfully after install

---

**Total deviations:** 3 auto-fixed (2 bugs, 1 blocking)
**Impact on plan:** Bug fixes were essential for correct operation. The partner model loading fix is a fundamental architectural insight about Wan 2.2's expert system.

## Issues Encountered

- **Training samples were noise on first run:** Both bugs (single-expert sampling + diffusers regression) conspired to produce noisy samples during the 5-epoch training runs. Training itself was fine (loss decreasing, checkpoints valid). Fixed both bugs and validated with standalone inference.
- **Pod SSH proxy doesn't support PTY:** Had to use TCP connection (port 32427) instead of RunPod's SSH proxy for command execution.
- **No imageio on pod:** Video export fell back to individual frames. Used keyframe grid PNGs for visual validation instead.

## Training Results

### test4: Low-Noise Expert Isolation
- **Config:** test4-low-only.yaml (high_noise.enabled: false)
- **Epochs:** 5 (completed)
- **Checkpoints:** 5 (one per epoch) at `/workspace/outputs/test4-low-only/low_noise/`
- **Checkpoint size:** 306 MB each
- **W&B run:** `low-only-r16-lr5e-05` in project `dimljus-isolation`
- **Loss:** Decreasing trend (confirmed in W&B)

### test5: High-Noise Expert Isolation
- **Config:** test5-high-only.yaml (low_noise.enabled: false)
- **Epochs:** 5 (completed)
- **Checkpoints:** 5 (one per epoch) at `/workspace/outputs/test5-high-only/high_noise/`
- **Checkpoint size:** 306 MB each
- **W&B run:** `high-only-r16-lr5e-05` in project `dimljus-isolation`
- **Loss:** Decreasing trend (confirmed in W&B)

### Validation Results
- **Script:** validate_isolation_samples.py (test4 low-noise epoch 5 LoRA)
- **Pipeline:** WanPipeline with both experts (LoRA on transformer_2, base on transformer)
- **Quality checks:** ALL PASSED
  - First frame: mean=0.4064, std=0.3044 (healthy distribution)
  - Last frame: mean=0.3932, std=0.2209 (healthy distribution)
  - Motion: diff=0.2511 (significant inter-frame motion)
- **Visual:** Recognizable Holly Golightly on indoor staircase with morning light

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Expert isolation training validated: both experts produce valid checkpoints
- Sampling pipeline validated: dual-expert inference works with isolation LoRAs
- Ready for Plan 03-02: Post-training inference validation with full quality checks and Minta's visual review
- Pod is still running with all checkpoints available
- Key concern: Plan 03-02 inference should test BOTH checkpoints (high-noise and low-noise) and generate more prompts for comprehensive quality assessment

## Self-Check: PASSED

- All 3 created files found on disk
- All 4 commits (5b4f91f, 681d205, 387155c, e05f18b) found in git history
- Validation grid image confirms recognizable video output

---
*Phase: 03-expert-isolation-tests*
*Completed: 2026-02-28*
