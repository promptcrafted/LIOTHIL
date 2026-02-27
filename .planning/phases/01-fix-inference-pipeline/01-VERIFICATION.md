---
phase: 01-fix-inference-pipeline
verified: 2026-02-27T19:45:00Z
status: human_needed
score: 7/8 must-haves verified
human_verification:
  - test: "Run test_base_inference.py on RunPod with a trained LoRA checkpoint at /workspace/outputs/test2-unified-only/unified/test_lora_unified_epoch003.safetensors and confirm Test B output (grid.png) is visually different from Test A output"
    expected: "Test A and Test B .grid.png files show recognizably different content -- LoRA should shift content toward training subject, not just produce different noise"
    why_human: "INFER-02 requires a trained LoRA checkpoint on a GPU pod and human visual comparison. Test B in test_base_inference.py auto-skips if the LoRA file is absent. Cannot verify programmatically."
  - test: "Confirm inference.py boundary=0.6 + FlowMatchEuler produces coherent output (not dark/black frames) on a new RunPod session to ensure the validated settings survived the boundary update commit"
    expected: "Output grid shows coherent scene, not dark or near-black frames like boundary=0.5 produced"
    why_human: "GPU-only result. SUMMARY documents Minta approved this during the session, but the final boundary=0.6 commit (3216509) was made after the visual approval -- a fresh run would confirm the committed code, not the session-state code."
---

# Phase 1: Fix Inference Pipeline Verification Report

**Phase Goal:** Users can generate recognizable video from both the base Wan 2.2 model and a trained LoRA
**Verified:** 2026-02-27T19:45:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (from ROADMAP.md Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Running base model inference (no LoRA) produces a coherent video matching the text prompt -- not noise | ? HUMAN | GPU-validated by Minta during Plan 02 session. Code fixes verified (config=, embed_tokens, boundary=0.6). Cannot re-run programmatically. |
| 2 | Running inference with a trained LoRA produces output visually different from base model output | ? HUMAN | INFER-02 requires a LoRA checkpoint on pod. Test B in test_base_inference.py auto-skips if file absent. Human comparison required. |
| 3 | The same inference code produces identical results whether called from standalone script or training pipeline | ? PARTIAL | Structurally verified: both use WanInferencePipeline with same _fix_t5_embed_tokens and FlowMatchEuler scheduler. Identical GPU output cannot be confirmed without running both. |

**Score:** 0/3 truths can be fully verified programmatically (all require GPU or human). Code infrastructure: 7/8 must-haves verified (all automated checks pass).

### Required Artifacts (Plan 01-01 must_haves)

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `dimljus/training/wan/backend.py` | Fixed from_single_file() calls with explicit config= and subfolder= | VERIFIED | Line 207-212: `config="Wan-AI/Wan2.2-T2V-A14B-Diffusers"`, `subfolder=self._resolve_config_subfolder(expert)`. Helper `_resolve_config_subfolder()` at line 441. |
| `dimljus/training/wan/inference.py` | Keyframe grid saving utility, T5 embed_tokens fix, no VAE changes | VERIFIED | `_fix_t5_embed_tokens()` static method at line 285. Applied in `_load_t5_from_file()` and `_load_t5()` (diffusers path). VAE uses `AutoencoderKLWan.from_single_file()` with no config= (correct -- VAE detection works). boundary=0.6 in `_build_pipeline()`. |
| `runpod/test_base_inference.py` | Updated test script matching test3-inference.py's fix pattern | VERIFIED | Lines 176-183: both WanTransformer3DModel.from_single_file() calls use `config=HF_REPO, subfolder="transformer"` and `subfolder="transformer_2"`. T5 embed_tokens fix inline at lines 155-158. |
| `dimljus/training/sampler.py` | Keyframe grid saving alongside MP4 output | VERIFIED | `_save_keyframe_grid()` function at lines 82-157. Called from `_save_frames_to_video()` at line 191 (MP4 path) and line 223 (PNG fallback path). Both paths save grid. |
| `dimljus/encoding/text_encoder.py` | T5 embed_tokens fix in all three loading paths | VERIFIED | `_fix_t5_embed_tokens()` called after all three loading paths (line 161, 173, 192). All three paths: direct .pth file, Wan model directory, standalone HF model ID. |
| `runpod/test_scheduler_comparison.py` | Scheduler comparison script (created in Plan 02) | VERIFIED | File exists. Tests UniPC+0.875, FlowMatchEuler+0.6, FlowMatchEuler+0.875 with same seed. from_single_file calls at lines 146-153 include config=HF_REPO. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `dimljus/training/wan/backend.py` | `WanTransformer3DModel.from_single_file()` | `config="Wan-AI/Wan2.2-T2V-A14B-Diffusers"` | VERIFIED | Pattern found at line 211. Subfolder resolved via `_resolve_config_subfolder()`. |
| `runpod/test_base_inference.py` | `WanTransformer3DModel.from_single_file()` | `config=HF_REPO` matching test3-inference.py | VERIFIED | Lines 176-183 use `config=HF_REPO, subfolder="transformer"` and `subfolder="transformer_2"`. Matches test3-inference.py pattern exactly. |
| `dimljus/training/sampler.py` | `dimljus/training/wan/inference.py` | `pipeline.generate()` calls inside `generate_samples()` | VERIFIED | `generate_samples()` at line 409 calls `pipeline.generate()` at line 444. `WanInferencePipeline` wired via `__main__.py` `_create_inference_pipeline()` at line 174. |
| `runpod/test_base_inference.py` | `dimljus/training/wan/inference.py` | Both use `WanTransformer3DModel.from_single_file` with config= | VERIFIED | test_base_inference.py uses from_pretrained-then-swap pattern. Both scripts apply T5 embed_tokens fix. |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| INFER-01 | 01-01, 01-02 | Base model inference produces recognizable video (not noise) with Wan 2.2 T2V | ? HUMAN | Code fixes verified: config= parameter, T5 embed_tokens fix, boundary=0.6. GPU validation documented in SUMMARY (Minta approved). Needs fresh run to confirm committed code. |
| INFER-02 | 01-01, 01-02 | LoRA inference modifies base model output in a visually detectable way | ? HUMAN | Test B in test_base_inference.py has correct LoRA loading code but auto-skips if checkpoint absent. LoRA visual comparison requires human review. |
| INFER-03 | 01-01, 01-02 | Inference works both standalone (separate script) and integrated (from training pipeline) | PARTIAL | Structurally verified: WanInferencePipeline wired through __main__.py. Both paths use same T5 fix and scheduler settings. GPU output identity cannot be confirmed programmatically. |

No orphaned requirements found. INFER-01, INFER-02, INFER-03 are all declared in both PLAN files and traced in REQUIREMENTS.md as "Complete".

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None found | - | - | - | - |

No TODO, FIXME, placeholder, stub, or empty implementation patterns found in any of the phase-modified files. All implementations are substantive.

### Notable Observations

**test_base_inference.py LoRA test is conditional:** Test B (INFER-02) has a hard-coded path `/workspace/outputs/test2-unified-only/unified/test_lora_unified_epoch003.safetensors`. If this file does not exist on the pod, Test B prints "SKIPPED" and exits cleanly. This means INFER-02 was gated on a separate training run having already completed. SUMMARY.md marks INFER-02 as done but the Self-Check only verifies that "165 related tests pass" and "output grids confirm all 3 scheduler configs work" -- this is Test A (no LoRA), not Test B (LoRA). INFER-02 visual confirmation is marked complete but may not have been executed with an actual LoRA checkpoint during this phase.

**test_scheduler_comparison.py WanTransformer3DModel calls:** test_scheduler_comparison.py (created in Plan 02) correctly includes `config=HF_REPO` on both from_single_file calls (lines 146-153). No bare calls without config.

**test_isolate_bug.py is a legacy diagnostic:** This file contains from_single_file calls without config= but this is an older diagnostic script, not part of the fix. Its calls were intentionally left bare (they test the pre-fix behavior for comparison). Not a regression risk since it is not imported by any production code.

**All 1872 tests pass.** Confirmed by running the full test suite.

### Human Verification Required

#### 1. INFER-02: LoRA Inference Visual Difference

**Test:** On RunPod, with a trained LoRA checkpoint at `/workspace/outputs/test2-unified-only/unified/test_lora_unified_epoch003.safetensors`, run `python /workspace/dimljus/runpod/test_base_inference.py`. Compare `test_A_dual_base.grid.png` (no LoRA) against `test_B_lora_diffusers.grid.png` (with LoRA).

**Expected:** Test B grid shows recognizably different content from Test A — LoRA should shift output toward the training subject, not just produce different noise patterns.

**Why human:** Requires (a) a trained LoRA checkpoint from a previous training run, (b) a running GPU pod, and (c) visual judgment of whether the difference corresponds to training data. Cannot be verified programmatically.

#### 2. INFER-01 Confirmation (committed code)

**Test:** On a fresh RunPod session (not reusing session state), run `python /workspace/dimljus/runpod/test_base_inference.py` with no LoRA. Review `test_A_dual_base.grid.png`.

**Expected:** Grid shows a coherent scene matching the prompt ("A woman with dark hair walks down a city street, morning light") -- not dark/black frames, not noise.

**Why human:** The boundary=0.6 setting was approved visually during the session, but the final commit (3216509) updating inference.py boundary from 0.5 to 0.6 was made after the visual approval. A fresh run with the committed code (not session state) confirms the exact committed values are correct.

### Gaps Summary

No hard gaps found. All code-level requirements are implemented and wired correctly. The phase status is `human_needed` rather than `gaps_found` because:

1. The automated infrastructure is fully in place: config= fixes, T5 embed_tokens fix, keyframe grid output, correct boundary_ratio, all tests passing.
2. INFER-01 was GPU-validated by Minta during the session -- the code is correct.
3. INFER-02 requires a trained LoRA checkpoint which was not produced within this phase's scope (it requires prior training runs).
4. INFER-03 is structurally verified -- both code paths use the same WanInferencePipeline class with identical settings.

The phase goal "Users can generate recognizable video from both the base Wan 2.2 model and a trained LoRA" is conditionally achieved: base model inference is proven, LoRA inference requires a checkpoint from training to confirm.

---

_Verified: 2026-02-27T19:45:00Z_
_Verifier: Claude (gsd-verifier)_
