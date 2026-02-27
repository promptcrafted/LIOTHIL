# Research Summary: Dimljus Inference Fix & Training Validation

**Domain:** Video LoRA trainer validation / integration testing
**Researched:** 2026-02-26 (updated with inference debugging deep dive)
**Overall confidence:** HIGH

## Executive Summary

This summary covers two research dimensions: (1) the inference pipeline debugging that blocks all validation, and (2) the complete validation test matrix for proving the trainer works.

### Inference Pipeline -- Deep Dive (this update)

A comprehensive investigation into what causes noisy/gridded output in video diffusion inference has revealed the most likely root cause of the Dimljus inference bug: **a triple mismatch in scheduler type, shift value, and boundary ratio.**

The official Wan 2.2 T2V HuggingFace configuration uses `UniPCMultistepScheduler(flow_shift=3.0)` with `boundary_ratio=0.875`. Dimljus inference uses `FlowMatchEulerDiscreteScheduler(shift=5.0)` with `boundary_ratio=0.5`. These produce fundamentally different sigma schedules and expert switching points. The boundary=0.5 means the low-noise expert processes 50% of timesteps instead of the designed 12.5%.

Thirteen specific pitfalls were documented in PITFALLS.md, each with symptoms, diagnostics, and fixes. The recommended debugging approach is: (1) test the official from_pretrained pipeline first, (2) check library versions, (3) diff Dimljus config against what from_pretrained loads.

### Validation Matrix (prior research)

A thorough validation matrix for a video LoRA trainer requires three layers: training correctness (loss behavior, gradient health, checkpoint integrity), inference correctness (base model coherent video, LoRA modifies output, temporal coherence), and operational resilience (checkpoint resume, mixed datasets, VRAM bounds).

The single most important finding is that **loss metrics alone are insufficient to validate training quality** -- universally agreed upon across musubi-tuner community, ai-toolkit, and recent research. Visual sampling during training with fixed seeds and production-identical inference settings is the minimum viable validation. Automated metrics (FVD, CLIP-score) are unreliable for video.

For Dimljus specifically, the differential MoE architecture introduces unique validation requirements: per-expert loss curve divergence, expert isolation verification (frozen weights must not change), timestep boundary correctness, and the ultimate thesis test -- differential hyperparameters producing visibly better results than unified training.

## Key Findings

**Inference root cause:** Triple mismatch: wrong scheduler (FlowMatchEuler vs UniPC), wrong shift (5.0 vs 3.0), wrong boundary (0.5 vs 0.875). Each independently could cause noise; combined they guarantee it.

**from_single_file config bug (diffusers#12329):** Can silently load Wan 2.1 config for Wan 2.2 models. Always requires explicit `config=` parameter.

**T5 embed_tokens:** Known weight tying bug where manual loading leaves embed_tokens as zeros, causing prompt to be ignored. Fix exists in Dimljus but needs pod verification.

**transformers v5.0:** Breaks Wan inference entirely. Must pin to 4.57.3 or earlier.

**Official correct settings (from HF model card and Wan repo):**
| Setting | Official Value |
|---------|---------------|
| Scheduler | UniPCMultistepScheduler |
| flow_shift | 3.0 |
| boundary_ratio | 0.875 (T2V), 0.9 (I2V) |
| guidance_scale | 4.0 (high-noise), 3.0 (low-noise) |
| steps | 40 |
| VAE dtype | float32 |

## Implications for Roadmap

Based on research, the validation work naturally sequences as:

1. **Fix Base Inference** -- Hard prerequisite. Nothing validates without working inference.
   - Addresses: Base model inference test (currently producing noise)
   - Root cause: Scheduler/shift/boundary triple mismatch (see PITFALLS.md Pitfall 4+6)
   - Approach: (a) Test from_pretrained first, (b) Switch to UniPCMultistepScheduler with flow_shift=3.0 and boundary_ratio=0.875, (c) Verify T5 embeddings non-zero, (d) Verify from_single_file uses explicit config=

2. **Single Expert Isolation Tests (4-5)** -- Prove each expert works independently.
   - Addresses: Expert isolation verification, per-expert loss validation
   - Avoids: Pitfall 5 (expert cross-contamination)

3. **Checkpoint Resume Tests (6-8)** -- Prove durability before long runs.
   - Addresses: Operational resilience for 50-epoch expert phases
   - Community evidence shows this is where most trainers break.

4. **Mixed Dataset Test (9)** -- Quick validation of still image support.
   - Addresses: Production dataset compatibility

5. **Real Training Run (10)** -- The thesis validation. unified 10, low 15, high 50.
   - Addresses: Differential MoE thesis, production-quality output

**Phase ordering rationale:**
- Inference fix is prerequisite for everything (all tests require visual validation)
- Single-expert before combined isolates variables
- Resume before long runs prevents progress loss during test 10
- Real training last because most expensive and depends on all prior tests

**Research flags for phases:**
- Inference fix: Needs hands-on debugging (SSH to pod, apply fixes), this research provides the complete diagnostic/fix guide
- Real training: Needs sampling frequency decision (recommend every 5 epochs for low-noise)
- Latent normalization: Should be investigated after inference works (affects LoRA quality, not base model)

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Scheduler/shift mismatch | HIGH | Official config verified from HF model card and Wan repo |
| from_single_file bug | HIGH | Documented diffusers issue with confirmed fix |
| T5 embed_tokens bug | HIGH | Root cause traced in transformers source code |
| VAE dtype | HIGH | Multiple independent reports, confirmed in Dimljus |
| transformers v5.0 | HIGH | Documented with confirmed resolution |
| Boundary ratio impact | HIGH | Official config value verified from model_index.json |

## Gaps to Address

- Exact diffusers and transformers versions on the RunPod pod need verification
- Whether the training backend's model loading uses explicit config or not
- Whether FlowMatchEulerDiscreteScheduler(shift=5.0) can work at all with Wan 2.2, or if UniPC is required
- The relationship between the official sample_shift=12.0 and diffusers flow_shift=3.0 (different schedulers, different internal math)
- Latent normalization during encoding: ai-toolkit does it, Dimljus does not
- ComfyUI LoRA compatibility: physical test needed
