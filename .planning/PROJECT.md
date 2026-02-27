# Dimljus — Inference Fix & Training Validation

## What This Is

Dimljus is a purpose-built video LoRA training toolkit for diffusion transformer models, starting with Wan 2.2's dual-expert MoE architecture. The foundation is built (config, data pipeline, captioning, training loop, checkpoint I/O) — this scope covers fixing the broken inference pipeline and running the test matrix to prove training actually works, culminating in a real differential MoE training run.

## Core Value

The inference pipeline must produce recognizable video — without working sampling, nothing else in the trainer can be validated.

## Requirements

### Validated

- ✓ Training config schema with differential MoE support — existing
- ✓ Wan 2.2 model backend (dual-expert loading, switching, forward pass) — existing
- ✓ Training loop with expert-aware noise scheduling — existing
- ✓ Checkpoint save/load with diffusers prefix format — existing
- ✓ VRAM management (subprocess encoding, gc.collect, expert swap) — existing
- ✓ Dataset caching (VAE latents + T5 embeddings) — existing
- ✓ Test 1 complete: full pipeline unified→low→high (3 epochs each) — existing
- ✓ Test 2 complete: unified only (3 epochs) — existing
- ✓ Test 3 complete: experts only, no unified (3 epochs each) — existing

### Active

- [ ] Fix base model inference (currently produces noisy grid)
- [ ] Fix LoRA inference (depends on base model fix)
- [ ] Run test 4: low-noise only training (no unified, no high)
- [ ] Run test 5: high-noise only training (no unified, no low)
- [ ] Run tests 6-8: checkpoint resume for unified, low, high
- [ ] Run test 9: still frame training (5 frames as single-frame videos)
- [ ] Run real training: unified 10 → low 15 → high 50 epochs
- [ ] Validate MoE theory: differential hyperparams produce visibly better results
- [ ] Log metrics: loss curves, training time, VRAM usage per test

### Out of Scope

- New model backends (LTX-2, SkyReels) — future milestone
- Control signal expansion (depth, edge, pose, VACE) — future milestone
- Data pipeline improvements — Phase 4 work, not this scope
- Packaging/release/documentation — after training is proven
- I2V training — T2V validation first

## Context

The inference pipeline worked at one point in a standalone script after researching how musubi-tuner and ai-toolkit handle sampling. Key differences found: both merge LoRA into base weights for inference (Dimljus keeps as wrapper), musubi uses FlowUniPCMultistep with shift=12.0 for T2V 480p, ai-toolkit uses UniPCMultistep with flow_shift=3.0. The fix worked in isolation but broke when integrated back into the main pipeline.

Training itself appears functional — loss decreases across epochs, checkpoints save correctly at expected sizes (293MB per expert, 586MB merged). The noisy grid is specifically an inference/sampling issue, not a training issue.

Current test environment: RunPod with NVIDIA RTX PRO 6000 Blackwell 98GB, 10 Holly clips dataset with 29 VAE + 9 T5 cached embeddings.

Current inference settings (from memory, Minta-validated): shift=5.0, boundary_ratio=0.5, FlowMatchEulerDiscreteScheduler. Training uses boundary_ratio=0.875 (intentionally different).

## Constraints

- **Hardware**: RunPod GPU pod (RTX PRO 6000 98GB) — SSH access, no GUI
- **Model weights**: Wan 2.2 T2V A14B (~27GB per expert), stored on pod
- **Validation method**: Visual inspection of generated video + loss metrics
- **Reference baseline**: musubi-tuner output on same dataset (known-good)
- **Settings lock**: Do not change Minta-validated inference params (shift=5.0, boundary=0.5) without discussion

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| FlowMatchEulerDiscrete over UniPC | Minta validated these settings previously | — Pending (under investigation) |
| Training boundary 0.875, inference boundary 0.5 | Different contexts need different cutoffs | — Pending |
| T2V first, I2V later | Simpler case (one control signal) isolates pipeline bugs | ✓ Good |
| Process isolation for encoding | Prevents VRAM leaks from encoder objects | ✓ Good |

---
*Last updated: 2026-02-26 after GSD initialization*
