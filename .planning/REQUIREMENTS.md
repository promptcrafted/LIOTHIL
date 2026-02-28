# Requirements: Dimljus -- Inference Fix & Training Validation

**Defined:** 2026-02-26
**Core Value:** The inference pipeline must produce recognizable video -- without working sampling, nothing else in the trainer can be validated.

## v1 Requirements

Requirements for this milestone. Each maps to roadmap phases.

### Inference

- [x] **INFER-01**: Base model inference produces recognizable video (not noise) with Wan 2.2 T2V
- [x] **INFER-02**: LoRA inference modifies base model output in a visually detectable way
- [x] **INFER-03**: Inference works both standalone (separate script) and integrated (from training pipeline)

### Training Isolation

- [x] **TRAIN-01**: Low-noise-only training (no unified, no high) completes without crash and produces checkpoints
- [x] **TRAIN-02**: High-noise-only training (no unified, no low) completes without crash and produces checkpoints
- [ ] **TRAIN-03**: Checkpoint resume for unified training continues from saved state without loss spike
- [ ] **TRAIN-04**: Checkpoint resume for low-noise training continues from saved state without loss spike
- [ ] **TRAIN-05**: Checkpoint resume for high-noise training continues from saved state without loss spike
- [ ] **TRAIN-06**: Still frame training (5 reference images as single-frame videos) completes and produces checkpoints

### Base Strategy Comparison

- [ ] **BASE-01**: Low-noise-only LoRA used as starting base for subsequent differential training (instead of unified)
- [ ] **BASE-02**: Merged (low+high) checkpoint used as starting base for subsequent differential training (check if ai-toolkit has a pre-merged model available)
- [ ] **BASE-03**: Compare output quality: unified base vs low-noise base vs merged base -- determine which produces the best downstream results

### Production Training

- [ ] **PROD-01**: Real training run completes using best base strategy with expert freezing
- [ ] **PROD-02**: Real training produces a LoRA that visibly modifies inference output toward training subject
- [ ] **PROD-03**: MoE theory test -- compare differential training output against uniform baseline to assess whether per-expert hyperparams are worth pursuing

### Metrics & Logging

- [x] **METR-01**: Loss curves logged per training phase (unified, low-noise, high-noise)
- [x] **METR-02**: Training wall-clock time logged per test run
- [x] **METR-03**: VRAM usage tracked during training
- [x] **METR-04**: Per-expert loss divergence tracked (how expert losses separate over epochs)
- [x] **METR-05**: Fixed-seed visual samples generated at regular intervals during training
- [x] **METR-06**: Expert weight checksums verified (frozen expert weights don't change during other expert's training)

## v2 Requirements

Deferred to future milestone. Tracked but not in current roadmap.

### Expanded Model Support

- **MODEL-01**: I2V training support (reference image as control signal)
- **MODEL-02**: VACE context block training support
- **MODEL-03**: Additional model backends (LTX-2, SkyReels)

### Advanced Features

- **ADV-01**: Control signal expansion (depth, edge, pose generation + control LoRA training)
- **ADV-02**: Audio control signal interface
- **ADV-03**: Latent normalization during VAE encoding (ai-toolkit pattern)

### Release

- **REL-01**: Public packaging and documentation
- **REL-02**: ComfyUI LoRA compatibility validation

## Out of Scope

| Feature | Reason |
|---------|--------|
| Changing validated inference params (shift=5.0, boundary=0.5) | Minta tested extensively; investigate code bugs first |
| I2V training | T2V validation first; I2V adds reference image complexity |
| Automated quality metrics (FVD, CLIP-score) | Research shows unreliable for video; visual inspection preferred |
| New data pipeline features | Data pipeline is complete; this scope is training/inference only |
| Multi-GPU / distributed training | Single-GPU validation first |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFER-01 | Phase 1 | Complete |
| INFER-02 | Phase 1 | Complete |
| INFER-03 | Phase 1 | Complete |
| METR-01 | Phase 2 | Complete |
| METR-02 | Phase 2 | Complete |
| METR-03 | Phase 2 | Complete |
| METR-04 | Phase 2 | Complete |
| METR-05 | Phase 2 | Complete |
| METR-06 | Phase 2 | Complete |
| TRAIN-01 | Phase 3 | Complete |
| TRAIN-02 | Phase 3 | Complete |
| TRAIN-03 | Phase 4 | Pending |
| TRAIN-04 | Phase 4 | Pending |
| TRAIN-05 | Phase 4 | Pending |
| TRAIN-06 | Phase 5 | Pending |
| BASE-01 | Phase 6 | Pending |
| BASE-02 | Phase 6 | Pending |
| BASE-03 | Phase 6 | Pending |
| PROD-01 | Phase 7 | Pending |
| PROD-02 | Phase 7 | Pending |
| PROD-03 | Phase 7 | Pending |

**Coverage:**
- v1 requirements: 21 total
- Mapped to phases: 21
- Unmapped: 0

---
*Requirements defined: 2026-02-26*
*Last updated: 2026-02-28 -- TRAIN-01 and TRAIN-02 completed in Plan 03-01*
