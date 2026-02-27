# Roadmap: Dimljus -- Inference Fix & Training Validation

## Overview

The training infrastructure exists and produces checkpoints with decreasing loss, but inference outputs noisy grids -- nothing can be validated visually until sampling works. This roadmap fixes the inference pipeline first, stands up metrics infrastructure so every subsequent test is logged, then systematically proves each component of the trainer (single experts, checkpoint resume, still frames), compares base strategies for differential training, and culminates in a real production training run that tests the core MoE thesis.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Fix Inference Pipeline** - Make base model and LoRA inference produce recognizable video
- [ ] **Phase 2: Metrics Infrastructure** - Build logging so every subsequent test is captured
- [ ] **Phase 3: Expert Isolation Tests** - Prove each expert trains correctly in isolation
- [ ] **Phase 4: Checkpoint Resume Tests** - Prove training can survive interruption and resume cleanly
- [ ] **Phase 5: Still Frame Training** - Prove the pipeline handles single-frame video edge cases
- [ ] **Phase 6: Base Strategy Comparison** - Determine which starting point produces the best downstream differential training
- [ ] **Phase 7: Production Training & MoE Thesis** - Run real training and validate that differential hyperparameters are worth it

## Phase Details

### Phase 1: Fix Inference Pipeline
**Goal**: Users can generate recognizable video from both the base Wan 2.2 model and a trained LoRA
**Depends on**: Nothing (first phase -- hard prerequisite for all validation)
**Requirements**: INFER-01, INFER-02, INFER-03
**Success Criteria** (what must be TRUE):
  1. Running base model inference (no LoRA) produces a coherent video matching the text prompt -- not noise, not a grid, not a static frame
  2. Running inference with a trained LoRA produces output that is visually different from base model output in a way that corresponds to the training data
  3. The same inference code produces identical results whether called from a standalone test script or triggered by the training pipeline's sample generation
**Plans**: 2 plans

Plans:
- [ ] 01-01-PLAN.md -- Fix from_single_file() config bug in all code paths + add keyframe grid output
- [ ] 01-02-PLAN.md -- GPU validation of base model and LoRA inference on RunPod

### Phase 2: Metrics Infrastructure
**Goal**: Every training run automatically logs the data needed to compare experiments
**Depends on**: Phase 1 (visual samples require working inference)
**Requirements**: METR-01, METR-02, METR-03, METR-04, METR-05, METR-06
**Success Criteria** (what must be TRUE):
  1. After any training run, per-phase loss curves (unified, low-noise, high-noise) are available as logged data that can be plotted or compared
  2. Wall-clock time for each training run is recorded automatically without manual timing
  3. VRAM usage is captured at regular intervals during training and peak usage is reported at the end
  4. When both experts are trained, their loss values are logged separately so divergence is visible over epochs
  5. Fixed-seed visual samples are generated at configurable intervals during training (using the working inference from Phase 1)
  6. After training a phase where one expert is frozen, the frozen expert's weight checksums before and after confirm no modification occurred
**Plans**: TBD

Plans:
- [ ] 02-01: TBD
- [ ] 02-02: TBD

### Phase 3: Expert Isolation Tests
**Goal**: Each expert (low-noise and high-noise) can be trained independently and produces valid checkpoints
**Depends on**: Phase 1 (need inference to visually validate), Phase 2 (need metrics to log results)
**Requirements**: TRAIN-01, TRAIN-02
**Success Criteria** (what must be TRUE):
  1. Low-noise-only training (no unified warm-up, no high-noise phase) runs to completion without crash and saves a loadable checkpoint
  2. High-noise-only training (no unified warm-up, no low-noise phase) runs to completion without crash and saves a loadable checkpoint
  3. Loss curves for both isolation tests show decreasing loss over epochs (logged via Phase 2 metrics)
**Plans**: TBD

Plans:
- [ ] 03-01: TBD

### Phase 4: Checkpoint Resume Tests
**Goal**: Training can be interrupted and resumed without losing progress or corrupting the model
**Depends on**: Phase 3 (expert isolation must work before testing resume)
**Requirements**: TRAIN-03, TRAIN-04, TRAIN-05
**Success Criteria** (what must be TRUE):
  1. Unified training stopped mid-run and resumed from checkpoint continues with no loss spike at the resume point
  2. Low-noise training stopped mid-run and resumed from checkpoint continues with no loss spike at the resume point
  3. High-noise training stopped mid-run and resumed from checkpoint continues with no loss spike at the resume point
  4. Resumed training produces checkpoints that are functionally identical to uninterrupted training (verified by inference comparison)
**Plans**: TBD

Plans:
- [ ] 04-01: TBD

### Phase 5: Still Frame Training
**Goal**: The training pipeline handles degenerate video data (single frames) without crashing
**Depends on**: Phase 1 (need inference for validation)
**Requirements**: TRAIN-06
**Success Criteria** (what must be TRUE):
  1. Training on 5 reference images (treated as single-frame videos) completes without crash and produces a loadable checkpoint
  2. Inference with the still-frame LoRA produces output that shows influence from the training images
**Plans**: TBD

Plans:
- [ ] 05-01: TBD

### Phase 6: Base Strategy Comparison
**Goal**: Determine which base training strategy produces the best starting point for differential expert training
**Depends on**: Phase 3 (need expert isolation to work), Phase 4 (need resume for long runs)
**Requirements**: BASE-01, BASE-02, BASE-03
**Success Criteria** (what must be TRUE):
  1. A low-noise-only LoRA is used as the starting base for a subsequent differential training run (instead of unified warm-up)
  2. A merged low+high checkpoint is used as the starting base for a subsequent differential training run
  3. Visual comparison of outputs from all three base strategies (unified, low-noise-only, merged) is available and Minta can identify which produces the best downstream results
**Plans**: TBD

Plans:
- [ ] 06-01: TBD
- [ ] 06-02: TBD

### Phase 7: Production Training & MoE Thesis
**Goal**: Run real training with the best base strategy and validate that differential MoE hyperparameters produce better results than uniform training
**Depends on**: Phase 6 (need best base strategy determined), Phase 2 (need full metrics)
**Requirements**: PROD-01, PROD-02, PROD-03
**Success Criteria** (what must be TRUE):
  1. A full production training run (using the best base strategy from Phase 6, with expert-specific epochs and learning rates) completes without crash
  2. Inference with the production LoRA produces video that visibly depicts the training subject (Holly) -- recognizable person, not just style transfer
  3. Side-by-side comparison of differential training output vs uniform-hyperparameter training output is available, and Minta can assess whether per-expert tuning produces visibly better results
**Plans**: TBD

Plans:
- [ ] 07-01: TBD
- [ ] 07-02: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Fix Inference Pipeline | 0/2 | Not started | - |
| 2. Metrics Infrastructure | 0/2 | Not started | - |
| 3. Expert Isolation Tests | 0/1 | Not started | - |
| 4. Checkpoint Resume Tests | 0/1 | Not started | - |
| 5. Still Frame Training | 0/1 | Not started | - |
| 6. Base Strategy Comparison | 0/2 | Not started | - |
| 7. Production Training & MoE Thesis | 0/2 | Not started | - |
