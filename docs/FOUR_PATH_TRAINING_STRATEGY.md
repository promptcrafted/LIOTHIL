# Four-Path Expert Training Strategy

**Date:** 2026-02-24
**Status:** Working hypothesis — requires experimental validation
**Based on:** [Expert Weight Analysis](EXPERT_WEIGHT_ANALYSIS.md), [Fork-and-Specialize](FORK_AND_SPECIALIZE.md), [Training Methodology](TRAINING_METHODOLOGY.md)

---

## Overview

Dimljus's fork-and-specialize strategy trains a unified LoRA warmup, then forks into per-expert LoRAs. But the four expert paths (T2V high/low, I2V high/low) each behaved completely differently during Alibaba's 2.1 -> 2.2 specialization. This means each path needs a different training budget, different rank allocation, and different expectations for what the fork phase needs to accomplish.

This document breaks down each path with data from the expert weight analysis, proposes per-path training hypotheses, and flags the key experiments needed to validate them.

---

## The Four Paths at a Glance

| Path | Movement from 2.1 | TV Cosine | Specialization Type | Hypothesis |
|------|-------------------|-----------|--------------------|----|
| **T2V High-Noise** | 46.3% | ~0 (orthogonal) | Heavy, one-sided | Needs aggressive post-fork training |
| **T2V Low-Noise** | 0.04% | ~0 (orthogonal) | Essentially none | Minimal post-fork refinement (IS Wan 2.1) |
| **I2V High-Noise** | 50.1% | 0.58 (similar dir) | Heavy, coordinated | Heavy post-fork, but unified phase carries more |
| **I2V Low-Noise** | 36.4% | 0.58 (similar dir) | Substantial, coordinated | Real post-fork training needed (NOT like T2V low) |

---

## Path 1: T2V High-Noise Expert

### What the data says
- **46.3% average movement** from Wan 2.1 across LoRA targets
- Movement gradient: early blocks 55.2%, middle 47.5%, late 34.9%
- Component ranking: FFN 59.0% > self-attn 52.3% > cross-attn 33.9%
- Task vector direction: orthogonal to low-noise (cosine ~0)
- This expert was completely retrained by Alibaba for composition and motion at high noise levels

### What this means for LoRA training
This is the expert that does the most "new work" during personalization. When you teach a model a new character's motion style, walk cycle, or how they move through space, it's predominantly the high-noise expert learning that.

The high-noise expert converges faster on these coarse features (they're structurally simpler than fine detail), but needs real capacity to capture them.

### Training hypothesis
- **Rank:** 16-24 (moderate — coarse features need less capacity than fine detail)
- **Learning rate:** 1e-4 (higher — faster convergence on compositional features)
- **Post-fork epochs:** 20-30 (converges earlier than low-noise)
- **Block targeting priority:** Early blocks 0-11 (55%+ movement, where composition lives)
- **Component priority:** FFN first (59%), then self-attention (52%)

### Current production settings (musubi baseline)
- Rank 16 / lr 1e-4 / 30 epochs — these settings work and are the known-good baseline

---

## Path 2: T2V Low-Noise Expert

### What the data says
- **0.04% average movement** from Wan 2.1 — essentially zero
- Asymmetry ratio: high-noise moved **1,100x more** than low-noise
- 65% of LoRA-target layers (258/400) moved less than 5%
- Task vector: orthogonal to high-noise (no shared direction)
- **The T2V low-noise expert IS Wan 2.1.** Alibaba barely touched it.

### What this means for LoRA training
The low-noise expert was already good at fine detail rendering. It didn't need new skills for Wan 2.2 — it just needed to stay stable while the high-noise expert specialized. For LoRA training, this means:

- The low-noise expert needs to learn your subject's fine details (textures, facial features, skin quality)
- It starts from an excellent foundation (Wan 2.1 is a proven detail renderer)
- **It overfits rapidly when pushed too hard** — the "strawberryman" finding: aggressive LR on low-noise produces washed-out artifacts, collapsed identity, lost texture fidelity

### Training hypothesis
- **Rank:** 24-32 (higher — fine details are information-dense)
- **Learning rate:** 6e-5 to 8e-5 (lower — prevents overfitting artifacts)
- **Post-fork epochs:** 40-50 (needs longer, gentler refinement)
- **Block targeting:** Broader than high-noise — fine detail uses the full transformer depth
- **Component priority:** FFN still most important, but cross-attention matters more here (text guidance for detail)

### The Wan 2.1 swap experiment
Since T2V low-noise moved only 0.04% from Wan 2.1, you could theoretically swap in Wan 2.1 weights for the low-noise expert and get near-identical results. This would be a useful validation experiment:

1. Run inference with Wan 2.2 T2V normally (high-noise expert 1 + low-noise expert 2)
2. Run inference swapping low-noise expert 2 with Wan 2.1 T2V weights
3. Compare output quality — if they're perceptually identical, it confirms the analysis

If successful, this also means **Wan 2.1 T2V LoRAs you've already trained should work as T2V low-noise expert LoRAs** with minimal or no re-training. That's a direct value proposition: existing LoRA assets are reusable.

### Current production settings (musubi baseline)
- Rank 24 / lr 8e-5 / 50 epochs — these settings work and are the known-good baseline

---

## Path 3: I2V High-Noise Expert

### What the data says
- **50.1% average movement** from Wan 2.1 I2V — slightly more than T2V high-noise (46.3%)
- Movement gradient: early blocks 59.9%, middle 51.2%, late 38.0%
- Component ranking: FFN 64.0% > self-attn 56.8% > cross-attn 36.4%
- Task vector cosine with low-noise: **0.58** (moved in similar direction!)
- Expert-to-expert cosine: 0.9298 (more similar to its partner than T2V experts are)
- Cross-model comparison: T2V vs I2V high-noise Pearson r = 0.9992 (near-identical pattern)

### Why it's different from T2V high-noise
The high-noise expert's JOB is similar in both models (global composition, motion at high noise levels). The movement patterns are almost identical (0.9992 correlation). But there's a critical difference: **the reference image**.

In I2V, the reference image is VAE-encoded and channel-concatenated with the noisy latents (16 noise + 16 image + 4 other = 36 channels). The high-noise expert has to learn composition FROM a reference image, not just from text. This extra conditioning channel is why it moved slightly more than T2V (50.1% vs 46.3%).

The TV cosine of 0.58 means the high-noise expert moved in a similar direction to the low-noise expert — both were adapting to the image conditioning. The high-noise expert adapted MORE (composition from reference image is its specialty), but both experts had to learn "how to use this reference image."

### Training hypothesis
- **Rank:** 16-24 (similar to T2V high-noise)
- **Learning rate:** 1e-4 (similar to T2V high-noise)
- **Post-fork epochs:** 25-35
- **Key difference from T2V:** The unified phase carries more information here. Because both I2V experts moved in similar directions (cosine 0.58), the unified LoRA captures a meaningful shared signal about "how this subject looks in the reference image." The fork phase adds "how to compose this subject at high noise" on top of that shared foundation.
- **Block targeting:** Same early-block priority (blocks 0-11)

### Implication for fork-and-specialize
In T2V, the unified phase can't really optimize for both experts on divergent layers because they moved orthogonally (cosine ~0). In I2V, the unified phase IS useful for divergent layers because they moved in similar directions (cosine 0.58). This means:

- **I2V fork-and-specialize should produce better results than T2V fork-and-specialize** — the unified phase does real work for both experts
- The unified phase may need more epochs for I2V (to learn the shared reference-image signal)
- The per-expert phase may need fewer epochs (the shared foundation covers more ground)

---

## Path 4: I2V Low-Noise Expert

### What the data says
- **36.4% average movement** from Wan 2.1 I2V — this is the critical difference from T2V
- Movement gradient: early blocks 44.3%, middle 36.5%, late 28.1%
- Component ranking: FFN 46.8% > self-attn 41.9% > cross-attn 25.6%
- NOT A SINGLE LoRA-target layer moved less than 5% — every layer changed substantially
- Task vector cosine with high-noise: **0.58** (coordinated movement)

### Why this is NOT like T2V low-noise
This is the most important insight in the analysis. In T2V, the low-noise expert IS Wan 2.1 (0.04% movement). In I2V, the low-noise expert changed **36.4%** — nearly as much as the high-noise expert (50.1%). The asymmetry ratio is only 1.4x (vs 1,100x for T2V).

**Why?** Because the reference image changes everything. The I2V low-noise expert has to learn how to use the reference image's fine detail information for its specific job (texture rendering, identity preservation, detail fidelity). The original Wan 2.1 I2V weights didn't have this dual-expert specialization, so the low-noise expert had to develop a new skill: "render fine details guided by this specific reference image."

### Training hypothesis
- **Rank:** 32 (higher than T2V low-noise — more adaptation needed)
- **Learning rate:** 6e-5 to 8e-5 (still conservative — overfitting risk remains)
- **Post-fork epochs:** 40-50 (substantial training needed, not just gentle refinement)
- **Block targeting:** Broad — every layer changed significantly
- **Key difference from T2V low-noise:** Cannot be treated as "already done." Real training budget required.

### Minta's production finding
For I2V specifically, Minta found that applying only the low-noise LoRA to BOTH experts at inference produced better results than separate overfit LoRAs. This aligns with the analysis:

- TV cosine of 0.58 means both experts moved in similar directions
- A well-trained low-noise LoRA captures signal that's useful for BOTH experts
- This is why unified/single-LoRA approaches may work better for I2V than for T2V

---

## T2V vs I2V: Two Different Strategies

### T2V: Asymmetric Fork

```
Unified Warmup (10 epochs)
├── Captures: general subject identity at all noise levels
├── Useful for: base features, subject recognition
└── Limited by: experts want orthogonal things (cosine ~0)
    │
    ├── Fork → High-Noise Expert (20-30 epochs, aggressive)
    │   ├── Learns: composition, motion, spatial layout for this subject
    │   ├── Rank: 16-24, LR: 1e-4
    │   └── Most adaptation needed (46.3% movement from 2.1)
    │
    └── Fork → Low-Noise Expert (40-50 epochs, conservative)
        ├── Learns: fine detail, texture, identity preservation
        ├── Rank: 24-32, LR: 8e-5
        └── Already has strong foundation (IS Wan 2.1, 0.04% movement)
```

The unified phase in T2V is primarily useful for shared/norm/modulation layers. The divergent layers (FFN, attention) can't learn a useful shared signal because the two experts want completely different things from those layers.

### I2V: Coordinated Fork

```
Unified Warmup (15 epochs — longer, carries more signal)
├── Captures: "how this subject looks in the reference image" — shared signal
├── Useful for: BOTH experts (cosine 0.58 = similar direction)
└── The unified phase does real work for divergent layers too
    │
    ├── Fork → High-Noise Expert (25-35 epochs)
    │   ├── Learns: composition from reference, motion planning
    │   ├── Rank: 16-24, LR: 1e-4
    │   └── Builds on strong shared foundation
    │
    └── Fork → Low-Noise Expert (40-50 epochs, conservative)
        ├── Learns: detail rendering from reference, texture fidelity
        ├── Rank: 32, LR: 6e-5 to 8e-5
        └── Needs real training (36.4% movement — NOT Wan 2.1)
```

The I2V unified phase should produce a better shared foundation because both experts moved in similar directions. This means:
- **Longer unified phase is justified** (more shared signal to capture)
- **Shorter per-expert phases may suffice** (less distance to cover from fork point)
- **Single LoRA (no fork) is a viable fallback** if fork doesn't outperform it

---

## Experiment Plan

### Experiment 1: T2V Fork-and-Specialize (Core Validation)
Compare against musubi baseline on identical dataset.

| Run | Strategy | Settings |
|-----|----------|----------|
| A (baseline) | musubi per-expert from scratch | H: rank 16, lr 1e-4, 30ep / L: rank 24, lr 8e-5, 50ep |
| B | Fork-and-specialize | 10ep unified → H: 20ep post-fork / L: 40ep post-fork |
| C | Fork, high-noise only | 10ep unified → H: 20ep post-fork / L: keep unified weights |

Run C tests whether T2V low-noise even needs a fork phase (given it's essentially Wan 2.1).

### Experiment 2: Wan 2.1 Swap Test (T2V Low-Noise Validation)
No training needed — pure inference test.

| Run | Low-Noise Expert Weights |
|-----|--------------------------|
| A | Wan 2.2 T2V low-noise expert (normal) |
| B | Wan 2.1 T2V weights (swapped in) |

If perceptually identical: confirms the 0.04% finding and validates that existing Wan 2.1 LoRAs transfer.

### Experiment 3: I2V Fork-and-Specialize
Compare unified vs fork approaches for I2V.

| Run | Strategy | Settings |
|-----|----------|----------|
| A (baseline) | musubi per-expert from scratch | H: rank 24, lr 1e-4, 30ep / L: rank 32, lr 8e-5, 50ep |
| B | Fork-and-specialize | 15ep unified → H: 25ep / L: 40ep |
| C | Single unified LoRA (no fork) | 50ep unified on both experts |

Run C tests whether I2V even benefits from forking (given TV cosine 0.58 and Minta's finding that single low-noise LoRA on both experts works well).

### Experiment 4: Unified Phase Duration
How many unified epochs before returns diminish?

| Run | Unified Epochs | Post-Fork Epochs |
|-----|---------------|-----------------|
| A | 5 | 45 total per-expert |
| B | 10 | 40 total per-expert |
| C | 20 | 30 total per-expert |
| D | 30 | 20 total per-expert |

This tests whether the unified phase is doing real work or just burning compute.

---

## Key Numbers Reference

### Movement from Wan 2.1 (LoRA targets, per-block range)

| Path | Early (0-9) | Middle (10-29) | Late (30-39) |
|------|-------------|---------------|--------------|
| T2V High | 55.2% | 47.5% | 34.9% |
| T2V Low | 0.04% | 0.05% | 0.04% |
| I2V High | 59.9% | 51.2% | 38.0% |
| I2V Low | 44.3% | 36.5% | 28.1% |

### Movement by Component (LoRA targets)

| Path | FFN | Self-Attn | Cross-Attn |
|------|-----|-----------|------------|
| T2V High | 59.0% | 52.3% | 33.9% |
| T2V Low | 0.06% | 0.05% | 0.02% |
| I2V High | 64.0% | 56.8% | 36.4% |
| I2V Low | 46.8% | 41.9% | 25.6% |

### Expert Relationship Metrics

| Metric | T2V | I2V |
|--------|-----|-----|
| TV cosine (direction agreement) | 0.0005 | 0.58 |
| Expert-to-expert cosine | 0.9103 | 0.9298 |
| Asymmetry ratio (H/L) | 1,100x | 1.4x |
| Specialization type | One-sided | Two-sided, coordinated |

### Block Targeting (Both Models)

| Priority | Blocks | Why |
|----------|--------|-----|
| **Highest** | 0-11 | Most divergent, composition/motion processing |
| **Standard** | 12-29 | Moderate divergence, general processing |
| **Lowest** | 30-39 | Least divergent, output refinement |

Spearman rank correlation between T2V and I2V block ordering: **0.9959** — same targeting strategy works for both.
