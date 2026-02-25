# Fork-and-Specialize LoRA Training Strategy

**Status:** Hypothesis — requires experimental validation
**Date:** 2026-02-22
**Based on:** Expert weight divergence analysis of Wan 2.2 T2V and I2V models

---

## Core Idea

Instead of training two independent per-expert LoRAs from scratch, train a shared foundation first and then fork into per-expert specialization. This mirrors how the base model itself was trained (shared pretrained weights → MoE specialization).

The expert divergence analysis showed a clear partition in the base model:
- **Layers that barely diverged between experts** (cosine > 0.95): norms, modulation/adaLN, QK-norm weights
- **Layers that diverged significantly** (cosine < 0.95): attention projections (especially cross-attention values), FFN, embeddings

This suggests that some layers want the same correction regardless of expert, while others need expert-specific treatment.

---

## The Strategy (Both Theories)

### Steps Common to Both

1. **Choose rank** that captures the full representation needed. For a 14B-active-parameter model, at least rank 32-64. (Needs tuning — different from per-expert-from-scratch because the LoRA carries more signal.)

2. **Unified phase** — train a single LoRA as a shared foundation across both experts. (See Theory A vs Theory B below for how this works.)

3. **Fork** — copy the LoRA.

4. **Per-expert phase** — continue training each copy on its respective expert's noise range. Target only the layers that the divergence analysis showed need specialization (attn2, FFN, potentially attn1). Layers that don't need specialization (norms, modulation, QK-norm) can be frozen or excluded from training.

5. **Deliver** two standard LoRA files. The shared/per-expert split is a training strategy, not a delivery format. Users load per-expert LoRAs the same way they always have.

---

### Theory A: Shared LoRA on Both Experts

**Hypothesis:** A single LoRA can be meaningfully optimized against both experts simultaneously during the unified phase.

**How it works:**
- Inject the same LoRA weights into both experts
- Each training step: sample a timestep → SNR determines which expert is active → forward pass through that expert with the shared LoRA → backprop → update LoRA
- Over the course of training, the LoRA sees both experts naturally because different timesteps route to different experts
- The LoRA learns weight corrections that work on both sets of base weights — this IS the shared signal

**Then fork:**
- Copy the LoRA
- Continue training each copy restricted to its expert's noise range
- Optionally restrict to only the divergent layers (attn2, FFN)
- Each copy specializes for its expert

**What to watch for:**
- Loss curve during unified phase: does it converge smoothly, or oscillate?
- If oscillating: the two experts may want incompatible corrections on certain layers, meaning the LoRA can't find a good compromise
- Gradient magnitude per layer: layers where both experts agree should show consistent gradients; layers where they disagree should show noisy/conflicting gradients

**Expected outcome based on divergence data:**
- Layers with cosine > 0.95 (norms, modulation): should converge cleanly — both experts want similar corrections
- Layers with cosine < 0.92 (attn2.to_v, FFN): may show conflicting gradients — experts want different things here
- Overall loss should still converge because the majority of parameters are in the "similar" category

**Validation test:**
1. Train Theory A unified phase for N epochs
2. Compare loss curve against a baseline (per-expert from scratch for N epochs)
3. After forking, compare convergence speed and final quality against per-expert from scratch
4. Re-run divergence analysis on the two output LoRAs — divergence pattern should mirror the base model pattern (attn2/FFN diverge, norms stay shared)

---

### Theory B: Unified Phase on Shared Layers Only

**Hypothesis:** The unified phase should only target layers where both experts truly agree. Divergent layers should be per-expert from the start.

**How it works:**
- During the unified phase, only apply LoRA to the layers that the divergence analysis showed barely changed between experts:
  - Modulation/adaLN (0.978 cosine)
  - Norm layers (0.957 cosine)
  - Possibly QK-norm (0.9999 — though these may not need LoRA at all)
- Train across both experts / full noise range as in Theory A
- These layers should converge cleanly because both experts want the same thing

**Then fork:**
- Copy the LoRA (which only has shared-layer weights)
- ADD new LoRA weights for the divergent layers (attn2, FFN, attn1, embeddings)
- Train each copy on its expert's noise range, with the new per-expert layers training from initialization and the shared layers continuing/frozen

**What to watch for:**
- Does starting the per-expert layers from scratch (no unified foundation for those layers) lose quality compared to Theory A?
- Is the shared-layer-only unified phase even useful, or is it so few parameters that it doesn't contribute much?

**Expected outcome:**
- Unified phase converges very cleanly (no gradient conflict — only targeting agreed-upon layers)
- Per-expert phase has to do more work (divergent layers start from scratch instead of a shared foundation)
- Total training time might be similar or longer than Theory A

**Validation test:**
1. Train Theory B unified phase for N epochs (shared layers only)
2. Fork and train per-expert phase for M epochs
3. Compare final quality against Theory A and against per-expert from scratch baseline
4. Compare total compute (unified + per-expert epochs) against alternatives

---

## Key Question: Does Theory A's Unified Phase Actually Help the Divergent Layers?

This is the crux. In Theory A, the divergent layers (attn2, FFN) get trained on both experts during the unified phase. This gives them a "compromise" initialization before forking. The question is whether that compromise is:

**(a) A useful foundation** — the shared signal exists and is valuable, even for layers that later specialize. The compromise captures "what does this subject look like in general" and the fork adds "specifically at this noise level."

**(b) A conflicting signal** — the two experts want such different things from these layers that the compromise is noise, not signal. The fork phase has to undo the compromise before it can specialize, wasting epochs.

The divergence data suggests a spectrum:
- attn2.to_v.bias at 0.44 cosine: probably (b) — these are nearly orthogonal between experts
- FFN at 0.89 cosine: could go either way — divergent but not opposing
- attn1 at 0.92 cosine: probably (a) — different but with substantial shared component

**This might mean Theory A works for some layers and Theory B works for others.** A hybrid where the unified phase targets everything EXCEPT the most extreme outliers (attn2.to_v specifically) could be optimal.

---

## Comparison Against Baseline

Whatever approach we use, it must be compared against the known-good baseline:

**Baseline: Independent per-expert training (current approach)**
- Expert 1: rank 16, lr 1e-4, 30 epochs, high-noise range only
- Expert 2: rank 24, lr 8e-5, 50 epochs, low-noise range only
- No shared training phase
- Known-good results from production

**Comparison metrics:**
- Training loss curves (should converge at least as well)
- Inference quality on identical prompts (visual assessment by Minta)
- Total training time (unified + per-expert should be competitive)
- Parameter efficiency (same or fewer total LoRA parameters for same quality)

---

## Divergence Data Reference

From the Wan 2.2 expert weight analysis:

### T2V Within-Expert Divergence (by layer type)
| Layer Type | Avg Cosine | Interpretation |
|-----------|-----------|----------------|
| FFN | 0.892 | Most divergent — different information transforms |
| Embedding | 0.907 | Significantly different |
| Attention | 0.915 | Significantly different |
| Norm | 0.957 | Mostly shared |
| Modulation | 0.978 | Mostly shared |
| QK-norm | 0.9999 | Essentially identical |

### T2V Block Position Pattern
| Blocks | Avg Cosine | Pattern |
|--------|-----------|---------|
| 0 | 0.887 | Entry block — most divergent |
| 6-12 | 0.925-0.930 | Middle — most similar |
| 25-34 | 0.900-0.913 | Late — second divergence peak |
| 39 | 0.950 | Output — converges back |

### Most Divergent Individual Tensors
All are `attn2.to_v.bias` (cross-attention value bias):
- Block 0: **0.440** cosine
- Block 34: **0.467**
- Block 26: **0.503**
- Block 33: **0.512**

### Cross-Model Findings
| Comparison | Mean Cosine |
|-----------|-----------|
| T2V E1 vs E2 (within-model) | 0.917 |
| I2V E1 vs E2 (within-model) | 0.936 |
| T2V vs I2V high-noise expert | 0.984 |
| T2V vs I2V low-noise expert | 0.949 |

I2V experts diverge ~40% less. The reference image reduces the need for expert specialization. The low-noise expert changed much more between T2V and I2V than the high-noise expert.

---

## Open Questions

1. **Rank selection for fork-and-specialize:** Does the unified phase need higher rank than independent per-expert? Or does the better initialization mean the same rank goes further?

2. **Unified phase duration:** How many epochs for the unified phase before forking? Too few = no shared foundation. Too many = the divergent layers are stuck in a bad compromise.

3. **I2V vs T2V strategy divergence:** I2V experts are more similar (0.936 vs 0.917). Does I2V benefit from fork-and-specialize, or is the divergence small enough that a single LoRA on both experts works without forking?

4. **Block targeting during fork phase:** Should the fork phase only target blocks 0 and 25-34 (highest divergence) and leave middle blocks frozen?

5. **Hybrid Theory A/B:** Should extremely divergent layers (attn2.to_v.bias, cosine < 0.6) be excluded from the unified phase and only trained per-expert, while moderately divergent layers (FFN, cosine ~0.89) get both phases?
