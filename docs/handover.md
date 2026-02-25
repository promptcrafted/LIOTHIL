# Dimljus — Research Partner Handover

**Date:** 2026-02-23
**Author:** Minta (via Claude session)

This document brings a new research partner up to speed on the Dimljus project — what it is, where we are, what's been decided, what's still open, and what's actively in progress.

---

## What Is Dimljus?

A purpose-built video LoRA training toolkit for diffusion transformer models. The core thesis: **video training is a control signal routing problem.** Video is the sole target the model learns to produce; everything else (captions, reference images, depth maps, poses) is a first-class control signal with its own preparation, validation, encoding, and weighting.

**This is NOT a general-purpose trainer.** It encodes Minta's specific methodology for production video LoRA training, built from the ground up rather than extending existing image-focused tools.

**Philosophy:** Curator-first, not math-first. Datasets, aesthetic judgment, and production quality drive technical decisions.

---

## Who Is Building This

**Minta** — Senior AI Architect, cofounder of Alvdansen Labs. Thousands of hours of training experience, deep understanding of model behavior, but a code beginner. She defines what the tool should do, evaluates output quality, and provides the domain expertise.

**Claude** — Implementation partner. Writes clean Python, explains every design decision, respects Minta's intelligence without assuming syntax knowledge.

**Working style:** Discussion-first, action-later. This project requires deep architectural discussion before implementation. Minta signals when she's ready to move from discussion to building. Step by step — verify each step before moving on.

---

## Target Models

**Primary:** Wan 2.2 (dual MoE, 14B active / ~27B total parameters)
- T2V (text-to-video): single control signal (text)
- I2V (image-to-video): two control signals (text + reference image)
- Two experts: high-noise (early denoising, global composition) and low-noise (late denoising, fine detail)
- Expert switching at SNR boundary (~0.875 for T2V, ~0.900 for I2V)

**Validation path:** Wan 2.1 T2V first (single transformer, simplest), then Wan 2.2 T2V (adds MoE), then Wan 2.2 I2V (adds image control signal).

**Future:** LTX-2, SkyReels-V2, models yet to come. Infrastructure is model-agnostic.

---

## The Key Discovery: Differential MoE Training

This is the core insight that no existing trainer supports natively.

The two Wan 2.2 experts need **different hyperparameters**. Training both identically produces suboptimal results:
- High-noise expert converges faster on coarse features -> lower rank, higher LR, fewer epochs
- Low-noise expert needs gentler treatment for fine detail -> higher rank, lower LR, more epochs
- Aggressive low-noise training causes washed-out artifacts, texture loss, identity collapse

**Current experimental parameters:**

| Parameter | High-Noise Expert | Low-Noise Expert |
|-----------|------------------|------------------|
| LoRA Rank | 16 | 24 |
| Learning Rate | 1e-4 | 8e-5 |
| Epochs | 30 | 50 |

**Training strategy:** Mirror the base model — unified foundation phase, then fork into per-expert specialization. This is decided (Decision D9).

---

## Current Status (2026-02-24)

**Phase 0: COMPLETE** — Data Config Schema (76 tests)
**Phase 1: COMPLETE** — Video Ingestion & Scene Detection (114 tests)
**Phase 2: COMPLETE** — Caption Generation Pipeline (143 tests)
**Phase 3: COMPLETE** — Image Extraction & Processing (88 tests)
**Phase 4: COMPLETE** — Dataset Validation & Organization (205 tests)
**Triage Module: COMPLETE** — CLIP-based content sorting (45 tests)
**Batch Ingest + CLI: COMPLETE** — Directory ingest, max-frames, organize flags (69 tests)
**Phase 5: IN REVIEW** — Training Config Schema (204 tests implemented, under file-by-file review)

**Total: 1085 tests passing (881 existing + 204 Phase 5).**

### Phase 5 Status Detail

Initial implementation complete (schema, loader, defaults, examples, tests). Currently undergoing Minta's file-by-file review. Major restructuring in progress:

- **Template architecture revised**: Shared `training_defaults.py` → per-model template files. First: `wan22_t2v_training_template.py` (self-contained, no shared constants).
- **Fork-and-specialize is PRIMARY**: Not optional/experimental. Four training paths supported (unified→fork, unified-only, expert-from-scratch, expert-from-file).
- **Quality-first defaults**: LR 5e-5 (not 2e-4), base precision bf16 (not fp8), no speed-over-quality shortcuts.
- **Two-level fork targeting**: Component-level (ffn, self_attn, cross_attn) and projection-level (cross_attn.to_v).
- **Per-expert rank removed**: Rank determines matrix shape — can't change after fork.
- **Remaining review**: training_schema.py, training_loader.py, example YAMLs, tests need revision to match new template structure.

See `docs/PHASE5_REVIEW_NOTES.md` for detailed change log.

### End-to-End Pipeline Validation (2026-02-23)

Tested the full pipeline on 25 Breakfast at Tiffany's YouTube videos:
1. **Ingest**: 25 videos -> 1720 clips (scene detect + split + normalize + subdivide at 81 frames max)
2. **Triage**: CLIP matching against reference images -> holly(162), cat(356), tiffanys(233), text_overlay(84), unmatched(885)
3. **Caption**: 162 Holly clips captioned via Replicate -> short, factual, natural language confirmed production-quality
4. **Validated**: CHARACTER triage is production-ready. OBJECT/SETTING triage is experimental (matches visual similarity, not specific identity).

### What Exists

- Complete data preparation pipeline: ingest -> triage -> caption -> extract -> validate
- 6 packages: `dimljus/config/`, `dimljus/video/`, `dimljus/caption/`, `dimljus/dataset/`, `dimljus/triage/`
- CLI tools: scan, ingest, normalize, caption, score, audit, extract, triage, validate
- Multiple caption backends: Gemini (API), Replicate (raw HTTP), OpenAI-compatible (local)
- CLIP-based triage with text overlay detection
- Dataset validation with quality metrics (blur, exposure, motion, duplicates)
- Manifest generation and bucketing preview
- Rich terminal output with ASCII-safe plaintext fallback
- Extensive architecture documentation (see `docs/` directory)
- 18 resolved decisions (D1-D18) and 16 open questions (Q3-Q16)

---

## Package Architecture

```
dimljus/
  config/         ← Data config schema (Pydantic v2, YAML, 3-tier design)
  video/          ← Video pipeline (probe, validate, scene detect, split, extract, quality)
  caption/        ← Caption pipeline (Gemini, Replicate, OpenAI-compat, scoring, audit)
  dataset/        ← Dataset validation (discover, quality metrics, validate, manifest, bucketing, reports)
  triage/         ← CLIP-based content sorting (embeddings, subjects, filters, organize)
```

### Pipeline Flow

```
Raw footage
    |
    v
Phase 1: Ingest (scene detect -> split -> normalize -> subdivide)     <- MODEL-AGNOSTIC
    |
    v
Triage: CLIP match against subject references -> organize into folders <- MODEL-AGNOSTIC
    |
    v
Phase 2: Caption (Gemini/Replicate/local VLM -> .txt sidecars)       <- MODEL-AGNOSTIC
    |
    v
Phase 3: Extract references (first frame, best frame -> .png)         <- MODEL-AGNOSTIC
    |
    v
Phase 4: Validate & manifest (completeness, quality, bucketing)       <- MODEL-AGNOSTIC
    |
    v
Phase 6: Latent pre-encoding (VAE + text encoder -> cached tensors)   <- MODEL-SPECIFIC
    |
    v
Phase 7-8: Training (noise schedule, LoRA, forward pass, loss)        <- MODEL-SPECIFIC
```

---

## Active Research: Expert Weight Divergence Analysis

We measured how much the two Wan 2.2 MoE experts diverged from their shared initialization. This directly informs the fork-and-specialize LoRA strategy.

### Headline Numbers

| Comparison | Mean Cosine | What It Means |
|-----------|-----------|--------------|
| T2V Expert 1 vs Expert 2 | **0.917** | Major divergence. Fork-and-specialize is justified. |
| I2V Expert 1 vs Expert 2 | **0.936** | Less divergent. Reference image reduces need for specialization. |
| T2V vs I2V high-noise expert | **0.984** | Barely changed. Ref image doesn't affect high-noise expert. |
| T2V vs I2V low-noise expert | **0.949** | Changed significantly. Ref image reshaped low-noise expert. |

Natural partition at ~0.95 cosine: norms/modulation above (shared), attention/FFN below (per-expert).

Full details in `docs/FORK_AND_SPECIALIZE.md`.

---

## Resolved Decisions

| ID | Decision | Rationale |
|----|----------|-----------|
| D1 | Build in `C:\Users\minta\Projects\dimljus\` | Clean separation from bootstrap repo |
| D2 | YAML for configs | Readable, standard for ML |
| D3 | pyproject.toml from start | Modern Python standard |
| D4 | Experimental params as structural placeholders | Config shape > exact numbers |
| D5 | Pydantic for validation | Rich error messages |
| D6 | Flat package layout | Matches ML ecosystem |
| D7 | Config flows from USE CASE | User intent drives model/data choice |
| D8 | `dimljus init` wizard built LAST | First thing users see, last built |
| D9 | MoE: unified -> fork -> specialize | Mirrors base model training |
| D10 | Separate data and training configs | Independent concerns |
| D11 | Three config tiers (new/standard/internal) | Progressive complexity |
| D12 | Training config AFTER data tools | Design from experience, not speculation |
| D13 | Dimljus hierarchical structure (targets/signals) | Structure teaches data meaning |
| D14 | JSON manifest (folders+extensions, not per-file) | Compact, human-readable |
| D15 | filetype + rich as Phase 4 deps | Magic bytes + curator-friendly output |
| D16 | Manual dHash, opt-in duplicates | Zero deps, user controls |
| D17 | All three quality metrics (blur/exposure/motion) | Infrastructure ready for tuning |
| D18 | Opt-in bucketing preview, 16px steps | Know distribution before GPU time |

---

## Open Questions

| ID | Question | Status |
|----|----------|--------|
| Q3 | Model registry: full interface vs incremental | Open, leaning incremental |
| Q4 | Config merging: deep merge vs overlay | Open, needs research |
| Q6 | Standalone tools: where they live | Open, leaning entry points |
| Q8 | Logging strategy | Open, lower priority |
| Q9 | Fork-and-specialize: Theory A vs B | Needs training infra (Phase 6-7) |
| Q10 | Rank selection for fork phase | Needs experimental testing |
| Q11 | Unified phase duration | Needs experimental testing |
| Q12 | Does I2V need fork-and-specialize? | Needs experimental testing |
| Q13 | Block targeting during fork phase | Open |
| Q14 | Smarter unmatched clip handling | v1 ships flat, refine later |
| Q15 | Phase 4 organization notes from testing | Notes for implementation |
| Q16 | Batch captioning across organized folders | Post-release enhancement |

---

## Documentation Map

All in `dimljus-kit/docs/`:

| Document | What's In It |
|----------|-------------|
| `TECHNICAL_ARCHITECTURE.md` | How video training works, four hard problems, Dimljus architecture |
| `CONTROL_SIGNAL_ARCHITECTURE.md` | Image taxonomy, signal registry, injection methods, MoE interaction |
| `TRAINING_METHODOLOGY.md` | Curator-first approach, differential MoE, caption strategy |
| `EXISTING_TRAINERS.md` | Analysis of musubi-tuner, ostris, finetrainers, LTX trainer |
| `MODEL_LANDSCAPE.md` | Model coverage matrix, LTX-2, SkyReels analysis |
| `FORK_AND_SPECIALIZE.md` | Fork-and-specialize strategy with divergence data |
| `FOUR_PATH_TRAINING_STRATEGY.md` | Per-expert training paths (T2V high/low, I2V high/low) with hypotheses |
| `PHASE5_REVIEW_NOTES.md` | Phase 5 review change log |
| `WAN_TRAINING_DATA.md` | What Wan 2.2 was trained on |
| `WALKTHROUGH_SCRATCH.md` | Three scenario walkthrough with real commands |
| `handover_timothy.md` | LTX-specific handover (data architecture is settled) |

Memory files (persist across conversations):
- `MEMORY.md` — project status, key context, implementation details
- `decisions.md` — all resolved architectural decisions (D1-D18)
- `open_questions.md` — running question log (Q3-Q16)

---

## Phase Plan (Ground Up)

Each phase produces something independently useful. Data preparation tools work with any trainer.

| Phase | What | Standalone? | Status |
|-------|------|-------------|--------|
| 0 | Data config schema | -- | **COMPLETE** (76 tests) |
| 1 | Video ingestion & scene detection | Yes | **COMPLETE** (114 tests) |
| 2 | Caption generation pipeline | Yes | **COMPLETE** (143 tests) |
| 3 | Image extraction & processing | Yes | **COMPLETE** (88 tests) |
| 4 | Dataset validation & organization | Yes | **COMPLETE** (205 tests) |
| -- | Triage module + batch ingest | Yes | **COMPLETE** (114 tests) |
| 4G | Documentation (guides, references) | -- | Deferred |
| 5 | Training config schema | -- | **IN REVIEW** (204 tests) |
| 6 | Latent pre-encoding | No (needs model) | Not started |
| 7 | Training infrastructure (model-agnostic) | -- | Not started |
| 8 | Wan model implementations | -- | Not started |
| 9 | Differential MoE training strategy | -- | Not started |
| 10 | Control signal expansion (depth/edge/pose/VACE) | -- | Not started |
| 11+ | Extensions (audio, LTX-2, SkyReels) | -- | Future |

---

## Environment

- **OS:** Windows 11 (Lenovo PC, NVIDIA GPUs)
- **Python:** 3.12.10 in venv at `C:\Users\minta\Projects\dimljus\.venv`
- **Packages:** pydantic 2.12.5, pyyaml 6.0.3, scenedetect 0.6.7.1, opencv-python 4.13, requests 2.32.5
- **ffmpeg:** 8.0.1 via WinGet
- **Production trainer:** musubi-tuner (personal fork, active client work)
- **Dimljus code:** `C:\Users\minta\Projects\dimljus-kit\` (main repo)
- **Test data:** `C:\Users\minta\Work\Marey Lora Proj\jinx\` (Jinx clips + Breakfast at Tiffany's)

---

## Key Principles (Non-Negotiable)

1. **Video is the target. Everything else is a control signal.**
2. **Ground up.** Every phase is a true foundation. No skipping.
3. **Standalone tools.** Data prep works with any trainer, not just Dimljus.
4. **Validate before compute.** Catch problems before burning GPU time.
5. **Known-good baselines.** Every component compared against musubi-tuner.
6. **Document the WHY.** Comments explain reasoning.
7. **Windows-first.** No Linux assumptions.
8. **Model-agnostic infrastructure.** New model = new backend, not rewrite.
9. **Teach the user.** Structure teaches how to think about inputs and signals.
10. **Discussion-first.** Architect thoroughly before building.
