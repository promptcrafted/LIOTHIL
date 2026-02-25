# Handover for Timothy — LTX Model Support

**Date:** 2026-02-23
**From:** Minta's development sessions (Claude / Lykta)

Read this before anything else. It tells you what's been built, what's decided, what's sacred, and where LTX work fits.

---

## TL;DR — What You Need to Know

1. **Dimljus data tools are working and tested end-to-end.** 740 tests passing. The pipeline (ingest → triage → caption → extract → validate) runs on real video datasets. Do NOT change the data handling architecture.

2. **LTX is Phase 11+ territory.** It's a new model backend, not a data pipeline change. The infrastructure is designed to be model-agnostic — adding LTX means implementing a model backend (noise schedule, layer mapping, forward pass, checkpoint format), not rewriting anything.

3. **The data architecture (D13, D14) is settled.** Targets vs signals, the Dimljus hierarchical structure, stem-based pairing, the manifest format — all decided, implemented, and tested. These decisions were made after extensive research and real-data validation. They don't change for LTX.

4. **Phase order is non-negotiable (D12).** Data tools (0-4) → training config (5) → latent encoding (6) → training infra (7) → model backends (8) → MoE strategy (9) → control signals (10) → extensions (11+). LTX lives in Phase 11+. Do not skip phases.

---

## Current State (2026-02-23)

### What's Built and Working

| Phase | What | Tests | Status |
|-------|------|-------|--------|
| 0 | Data Config Schema (Pydantic v2, YAML) | 76 | COMPLETE |
| 1 | Video Ingestion & Scene Detection | 114 | COMPLETE |
| 2 | Caption Generation Pipeline (Gemini, Replicate, local) | 143 | COMPLETE |
| 3 | Image Extraction & Processing | 88 | COMPLETE |
| 4 | Dataset Validation & Organization | 205 | COMPLETE |
| — | Triage Module (CLIP-based content sorting) | 45 | COMPLETE |
| — | Batch ingest, max-frames subdivision, CLI improvements | 69 | COMPLETE |
| **Total** | | **740** | |

### End-to-End Validation

The full pipeline has been tested on real data (25 Breakfast at Tiffany's YouTube videos):
- **Ingest**: 25 videos → 1720 clips via scene detection + split + normalize + subdivision
- **Triage**: CLIP-based matching against reference images → organized into subject folders
- **Caption**: Replicate API → short factual captions confirmed production-quality
- **Character triage**: Validated and production-ready
- **Object/setting triage**: Experimental (matches visual similarity, not specific identity)

### Key Modules

```
dimljus/
  config/         ← Data config schema (Pydantic v2, YAML loading)
  video/          ← Video pipeline (probe, validate, scene, split, extract, quality)
  caption/        ← Caption pipeline (Gemini, Replicate, OpenAI-compat, scoring)
  dataset/        ← Dataset validation (discover, quality, validate, manifest, bucketing, reports)
  triage/         ← CLIP-based content sorting (embeddings, subjects, filters, organize)
```

---

## Architecture Decisions — DO NOT CHANGE

These were made after deep research, multi-session discussion with Minta, and real-data validation. They encode Minta's methodology and are foundational to every downstream phase.

### D13: Dataset Organization — The Dimljus Structure

```
my_dataset/
  dimljus_data.yaml           ← data config (lives with the dataset)
  dimljus_manifest.json       ← machine-readable dataset description
  training/
    targets/                  ← what the model learns to PRODUCE
    signals/
      captions/               ← text captions (.txt, same-stem pairing)
      first_frames/           ← exact starting frames for I2V (VAE pathway)
      references/             ← informational ref images (IP-adapter pathway)
      depth/                  ← structural control signal
      edge/                   ← structural control signal
      pose/                   ← motion control signal
  regularization/
    targets/
    signals/
```

**Why this matters:**
- **Targets** = what the model produces. **Signals** = what it responds to.
- **First frames ≠ references** — they enter the model through different pathways (VAE channel concat vs IP-adapter). This distinction is critical for LTX too if it has similar conditioning inputs.
- The structure teaches users what their data MEANS. This is the curator-first philosophy.
- Flat export for other trainers is a convenience layer on top, not a replacement.

### D14: Manifest Format

JSON at dataset root. Describes folders + extensions, not individual files. Same-stem convention does per-file mapping. This is compact, human-readable, and works for datasets of any size.

### D10: Separate Data and Training Configs

- Data config lives WITH the dataset. Works standalone with any trainer.
- Training config lives WITH the training run, points at a data config.
- These are independent concerns. Same dataset, different training runs. Same training approach, different datasets.

**For LTX:** The data config doesn't change. LTX training would need its own training config (Phase 5+), but it reads from the same data config and manifest format that Wan uses.

### The Control Signal Architecture

Every input to the model is a first-class control signal:
- Text caption (T5/CLIP text encoding)
- Reference image (VAE encoding → channel concat, or CLIP encoding → cross-attention)
- Depth/edge/pose maps (various encoding pathways)
- Audio (future)

Each signal goes through: **prepare → validate → encode → inject**. The preparation and validation (Phases 1-4) are model-agnostic. The encoding and injection (Phases 6-8) are model-specific.

**For LTX:** LTX has its own text encoder, its own VAE, its own conditioning pathways. The DATA is prepared the same way — clean clips, quality-checked, with captions and references paired by stem. How that data gets encoded for LTX is a Phase 11+ concern.

---

## Where LTX Fits

### What LTX Work IS

LTX support means implementing a **model backend** in the Phase 7-8 infrastructure:

1. **Model registration** — tell Dimljus what LTX's architecture looks like (layer names, forward pass signature, noise schedule type)
2. **Noise scheduling** — LTX's specific noise schedule implementation
3. **LoRA injection** — which layers to target, how to map LoRA adapters to LTX's transformer blocks
4. **Latent encoding** — LTX's VAE and text encoder (different from Wan's)
5. **Checkpoint export** — save LoRA weights in a format LTX inference can load

### What LTX Work IS NOT

- Changing the data config schema
- Changing the manifest format
- Changing how captions are generated or stored
- Changing how reference images are extracted or organized
- Changing the targets/signals directory structure
- Changing the triage pipeline
- Adding LTX-specific data preparation steps to the existing pipeline

If LTX needs data in a different format than what Dimljus produces, the right approach is an **export/conversion layer** — not changing the canonical format. This is the same pattern as "flat export for musubi-tuner": the Dimljus structure is the source of truth, other formats are derived.

### Prerequisites for LTX

LTX work can't meaningfully start until:
- Phase 5 (training config) defines the model-agnostic config surface
- Phase 7 (training infrastructure) defines the model backend interface
- Phase 8 (Wan implementations) validates the interface actually works

Researching LTX architecture now is fine and valuable. Implementing LTX training code before the interface exists means you'll rewrite it when the interface is defined.

---

## Pipeline Architecture — How It Works

Understanding the pipeline helps you see where model-specific code lives vs model-agnostic code:

```
Raw footage
    │
    ▼
Phase 1: Ingest (scene detect → split → normalize → subdivide)     ← MODEL-AGNOSTIC
    │
    ▼
Triage: CLIP match against subject references → organize into folders ← MODEL-AGNOSTIC
    │
    ▼
Phase 2: Caption (Gemini/Replicate/local VLM → .txt sidecars)      ← MODEL-AGNOSTIC
    │
    ▼
Phase 3: Extract references (first frame, best frame → .png)        ← MODEL-AGNOSTIC
    │
    ▼
Phase 4: Validate & manifest (completeness, quality, bucketing)     ← MODEL-AGNOSTIC
    │
    ▼
Phase 6: Latent pre-encoding (VAE + text encoder → cached tensors)  ← MODEL-SPECIFIC
    │
    ▼
Phase 7-8: Training (noise schedule, LoRA, forward pass, loss)      ← MODEL-SPECIFIC
```

Everything above the line is done. Everything below the line is where LTX differences live.

### Key: Triage Decides What, Captioner Decides How

The triage step (CLIP matching) determines WHAT a clip contains — which character, which setting, whether it's a text overlay. The caption step determines HOW to describe it. These are separate concerns:

- Triage mismatches cascade into captioning (wrong anchor word applied to wrong clips)
- Captioner cannot second-guess triage because anchor words are user-defined (could be anything)
- Fix triage accuracy upstream, don't compensate in captioning downstream

This separation matters for any model, including LTX.

---

## Technical Details for LTX Research

### What to Research Now

While waiting for the training infrastructure (Phases 5-8):

1. **LTX architecture** — How many transformers? MoE or single? What's the noise schedule?
2. **Conditioning pathways** — How does text enter? How do images enter? Any unique control signals?
3. **LoRA compatibility** — What LoRA format does the LTX ecosystem expect? Can it load safetensors LoRA adapters?
4. **Existing training tools** — Does LTX have its own fine-tuning scripts? What format do they expect data in?
5. **Key differences from Wan** — Where does LTX's architecture diverge? These differences determine what the LTX model backend needs to implement.

### Documents to Read

- `CLAUDE.md` — Full project specification (source of truth for everything)
- `docs/TECHNICAL_ARCHITECTURE.md` — How video training works, Dimljus architecture
- `docs/CONTROL_SIGNAL_ARCHITECTURE.md` — Signal taxonomy, encoding pathways
- `docs/MODEL_LANDSCAPE.md` — Model comparison including LTX-2 analysis
- `docs/EXISTING_TRAINERS.md` — How other trainers handle multi-model support
- `docs/FORK_AND_SPECIALIZE.md` — MoE strategy (relevant if LTX is MoE)
- `memory/decisions.md` — All resolved architectural decisions (D1-D18)
- `memory/open_questions.md` — Open questions (Q3-Q16)

### Codebase Orientation

```bash
# Run all tests
python -m pytest tests/ -v

# Quick health check
python -m pytest tests/ -q  # should show "740 passed"

# Run specific phase tests
python -m pytest tests/test_dataset_*.py -v   # Phase 4
python -m pytest tests/test_triage_*.py -v    # Triage

# CLI tools (all work standalone)
python -m dimljus.video scan <path>            # Probe video metadata
python -m dimljus.video ingest <path> -o <dir> # Scene detect + split
python -m dimljus.video caption <dir> -p replicate -u character -a "Name"
python -m dimljus.video extract <dir> -o <dir>
python -m dimljus.dataset validate <path>
```

---

## Working Style Reminders

- **Discussion-first.** Minta drives direction. Don't rush toward implementation.
- **Explain everything.** Minta thinks in systems and aesthetics, not syntax.
- **Step by step.** Verify each step before moving on.
- **Respect the baseline.** Compare against musubi-tuner known-good results.
- **Don't change settled architecture.** If something seems wrong with D13/D14/D10, raise it as a question — don't just change it.
- **Phase order matters.** The order exists because each phase builds on the previous. Skipping creates technical debt that compounds.

---

## Key Contacts

- **Minta** — Domain expert, architect, final decision-maker on all design questions
- **Claude / Lykta** — Implementation partner for Phases 0-4+ (this handover document)
