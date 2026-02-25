# Handover for Timothy

**Date:** 2026-02-24
**From:** Minta's development sessions (Claude / Lykta)
**Repo:** https://github.com/alvdansen/dimljus (private)

---

## TL;DR

1. **Dimljus is built through Phase 8.** 1857 tests passing. The full pipeline — data preparation, captioning, triage, dataset organization, latent encoding infrastructure, model-agnostic training loop, and Wan 2.2 model implementations — is implemented and tested.

2. **RunPod training works today.** `runpod/train.py` runs Wan 2.2 T2V and I2V LoRA training on RunPod using musubi-tuner as the backend. One script, both variants, both experts. Setup takes one command.

3. **The data architecture is settled.** Targets vs signals, the Dimljus hierarchical structure, stem-based pairing, the manifest format — all decided, implemented, and tested against real datasets. These don't change.

4. **Phase order is non-negotiable.** Everything is built ground-up. Each phase is a foundation for what follows. The infrastructure is model-agnostic — adding a new model means implementing a backend, not rewriting the trainer.

---

## Current State (2026-02-24)

### What's Built

| Phase | What | Tests | Status |
|-------|------|-------|--------|
| 0 | Data Config Schema (Pydantic v2, YAML) | 76 | COMPLETE |
| 1 | Video Ingestion & Scene Detection | 114 | COMPLETE |
| 2 | Caption Generation Pipeline (Gemini, Replicate, local) | 143 | COMPLETE |
| 3 | Image Extraction & Processing | 88 | COMPLETE |
| 4 | Dataset Validation & Organization | 293 | COMPLETE |
| — | Triage Module (CLIP-based content sorting) | 103 | COMPLETE |
| 5 | Training Config Schema | 264 | COMPLETE |
| 6 | Latent Pre-Encoding (GPU-free infra) | 221 | COMPLETE (GPU stubs remain) |
| 7 | Training Infrastructure + Differential MoE | 235 | COMPLETE |
| 8 | Wan Model Implementations | 256 | COMPLETE |
| — | RunPod Training Scripts | — | READY |
| **Total** | | **1857** | |

### What's NOT Done Yet

- **Phase 6 GPU code** — Real VAE and T5 encoding (stubs exist in `vae_encoder.py`, `text_encoder.py`). The encoding infrastructure is complete, just needs the actual GPU encode calls filled in.
- **Phase 9** — Differential MoE training strategy as a first-class feature (the infrastructure supports it, but hasn't been validated end-to-end on GPU).
- **Phase 10** — Control signal expansion (depth/edge/pose + control LoRA training).
- **Phase 11+** — LTX, SkyReels, audio signals, VACE support.

### Module Map

```
dimljus/
  config/          Data config + training config schemas, YAML loading
  video/           Video pipeline (probe, validate, scene detect, split, extract)
  caption/         Caption pipeline (Gemini, Replicate, OpenAI-compat, scoring)
  dataset/         Dataset validation (discover, quality, validate, manifest, bucketing)
  triage/          CLIP-based content sorting (embeddings, concepts, filters)
  encoding/        Latent pre-encoding pipeline (expand, bucket, cache, dataset)
  training/        Training loop, noise scheduling, LoRA state, checkpoints, metrics
  training/wan/    Wan-specific: constants, registry, PEFT bridge, checkpoint I/O, backend

runpod/
  setup.sh         One-command pod setup (system packages, musubi-tuner, models)
  train.py         Unified T2V + I2V training script
  dataset-config.toml   Template dataset config for musubi-tuner
  README.md        Quick-start guide

examples/
  full_train.yaml  Complete dimljus training config (user-facing)

docs/              Architecture docs, research, methodology
scripts/           Analysis scripts (task vectors, expert comparison)
tests/             1857 tests (all GPU-free, all passing)
```

---

## Getting Started on RunPod

### Setup

```bash
# Clone the repo
cd /workspace
git clone https://github.com/alvdansen/dimljus.git

# Run setup — installs everything, downloads ~35GB of models
bash /workspace/dimljus/runpod/setup.sh
```

**Pod requirements:**
- GPU: H100 80GB or A100 80GB
- Template: RunPod PyTorch 2.x
- Container Disk: **50 GB** (default 20GB runs out)
- Volume Disk: 200 GB
- Environment Variables: `HF_TOKEN` = your HuggingFace token

### Upload Dataset

Via Jupyter Lab file browser:
```
/workspace/datasets/my_dataset/
    Videos/           ← training clips (.mp4)
    Videos/*.txt      ← caption files (same stem as video)
    Images/           ← reference images for I2V (optional)
```

Then edit `/workspace/dimljus/runpod/dataset-config.toml` to point to your dataset.

### Train

```bash
cd /workspace/musubi-tuner
tmux new -s train

# T2V — train both experts sequentially
python /workspace/dimljus/runpod/train.py --variant t2v --noise_level both

# I2V — train one expert
python /workspace/dimljus/runpod/train.py --variant i2v --noise_level high

# Custom hyperparameters
python /workspace/dimljus/runpod/train.py --variant t2v --noise_level high \
    --lr 5e-5 --rank 32 --epochs 30

# Resume from checkpoint
python /workspace/dimljus/runpod/train.py --variant t2v --noise_level high \
    --resume_from /workspace/outputs/my-lora-e25.safetensors

# Train on top of a speed LoRA (Lightning)
python /workspace/dimljus/runpod/train.py --variant t2v --noise_level high \
    --merge lightning
```

### Default Hyperparameters

| Setting | High-Noise Expert | Low-Noise Expert |
|---------|-------------------|------------------|
| Learning Rate | 1e-4 | 8e-5 |
| LoRA Rank | 16 | 16 |
| LoRA Alpha | 16 | 16 |
| Max Epochs | 30 | 50 |
| Save Every | 5 epochs | 5 epochs |

Shared: adamw8bit, cosine_with_min_lr scheduler, 0.01 weight decay, seed 42.
T2V: flow shift 3.0, boundary 875. I2V: flow shift 5.0, boundary 900.

### After Pod Restart

```bash
bash /workspace/dimljus/runpod/setup.sh
```

Models stay cached on `/workspace`. Only Python packages need reinstalling.

---

## How the RunPod Script Works

The training script (`runpod/train.py`) is a bridge. It uses **musubi-tuner** as the GPU backend while dimljus's native training pipeline matures. The 3-step pipeline:

1. **Cache latents** — VAE encodes all videos/images to latent space (musubi's `wan_cache_latents.py`)
2. **Cache text** — T5 encodes all captions (musubi's `wan_cache_text_encoder_outputs.py`)
3. **Train LoRA** — Gradient descent on cached latents (musubi's `wan_train_network.py` via accelerate)

When dimljus's native GPU code is ready (Phase 6 Steps 10-11 + Phase 8 GPU validation), we swap the musubi subprocess calls for `python -m dimljus.training train --config train.yaml`.

---

## Architecture Decisions — DO NOT CHANGE

These were made after deep research, multi-session discussion, and real-data validation.

### Target/Signal Separation (D13)

```
my_dataset/
  training/
    targets/              ← what the model learns to PRODUCE
    signals/
      captions/           ← text control signal
      first_frames/       ← exact starting frame for I2V (VAE pathway)
      references/         ← informational ref images (IP-adapter pathway)
      depth/              ← structural control signal
```

**Targets** = what the model produces. **Signals** = what it responds to. First frames and references enter the model through different pathways. This distinction is critical.

### Separate Data and Training Configs (D10)

- Data config lives WITH the dataset. Works standalone with any trainer.
- Training config lives WITH the training run, points at a data config.
- Same dataset, different training runs. Same training approach, different datasets.

### Model-Agnostic Training Infrastructure (Phase 7)

The training loop, noise scheduling interface, LoRA state management, and checkpoint system don't know about Wan specifically. They define **protocols** that a model backend must implement:

- `NoiseSchedule` — how to sample and apply noise
- `ModelBackend` — how to load the model, get LoRA targets, run forward pass
- `InferencePipeline` — how to generate samples during training

Adding a new model means implementing these three protocols. The training loop, optimizer, scheduler, checkpoint manager, metrics tracking — all of that is shared.

### Differential MoE (Phase 7 + 9)

Wan 2.2 has two noise-level experts. Key finding: they need different training strategies. High-noise expert converges fast on coarse features (composition, motion). Low-noise expert overfits rapidly on fine details.

The training infrastructure supports:
- **Fork-and-specialize**: unified warmup → fork into per-expert LoRAs (recommended)
- **Unified only**: single LoRA, both experts see all timesteps
- **Expert from scratch**: skip unified, train experts independently

---

## Key Technical Context

### Wan 2.2

- 14B active parameters per step, ~27B total (dual MoE)
- Two experts: high-noise (early denoising) and low-noise (late denoising)
- Expert switching at boundary ratio (T2V: 0.875, I2V: 0.900)
- Flow matching + Diffusion Transformer
- T5 text encoder (cross-attention), Wan-VAE (3D causal, 4x temporal compression)
- I2V: reference image VAE-encoded, concatenated in latent channel dimension

### Key Finding: Low-Noise Expert IS Wan 2.1

The Wan 2.2 low-noise expert is byte-level identical to Wan 2.1. Alibaba only retrained the high-noise expert. This means "unified warmup" = training a Wan 2.1-compatible LoRA, then forking for high-noise adaptation.

### LoRA Checkpoint Format

Dimljus uses ai-toolkit/diffusers format: `blocks.0.attn1.to_q.lora_A.weight`. Conversion to/from musubi format (`lora_unet_blocks_0_attn1_to_q.lora_down.weight`) is handled by `training/wan/checkpoint_io.py`.

---

## Running Tests Locally

```bash
# All tests (no GPU needed)
python -m pytest tests/ -q       # should show "1857 passed"

# Specific areas
python -m pytest tests/test_wan_*.py -v        # Phase 8 Wan implementations
python -m pytest tests/test_training_*.py -v   # Phase 7 training infra
python -m pytest tests/test_dataset_*.py -v    # Phase 4 dataset tools
python -m pytest tests/test_triage_*.py -v     # Triage module
```

---

## Working Style

- **Discussion-first.** Minta drives direction. Don't rush to implementation.
- **Explain everything.** Minta thinks in systems and aesthetics, not syntax.
- **Respect the baseline.** Compare against musubi-tuner known-good results.
- **Don't change settled architecture.** If something seems wrong, raise it as a question.
- **Phase order matters.** Each phase builds on the previous.
