# Technical Architecture — How Video LoRA Training Works

> Last updated: 2026-02-22 (v3 — Phase 0+1 complete)
> Status: Phase 0 (Data Config) and Phase 1 (Video Ingestion) implemented and verified

This document explains the mechanics of video LoRA training from first principles, how existing trainers (kohya, ostris, musubi-tuner) implement it, and where Dimljus's architecture diverges.

## How Existing Trainers Are Structured

Every training framework follows the same three-layer pattern:

### Layer 1: Config Surface (what the user touches)
A YAML/TOML file with a curated subset of parameters. The author makes editorial choices about what to surface (learning rate, rank, dataset path) vs what gets sensible defaults (optimizer type, scheduler, precision) vs what's completely hidden (internal tensor operations, memory management).

### Layer 2: Orchestrator (the manager)
A Python script that reads config, validates it, builds all PyTorch objects, and runs the training loop. Translates "what you want" into actual operations.

### Layer 3: Utilities (the workers)
Isolated modules for specific jobs — dataset loading, noise scheduling, LoRA injection, checkpoint saving. The orchestrator calls them; they don't know about each other.

### How They Differ
- **Kohya (sd-scripts)**: TOML configs + argparse. `train_util.py` defines hundreds of arguments with defaults. The TOML only overrides what you change. Massive codebase supporting every model and training type.
- **Ostris (ai-toolkit)**: YAML configs with factory pattern. `type` fields map to Python classes. More modular, object-oriented.
- **musubi-tuner**: Fork of kohya with video-specific additions. Minta's current production tool.

## The Training Loop (Plain English)

```python
for epoch in range(num_epochs):
    for batch of video clips from dataset:
        1. Encode video through VAE → latent representation
        2. Sample random noise level (timestep)
        3. Add noise to latents at that level
        4. Prepare conditioning (text embeddings, reference image if I2V)
        5. Ask model to predict the noise (or velocity, for flow matching)
        6. Compare prediction to actual noise → calculate loss
        7. Backpropagate through LoRA weights only (base model frozen)
        8. Update LoRA weights based on gradients
        9. Log loss for monitoring

    if it's time to save:
        save checkpoint
```

## Four Hard Problems in Video Training

### 1. Noise Scheduling
Every diffusion model has a specific noise schedule that defines how much noise is added at each timestep. Getting this wrong means the model learns to denoise at the wrong levels.

**Wan 2.2 uses flow matching**, not standard DDPM noise scheduling. Flow matching defines a straight-line path from data to noise (or vice versa), parameterized by a continuous time variable t ∈ [0, 1]. The model predicts the velocity field that transports samples along this path.

For Wan 2.2's MoE architecture, this has a critical implication: the high-noise expert handles timesteps where the signal-to-noise ratio is low (early denoising, t near 1), and the low-noise expert handles timesteps where SNR is high (late denoising, t near 0). The boundary is at approximately SNR ≈ 875/1000 of the schedule.

**What trainers do**: They look up the model's scheduler configuration (usually stored in the model's config files) and use matching noise sampling during training. Kohya and musubi-tuner read this from the model's `scheduler_config.json` or hardcode it per model type.

### 2. LoRA Injection
LoRA works by adding small trainable matrices to specific layers of the frozen base model. The choice of which layers to target matters.

**Standard targets**: The query (Q), key (K), and value (V) projection matrices in attention layers, and sometimes the MLP layers. In Wan's DiT architecture, these are within the transformer blocks.

**For Wan 2.2 MoE**: Each expert is a separate transformer. A LoRA can target one expert, the other, or both with different configurations. This is the basis of differential MoE training — different rank/lr/epochs per expert.

**What trainers do**: They define a mapping of "which layers in this specific model architecture should get LoRA adapters." This mapping is model-specific and usually hardcoded per model type in the trainer.

**Weight format**: LoRA weights must be saved in a format compatible with inference tools. For Wan, this means safetensors files with keys that match what ComfyUI/diffusers expect. Getting the key naming wrong means the LoRA loads but does nothing (or crashes).

### 3. Memory Management
Video models are enormous. Wan 2.2 I2V is 14B active parameters. Training requires:
- The frozen base model weights in VRAM
- The LoRA adapter weights (trainable)
- Optimizer states (2x the LoRA weights for AdamW)
- Activation cache for backpropagation
- The video latents themselves

**Key techniques**:
- **Gradient checkpointing**: Trade compute for memory — don't store all activations, recompute them during backward pass. Saves ~60% VRAM at ~20% speed cost.
- **Mixed precision (bf16)**: Keep weights in bfloat16 instead of float32. Halves memory for model weights.
- **Model offloading**: Move the text encoder to CPU after encoding captions. Move the VAE to CPU after encoding video latents.
- **Gradient accumulation**: Process one video at a time but accumulate gradients over N steps before updating. Simulates larger batch sizes without the memory cost.

### 4. Video-Specific Considerations
**Temporal compression**: Wan-VAE is a 3D causal VAE with ~4x temporal compression. 81 frames → ~21 temporal tokens. This means clip preparation directly affects what the model sees — frame rate, clip length, and temporal alignment all matter.

**Frame count constraints**: Wan requires frame counts satisfying `(frames - 1) % 4 == 0`. Valid counts: 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81.

**Resolution buckets**: Like image training, video training benefits from bucketing — grouping clips by resolution to minimize padding waste. But video adds the temporal dimension, so buckets are 3D (width × height × frames).

## Where Existing Trainers Fall Short for Video

### Problem 1: Image-first architecture
Kohya and ostris were designed for image LoRA training. Video was added as an extension — the dataset loader learned to read video files, but the pipeline still fundamentally processes each clip as "a stack of images" rather than a temporally coherent sequence. There's no temporal validation, no cut detection, no frame rate awareness baked into the pipeline.

### Problem 2: No control signal routing
In existing trainers, the reference image (for I2V) and the text caption are handled implicitly by whatever the model expects. There's no abstraction layer that says "this is a control signal with its own preparation and validation." If you want to add depth conditioning, you'd need to modify the training loop itself.

### Problem 3: No differential training
For Wan 2.2's MoE, existing trainers either train both experts identically (same rank, same lr, same epochs) or let you pick one expert to train. There's no native concept of "train the high-noise expert for 30 epochs at rank 16 then the low-noise expert for 50 epochs at rank 24."

### Problem 4: Caption-centric conditioning
Training UIs and configs treat captions as the primary conditioning mechanism. Caption quality is emphasized above all else. But in production video training, the reference image carries far more information than the caption, and other control signals (depth, pose) can be even more important for specific use cases. Captions are ONE signal, not THE signal.

## Dimljus's Architectural Divergence

### Control Signal Registry
Every input to training is registered as a named control signal with:
- **Preparation function**: How to extract/load this signal from raw data
- **Validation function**: What makes this signal valid (resolution, duration, format)
- **Encoding function**: How to transform it into the representation the model expects
- **Injection method**: How it enters the model (cross-attention, channel concatenation, AdaLN modulation)
- **Config surface**: What parameters the user can set (weight, dropout rate, etc.)

Adding a new control type = registering a new signal. No training loop changes needed.

### Differential MoE as First-Class Feature
The config schema natively supports per-expert training parameters:
```yaml
model:
  type: wan22_t2v  # or wan22_i2v, wan21_t2v, etc.
  
training:
  experts:
    high_noise:
      rank: 16
      learning_rate: 1e-4
      epochs: 30
    low_noise:
      rank: 24
      learning_rate: 8e-5
      epochs: 50
```

For non-MoE models (Wan 2.1), the config collapses to a single set of parameters. The architecture handles both cases.

### Video-Native Dataset Pipeline
Before anything touches the training loop:
1. **Temporal validation**: Verify no scene cuts within clips, consistent frame rate, appropriate duration for the model's temporal window
2. **Resolution bucketing**: 3D buckets (W × H × frames) with intelligent grouping
3. **Control signal alignment**: Verify all control signals match the video (same duration, compatible resolution)
4. **Latent pre-encoding**: Optionally pre-encode videos through VAE to avoid re-encoding every epoch (standard in musubi-tuner, essential for video due to cost)

### Data Preparation as Standalone Tooling
The biggest gap in existing trainers isn't the training loop — it's everything BEFORE the training loop. Existing trainers assume you show up with clean clips, good captions, and properly formatted data. In reality, the data journey is:

1. Raw video (Blu-ray rip, downloaded footage, client assets)
2. Scene detection and cutting into clean clips
3. Caption generation
4. Reference image extraction (for I2V)
5. Control signal computation (depth, edge, pose — for control LoRAs)
6. Validation and organization
7. Latent pre-encoding
8. THEN training

Steps 1–6 are where 80% of the time and failure happens. Dimljus's data tools solve these steps as standalone utilities that produce standard output formats, usable with ANY trainer. This makes them valuable even to users who never use Dimljus's training loop.

### Model-Agnostic Training Infrastructure
The training loop itself should not know about Wan, LTX, or any specific model. Instead, Dimljus defines interfaces that every model backend must implement:

- **Noise schedule**: How noise is added/removed. Wan uses flow matching. LTX-2 uses flow matching with different parameters. SkyReels uses Diffusion Forcing (non-decreasing noise). The interface is the same; the implementation differs.
- **Layer mapping**: Which transformer layers get LoRA adapters. Every model has different layer names, different numbers of layers, and different internal structure. The LoRA injection code asks the model backend "which layers should I target?"
- **Forward pass**: How the model processes inputs. T2V models just take noisy latents + text embeddings. I2V models also take a reference image latent. VACE models also take context tokens. The model backend defines its own forward pass.
- **Checkpoint format**: How to save weights so inference tools can load them. ComfyUI expects specific key names; diffusers expects different ones. The model backend knows its own format.

Adding a new model = implementing these four interfaces. The training loop, optimizer, logging, and everything else stays the same.

## File Structure (Current)

What actually exists today after Phase 0 + Phase 1:

```
dimljus-kit/
├── CLAUDE.md                      ← Primary instruction file for Claude Code
├── pyproject.toml                 ← Package config (hatchling, optional deps)
├── examples/                      ← Example YAML data configs (minimal, standard, full)
│
├── dimljus/
│   ├── __init__.py                ← Package root (version 0.1.0)
│   │
│   ├── config/                    ← [Phase 0] Data config schema
│   │   ├── __init__.py
│   │   ├── data_schema.py         ← Pydantic v2 models (DimljusDataConfig + 11 sub-models)
│   │   ├── loader.py              ← YAML loading, path resolution, error formatting
│   │   └── defaults.py            ← Constants (WAN_TRAINING_FPS, VALID_FRAME_COUNTS, etc.)
│   │
│   ├── video/                     ← [Phase 1] Video ingestion & processing
│   │   ├── __init__.py            ← Public API + ffmpeg auto-discovery
│   │   ├── _ffmpeg.py             ← Auto-discover ffmpeg from WinGet/Chocolatey/Scoop
│   │   ├── errors.py              ← DimljusVideoError, FFmpegNotFoundError, etc.
│   │   ├── models.py              ← VideoMetadata, ClipValidation, ScanReport, etc.
│   │   ├── probe.py               ← ffprobe wrapper (metadata extraction)
│   │   ├── validate.py            ← Structural validation against VideoConfig
│   │   ├── scene.py               ← PySceneDetect wrapper (scene detection)
│   │   ├── split.py               ← ffmpeg wrapper (normalize, split at scenes)
│   │   └── __main__.py            ← CLI: python -m dimljus.video {scan,ingest,normalize,...}
│   │
│   └── caption/                   ← [Phase 1] VLM captioning pipeline
│       ├── __init__.py            ← Public API (caption_clips, audit_captions)
│       ├── models.py              ← CaptionConfig, CaptionResult, AuditResult
│       ├── prompts.py             ← Use-case prompt templates (character, style, motion, object)
│       ├── base.py                ← VLMBackend abstract base class
│       ├── gemini.py              ← Google Gemini direct API backend
│       ├── replicate.py           ← Replicate raw HTTP backend (schema auto-detection)
│       └── captioner.py           ← Batch captioner orchestrator with progress/retry
│
├── tests/
│   ├── conftest.py                ← Shared fixtures, @requires_ffmpeg markers
│   ├── fixtures/jinx_subset/      ← 5-clip Jinx test dataset
│   ├── test_data_config.py        ← 76 tests (Phase 0)
│   ├── test_video_models.py       ← 22 tests (pure Python)
│   ├── test_video_validate.py     ← 29 tests (pure Python)
│   ├── test_video_probe.py        ← 21 tests (11 unit + 10 integration)
│   ├── test_video_scene.py        ← 5 tests (1 unit + 4 integration)
│   ├── test_video_split.py        ← 7 integration tests
│   ├── test_caption_models.py     ← 16 tests (pure Python)
│   └── test_caption_prompts.py    ← 14 tests (pure Python)
│
└── docs/                          ← Methodology documentation
```

## File Structure (Planned)

Future phases will add:

```
dimljus/
│   ├── data/                      ← [Phase 4] Dataset validation & organization
│   │   ├── dataset.py             ← Video dataset loading
│   │   ├── buckets.py             ← 3D resolution/temporal bucketing
│   │   └── organize.py            ← Dataset structure management
│   │
│   ├── signals/                   ← [Phase 10] Control signal registry
│   │   ├── registry.py            ← Control signal type registry
│   │   ├── base.py                ← Abstract base for all signals
│   │   ├── text.py, image.py, depth.py, ...
│   │
│   ├── models/                    ← [Phase 8] Model backends
│   │   ├── registry.py            ← Model backend registry
│   │   ├── base.py                ← Abstract base: what every model backend must provide
│   │   ├── wan21_t2v.py, wan22_t2v.py, wan22_i2v.py
│   │
│   ├── network/                   ← [Phase 7] LoRA injection
│   │   ├── lora.py                ← Model-agnostic LoRA interface
│   │   └── moe.py                 ← MoE-aware expert targeting
│   │
│   ├── training/                  ← [Phase 7] Training infrastructure
│   │   ├── loop.py, noise.py, loss.py
│   │   └── strategies/
│   │       └── differential_moe.py  ← [Phase 9] Per-expert hyperparameters
│   │
│   └── checkpoint/                ← [Phase 7] Checkpoint export
│       ├── save.py
│       └── compat.py              ← ComfyUI/diffusers format compatibility
```

Key structural decisions:
- **Data tools are standalone.** `dimljus/video/` and `dimljus/caption/` work independently of the training pipeline. They output standard formats (video clips, `.txt` caption files) that work with musubi-tuner, ostris, or any trainer.
- **dimljus/models/ will be the model registry.** Each model backend implements a standard interface. The training loop won't know about Wan — it calls the model backend's methods.
- **dimljus/training/strategies/ is for training strategies.** Differential MoE is a strategy that wraps the base training loop, not a modification to it.

## Phase Plan (Ground Up)

Each phase produces something usable on its own. The data preparation phases (1–4) are valuable even without the training pipeline — they solve the #1 barrier to entry in video LoRA training.

### Phase 0: Data Config Schema -- COMPLETE
**Goal**: Define what a dataset looks like to Dimljus. YAML config with validation and defaults.
**Delivered**: `dimljus/config/` — Pydantic v2 models (DimljusDataConfig + 11 sub-models), YAML loader with path resolution, 76 tests. Minimum viable config = just a dataset path; everything else defaults to Wan training priors (16 FPS, 480p, 4n+1 frames).
**Verified**: Integration tests against real Jinx character LoRA dataset. All validation errors include what's wrong + how to fix it.

### Phase 1: Video Ingestion & Scene Detection -- COMPLETE
**Goal**: Take raw source material and produce clean training clips, with optional VLM captioning.
**Delivered**: `dimljus/video/` + `dimljus/caption/` — a comprehensive standalone toolkit:
- **Probe**: ffprobe metadata extraction (fps, resolution, frame count, SAR, codec)
- **Validate**: Structural validation against VideoConfig (resolution, fps, frame count 4n+1, SAR)
- **Scene detect**: PySceneDetect wrapper (find cuts, verify pre-cut clips have no cuts)
- **Split/Normalize**: ffmpeg wrapper (fps conversion, resolution scaling, SAR correction, frame count trimming)
- **Caption**: VLM backends for Gemini (direct API) and Replicate (raw HTTP with schema auto-detection)
- **Prompts**: Use-case-specific templates (character, style, motion, object)
- **Audit**: Compare existing captions against fresh VLM output
- **CLI**: `python -m dimljus.video {scan, ingest, normalize, caption, audit}`
- **114 tests** (190 total with Phase 0)
**Standalone**: Yes — output clips and `.txt` captions work with musubi-tuner, ostris, or any trainer.
**Verified**: Tested on real video (puppy clip at 1280x720@60fps). Replicate backend produces accurate captions. Scan tested on 26-clip Jinx dataset (1920x1080@23.976fps). Ingest pipeline produces correct normalized output with JSON manifest.

### Phase 2: Caption Generation Pipeline
**Goal**: Refine and harden the captioning infrastructure built in Phase 1.
**Delivers**: Improvements to `dimljus/caption/`:
- Local VLM model backends (no API dependency)
- Advanced prompt engineering and caption quality scoring
- Caption editing/merging workflows
- Production hardening (better error recovery, resume interrupted batches)
**Standalone**: Yes — output `.txt` files are the universal caption format for all trainers.
**Validation**: Generate captions for 20+ clips → manually review quality and accuracy.

### Phase 3: Image Extraction & Processing
**Goal**: Extract reference images from clips for I2V training or other image input needs.
**Delivers**: `tools/extract_frames.py` — a standalone CLI tool that:
- Extracts first frame (standard I2V reference)
- Extracts "best quality" frame (sharpest, best-exposed — using simple heuristics)
- Supports user-specified frame numbers
- Outputs images with standard naming conventions
**Standalone**: Yes — reference images work with any I2V training setup.
**Validation**: Extract frames from 20 clips → verify quality and naming.

### Phase 4: Dataset Validation & Organization
**Goal**: Bring all data together into Dimljus's target/controls/text/metadata structure. **Teach users how to think about their data.**
**Delivers**: `tools/validate.py` + `dimljus/data/organize.py`:
- Validates the full dataset (frame counts, caption existence, resolution, control signal alignment)
- Produces a detailed report: what's good, what's missing, what's broken
- Organizes files into the training structure
- Classifies image inputs by role: reference (I2V conditioning), subject (IP adapter/character), style (aesthetic), structural (depth/edge/pose)
- Generates metadata files documenting what each piece of data means and how it will be used
**Key insight**: This is where users learn the target/control distinction. The organizational structure itself is pedagogical — it makes explicit what existing trainers leave implicit.
**Standalone**: Partially — the validation tool works independently, but the organization tool produces Dimljus's specific structure.
**Validation**: Run on a complete dataset → report catches known problems → organized output passes manual inspection.

### Phase 5: Training Config Schema
**Goal**: Design the training config — model selection, hyperparameters, optimizer, LoRA settings, MoE expert settings, checkpoint output. This comes AFTER all data tools are built so the training config is designed from real experience with the data pipeline, not speculation. Points at a data config.
**Delivers**: Training config YAML schema with validation and sensible defaults per model type.

### Phase 6: Latent Pre-Encoding
**Goal**: Encode all training data into cached latent representations so training doesn't re-encode every epoch.
**Delivers**: `tools/pre_encode.py` — a CLI tool that:
- Encodes video clips through the VAE → saves latent tensors to disk
- Encodes text captions through T5 → saves embedding tensors to disk
- Encodes reference images through VAE → saves latent tensors to disk
- Reports encoding progress, VRAM usage, and disk space
- Validates that all samples encode successfully before training begins
**Model-specific**: Yes — this is the bridge between model-agnostic data prep and model-specific training. Wan uses Wan-VAE + T5; other models will use different encoders.
**Standalone**: No — requires the target model's encoder components.
**Validation**: Pre-encode a small dataset → verify tensors load correctly → verify tensor shapes match what musubi-tuner produces for the same clips.

### Phase 7: Training Infrastructure (Model-Agnostic)
**Goal**: Build the core training machinery that works for ANY model.
**Delivers**:
- **Noise scheduling interface**: Every diffusion model adds/removes noise differently (DDPM, flow matching, diffusion forcing). The interface defines what a noise schedule must provide; the model backend implements it.
- **LoRA injection interface**: Every model has transformer layers that get LoRA adapters, but layer names differ. The interface defines how to inject LoRA; the model backend provides the layer mapping.
- **Training loop**: The actual for-each-epoch, for-each-batch, compute-loss, update-weights cycle. Model-agnostic — calls model backend methods for the model-specific parts.
- **Checkpoint export interface**: Saving weights in formats inference tools understand. The interface defines export; the model backend provides format details.
- **Model registry**: Register a new model by providing: noise schedule, layer mapping, forward pass, checkpoint format. Adding a model doesn't require touching the training loop.
**Validation**: All interfaces have clear contracts. Registry loads and validates model backends.

### Phase 8: Wan Model Implementations
**Goal**: Implement Wan-specific backends that plug into Phase 7's infrastructure.
**Delivers**:
- **Wan 2.1 T2V backend**: Flow matching noise schedule, single-transformer LoRA injection, forward pass, ComfyUI-compatible checkpoint export. The simplest case — validates the entire pipeline works.
- **Wan 2.2 T2V backend**: Adds MoE-aware expert routing. Two experts with independent LoRA configurations. Proves MoE support works.
- **Wan 2.2 I2V backend**: Adds reference image as a control signal (VAE encoding → latent concatenation in channel dimension). Proves control signal routing works.
**Validation for each sub-step**: Train a LoRA on the same dataset with the same hyperparameters in both Dimljus and musubi-tuner. Compare: loss curves should roughly match, checkpoint weights should be compatible with ComfyUI, visual output quality should be comparable.

### Phase 9: Differential MoE Training Strategy
**Goal**: Per-expert hyperparameters as a training strategy that sits ABOVE the model layer.
**Delivers**: `dimljus/training/strategies/differential_moe.py`:
- Configurable per-expert rank, learning rate, and epoch count
- Works with any MoE model backend (Wan 2.2 is validation target, but architecture supports any dual/multi-expert model)
- Validation/warnings for aggressive low-noise expert parameters (the "strawberryman" discovery)
**Validation**: Train with differential parameters → verify results match Minta's known-good differential training outcomes.

### Phase 10: Control Signal Expansion
**Goal**: Full control signal architecture proven with real signals.
**Delivers**:
- Depth/edge/pose generation tools (standalone)
- Control LoRA training (Flux-tools-style channel concatenation approach)
- VACE context block support
- Abstract control signal base class proven with multiple signal types
**Validation**: Train a depth control LoRA → verify it actually controls model output.

### Phase 11+: Extensions
- Audio control signal interface (trigger: Wan 2.5 open release)
- Additional model backends (LTX-2, SkyReels)
- SkyReels-V2 Diffusion Forcing-aware training (research — genuinely novel)
- Community release preparation
