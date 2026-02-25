# Existing Trainer Analysis

> Last updated: 2026-02-22
> Status: Research complete

This document analyzes the three main training frameworks relevant to Dimljus's development. Understanding how they work is essential for both compatibility and identifying what Dimljus should do differently.

## musubi-tuner (Minta's Current Production Tool)

**Repository**: kohya's musubi-tuner (Minta runs a personal fork)
**Architecture**: Extension of sd-scripts for video models
**Config format**: TOML + command-line arguments

### What It Does Well
- Proven production quality for Wan 2.1/2.2 LoRAs
- HunyuanVideo and FramePack support
- Latent pre-encoding (caches VAE-encoded video latents to avoid re-encoding every epoch)
- Resolution bucketing for videos
- Well-understood by Minta's team

### What It Lacks
- No differential MoE support (trains both experts identically or you pick one)
- No control signal abstraction — reference images and captions are handled implicitly
- Video treated as "images with temporal dimension" rather than first-class temporal sequences
- No temporal validation (cut detection, frame rate verification)
- Caption-centric conditioning model
- TOML config is functional but not opinionated — exposes too many knobs without guidance

### Key Implementation Details (For Compatibility)
- Saves LoRA weights as safetensors with specific key naming conventions
- Uses `train_util.py` for argument parsing with hundreds of defaults
- `NetworkTrainer` class manages the training loop
- Flow matching noise scheduling for Wan models
- Mixed precision (bf16) with gradient checkpointing

## ostris / ai-toolkit

**Repository**: ostris/ai-toolkit
**Architecture**: Modular, factory-pattern based
**Config format**: YAML with `type` field dispatching

### What It Does Well
- Clean modular architecture — each model type is a separate trainer class
- YAML config with factory pattern (type fields select Python classes)
- Recently added Wan 2.2 I2V support
- RunComfy cloud integration for easy access
- Multi-stage training support for Wan 2.2 MoE (high/low noise stages)

### What It Lacks
- Audio-awareness for LTX-2 is incomplete — may only train video transformer, not joint audio-video system
- Same caption-centric approach as all existing trainers
- No control signal routing abstraction
- Video validation is basic

### Key Design Pattern Worth Studying
The factory pattern in ostris is worth understanding:
```yaml
config:
  process:
    - type: "sd_trainer"      # dispatches to SDTrainer class
      network:
        type: "lora"          # dispatches to LoRA network builder
```
This makes the system extensible — adding a new model type means adding a new class, not modifying the existing code. Dimljus should use a similar pattern for model type registration.

## finetrainers

**Repository**: a]
**Architecture**: Built on Hugging Face diffusers
**Config format**: Python scripts / YAML

### What It Does Well
- Tight integration with diffusers library
- Supports CogVideoX (which neither kohya nor ostris does)
- HunyuanVideo support
- Wan support
- Leverages diffusers' existing model loading and pipeline infrastructure

### What It Lacks
- Less production-proven for Wan compared to musubi-tuner
- Same fundamental limitations around control signals and video-native handling
- More research-oriented than production-oriented

## Lightricks ltx-trainer (LTX-2 Official)

**Repository**: Lightricks/LTX-2 (monorepo, ltx-trainer package)
**Architecture**: Custom training scripts
**Config format**: YAML

### What Sets It Apart
- **Joint audio-video training** with explicit `with_audio` toggle
- First-frame conditioning training for I2V quality
- Automatic scene splitting for long videos
- Validation sampling during training
- Low-VRAM config with INT8 quantization for 32GB GPUs

### What It Still Lacks
- No differential training strategies
- No control signal sophistication beyond video + caption + audio
- No methodology opinions — it's a tool, not a framework with craft knowledge
- Hardware floor: H100 80GB recommended, 32GB minimum (no 24GB path)

### Assessment: NOT Nerfed
Unlike typical company-released trainers, the LTX trainer is genuinely complete for its model. The limitation isn't features — it's that it only supports LTX-2, and it doesn't encode any training methodology or production workflow knowledge. It's an engine, not a vehicle.

## Common Patterns Across All Trainers

### What They All Do
1. Config-driven training with defaults
2. Mixed precision (bf16/fp16) training
3. Gradient checkpointing for memory management
4. AdamW or similar optimizers
5. LoRA injection into attention layers (Q, K, V projections)
6. Safetensors checkpoint format
7. Latent pre-encoding (VAE cache)

### What None of Them Do (Dimljus's Opportunity)
1. **Control signal routing** — abstracting every input as a registered control signal
2. **Differential MoE training** — per-expert hyperparameters as a native concept
3. **Temporal validation** — scene cut detection, frame rate verification, temporal coherence checking
4. **Methodology encoding** — opinionated defaults that reflect production training best practices
5. **Caption demotion** — treating text as one signal among many rather than the primary conditioning mechanism
6. **Control LoRA training** — teaching models to respond to new control types (depth, edge, pose) during training
7. **VACE-aware training** — native support for VACE's context block architecture

## Compatibility Requirements

Dimljus's output must be compatible with the existing ecosystem. Specifically:

### LoRA Weight Format
- **File format**: safetensors
- **Key naming**: Must match what ComfyUI and diffusers expect for the target model
- For Wan, reference musubi-tuner's output key naming
- For Wan 2.2 MoE, separate weight files per expert OR a single file with expert-prefixed keys

### Inference Compatibility
- Trained LoRAs must load and work in:
  - ComfyUI (primary inference tool for Minta's workflow)
  - diffusers (for API/pipeline usage)
- Weight strength/scale should behave predictably (1.0 = full strength)

### Dataset Format
- Dimljus should be able to consume datasets prepared for musubi-tuner (backward compatibility)
- Should also support its own richer dataset format with control signal metadata
