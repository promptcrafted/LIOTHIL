# Dimljus — Video LoRA Training Toolkit

You are assisting Minta, Senior AI Architect and cofounder of Alvdansen Labs, in building **Dimljus** — a purpose-built video LoRA training toolkit for diffusion transformer models (starting with the Wan model family: Wan 2.2 T2V/I2V and Wan 2.1 T2V).

## Thesis

Existing training frameworks were built for image generation and treat video as a bolted-on extension — they don't distinguish between what the model should produce and what it should obey, they handle frames as stacked images rather than temporally coherent sequences, and they offer no native support for the control signal architectures (like VACE's context blocks) that define the frontier of video AI. Dimljus inverts this by treating video training as a **control signal routing problem**: video is the sole target the model learns to produce, and every input — reference images, depth maps, poses, captions, and signals yet to be invented — is a first-class control signal with its own preparation, validation, and weighting.

## What Dimljus Is (and Isn't)

Dimljus is NOT a general-purpose replacement for kohya (sd-scripts) or ostris (ai-toolkit). It is a focused, opinionated training framework built around Minta's specific methodology for production video LoRA training.

Core philosophy: **curator-first, not math-first** — datasets, aesthetic judgment, and production quality drive technical decisions.

Key differentiators from existing tools:
- **Video-native architecture**: videos are temporally coherent sequences, not stacked images. The dataset pipeline validates temporal coherence, detects cuts, verifies frame rates, and understands video as video.
- **Control signal routing**: every conditioning input (reference images, depth maps, poses, edge maps, captions) is a first-class control signal with its own preparation, validation, encoding, and weighting pathway. Captions are ONE control signal, not THE control signal.
- **Target/control separation**: clear architectural distinction between what the model learns to produce (video) and what it learns to obey (control signals). This mirrors how the models actually work — in Wan I2V, the reference image is VAE-encoded and concatenated in latent space as conditioning channels, not treated as training target data.
- **Differential MoE training**: native support for different hyperparameters per noise-level expert (Wan 2.2's dual MoE architecture)
- **VACE-aware**: designed to support training LoRAs that work with VACE's context block architecture
- **Control LoRA training**: ability to teach models to respond to new control signals (depth, edge, pose) via Flux-tools-style channel concatenation
- **Production defaults**: opinionated config surface encoding Minta's methodology for enterprise client delivery

This project runs in parallel with Minta's continued use of her musubi-tuner fork for active client work. Dimljus is experimental and exploratory — build incrementally, validate each piece against known-good results before moving forward.

## Design Philosophy: Ground Up, Standalone Tools

Dimljus is built from the ground up. Every phase produces something usable on its own, and the data preparation tools work independently of Dimljus's training pipeline — they produce standard output formats that work with musubi-tuner, ostris, or any other trainer.

This matters because **data preparation is the #1 barrier to entry** in video LoRA training. Existing trainers assume you've already solved this. Dimljus's data tools (scene detection, caption generation, reference extraction) are valuable even if you never use Dimljus's training loop.

## Model Support Strategy

**Architecture supports**: All Wan model variants (2.1, 2.2, T2V, I2V). Extensible to future models via a model registry pattern.

**Training infrastructure is model-agnostic.** The training loop, noise scheduling interface, LoRA injection interface, and checkpoint export interface don't know about Wan specifically — they define what information a model needs to provide. Adding a new model means implementing a model backend (noise schedule, layer mapping, forward pass, checkpoint format), not rewriting the trainer.

**Validation sequence**: Wan 2.1 T2V first (single transformer, simplest case), then Wan 2.2 T2V (adds MoE routing), then Wan 2.2 I2V (adds image control signal). Each exercises more of the infrastructure.

**Why T2V first**: T2V has only one control signal (text caption). I2V adds the reference image as a second control signal with its own encoding pathway (VAE → latent concatenation). Proving the pipeline on the simpler case avoids conflating pipeline bugs with control signal routing bugs.

## Technical Context

### Wan 2.2 Architecture
- 14B active parameters per step, ~27B total (dual MoE)
- Two-expert design: high-noise expert (early denoising, global composition/motion) and low-noise expert (late denoising, fine detail/texture)
- Expert switching determined by signal-to-noise ratio (SNR), boundary at ~875/1000 of noise schedule
- Flow matching within Diffusion Transformers
- Text enters via T5 encoder through cross-attention
- I2V: reference image VAE-encoded and concatenated with noisy latents (channel dimension)
- VACE: separate context blocks and context tokens for control signals
- Wan-VAE: 3D causal VAE with ~4x temporal compression. 81 frames → ~21 temporal tokens in latent space

### Differential MoE Training (Core Finding)
High noise experts converge faster on coarse compositional features. Low noise experts need longer training for fine detail work. Low noise experts overfit rapidly when trained aggressively, producing washed-out artifacts and failing on dynamic expressions.

Current experimental parameters:
- High noise expert: rank 16 / lr 1e-4 / 30 epochs
- Low noise expert: rank 24 / lr 8e-5 / 50 epochs

This is the key insight that no existing trainer supports natively.

### Wan 2.1 Architecture
- Single transformer (not MoE) — same DiT architecture but without expert splitting
- Same T5 text encoder, same VAE
- Simpler training target — good for proving the base pipeline works correctly

## Data Architecture

Every training sample in Dimljus follows this structure:

```
Training Sample:
├── target/
│   └── video clip              ← what the model learns to PRODUCE
│       (validated: frame rate, temporal coherence, no cuts, resolution)
│
├── controls/
│   ├── reference_image         ← primary visual conditioning (I2V first frame)
│   ├── subject_image           ← character/object reference (IP adapter, subject LoRA)
│   ├── style_image             ← aesthetic reference (style transfer, mood board)
│   ├── depth_map (optional)    ← structural control signal
│   ├── edge_map (optional)     ← structural control signal
│   ├── pose (optional)         ← motion control signal
│   ├── audio_track (optional)  ← temporal/semantic control signal (future)
│   └── [extensible]            ← any future control signal type
│
├── text/
│   └── caption                 ← semantic control signal (one signal among many)
│       (optional, with configurable dropout rate)
│
└── metadata/
    ├── source_info             ← provenance, quality tags
    ├── control_config          ← per-sample control signal settings
    └── image_roles             ← what each image input MEANS (reference, subject, style, structure)
```

**Image inputs are not just "reference images."** Users bring many kinds of images into training, each with a different role:
- **I2V reference**: extracted first frame, tells the model "make video starting from this"
- **Subject reference**: a character sheet, product photo, or face crop — tells the model "this is what the subject looks like"
- **Style reference**: a mood board image, film still, or painting — tells the model "match this aesthetic"
- **Structural guide**: depth map, edge map, pose skeleton — tells the model "follow this structure"
- **Brand asset**: logo, UI element — tells the model "incorporate this element"

The organizational structure teaches users to think about what each image input MEANS, not just where to put it. This distinction drives how each image gets encoded and injected into the model.

Key principles:
- Controls are first-class citizens, not afterthoughts
- Every control signal goes through: prepare → validate → encode
- The config schema defines which controls are active and how they're weighted
- Caption dropout is a feature, not a bug — forcing reliance on visual control signals
- The system is extensible: adding a new control type should not require rewriting the pipeline

## Working Relationship

- **Minta is the domain expert and architect.** She defines what the tool should do, what training behaviors matter, and evaluates output quality. She has thousands of hours of training experience but is a code beginner.
- **Claude is the implementation partner.** Write clean, well-commented Python. Explain every design decision. Never assume she'll "just know" what code does — explain it, but respect her intelligence. She thinks in systems and aesthetics, not syntax.
- **Minta cannot test code on GPU through Claude.** Every training component must be validated by Minta running it locally. Build incrementally: write a piece → she tests it → we verify → move on.

## Code Standards

- Python 3.10+, type hints everywhere
- YAML for all config files (PyYAML)
- Every function gets a docstring explaining what it does and WHY
- Inline comments for anything non-obvious
- Clear error messages that tell the user what went wrong AND how to fix it
- File paths use pathlib.Path, always handle Windows paths correctly
- Print meaningful progress information during long operations
- Fail fast with helpful messages rather than silently producing bad results

## Architecture Layers

1. **Config Surface** — YAML files users edit. Small, opinionated, with sensible defaults.
2. **Orchestrator** — `train.py` reads config, validates, builds objects, runs training loop.
3. **Utilities** — Isolated modules: dataset loading, control signal processing, LoRA network setup, noise scheduling, checkpoint saving, logging.

## Hardware & Environment

- Windows PC (Lenovo), NVIDIA GPUs
- Python environment: Windows with VS Code and Claude Code
- Dataset scripts: `C:\Users\minta\Training\`
- Output path: `C:\Users\minta\Training\datasets\{name}\{video_clips,image_stills}\`

## Available MCP Tools

Claude Code has the following MCP servers configured globally. Use these proactively when they'd help with Dimljus work.

### Directly Relevant to Dimljus
- **Hugging Face** — Search models, datasets, and papers on HF Hub. Use for finding Wan model variants, LoRA checkpoints, and training datasets.
- **Replicate** — Run inference and search models. Dimljus already uses Replicate for captioning backends (`dimljus/caption/replicate.py`).
- **video-editor** — FFmpeg-based video editing via MCP. Useful for quick video inspection, conversion, and frame extraction alongside the dimljus pipeline.
- **arxiv** — Search and download ML papers. Use for researching video generation, LoRA training methods, MoE architectures, and diffusion models. Papers stored at `~/.arxiv-papers`.
- **wandb** — Weights & Biases experiment tracking. Use when training runs begin (Phase 8+) to compare hyperparameters and results across LoRA experiments.
- **filesystem** — Direct read/write access to `C:\Users\minta\Projects` and `C:\Users\minta\Training`. Useful for inspecting dataset directories and output files.
- **pypi** — Search Python packages, check versions, scan for vulnerabilities. Use for dependency management (pydantic, scenedetect, google-genai, etc.).
- **context7** — (Plugin marketplace) Live, version-specific documentation for libraries. Use when working with PyYAML, Pydantic, ffmpeg-python, scenedetect APIs.
- **docker-mcp** — Container management. Relevant when containerizing training environments.
- **git** — Enhanced git operations on the dimljus-kit repo.

### Research & Web
- **brave-search** — Web, image, and video search.
- **tavily** — AI-optimized deep research search with structured extraction.
- **youtube-transcript** — Pull transcripts from video tutorials and ML talks.

### Productivity & Communication
- **Notion** — Already connected. Use for project management, phase tracking, documentation.
- **slack** — Send/search Slack messages (OAuth required on first use).
- **linkedin** — Profile and company search (session required: `uvx linkedin-scraper-mcp --login`).
- **obsidian** — Read/write/search Minta's Obsidian vault for notes and research.
- **github-mcp** — GitHub PRs, issues, Actions (OAuth on first use).
- **sentry** — Error tracking for production issues (OAuth on first use).

### General Purpose
- **mcp-installer** — Install additional MCP servers via natural language.
- **sequential-thinking** — Structured reasoning for complex architecture decisions.
- **memory** — Persistent knowledge graph across sessions.
- **puppeteer** — Browser automation and web scraping.

## Phase Plan (Ground Up)

### Phase 0: Data Config Schema -- COMPLETE
Define the data config schema — what a dataset looks like to Dimljus. YAML config with validation and defaults for video specs, caption handling, control signals, quality thresholds. Data config only — training config comes later (D12). No processing, no training — just the organizational bones.

**Delivered:** `dimljus/config/` — Pydantic v2 models, YAML loader, 76 tests.

### Phase 1: Video Ingestion & Scene Detection -- COMPLETE
Take raw source material (Blu-ray rips, YouTube downloads, client footage, pre-cut clips) and produce clean training clips. Scene detection, quality filtering, frame rate normalization, temporal validation. VLM captioning via Gemini and Replicate APIs. **Fully standalone tool** — works with any trainer.

**Delivered:** `dimljus/video/` (probe, validate, scene detect, split/normalize, CLI) + `dimljus/caption/` (Gemini backend, Replicate raw HTTP backend, batch captioner, use-case prompts, audit mode). 190 tests total.

### Phase 2: Caption Generation Pipeline -- COMPLETE
Take clips from Phase 1, generate captions. Support multiple backends (local models, API calls). Output standard `.txt` sidecar files compatible with musubi-tuner, ostris, and Dimljus. **Fully standalone tool.**

**Delivered:** `dimljus/caption/` — Gemini, Replicate, OpenAI-compatible backends, batch captioner, quality scoring, audit mode, custom prompts. 143 tests.

### Phase 3: Image Extraction & Processing -- COMPLETE
Extract reference images from clips (first frame, best frame, user-selected). This is a standalone utility that lives alongside the broader image input organization built in Phase 4. **Fully standalone tool.**

**Delivered:** `dimljus/video/extract.py` + `image_quality.py` + `extract_models.py` — first frame, best frame, user-selected extraction, image pass-through for mixed datasets, Laplacian sharpness scoring, CLI command. 88 tests. Total: 427 tests.

### Phase 4: Dataset Validation & Organization
This is where all data comes together. Validate the full dataset: correct frame counts, captions for every clip, resolution checks, control signal alignment. Organize into the target/controls/text/metadata structure. **Teach users how to think about their data** — what each image input means (reference vs subject vs style vs structural), what role each piece plays. This is where Dimljus's data architecture manifests.

### Phase 5: Training Config Schema
Design the training config — model selection, hyperparameters, optimizer, LoRA settings, MoE expert settings, checkpoint output. This comes AFTER all data tools are built so the training config is designed from real experience with the data pipeline, not speculation. Points at a data config.

### Phase 6: Latent Pre-Encoding
Encode videos through VAE, captions through T5, reference images through VAE. Cache everything to disk so training doesn't re-encode every epoch. This is **model-specific** (Wan 2.1/2.2 use Wan-VAE + T5; other models use different encoders) and bridges the gap between model-agnostic data preparation and model-specific training.

### Phase 7: Training Infrastructure (Model-Agnostic)
Build the training loop, noise scheduling interface, LoRA injection interface, and checkpoint export interface. These define **what information a model must provide** — they don't know about Wan specifically. The model registry pattern: register a model by providing its noise schedule, layer mapping, forward pass, and checkpoint format.

### Phase 8: Wan Model Implementations
Implement Wan-specific backends. Validate Wan 2.1 T2V first (simplest: single transformer, text-only conditioning), then Wan 2.2 T2V (adds MoE expert routing), then Wan 2.2 I2V (adds reference image control signal). Each exercises more of the Phase 7 infrastructure. **Validate every component against musubi-tuner output on identical datasets.**

### Phase 9: Differential MoE Training Strategy
Per-expert hyperparameters as a training strategy that sits above the model layer. Should work with any MoE model, not just Wan 2.2. Wan 2.2 is the validation target.

### Phase 10: Control Signal Expansion
Depth/edge/pose generation tools + control LoRA training. VACE context block support. Abstract control signal base class proven with real signals.

### Phase 11+: Extensions
Audio control signal interface. Additional model backends (LTX-2, SkyReels). SkyReels Diffusion Forcing-aware training (research).

## Key Principles

1. **Video is the target. Everything else is a control signal.** This distinction drives every design decision.
2. **Ground up.** Every phase is a true foundation for what follows. No skipping ahead.
3. **Standalone tools.** Data preparation tools produce standard formats that work with ANY trainer, not just Dimljus.
4. **Validate before you compute.** Catch config errors and dataset problems before burning GPU time.
5. **One thing at a time.** Each module is independent and testable.
6. **Known-good baselines.** Every component gets compared against musubi-tuner output before we trust it.
7. **Document the WHY.** Code comments explain reasoning, not just mechanics.
8. **Windows-first.** Minta works on Windows. No Linux assumptions in paths, scripts, or tooling.
9. **Model-agnostic infrastructure.** Adding a new model should mean implementing a backend, not rewriting the trainer.
10. **Teach the user.** The data organization structure teaches users how to think about image inputs, control signals, and what each piece of training data MEANS.
