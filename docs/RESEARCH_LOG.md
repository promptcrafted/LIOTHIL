# Research Log

> Last updated: 2026-02-22
> Status: Active — capture new findings as they emerge

This document records insights, discoveries, and decisions from the research phase in chronological order. Claude Code should reference this for context on WHY certain decisions were made.

---

## Session: 2026-02-22 — Phase 0 Implementation (Data Config Schema)

### What Was Built
Implemented the complete data config schema — Dimljus's description of a dataset. This is the foundational schema that all data tools (Phases 1–4) will consume and produce.

### Key Design Decisions Baked Into Code
- **Minimum viable config = just a path.** `dataset: { path: ./video_clips }` is all a new user needs. Everything else defaults to Wan model training priors (16 FPS, 480p, 4n+1 frame counts).
- **Backwards compatibility shorthand.** `dataset.path` (singular) auto-wraps into a `datasets[]` list, so simple configs stay simple.
- **Multi-dataset with per-dataset settings.** repeats, loss_multiplier, is_regularization per folder. Addresses the common pattern of balancing a small hero dataset against larger supplementary data.
- **Two caption formats.** Sidecar `.txt` (default, one per clip) and consolidated `.jsonl`. Both from Minta's lora-gym methodology.
- **Anchor word, not trigger word.** Natural-language term that reads naturally in captions (`"annika A girl walks through a garden..."`). Minta's terminology.
- **3D bucketing.** aspect_ratio + frame_count + resolution — more sophisticated than the 1D aspect-ratio-only bucketing in image trainers.
- **SAR handling.** `auto_correct | reject` — explicitly addresses the SAR bug in musubi-tuner.
- **Caption dropout deliberately excluded.** It's a training-time behavior, not a data description. Deferred to training config (Phase 5).

### Verified Against Real Data
Integration tests use a 5-clip subset from a real Jinx character LoRA dataset (`tests/fixtures/jinx_subset/`). Real `.txt` captions, placeholder `.mov` files. Config validates correctly with `name: jinx`, `use_case: character`, `anchor_word: Jinx`, `source: first_frame`.

### Deferred to Training Config (Phase 5)
Documented in `memory/training_config_scratch.md` with a comparison table:
- **Caption dropout rate** — training-time, per-run
- **Training curriculum / data scheduling** — Dimljus-original, neither ai-toolkit nor musubi supports
- **Latent cache paths** — infrastructure concern
- **Per-expert data routing** — MoE training strategy

---

## Session: 2026-02-22 — Model Landscape & Audio Signals

### Research Questions Investigated
1. Should Dimljus focus only on Wan + LTX, or are there other models to consider?
2. Are there models with NO training support that represent opportunities?
3. How should audio conditioning be handled as a control signal?
4. Do models with built-in control signals generalize better to new control types?

### Key Findings

#### Trainer Coverage Matrix
Compiled a comprehensive analysis of which trainers support which models:

- **Wan 2.1/2.2**: Best ecosystem — musubi-tuner ✅, ostris ✅, finetrainers ✅
- **HunyuanVideo (13B)**: musubi-tuner ✅, finetrainers ✅, ostris ✗
- **FramePack**: musubi-tuner ✅ (native), others partial. Based on HunyuanVideo with next-frame prediction. Can load HunyuanVideo LoRAs with conversion (rename blocks, split QKV weights).
- **CogVideoX (2B/5B)**: finetrainers only
- **LTX-2 (19B)**: Official Lightricks trainer only (see detailed analysis below)
- **SkyReels-V2 (14B)**: NO TRAINING SUPPORT ANYWHERE (see below)
- **Mochi 1 (10B)**: No mainstream trainer support (Genmo released a method but nobody adopted it)

#### SkyReels-V2: Biggest Unoccupied Niche
SkyReels-V2 (14B) is based on Wan's architecture but uses **Diffusion Forcing** (DF) — a fundamentally different training paradigm where different frames in the same sequence can have different noise levels (non-decreasing noise schedule). Standard LoRA training assumes uniform noise across all frames. Nobody has figured out DF-aware LoRA training.

spacepxl made one SkyReels I2V smooth LoRA (proof of concept that it's possible), but no tools support it systematically. If Dimljus could crack DF-aware training, it would be genuinely novel. Phase 7+ ambition.

SkyReels-V3 (released January 2026) has the same DF challenges.

#### LTX-2 Official Trainer: NOT Nerfed (Inverted Pattern)
Investigated Minta's hypothesis that company-released trainers are typically nerfed. For LTX-2, the pattern is **inverted**: the official trainer is MORE capable than community alternatives.

**Why it's not nerfed:**
- Joint audio-video LoRA training with explicit `with_audio` toggle
- Config file (`ltx2_av_lora.yaml`) shows real production setup with separate directories for video latents, text embeddings, and audio latents
- First-frame conditioning training for I2V quality
- Low-VRAM config with INT8 quantization for 32GB GPUs (RTX 5090)
- They actually bothered to make a consumer-GPU path

**Why the official trainer is MORE capable than community alternatives:**
The audio-video joint training requires understanding LTX-2's internal architecture in a way that only Lightricks fully documents. Community trainers (ostris adding LTX-2 support) may only train the video transformer, NOT the joint audio-video system. A RunComfy guide for ostris + LTX-2 notes: "audio-aware finetuning depends on whether your trainer actually updates the audio pathway and cross-modal components (many third-party training stacks start by..."

**Real limitations (not feature-nerfing, but methodology gaps):**
1. Hardware floor: H100 80GB recommended, 32GB minimum, no 24GB path
2. No differential training strategies
3. No control signal sophistication — simple videos + captions + optional audio
4. Commercial gravity toward Lightricks' hosted services
5. No methodology opinions — it's an engine, not a vehicle

**Critical insight — Audio latent space entanglement:**
When you LoRA train LTX-2, the audio pathway gets modified EVEN IF you don't intend to train audio. One MachineDelusions I2V adapter LoRA documented unexpected audio changes from video-only training. The latent spaces are entangled. This validates Dimljus's "every signal is explicit" philosophy — ignoring a signal during training has unintended consequences on entangled models.

#### Three Audio Conditioning Paradigms
Identified three distinct approaches to audio in video generation:

1. **Post-hoc audio generation (MMAudio)**: Generate video first, then generate matching audio from video frames. MMAudio is a 157M parameter flow matching model. Current practical approach. Separate model, not a control signal during video training.

2. **Native joint audio-video generation (LTX-2 / Wan 2.5)**: Audio and video generated together in unified architecture. Audio part of same latent space. When LoRA training, same adapters affect both audio and video. Latent spaces are entangled.

3. **Audio-conditioned video generation (Wan-S2V)**: Audio drives video generation — input audio track → model generates matching video. Closest to Dimljus's control signal architecture philosophy.

#### Models With Built-in Control Signals Generalize Better
Confirmed Minta's hypothesis. Well-documented phenomenon: models with structured conditioning pathways generalize better to new control types.

**Mechanism**: When a model has been trained to attend to reference image conditioning via channel concatenation, attention layers already learned "there is dense spatial information in these extra channels that I should respect." Introducing depth maps through the same pathway means the model already has machinery for obeying dense spatial signals — the LoRA just teaches what depth maps look like.

**Why VACE works well**: Explicitly trained to handle multiple control types through context blocks. Each new control type has a pre-existing "slot."

**Why Flux-tools control LoRA approach is efficient**: Flux already knows how to process image inputs. Teaching it to interpret depth map as "structure I should follow" is a small adaptation.

**Implication**: Prioritize models with existing multi-modal conditioning (Wan I2V, VACE, LTX-2) as targets for control signal expansion. Pure T2V models are worse candidates.

### Decisions Made

1. **Model focus confirmed**: Wan 2.2 (primary) with Wan 2.1 for pipeline validation
2. **T2V first**: Prove pipeline on simpler case before adding I2V control signal complexity
3. **LTX-2 not near-term**: Lightricks ships a capable trainer; Dimljus's value-add would be differential training and control signal routing, not basic LoRA support
4. **Audio strategy**: Design the interface now (it's just another signal in the registry), implement when clear target emerges (Wan 2.5 open release or MMAudio pipeline integration)
5. **SkyReels-V2 DF-aware training**: Biggest unmet need in the ecosystem, but Phase 7+ ambition

### Closed vs Open Model Status (for future reference)
- **Wan 2.5/2.6**: NOT open source, API only via Alibaba Cloud. Community expected open release based on 2.1/2.2 pattern. Speculative timeline: mid-to-late 2026. Features native audio-video sync, 1080p HD, RLHF alignment.
- **Wan 2.1/2.2**: Open source, Apache 2.0
- **LTX-2**: Open source, Apache 2.0
- **SkyReels-V2/V3**: Open source, Apache 2.0
- **Everything else good** (Seedance, Kling, Hailuo, Veo): Closed commercial

---

## Session: 2026-02-22 — Phase 1 Implementation (Video Ingestion & Captioning)

### What Was Built
Implemented the complete video ingestion pipeline and VLM captioning system. This is the first real tool — takes raw video and produces clean training clips with optional AI-generated captions.

### Modules Delivered

**Video Pipeline (`dimljus/video/`):**
- `probe.py` — ffprobe wrapper extracting all training-relevant metadata (fps, resolution, frame count, SAR, codec, duration)
- `validate.py` — Structural validation against VideoConfig with machine-readable issue codes and severity levels (error/warning/info)
- `scene.py` — PySceneDetect wrapper for finding scene cuts and verifying pre-cut clip coherence
- `split.py` — ffmpeg wrapper for normalization (fps, resolution, SAR, frame count) and scene splitting
- `_ffmpeg.py` — Auto-discovers ffmpeg from WinGet/Chocolatey/Scoop install locations (solves the "ffmpeg not in PATH after winget install" problem on Windows)
- `__main__.py` — CLI: `python -m dimljus.video {scan, ingest, normalize, caption, audit}`

**Captioning Pipeline (`dimljus/caption/`):**
- `gemini.py` — Google Gemini direct API backend (video upload + poll, image inline)
- `replicate.py` — Replicate raw HTTP backend with model schema auto-detection
- `prompts.py` — Use-case-specific prompt templates (character: omit appearance, style: omit aesthetics, motion: focus movement, object: omit object description)
- `captioner.py` — Batch orchestrator with progress, retry, rate limiting, anchor word prepending

### Key Technical Decisions

1. **Replicate: raw HTTP, not the SDK.** The Replicate Python SDK's `client.run()` does not reliably pass video data — tested on a puppy video, the SDK-based approach produced a hallucinated caption about latte art (meaning the model never saw the video). Switching to raw `requests.post()` with base64 data URIs fixed it immediately. The model confirmed correct content (puppy) on the same video.

2. **Model schema auto-detection.** Different Replicate models use different input field names (`videos` array vs `media` string vs `video` string). The backend fetches the model's OpenAPI schema from Replicate's API on first use and sends data in the correct field. This means the same backend works with any Replicate model without hardcoded field names.

3. **ASCII-only console output.** Windows cp1252 encoding can't handle Unicode characters like arrows, em dashes, or checkmarks. All console output uses ASCII equivalents (`->` not `->`, `-` not `--`, `OK`/`FAIL` not checkmarks).

4. **ffmpeg auto-discovery.** WinGet installs ffmpeg to a deep path that only appears in PATH after shell restart. `_ffmpeg.py` scans known install locations at import time and adds to PATH if found. Zero-config for users.

5. **Validation issue codes.** Machine-readable `IssueCode` enum with `Severity` levels. Every issue includes a human-readable message explaining what's wrong AND how to fix it. This pattern carries forward to all future validation.

### Verified Against Real Data
- **Puppy video** (1280x720@60fps, 10s): Probe extracts correct metadata. Ingest normalizes to 480p@16fps with correct 4n+1 frame count. Replicate backend produces accurate caption ("small, black and white puppy lying on sandy ground").
- **Jinx dataset** (26 clips, 1920x1080@23.976fps .mov files): Scan correctly identifies all clips as needing re-encoding (resolution above target, fps mismatch, non-4n+1 frame counts). Report is readable and actionable.

### Test Coverage
- 190 tests total (76 Phase 0 + 114 Phase 1)
- Three tiers: pure Python (models + validate, always runs), integration (needs ffmpeg/scenedetect, skip markers), manual verification (real videos)
- Integration tests use tiny generated videos via ffmpeg's `testsrc` filter (~50KB each)

---

## Open Research Questions (Ongoing)

These surfaced during research and should be investigated as the project progresses:

1. **VACE training**: Why hasn't anyone set up LoRA training for VACE's context blocks? Is it a technical limitation (context blocks need different LoRA injection) or just a gap in tooling?

2. **DF-aware LoRA training**: Could Dimljus's control signal architecture be extended to handle Diffusion Forcing's per-frame noise levels? The non-decreasing noise schedule across frames is fundamentally different from standard uniform noise.

3. **Caption dropout × MoE interaction**: Does caption dropout affect high-noise and low-noise experts differently? The high-noise expert handles coarse features that might be more text-dependent.

4. **Expert boundary tuning**: Is Wan 2.2's default SNR boundary for expert switching optimal for LoRA training, or should it be a tunable parameter in Dimljus?

5. **Audio entanglement in training**: On models like LTX-2 where audio and video share latent space, how do you prevent unintended audio degradation when training video-only LoRAs? Is this a fundamental limitation of shared-latent architectures?

6. **Control signal weighting × noise level**: Should different control signals be weighted differently for different noise levels? (e.g., text more important at high noise, reference image more important at low noise)
