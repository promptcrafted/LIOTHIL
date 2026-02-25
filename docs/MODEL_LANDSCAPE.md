# Model Landscape & Trainer Coverage Analysis

> Last updated: 2026-02-22
> Status: Research complete, decisions made

This document captures the current state of open-source video generation models, which ones have training support, where the gaps are, and why Dimljus targets what it targets.

## Models With Strong Community Trainer Support

| Model | musubi-tuner | ostris (ai-toolkit) | finetrainers | Notes |
|---|---|---|---|---|
| **Wan 2.1/2.2 (T2V + I2V)** | ✅ | ✅ | ✅ | Best ecosystem. Our bread and butter. |
| **HunyuanVideo (13B)** | ✅ | — | ✅ | kohya's first video model support |
| **FramePack** | ✅ (native) | — | Partial | Based on HunyuanVideo with next-frame prediction. musubi has native support, others hack it. Can load HunyuanVideo LoRAs with conversion (rename blocks, split QKV weights). |
| **CogVideoX (2B/5B)** | — | — | ✅ | finetrainers/diffusers only |
| **Flux (image)** | ✅ | ✅ | ✅ | Not video, but in the ecosystem |

## Models With Official Trainer Only

### LTX-Video / LTX-2 (19B)
- **Lightricks ships their own trainer** (ltx-trainer) — LoRA, full finetune, IC-LoRA
- First open-source model with **native joint audio+video generation** — audio in shared latent space
- 19B parameters: 14B video + 5B audio
- Uses Gemma-3 text encoder (NOT T5), different VAE and DiT variant from Wan
- Apache 2.0 license

**LTX Trainer Assessment (NOT nerfed):**
The official trainer is surprisingly complete. It supports joint audio-video LoRA training with an explicit `with_audio` config toggle, first-frame conditioning, scene splitting, and validation sampling. They even provide a low-VRAM config with INT8 quantization for 32GB GPUs.

The situation is *inverted* from the usual pattern: the official trainer is MORE capable than community trainers for LTX-2, specifically because the audio-video joint training requires understanding the model's internal architecture that only Lightricks fully documents. Community trainers like ostris that are adding LTX-2 support may only train the video transformer, NOT the joint audio-video system.

**BUT the real limitations are:**
1. Hardware floor: H100 80GB recommended, 32GB minimum (no 24GB path)
2. No differential training strategies
3. No control signal sophistication — simple videos + captions + optional audio
4. Commercial gravity toward Lightricks' hosted services (LTX Studio, fal.ai, WaveSpeedAI)
5. No methodology opinions — it's a tool, not a framework with craft knowledge

**Key insight: Audio latent space entanglement.** When you LoRA train LTX-2, the audio pathway gets modified EVEN IF you don't intend it. One I2V adapter LoRA explicitly noted unexpected audio changes from video-only training. The latent spaces are entangled. This validates Dimljus's "every signal is explicit" philosophy.

## Open Weights, NO Training Support Anywhere

### SkyReels-V2 (14B) — BIGGEST UNOCCUPIED NICHE
- Based on Wan architecture with significant modifications
- Infinite-length video via **Diffusion Forcing** (DF)
- Apache 2.0 license
- **Why no trainer exists**: DF uses a fundamentally different training paradigm — non-decreasing noise schedule across frames, where different frames in the same sequence can have different noise levels. Standard LoRA training assumes uniform noise across all frames. Nobody has figured out DF-aware LoRA training yet.
- spacepxl made one SkyReels I2V smooth LoRA (proof it's possible), but no tools support it
- **If Dimljus could crack DF-aware training, it would be genuinely novel** (Phase 7+ ambition)

### SkyReels-V3
- Released January 2026
- Same DF architecture challenges as V2

### Mochi 1 (10B)
- Apache 2.0, AsymmDiT architecture
- Genmo released a LoRA method but no mainstream trainer adopted it
- Architecture different enough from standard DiT that existing tools need custom handling

## Closed / API-Only (Cannot Train)

| Model | Status |
|---|---|
| **Wan 2.5 / 2.6** | NOT open source. API only via Alibaba Cloud. Community frustrated — expected open release based on 2.1/2.2 pattern. Speculative timeline: mid-to-late 2026. Features native audio-video sync, 1080p HD, RLHF alignment. |
| **Seedance, Kling, Hailuo, Veo** | All closed commercial |

## Audio Conditioning Approaches (Three Paradigms)

### 1. Post-hoc Audio Generation (MMAudio)
- Generate video first, then generate matching audio from video frames
- MMAudio: 157M parameter flow matching model, trained jointly on video-audio-text data
- Separate model, not a control signal during video training
- Current practical approach people actually use today
- 1.23s to generate 8s clip, frame-level synchronization via conditional sync module

### 2. Native Joint Audio-Video Generation (LTX-2 / Wan 2.5)
- Audio and video generated together in unified architecture
- LTX-2: audio part of same latent space and diffusion process
- When LoRA training, same adapters affect both audio and video
- Latent spaces are entangled — unexpected cross-modal effects during training

### 3. Audio-Conditioned Video Generation (Wan-S2V)
- Audio drives video generation: input audio track → model generates matching video
- "Audio as input control signal" model
- Closest to Dimljus's control signal architecture philosophy

### Audio Strategy for Dimljus
Design the audio control signal interface NOW (it's just another entry in the signal registry), but don't implement encoding/training until a clear target model emerges. Two most likely triggers:
1. Wan 2.5 goes open source with native audio
2. Want to pair Dimljus-trained LoRAs with MMAudio in a pipeline

Audio maps into Dimljus's control signal framework:
```
controls/
├── audio_track         ← temporal/semantic conditioning
│   ├── foley_hints     ← "sounds of footsteps, door closing"
│   ├── music_track     ← rhythm/energy signal
│   ├── speech_dialogue ← lip sync driving signal
│   └── audio_embedding ← encoded audio features (CLAP/Synchformer)
```

## Why Models With Built-in Control Signals Generalize Better

Well-documented phenomenon: models with structured conditioning pathways generalize better to new control types.

**Mechanism**: When a model has been trained to attend to reference image conditioning via channel concatenation, the attention layers have already learned "there is dense spatial information in these extra channels that I should respect." Introducing depth maps through the same pathway means the model already has machinery for obeying dense spatial signals — the LoRA just teaches what depth maps look like and how to respond.

**Implications for Dimljus**: Prioritize models that already have I2V or multi-modal conditioning (Wan I2V, VACE, LTX-2) as targets for control signal expansion. Pure T2V models are worse candidates for control LoRA training because they lack pre-existing conditioning pathways.

## Strategic Recommendations (Confirmed)

### Near Term (Phases 0–5)
- Wan 2.2 T2V + Wan 2.1 T2V to prove pipeline
- Then Wan 2.2 I2V with differential MoE
- Solidify control signal architecture

### Medium Term
- VACE support for multi-control training
- Control LoRA training (depth, edge, pose via channel concatenation)
- Audio control signal interface design (no implementation yet)

### Opportunity Watching
- SkyReels-V2 DF-aware LoRA training (biggest unoccupied niche)
- LTX-2 joint audio-video LoRA training (interesting but Lightricks ships own trainer)
- Wan 2.5 open release (would change the landscape significantly)
