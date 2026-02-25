# Wan 2.2 Training Data Reference

**Status:** Reference document — grounding for data config design
**Date:** 2026-02-22
**Sources:** Wan Technical Report (arXiv:2503.20314), Wan 2.2 GitHub, HuggingFace model configs, EmergentMind analysis

---

## Why This Document Exists

Understanding what Wan 2.2 was trained on tells us what the model already **knows** — its learned priors, its natural distribution, the data patterns it was optimized for.

This is NOT about treating pretraining conventions as rules. Diverging from how the foundation model was trained is not a blocker — it needs to be an **informed choice**. When your LoRA data diverges from the pretraining distribution, you're teaching the model something outside its comfort zone. That may be exactly what you want. A style LoRA **deliberately** diverges from the model's aesthetic priors — that's the point. A character LoRA deliberately teaches the model a face it's never seen. But you need to understand what you're overriding to do it well.

**The frame**: "What does the model already know?" → "How does my data work with or against that?"

---

## Training Data Scale & Composition

Wan 2.2 was trained on a massive multi-modal dataset:

- **~1 billion+ videos and images** total, **~1 trillion tokens**
- Wan 2.2 expanded significantly over Wan 2.1: **+65.6% more images**, **+83.2% more videos**
- **Mixed image + video training** — the model learned from both simultaneously
- Global batch size during pretraining: **1,536** (mixed image/video samples)

The model learned from both images and video in the same training run, which means it carries strong priors from still images that inform its video generation. Low-resolution, coarse features were learned first from images; temporal coherence was layered on top.

### What This Means for LoRA Training

The pretraining dataset is billions of samples with automated quality filtering. A LoRA dataset is 15–60 curated clips with human aesthetic judgment. These are fundamentally different scales and approaches, and that's fine — the LoRA only needs to teach a delta from what the model already knows, not rebuild the foundation.

---

## Progressive Training Stages

Wan 2.2 was trained in four progressive stages, each building on the previous:

| Stage | Resolution | Content | FPS | Duration |
|-------|-----------|---------|-----|----------|
| **I** | 256px | Images only | — | — |
| **II** | 256px images + 192px videos | Mixed | 16 FPS | ~5s clips |
| **III** | 480px | Images + videos | — | — |
| **IV** | 720px | Images + videos | — | Final pretraining |

### What This Means

- **Image-first, then video.** The model's foundational visual understanding comes from still images. Video temporal coherence was learned later, on top of already-strong spatial features.
- **Coarse-to-fine resolution.** Low-resolution features (composition, color, shape) are deeply baked in — they were trained first and longest. High-resolution features (texture, fine detail) were trained last and are more "surface-level."
- **16 FPS is the training frame rate** from Stage II onward. This is the temporal rhythm the model learned.
- **The model can handle 480P and 720P** because it was explicitly trained at both resolutions in the later stages.

---

## Data Quality Filtering Pipeline

Wan's data underwent rigorous multi-step quality filtering before entering training:

### Step 1: Content Filtering
- **OCR/text coverage** — filter or flag videos with excessive on-screen text
- **NSFW filter** — remove inappropriate content

### Step 2: Technical Quality
- **Blur assessment** — reject blurry or unfocused footage
- **Exposure assessment** — reject over/under-exposed content

### Step 3: Deduplication
- **De-duplication** — remove near-duplicate videos and images

### Step 4: Quality Scoring
- **Clustering + expert scoring** for both **visual quality** and **motion quality**
- **Camera motion classification** — tiers assigned (pan, tilt, dolly, static, handheld, etc.)
- The model learned to associate motion types with visual patterns

### What This Means

The model was trained exclusively on quality-assessed, filtered data. It has never seen training-time blur, exposure issues, or low-quality motion. Feeding it unfiltered data doesn't just waste compute — it teaches the LoRA to produce artifacts the base model was never trained to generate.

**For LoRA data preparation:** Basic quality validation (blur detection, exposure check) should be part of the pipeline. Not because we need billion-sample filtering, but because the model expects clean inputs and the LoRA shouldn't have to fight that expectation.

---

## Caption Generation

Training captions came from three complementary sources:

### 1. Synthesized Captions
- Generated in both **Chinese and English**
- Template-based or rule-based descriptions

### 2. OCR Extraction
- Text detected within video frames was extracted
- On-screen text is signal, not noise — the model learned to read it

### 3. Dense VLM Captions
- **LLaVA-style architecture**: ViT vision encoder + Qwen LLM
- Produces **dense, descriptive captions** — detailed paragraph-style descriptions of what's happening
- Not short prompts or tag lists

### Text Encoder
- **UMT5-XXL** (5.3B parameters)
- **Max 512 tokens** — generous limit, most captions are well under this
- **Multilingual** (Chinese + English natively)

### What This Means

- **LoRA captions should be detailed descriptions**, ideally VLM-generated. Short tag-style captions ("a woman walking, cinematic") work against the model's learned distribution of rich descriptions.
- **512 tokens is plenty** — even dense VLM captions rarely approach this limit. Don't worry about token budgets for individual captions.
- **Caption dropout is important** (Minta uses 10–20%). The model learned from captions as ONE signal among many. Dropping captions during LoRA training forces the model to rely on visual features, producing more robust LoRAs that work across varied prompts.
- **Trigger words + dense description** is the pattern that aligns best with how the model was trained: a recognizable tag (trigger) followed by detailed visual description.

---

## Video Specifications

This section distinguishes between what's architecturally required (non-negotiable) and what the model was trained on (informative context for decisions).

### Architectural Constraints (Hard Rules)

These come from the Wan-VAE architecture and cannot be violated:

| Constraint | Value | Why |
|-----------|-------|-----|
| **Frame count** | Must satisfy `(frames - 1) % 4 == 0` | VAE temporal compression: first frame processed separately, then groups of 4 |
| **Valid frame counts** | 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81 | Derived from the 4n+1 rule |
| **Spatial compression** | 8x (each dimension) | VAE spatial downsampling; input dimensions must be divisible by 8 after patching |
| **Temporal compression** | 4x (with special first frame) | Formula: `T' = 1 + (T-1)/4` |
| **Latent channels** | 16 (T2V) or 36 (I2V: 16 video + 16 ref image + 4 mask) | Model architecture input dimensions |

### Pretraining Conventions (Informative)

These are what the model was trained on — not hard constraints, but the model is optimized for them:

| Convention | Value | Can You Diverge? |
|-----------|-------|-----------------|
| **Frame rate** | 16 FPS | Yes, but the model learned temporal coherence at 16 FPS. Using 24 FPS means the LoRA must also adapt temporal expectations. |
| **Resolution** | 480P, 720P | Yes, but the model's learned distribution is these resolutions. Training at significantly different resolutions means more of the LoRA's capacity goes to resolution adaptation vs content learning. |
| **Flow shift** | 3.0 (480P), 5.0 (720P) | This is a noise schedule parameter tied to resolution. Using the matched value is strongly recommended. |
| **Upscaling policy** | Never upscale — downscale only | The model was trained on downscaled-only data. Upscaling introduces interpolation artifacts the model has never seen. |

### Practical Frame Counts

At 16 FPS, the valid frame counts translate to these clip durations:

| Frames | Latent Tokens | Duration at 16 FPS |
|--------|--------------|-------------------|
| 1 | 1 | Still image |
| 5 | 2 | 0.31s |
| 9 | 3 | 0.56s |
| 13 | 4 | 0.81s |
| 17 | 5 | 1.06s |
| 21 | 6 | 1.31s |
| 33 | 9 | 2.06s |
| 41 | 11 | 2.56s |
| 49 | 13 | 3.06s |
| 61 | 16 | 3.81s |
| 81 | 21 | 5.06s |

Most LoRA training clips should be in the **33–81 frame range** (2–5 seconds) — long enough to capture meaningful motion, short enough to be manageable in VRAM.

---

## Bucketing & Batching

### How Wan Was Trained

Multi-resolution aspect ratio bucketing is the standard approach. During pretraining:

- **Global batch size**: 1,536 (mixed image/video samples)
- **Bucketing dimensions**: aspect ratio, frame count, resolution (minimum)
- Multiple aspect ratios supported via bucketing — no single forced ratio
- Images and videos were mixed within batches

### What This Means for LoRA Training

- **Bucketing is required** for efficient training when clips have different properties
- For LoRA datasets (15–60 clips), bucketing is simpler than at pretraining scale, but still necessary
- **Bucketing dimensions for Dimljus** could extend beyond what Wan used:
  - **Aspect ratio** — standard, required
  - **Frame count** — standard, required
  - **Resolution** — standard, required for multi-res training
  - **Motion intensity** — not used in Wan pretraining, but valuable for LoRA curation
  - **Content type** — not used in Wan pretraining, but useful for organizing subject vs style vs motion clips
- No bucket should contain a single clip — minimum 2–3 for meaningful batching
- Pre-encoding latents (caching) is standard practice: encode all clips through VAE once, then train from cached latents

---

## Aesthetic Metadata (Wan 2.2 Innovation)

One of Wan 2.2's distinguishing features over Wan 2.1 is the introduction of **explicit aesthetic metadata** as first-class training data:

> "Meticulously curated aesthetic data, complete with detailed labels for lighting, composition, contrast, color tone" — Wan 2.2 README

This is not afterthought metadata — it was part of the training pipeline. The model learned associations between aesthetic properties and visual output.

### Aesthetic Dimensions
- **Lighting** — quality, direction, consistency
- **Composition** — framing, rule of thirds, visual balance
- **Contrast** — tonal range, dynamic range
- **Color tone** — warmth, saturation, palette coherence
- **Camera motion** — classified into tiers (pan, tilt, dolly, static, etc.)

### What This Means

- The model has explicit understanding of aesthetic properties — it doesn't just produce "good looking" video by accident
- **Style LoRAs** could potentially leverage this: training data with consistent aesthetic properties (all warm lighting, all high contrast) teaches the model a coherent aesthetic
- **Quality assessment** during data prep could compute similar metrics to ensure dataset consistency
- This is an area where Dimljus could add value: computing aesthetic metadata automatically and using it to inform data quality decisions

---

## What This Means for LoRA Data Preparation

### Working WITH the Model's Priors

When your LoRA data **matches** the model's training distribution, the LoRA has less work to do. All its capacity can go toward learning the new content (character, motion pattern, scene type) rather than also adapting to unfamiliar data properties.

| If your data... | The model already knows... | LoRA can focus on... |
|----------------|--------------------------|---------------------|
| 480P, 16 FPS | Resolution and temporal rhythm | Content (character, style, motion) |
| Dense VLM captions | How to parse detailed descriptions | The specific trigger + content association |
| Clean, well-exposed | Quality standards | Content, not compensating for artifacts |
| Diverse camera angles | Camera motion vocabulary | Subject identity across views |

### Working AGAINST the Model's Priors (Deliberately)

When your LoRA data **diverges** from pretraining, you're asking the LoRA to override learned behavior. This is sometimes exactly what you want, but it costs LoRA capacity and training time.

| If your data... | You're asking the LoRA to... | Cost |
|----------------|------------------------------|------|
| Different FPS (e.g., 24) | Re-learn temporal pacing | Moderate — model adapts but may look unnatural |
| Non-standard resolution | Adapt spatial features | Moderate — especially if far from 480P/720P |
| Short tag-style captions | Ignore its learned dense-caption distribution | Low — caption dropout helps, and short prompts still work |
| Specific aesthetic (film noir, oversaturated) | Override default aesthetic priors | This is the POINT — style LoRAs deliberately diverge |
| Upscaled source material | Learn interpolation artifacts as features | Bad — this degrades quality, never do this |

### Pretraining vs LoRA: Different Goals, Overlapping Tools

| Dimension | Pretraining | LoRA Fine-Tuning |
|-----------|-------------|------------------|
| **Scale** | ~1B samples | 15–60 clips |
| **Curation** | Automated quality filtering at massive scale | Human aesthetic judgment at intimate scale |
| **Goal** | Learn the visual world | Learn a specific delta (character, style, motion) |
| **Quality signal** | Computed metrics (blur score, motion score, aesthetic labels) | Curator's eye (Minta selects clips that "feel right") |
| **Caption source** | Multi-source automated VLM | Can be VLM-generated, but curator reviews and may edit |
| **Filtering** | Algorithmic — reject below threshold | Curatorial — "does this clip teach what I want?" |

**Key insight:** Pretraining is automated quality filtering at scale. LoRA training is human aesthetic judgment at intimate scale. The tools overlap (blur detection, motion quality, captioning) but the decision-making is fundamentally different. Dimljus's data tools should provide the measurement; the curator makes the judgment.

---

## I2V Reference Image Handling

For I2V models, the reference image enters through channel concatenation in the diffusers format:

- **16 channels** — VAE-encoded reference image latents
- **4 channels** — mask (indicating which regions to condition on)
- **16 channels** — noisy video latents
- **Total: 36 input channels** (vs 16 for T2V)

The reference image is **not** a training target — it's a control signal that tells the model "make video starting from this frame." During pretraining, the first frame of each clip was extracted as the reference.

### What This Means for LoRA Data Preparation

- I2V LoRA datasets need reference images paired with each clip
- The standard approach: extract the first frame as the reference image
- The model learned "reference image = first frame of the video" — this is the strongest pattern
- Using non-first-frame references (e.g., a character sheet, a different angle) is possible but diverges from pretraining distribution — the LoRA must adapt to this different relationship
- Reference image quality matters — it goes through VAE encoding, so resolution and clarity affect the conditioning signal

---

## Gaps & Open Questions

These are questions we haven't answered yet — they represent areas for future research or experimentation.

### Motion Measurement
- **How should motion intensity be measured?** Optical flow magnitude? Frame-to-frame pixel difference? What thresholds define "low/medium/high" motion?
- The pretraining pipeline used "motion quality scoring" but the specific metrics aren't published.

### Content Classification
- **How should clip content be classified?** Manual tags? Caption-derived categories? CLIP-based clustering?
- Pretraining used clustering — could Dimljus compute content clusters automatically?

### Aesthetic Scoring
- **Can we compute Wan-style aesthetic labels automatically?** Lighting quality, composition, contrast, color tone?
- Open-source aesthetic scoring models exist (LAION aesthetic predictor, etc.) — are they useful for video?

### Caption Style Impact
- **How does deviating from Wan's dense caption style affect LoRA training?** What's gained or lost with shorter captions?
- Intuition: shorter captions with higher dropout rates may produce more flexible LoRAs; dense captions may produce more precise ones. Needs testing.

### Bucketing Granularity
- **What's the right granularity for LoRA bucketing?** At 20–40 clips, aggressive bucketing creates buckets with 1–2 clips each.
- The "coin slot" approach (fill a bucket, start training when full) — how does this map to Wan's actual strategy?

### I2V Reference Divergence
- **What happens when the reference image ISN'T the first frame?** How far can you diverge before the model struggles?
- This matters for character LoRAs where you might want to condition on a character reference that isn't a video frame.

---

## Sources

- **Wan Technical Report**: [arXiv:2503.20314](https://arxiv.org/abs/2503.20314) — "Wan: Open and Advanced Large-Scale Video Generative Models"
- **Wan 2.2 GitHub**: [github.com/Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)
- **Wan 2.1 GitHub**: [github.com/Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1)
- **HuggingFace Model Cards**: [Wan-AI/Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B), [Wan-AI/Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)
- **EmergentMind Analysis**: [emergentmind.com/papers/2503.20314](https://www.emergentmind.com/papers/2503.20314)
- **Actual Model Configs**: From our downloaded models at `C:\Users\minta\Projects\dimljus\models\`
