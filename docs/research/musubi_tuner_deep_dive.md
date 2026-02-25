# Musubi-Tuner Data Loading: Deep Dive

**Date:** 2026-02-24
**Purpose:** Reference material for Dimljus Phase 6+ design
**Source:** kohya-ss/musubi-tuner codebase analysis

---

## Dataset Configuration (TOML)

Two-level structure with fallback chain:

```toml
[general]           # Shared defaults (optional)
resolution = [960, 544]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true

[[datasets]]        # One or more dataset blocks
video_directory = "/path/to/videos"
target_frames = [1, 25, 45]
frame_extraction = "head"
cache_directory = "/path/to/cache"

[[datasets]]        # Can mix image and video datasets
image_directory = "/path/to/images"
```

**Type detection is implicit**: `video_directory` present → VideoDataset. Otherwise → ImageDataset. No explicit `type` field.

**Parameter resolution**: dataset_config → general_config → argparse_config → runtime_params (first non-None wins).

---

## Three-Phase Workflow

The dataset pipeline runs THREE times, once per phase:

1. **`cache_latents.py`** — iterates data, encodes through VAE, saves to disk
2. **`cache_text_encoder_outputs.py`** — iterates captions, encodes through T5, saves to disk
3. **`train_network.py`** — loads cached `.safetensors` during training

This means VAE and T5 are NEVER loaded simultaneously with the DiT during training.

---

## Data Loading Pipeline

```
TOML file
  → ConfigSanitizer (voluptuous validation)
  → BlueprintGenerator (creates DatasetBlueprint objects)
  → generate_dataset_group_by_blueprint() (instantiates Dataset objects)
  → DatasetGroup (torch ConcatDataset wrapping ImageDataset/VideoDataset)
```

### The Blueprint Pattern

BlueprintGenerator creates parameter objects with the fallback chain resolved. Each dataset block becomes a DatasetBlueprint, which is then instantiated into an actual Dataset.

### Datasource Layer

```
ContentDatasource (base)
  ├── ImageDatasource
  │     ├── ImageDirectoryDatasource (globs from directory)
  │     └── ImageJsonlDatasource (reads from JSONL)
  └── VideoDatasource
        ├── VideoDirectoryDatasource (globs from directory)
        └── VideoJsonlDatasource (reads from JSONL)
```

Datasources return **fetcher functions** (lazy lambdas). The dataset submits these to a ThreadPoolExecutor for parallel loading.

### The ItemInfo Object

All sample data flows through `ItemInfo`:

```python
class ItemInfo:
    item_key: str           # unique identifier (file path stem)
    caption: str
    original_size: tuple    # (width, height) of source
    bucket_size: tuple      # assigned bucket resolution
    frame_count: int        # for video
    content: ndarray        # raw pixels (HWC or FHWC)
    control_content: ndarray/list  # control signal pixels
    latent_cache_path: str
    text_encoder_output_cache_path: str
```

---

## Video vs Image Handling

### Class Hierarchy

```
BaseDataset(torch.utils.data.Dataset)
  ├── ImageDataset
  └── VideoDataset
DatasetGroup(torch.utils.data.ConcatDataset)
```

**Images**: Single frame `(H, W, C)`, bucketed by `(width, height)`.

**Videos**: Loaded via PyAV as `(F, H, W, C)`. Bucketed by `(width, height, frame_count)` — frame count is part of the bucket key.

### Frame Sampling Strategies

Five modes for extracting training clips from source videos:

1. **head**: First N frames (fastest, most common)
2. **chunk**: Non-overlapping N-frame segments
3. **slide**: Sliding window with configurable stride
4. **uniform**: N evenly-spaced starting points
5. **full**: Entire video, trimmed to `N*4+1`

`target_frames` is a **list** (e.g., `[1, 25, 45]`), generating **multiple training samples per video** at different frame counts. All forced to `N*4+1` for VAE compatibility.

### FPS Conversion

If `source_fps` specified, frame-skipping to match model target (16 for Wan). Mechanical (every Nth frame), not content-aware.

---

## Resolution Bucketing

`BucketSelector` generates buckets from a target area:

```
1. target_area = width * height (e.g., 960×544 = 522,240)
2. Generate candidate (w, h) pairs with w*h ≈ target_area
3. Both dimensions divisible by 16
4. Store (w, h) and (h, w) variants
5. Assign each sample to closest aspect ratio match
```

For video, bucket key = `(width, height, frame_count)`. Different frame lengths never share a bucket.

`BucketBatchManager` handles shuffling and iteration across all buckets. Supports optional timestep stratification.

---

## Latent Cache Format

All caches use **safetensors** with metadata.

**Naming**:
- Latent: `{stem}_{crop:05d}-{frames:03d}_{W}x{H}_{arch}.safetensors`
- Text encoder: `{stem}_{arch}_te.safetensors`

**Tensor keys** (self-describing):
```
latents_{F}x{H}x{W}_{dtype}          — target video latent
latents_image_{F}x{H}x{W}_{dtype}    — I2V conditioning (mask + first frame)
latents_control_{F}x{H}x{W}_{dtype}  — control signal latent
clip_{dtype}                           — CLIP visual embedding
varlen_t5_{dtype}                      — T5 text embedding (variable length)
```

**`varlen_` prefix** prevents stacking across batch dimension (text embeddings have variable token counts).

**Loading during training**: `BucketBatchManager.__getitem__()` loads safetensors on-the-fly, parses keys, strips dtype/dimension suffixes, accumulates per content_key. Regular keys: `torch.stack()`. `varlen_` keys: kept as list.

**Text encoder cache merging**: If cache file exists, new tensors merge into it (preserving existing keys). Supports multi-encoder models.

**Cache cleanup**: After caching, files that exist on disk but aren't in the current dataset are deleted (unless `--keep_cache`).

---

## Wan I2V Reference Image Flow

The reference image is NOT a separate "control" signal — it's automatically extracted as the first frame during latent caching:

```
1. Extract first frame: contents[:, :, 0:1, :, :]
2. Encode through CLIP → clip_context
3. Create mask (1 for first frame, 0 for rest)
4. Zero-pad remaining frames, encode through VAE → y
5. Concatenate: [mask, y] → shape (4+C, F, H, W)
6. Save as part of latent cache
```

During training:
```python
model(noisy_input, t=timesteps, context=text_context,
      clip_fea=clip_fea, y=image_latents)
```

---

## Control Signal Handling

### Directory-Based Pairing

```toml
[[datasets]]
video_directory = "/path/to/videos"
control_directory = "/path/to/controls"
```

Stem matching: `video1.mp4` pairs with `control1.mp4`. Multiple controls via numbered suffixes: `control1_0.png`, `control1_1.png`.

Control images can have independent resolution (`control_resolution` setting, `no_resize_control` flag).

### Wan 2.2 MoE Handling

State-swapping approach (NOT routing):
- Load BOTH expert weight files into memory
- For each batch, sample timestep, check against boundary (0.875)
- Swap entire model state_dict if needed
- Inactive weights go to CPU when `offload_inactive_dit` is set
- Single LoRA shared between both experts (same rank, same alpha)

**Key limitation for Dimljus**: musubi does NOT support per-expert hyperparameters. Same LoRA rank and learning rate for both experts. Dimljus's differential MoE requires a fundamentally different architecture.

---

## What Musubi Does NOT Do (Dimljus Opportunities)

1. No temporal coherence validation (trusts user)
2. No concept of control signal types (generic `control_directory`)
3. No differential MoE training in data pipeline
4. No caption validation or quality scoring
5. No dataset provenance/metadata (no manifest)
6. No incremental caching (all-or-nothing with `--skip_existing`)
7. No per-sample control signal configuration
8. No hash-based cache invalidation
