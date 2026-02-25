# Multi-Format Data Loading for Video Diffusion Training: Ecosystem Research Report

**Date:** 2026-02-24
**Purpose:** Phase 5-7 design input for Dimljus training infrastructure
**Scope:** How the diffusion model training ecosystem handles dataset loading, latent caching, resolution bucketing, and control signal routing — with focus on video LoRA training.

---

## Executive Summary

After surveying the five major open-source video training frameworks (HuggingFace diffusers examples, HuggingFace finetrainers, kohya-ss/musubi-tuner, ostris/ai-toolkit, and VideoX-Fun/WanControl), several clear patterns emerge:

1. **Pre-caching latents and text embeddings to disk is universal.** Every production framework does this. The dominant format is safetensors with resolution-encoded filenames. On-the-fly encoding exists only as a fallback for very large datasets that won't fit cached.

2. **Resolution bucketing is universal but implemented differently.** All frameworks group samples by resolution to avoid padding waste. The approaches range from fixed bucket lists (CogVideoX) to aspect-ratio-matched dynamic bucketing (musubi-tuner, ai-toolkit).

3. **Mixed image/video training is supported but poorly abstracted.** Most frameworks bolt video onto image training rather than designing for it natively. Only finetrainers and musubi-tuner have first-class mixed support.

4. **Control signal routing is the least mature area.** Each framework has its own ad-hoc approach. There is no established pattern for extensible multi-signal conditioning — this is a genuine gap Dimljus can fill.

5. **Map-style datasets dominate for cached data; iterable datasets for streaming.** When latents are pre-cached to disk, map-style `Dataset` with `__getitem__` is the norm. Iterable datasets are used only for streaming from remote sources.

---

## 1. HuggingFace Diffusers Training Examples

### Architecture

The official diffusers examples (particularly `train_cogvideox_lora.py`) use a straightforward approach:

```python
class VideoDataset(Dataset):
    def __init__(self, instance_data_root, dataset_name, caption_column,
                 video_column, height, width, fps, max_num_frames, ...):
        # Two loading paths:
        # 1. HuggingFace Hub via datasets.load_dataset()
        # 2. Local files via text file lists (videos.txt + prompts.txt)
```

**Key patterns:**
- Fixed resolution: all videos resized to a single target (e.g., 480x720). No bucketing.
- Frame count fixed: all videos padded to `4k+1` frames for VAE compatibility.
- In-memory latent caching: videos are VAE-encoded once before training and stored as `latent_dist` objects in a list, replacing the raw frames in `train_dataset.instance_videos`.
- Simple collation: `collate_fn` samples from cached latent distributions and stacks.

**Limitations:**
- No resolution bucketing (everything resized to one target).
- No mixed image/video support.
- Latent cache lives in GPU memory, not on disk — limits dataset size.
- No control signal handling.
- No caption dropout implementation in the dataset layer.

**Assessment:** Good reference implementation for understanding the basic flow, but too simple for production use. Dimljus should NOT follow this pattern — it doesn't scale.

### HuggingFace finetrainers (formerly cogvideox-factory)

This is the more mature HuggingFace offering, actively maintained by a-r-r-o-w. Significantly more sophisticated:

**Dataset architecture:**
```
IterableDatasetPreprocessingWrapper
    ├── ImageCaptionFilePairDataset  (image.jpg + image.txt pairs)
    ├── VideoCaptionFilePairDataset  (video.mp4 + video.txt pairs)
    ├── ImageFileCaptionFileListDataset  (images.txt + captions.txt)
    ├── VideoFileCaptionFileListDataset  (videos.txt + captions.txt)
    ├── ImageFolderDataset  (HF imagefolder builder)
    ├── VideoFolderDataset  (HF videofolder builder)
    ├── ImageWebDataset  (streaming from Hub)
    └── VideoWebDataset  (streaming from Hub)
```

**Key innovations:**
- `IterableCombinedDataset` merges multiple datasets with weighted sampling via buffer-based interleaving.
- `ResolutionSampler` groups samples by dimension tuple for uniform-resolution batches.
- Two-mode precomputation: `InMemoryDistributedDataPreprocessor` (small datasets, keep in RAM) vs `PrecomputedDistributedDataPreprocessor` (large datasets, serialize to `.pt` files).
- Control signal handling via `IterableControlDataset` with pluggable processors (Canny, identity copy).
- Frame conditioning strategies: INDEX, PREFIX, RANDOM, FIRST_AND_LAST, FULL — for masking which latent frames are conditioning vs target.

**Precomputation details:**
- Cache files: `{data_type}-{index}.pt` using `torch.save()`.
- Supports `enable_reuse` to skip reprocessing on restart.
- Processor functions are passed as a dictionary keyed by data type — the precomputer is model-agnostic.
- Caption dropout is handled externally (not in the dataset layer).

**Limitations:**
- Iterable-only design makes random access and deterministic resumption harder.
- Resolution bucketing is simple: nearest-bucket resize, not dynamic area-based bucketing.
- Control routing is limited to single control type per training run (Canny OR copy, not both).
- `.pt` files are less efficient than safetensors (no memory mapping, no metadata).

**Assessment:** Good architecture for streaming/distributed training. The processor-function pattern for precomputation is elegant and worth adopting. The control signal handling is a good starting point but too rigid for Dimljus's multi-signal vision.

---

## 2. PyTorch Dataset/DataLoader Patterns for Video

### Map-Style vs Iterable-Style

The ecosystem has converged on a clear division:

| Use Case | Dataset Type | Rationale |
|----------|-------------|-----------|
| Pre-cached latents on disk | Map-style (`Dataset`) | Random access via `__getitem__`, deterministic epoch ordering, easy resumption |
| Streaming from remote/huge datasets | Iterable (`IterableDataset`) | Lazy loading, no need to know dataset size upfront |
| Cached latents + bucketing | Map-style with custom `BatchSampler` | Bucket assignment needs random access; sampler controls batch composition |

**Recommendation for Dimljus:** Map-style dataset for Phase 6+ (latent caching). The entire Dimljus philosophy is "validate before you compute" — we know the dataset size, structure, and contents before training starts. Map-style gives us deterministic resumption, easier debugging, and simpler bucketing.

### Variable-Length Video Sequences

Three approaches exist in the wild:

1. **Fixed frame count (CogVideoX approach):** All videos padded/trimmed to one target length. Simple but wasteful — short clips get padded with zeros, long clips lose information.

2. **Frame-count bucketing (musubi-tuner approach):** `target_frames = [1, 25, 45]` — videos are assigned to the nearest valid frame count bucket. Combined with resolution bucketing, each batch has uniform `(F, H, W)`. This is the dominant production approach.

3. **Token-length bucketing (VideoX-Fun approach):** `--training_with_video_token_length` — bucket by total token count (F * H * W in latent space) rather than individual dimensions. More flexible but harder to implement.

**Recommendation for Dimljus:** Frame-count bucketing (approach 2). It's proven, compatible with musubi-tuner for comparison, and natural for Wan's `4n+1` frame constraint. Token-length bucketing is an interesting extension for later.

### Resolution Bucketing Algorithms

Two main algorithms exist:

**Area-based bucketing (musubi-tuner):**
```
1. Compute target area = width * height from config resolution
2. Generate candidate buckets:
   - sqrt_size = sqrt(area)
   - Iterate widths from sqrt_size/2 to sqrt_size
   - For each width, compute height = area / width
   - Round both to nearest multiple of resolution_step (e.g., 16px)
   - Store (w, h) and (h, w)
3. For each image: find bucket with closest aspect ratio
```

**Resolution-list bucketing (ai-toolkit, finetrainers):**
```
1. User provides list of target resolutions: [512, 768, 1024]
2. For each image: find nearest bucket by area/aspect ratio
3. Resize (downscale only if bucket_no_upscale=true)
```

**Key constraint for video:** Bucketing must include the temporal dimension. Musubi-tuner keys buckets as `(W, H, F)` tuples. A batch might be `(640, 480, 25)` — all items in that batch have the same spatial resolution AND frame count.

**Recommendation for Dimljus:** Area-based bucketing with frame count as a third dimension, matching musubi-tuner's approach. This gives us maximum dataset utilization while ensuring batches are uniform.

### Batch Collation

When all items in a batch share dimensions (because of bucketing), collation is trivial — just `torch.stack()`. The complexity moves to the sampler, not the collate function.

musubi-tuner's `BucketBatchManager` handles this elegantly:
1. Assign all items to buckets at epoch start
2. Shuffle within each bucket
3. Create batch indices: `[(bucket_reso, batch_idx), ...]`
4. Shuffle batch order across all buckets
5. `__getitem__` loads and stacks items from the selected bucket/batch

For variable-length keys (like text embeddings with different token counts), musubi-tuner uses a `varlen_` prefix convention — these keys are kept as lists rather than stacked.

---

## 3. Latent Caching Strategies

### Universal Pattern: Two-Phase Caching

Every production framework separates caching into two phases, each with its own script:

1. **Latent caching** (VAE encoding): `cache_latents.py`
   - Input: raw images/videos + dataset config
   - Output: VAE-encoded tensors saved to disk
   - Requires: VAE model loaded on GPU

2. **Text encoder output caching**: `cache_text_encoder_outputs.py`
   - Input: caption text files + dataset config
   - Output: text embeddings saved to disk
   - Requires: text encoder(s) loaded on GPU

This separation allows the VAE and text encoder to be completely unloaded during training, freeing massive amounts of VRAM. For Wan 2.1/2.2, the T5 encoder alone is ~10GB.

### Cache File Formats

**musubi-tuner (safetensors — recommended):**
```
Filename: {stem}_{F}x{H}x{W}_{dtype}.safetensors
Contents:
  - latents_{latent_F}x{latent_H}x{latent_W}_{dtype}  (tensor)
  - latents_flipped_...  (optional, for flip augmentation)
  - alpha_mask_...  (optional)
  - Metadata JSON: { architecture, width, height, crop_ltrb, format_version }

Text encoder outputs (separate file):
  - varlen_vl_embed_{dtype}  (variable-length text embeddings)
  - clip_l_pooler_{dtype}  (pooled CLIP features)
  - llama_attention_mask  (attention mask)
```

Benefits:
- Memory-mappable (safetensors supports zero-copy reading)
- Self-describing (metadata embedded)
- Resolution encoded in filename (fast filtering without opening files)
- Multi-resolution support (multiple latent tensors per file)
- Type hierarchy (fp32 > bf16 > fp8, with automatic cleanup)

**finetrainers (PyTorch .pt files):**
```
Filename: {data_type}-{index}.pt
Contents: dictionary of preprocessed tensors (via torch.save)
```

Simpler but worse: no memory mapping, no metadata, pickle-based (security concerns), no multi-resolution support.

**ai-toolkit (safetensors with mixin pattern):**
```
Latent cache: .safetensors files in dataset directory
Text embeddings: .text_embeds files
CLIP vision: .clip_vision files
Size metadata: .aitk_size.json (dimensions cache)
```

Uses lazy loading — cached files aren't loaded until `__getitem__` is called.

**WanControl (PyTorch .pth files):**
```
Filename: {stem}.tensors.pth
Contents: all preprocessed tensors for one sample
```

Simplest approach but least efficient.

### Cache Invalidation

This is handled poorly across the ecosystem:

- **musubi-tuner:** Implicit — if you change resolution or augmentation settings, you must manually delete the cache directory and re-run caching. The filename encodes resolution and dtype, so different resolutions coexist.
- **finetrainers:** `enable_reuse` flag skips re-caching. No automatic invalidation.
- **ai-toolkit:** Dimension metadata in `.aitk_size.json` enables some validation, but no hash-based invalidation.
- **CogVideoX:** Documentation says "delete cached latent directories" after dataset changes.

**Recommendation for Dimljus:** Implement content-based cache invalidation. Store a hash of the source file (or its mtime + size) in the cache metadata. On load, verify the source hasn't changed. This is a genuine improvement over the ecosystem — Dimljus's "validate before you compute" philosophy should extend to caches.

### Memory-Mapped Files

The research shows memory-mapped approaches are significantly faster for large datasets:
- `numpy.load(path, mmap_mode='r')` — 175x speedup over standard loading for COCO-scale datasets
- safetensors supports zero-copy memory mapping natively
- PyTorch's `torch.load(path, mmap=True)` is available but less commonly used

For Dimljus, safetensors with memory mapping is the clear winner:
- Zero-copy reading (no deserialization overhead)
- Safe (no arbitrary code execution)
- Metadata support
- Widely adopted in the ecosystem (compatibility with musubi-tuner caches)

---

## 4. Control Signal Routing Patterns

### The Landscape

Control signal handling is the most fragmented area across frameworks. Here's how each does it:

#### ControlNet Pattern (foundational)
```
Input image → Preprocessor (Canny/depth/pose) → Encoder (copy of UNet encoder)
    → Zero convolutions → Inject into UNet decoder at multiple scales
```
- Separate trainable copy of encoder blocks
- Zero-initialized connections (safe to start from)
- Each control type has its own preprocessor + ControlNet weights
- Composable: multiple ControlNets can be applied simultaneously

#### Flux Tools / Channel Concatenation Pattern
```
Control signal → VAE encode → Concatenate with noisy latent in channel dimension
    → Expanded patch embedding (doubled input channels) → Standard transformer
```
- No separate encoder — control information enters through the main pathway
- Requires expanding the first layer's input channels (zeroed new channels)
- Simpler architecture but less modular
- This is what Wan I2V uses for reference images

#### IP-Adapter / Cross-Attention Pattern
```
Reference image → CLIP image encoder → Separate cross-attention layers
    → Added to text cross-attention outputs
```
- Frozen image encoder (CLIP ViT-H typically)
- New cross-attention layers trained alongside frozen UNet
- Decoupled from text conditioning (separate key/value projections)
- Lightweight: ~22M trainable parameters

#### VACE / Context Block Pattern (Wan 2.1)
```
Control signals → VCU (Video Conditioning Unit) → Context tokens
    → Context Blocks (parallel to DiT blocks) → Additive injection into DiT
```
- Pre-trained DiT weights frozen
- Separate lightweight Context Blocks process control tokens
- Context representations injected additively into DiT layers
- Res-Tuning architecture: faster convergence, preserves base capabilities
- Single unified model handles multiple tasks (inpaint, outpaint, edit, reference)

### Per-Sample Control Signal Specification

**WanControl (CSV-based):**
```csv
file_name,text,control_name
video_00001.mp4,"description",video_00001_c.mp4
```
Simple pairing via `_c` suffix convention. One control per sample.

**musubi-tuner (directory-based):**
```toml
[[datasets]]
video_directory = "/path/to/videos"
control_directory = "/path/to/controls"
```
File matching by stem name: `video1.mp4` pairs with `control1.mp4`. Multiple controls via `control1_0.png`, `control1_1.png`. Resolution can differ from target (`control_resolution` setting).

**finetrainers (integrated in dataset class):**
```python
class IterableControlDataset:
    def __init__(self, dataset, control_type="Canny"):
        # Processes each sample through control_type processor
        # Outputs "control_image" or "control_video" key
```
Single control type per dataset. Control is generated on-the-fly from the training data itself (e.g., Canny edges from the target image).

**VideoX-Fun (JSON metadata):**
```json
{"video_path": "video.mp4", "text": "caption", "control_path": "control.mp4"}
```
Per-sample control specification in metadata. Supports training modes: `normal` (T2V), `inpaint` (I2V with mask), `control` (ControlNet-style).

### Encoder Dispatch/Routing

No framework has a clean abstraction for this. The typical pattern is:

```python
# Pseudo-code from finetrainers control trainer
if has_control_image:
    control_latent = vae.encode(control_image)
    # Channel concat: [noisy_latent; control_latent] along channel dim
    # OR: separate ControlNet forward pass
    # OR: VACE context token injection
```

The routing is hardcoded per model backend, not abstracted.

### Training-Time Augmentation of Control Signals

**finetrainers' frame conditioning** is the most interesting pattern:
```python
def apply_frame_conditioning_on_latents(latents, condition_type):
    # INDEX: keep single frame at position
    # PREFIX: retain first N frames (random N)
    # RANDOM: preserve random subset
    # FIRST_AND_LAST: keep boundary frames only
    # FULL: retain all frames
    mask = compute_mask(condition_type, num_frames)
    return latents * mask  # Zero out non-conditioning frames
```

This is used to train models that can condition on partial frame sequences — essential for video continuation/editing tasks.

**Caption dropout** is universal:
- Implemented by replacing text embeddings with zero/null embeddings at a configurable rate (typically 5-10%)
- Enables classifier-free guidance at inference
- All frameworks support this; most implement it in the training loop, not the dataset

**Recommendation for Dimljus:** This is where Dimljus can differentiate. Design a `ControlSignalSpec` abstraction:

```python
@dataclass
class ControlSignalSpec:
    signal_type: str            # "reference_image", "depth_map", "caption", etc.
    encoder: str                # "vae", "clip", "t5", "identity"
    injection_method: str       # "channel_concat", "cross_attention", "context_block"
    dropout_rate: float         # Per-signal dropout
    weight: float               # Conditioning scale

    # Per-sample: what file/data provides this signal
    source_path: Path | None
```

This makes control signals truly first-class, extensible, and independently configurable.

---

## 5. CogVideoX and Video Diffusion Training Codebases

### CogVideoX (official + diffusers)

**Dataset format:**
```
dataset/
├── prompts.txt       # One caption per line
├── videos/           # MP4 files
├── videos.txt        # Filename list
├── images/           # Optional I2V reference images
└── images.txt        # Optional filename list
```

**Key characteristics:**
- Fixed resolution: all videos resized to target (480x720 or 768x1360)
- Fixed frame count: must be `8N+1` (e.g., 49 frames)
- Auto-caching: framework encodes and caches latents before training
- No bucketing: single resolution target
- If no reference image provided, first frame extracted automatically (I2V)

**Data flow:**
```
Video files → decord frame extraction → resize → normalize [-1, 1]
    → VAE encode (in-memory) → Training loop
```

Simple and functional but rigid. Not suitable as a pattern for Dimljus.

### VideoX-Fun / Wan2.1-Fun

**Dataset format:**
```json
{"video_path": "path.mp4", "text": "caption"}
{"video_path": "path.mp4", "text": "caption", "control_path": "control.mp4"}
```

**Key innovations:**
- `--random_hw_adapt`: automatic resolution adaptation for mixed image/video
- `--training_with_video_token_length`: bucket by total token count, not individual dimensions
- `--enable_bucket`: aspect-ratio bucketing
- `--video_sample_stride`: frame subsampling for temporal resolution control
- Separate training scripts for each mode: `train.py`, `train_lora.py`, `train_control.py`, `train_control_lora.py`

**Assessment:** Good production framework for Wan training specifically, but tightly coupled to its own infrastructure. The token-length bucketing idea is worth considering for Dimljus long-term.

### WanControl (shalfun)

**Dataset format:**
```
data/example_dataset/
├── metadata.csv                  # file_name, text, control_name
└── train/
    ├── video_00001.mp4           # Target video
    ├── video_00001_c.mp4         # Control video (paired by _c suffix)
    └── ...
```

**Preprocessing pipeline:**
```bash
python train_wan_t2v.py --task data_process \
    --dataset_path data/example_dataset \
    --text_encoder_path "path/to/t5.pth" \
    --vae_path "path/to/wan_vae.pth"
```

Generates `.tensors.pth` files per sample containing all pre-encoded data.

**Assessment:** Clean pairing model (CSV with explicit control_name column). Simple and practical. The `_c` suffix convention is crude but effective for single-control scenarios.

---

## Synthesis: Recommendations for Dimljus

### Data Loading Architecture

```
Phase 6 (Latent Pre-Encoding):
    dimljus cache-latents --config dataset.yaml --vae path/to/vae
    dimljus cache-text   --config dataset.yaml --text-encoder path/to/t5

Phase 7 (Training):
    Map-style Dataset reads from cache directory
    BucketBatchSampler groups by (W, H, F) tuples
    Simple collate_fn (torch.stack, since bucket ensures uniform shapes)
    Variable-length text embeddings handled via padding + attention masks
```

### Cache Format: safetensors (following musubi-tuner)

```
cache/
├── {stem}_{F}x{H}x{W}_{dtype}.safetensors    # Video latents
├── {stem}_text_{encoder}.safetensors           # Text embeddings
├── {stem}_control_{type}_{F}x{H}x{W}.safetensors  # Control latents
└── cache_manifest.json                          # Hash-based invalidation
```

**Why safetensors:**
- Memory-mappable (zero-copy reads)
- Safe (no pickle vulnerabilities)
- Metadata support (store source hash, encoding params)
- Ecosystem compatibility (musubi-tuner caches can be read/compared)
- Multi-tensor per file (latents + masks + metadata in one file)

**Dimljus addition: `cache_manifest.json`**
```json
{
    "format_version": "1.0.0",
    "vae_hash": "sha256:abc123...",
    "text_encoder_hash": "sha256:def456...",
    "samples": {
        "clip_001": {
            "source_path": "clips/clip_001.mp4",
            "source_mtime": 1740412800.0,
            "source_size": 15728640,
            "source_hash": "sha256:...",
            "latent_file": "clip_001_21x80x60_bf16.safetensors",
            "text_file": "clip_001_text_t5.safetensors",
            "control_files": {
                "reference_image": "clip_001_control_ref_1x80x60_bf16.safetensors"
            }
        }
    }
}
```

This enables:
- Automatic cache invalidation when source files change
- Verification that the right VAE/encoder was used
- Incremental re-caching (only changed samples)
- Manifest as single source of truth for training

### Resolution Bucketing

Follow musubi-tuner's area-based approach:

```python
class BucketSelector:
    def __init__(self, target_resolution, frame_counts, reso_step=16):
        """
        target_resolution: (W, H) base resolution
        frame_counts: [1, 17, 33, 49, 81] valid frame counts (4n+1 for Wan)
        reso_step: pixel alignment (16 for Wan VAE)
        """
        self.target_area = target_resolution[0] * target_resolution[1]
        self.frame_counts = frame_counts
        self.reso_step = reso_step
        self.spatial_buckets = self._generate_spatial_buckets()

    def assign(self, width, height, num_frames):
        """Returns (bucket_w, bucket_h, bucket_f) tuple."""
        spatial = self._nearest_spatial_bucket(width / height)
        temporal = self._nearest_frame_count(num_frames)
        return (*spatial, temporal)
```

### Mixed Image/Video Training

Treat images as single-frame videos:

```python
# Image: tensor shape (C, 1, H, W) — one frame
# Video: tensor shape (C, F, H, W) — F frames

# Bucket key: (W, H, F)
# Images go into (W, H, 1) buckets
# Videos go into (W, H, F) buckets
# Never mixed in the same batch (different F dimensions)
```

This is how musubi-tuner handles it. Simple, clean, no special cases.

### Control Signal Routing

Dimljus should implement a registry-based pattern:

```python
class ControlSignalRegistry:
    """Registry of control signal types and their encoding pipelines."""

    def register(self, signal_type: str, encoder: ControlEncoder):
        """Register a new control signal type."""

    def encode(self, signal_type: str, data: Any) -> torch.Tensor:
        """Encode a control signal through its registered encoder."""

class ControlEncoder(Protocol):
    """Protocol for control signal encoders."""

    def encode(self, data: Any) -> torch.Tensor: ...
    def cache_key(self) -> str: ...  # For cache filename generation
    def injection_point(self) -> str: ...  # Where in the model this injects

# Built-in encoders:
class VAEEncoder(ControlEncoder):
    """Encodes images/video through VAE (for I2V reference, depth maps)."""

class CLIPEncoder(ControlEncoder):
    """Encodes images through CLIP (for IP-Adapter style conditioning)."""

class T5Encoder(ControlEncoder):
    """Encodes text through T5 (for captions)."""

class IdentityEncoder(ControlEncoder):
    """Pass-through for pre-encoded signals."""
```

This gives Dimljus:
- Extensible: add new control types by implementing `ControlEncoder`
- Per-signal caching: each control type generates its own cache files
- Per-signal dropout: configurable in the training config
- Per-signal weighting: configurable conditioning scale
- Clear separation: the dataset knows about files, the encoder knows about tensors

### Caption Dropout

Implement at the training loop level (not dataset level), following ecosystem convention:

```python
# During training step:
if random.random() < config.caption_dropout_rate:
    text_embedding = torch.zeros_like(text_embedding)  # Null conditioning
```

This is simpler and more flexible than dataset-level dropout. It also enables different dropout rates for different control signals independently.

### Per-Sample Configuration

Dimljus's data architecture already defines per-sample control signal settings in the dataset config. For the training pipeline, this should manifest as:

```yaml
# In the training sample metadata
samples:
  clip_001:
    target: clips/clip_001.mp4
    controls:
      reference_image:
        path: stills/clip_001_ref.png
        weight: 1.0
        dropout: 0.0
      caption:
        path: captions/clip_001.txt
        weight: 1.0
        dropout: 0.05
      depth_map:
        path: depth/clip_001_depth.mp4
        weight: 0.5
        dropout: 0.1
```

No other framework supports this level of per-sample, per-signal granularity. This is Dimljus's key differentiator.

---

## Key Trade-Off Decisions

### 1. Pre-cache Everything vs On-the-Fly Encoding

| Approach | Pros | Cons |
|----------|------|------|
| **Pre-cache all** (musubi, ai-toolkit) | Fast training, minimal VRAM, deterministic | Disk space, cache invalidation, preprocessing step |
| **On-the-fly** (some finetrainers modes) | No disk overhead, always fresh | Slow training, needs VAE in VRAM during training |
| **Hybrid** (finetrainers precompute) | Flexible, handles large datasets | Complex implementation |

**Dimljus choice:** Pre-cache everything, with hash-based invalidation. Minta's workflow is iterative (prepare data, validate, train, adjust). Pre-caching fits this perfectly — you cache once, train many times with different hyperparameters. The disk space cost is acceptable (Wan-VAE compresses ~4x temporally and ~8x spatially per dimension).

### 2. safetensors vs .pt vs Custom Format

| Format | Memory Map | Metadata | Security | Ecosystem |
|--------|-----------|----------|----------|-----------|
| safetensors | Yes (zero-copy) | Yes (JSON) | Safe | musubi-tuner, HF |
| .pt (torch.save) | Limited | No | Unsafe (pickle) | finetrainers |
| .npy (numpy) | Yes (mmap) | No | Safe | Research |
| Custom binary | Possible | Custom | Custom | None |

**Dimljus choice:** safetensors. It wins on every axis and provides compatibility with musubi-tuner caches for comparison during validation.

### 3. Map-Style vs Iterable-Style Dataset

| Style | Random Access | Resumption | Bucketing | Distributed |
|-------|--------------|------------|-----------|-------------|
| Map-style | Yes | Easy (save index) | Natural (sampler-based) | DistributedSampler |
| Iterable | No | Hard (save state) | Unnatural (buffer-based) | Manual sharding |

**Dimljus choice:** Map-style. Pre-cached data is inherently finite and indexed. Bucketing requires knowing all items upfront to assign buckets. Deterministic training (same seed = same batches) is valuable for debugging and comparison.

### 4. One Cache File Per Sample vs Consolidated Files

| Approach | Pros | Cons |
|----------|------|------|
| **Per-sample** (musubi, ai-toolkit) | Easy incremental updates, simple invalidation | Many small files, filesystem overhead |
| **Consolidated** (finetrainers batched .pt) | Fewer files, potentially faster sequential read | All-or-nothing invalidation, harder to update |

**Dimljus choice:** Per-sample safetensors files. Matches musubi-tuner compatibility, enables incremental re-caching, and simplifies the cache manifest. Filesystem overhead is negligible for typical training dataset sizes (tens to hundreds of samples, not millions).

---

## Summary Table: Framework Comparison

| Feature | diffusers examples | finetrainers | musubi-tuner | ai-toolkit | VideoX-Fun |
|---------|-------------------|-------------|-------------|------------|------------|
| Video support | Basic | Good | Excellent | Good | Excellent |
| Mixed image/video | No | Yes | Yes | Yes (limited) | Yes |
| Resolution bucketing | No | Basic | Full (area-based) | Full (list-based) | Full + token-length |
| Frame count bucketing | Fixed | Fixed per config | Multi-target | Single target | Stride-based |
| Latent caching | In-memory only | .pt files or in-memory | safetensors | safetensors | .pth files |
| Text embedding cache | No | .pt files | safetensors | .text_embeds | .pth files |
| Cache invalidation | N/A | Reuse flag | Manual | Implicit | Manual |
| Control signals | None | Single type (Canny/copy) | Directory-based pairs | Vision encoder cache | CSV/JSON pairs |
| Multiple controls | No | No | Yes (numbered suffix) | CLIP vision only | No |
| Caption dropout | No (in collate) | External | Training loop | Config param | Training loop |
| Per-sample config | No | No | No (per-dataset only) | No | JSON metadata |
| Dataset type | Map-style | Iterable | Map-style (custom) | Map-style | Map-style |
| Wan support | No | Yes (2.1) | Yes (2.1 + 2.2) | Yes (2.1) | Yes (2.1) |
| MoE awareness | No | No | No | No | No |

**Dimljus's opportunity:** No existing framework supports per-sample multi-signal configuration, hash-based cache invalidation, MoE-aware data loading (different training parameters per expert), or a clean control signal registry. These are all genuine gaps that align with Dimljus's design philosophy.

---

## Sources

### Primary Codebases Analyzed
- HuggingFace diffusers: https://github.com/huggingface/diffusers/tree/main/examples
- HuggingFace finetrainers: https://github.com/huggingface/finetrainers
- kohya-ss/musubi-tuner: https://github.com/kohya-ss/musubi-tuner
- ostris/ai-toolkit: https://github.com/ostris/ai-toolkit
- VideoX-Fun: https://github.com/aigc-apps/VideoX-Fun
- WanControl: https://github.com/shalfun/WanControl
- CogVideo: https://github.com/zai-org/CogVideo

### Documentation and Specifications
- musubi-tuner dataset config: https://github.com/kohya-ss/musubi-tuner/blob/main/docs/dataset_config.md
- sd-scripts latent cache format: https://github.com/kohya-ss/sd-scripts/issues/1750
- finetrainers dataset docs: https://github.com/huggingface/finetrainers/blob/main/docs/dataset/README.md
- finetrainers Wan docs: https://github.com/huggingface/finetrainers/blob/main/docs/models/wan.md
- ai-toolkit dataset loading: https://deepwiki.com/ostris/ai-toolkit/3.1-dataset-loading-and-caching

### Architecture References
- ControlNet: https://arxiv.org/abs/2302.05543
- IP-Adapter: https://arxiv.org/abs/2308.06721
- VACE: https://openaccess.thecvf.com/content/ICCV2025/papers/Jiang_VACE_All-in-One_Video_Creation_and_Editing_ICCV_2025_paper.pdf
- Classifier-Free Guidance: https://arxiv.org/abs/2207.12598
- Flux Tools depth control: https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux
