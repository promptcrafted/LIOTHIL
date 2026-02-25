# ai-toolkit (ostris) Data Loading: Deep Dive

**Date:** 2026-02-24
**Purpose:** Reference material for Dimljus Phase 6+ design
**Source:** ostris/ai-toolkit codebase analysis

---

## Dataset Configuration

Plain Python class with `kwargs.get()` defaults (no Pydantic, no schema):

```python
class DatasetConfig:
    def __init__(self, **kwargs):
        self.type = kwargs.get('type', 'image')
        self.folder_path = kwargs.get('folder_path', None)
        self.caption_ext = kwargs.get('caption_ext', '.txt')
        self.resolution = kwargs.get('resolution', 512)
        self.num_frames = kwargs.get('num_frames', 1)
        self.cache_latents_to_disk = kwargs.get('cache_latents_to_disk', False)
        # ~80 more fields
```

**YAML config example**:
```yaml
datasets:
  - folder_path: "/path/to/images/or/video/folder"
    caption_ext: "txt"
    caption_dropout_rate: 0.05
    num_frames: 1
    resolution: [512, 768, 1024]
```

**Multi-resolution splitting**: `resolution: [512, 768, 1024]` creates THREE separate DatasetConfig objects → three AiToolkitDataset instances → ConcatDataset. Not dynamic bucketing — separate datasets per base resolution.

**No validation**: Misspelled keys silently get default values.

---

## Data Loading Pipeline

### Mixin-Based Architecture

Main dataset class uses massive multiple inheritance:

```python
class AiToolkitDataset(
    LatentCachingMixin, ControlCachingMixin, CLIPCachingMixin,
    TextEmbeddingCachingMixin, BucketsMixin, CaptionMixin, Dataset
):
```

Per-sample data container:
```python
class FileItemDTO(
    LatentCachingFileItemDTOMixin,
    TextEmbeddingFileItemDTOMixin,
    CaptionProcessingDTOMixin,
    ImageProcessingDTOMixin,
    ControlFileItemDTOMixin,
    InpaintControlFileItemDTOMixin,
    ClipImageFileItemDTOMixin,
    MaskFileItemDTOMixin,
    AugmentationFileItemDTOMixin,
    UnconditionalFileItemDTOMixin,
    PoiFileItemDTOMixin,
    ArgBreakMixin,  # stops super().__init__() chains
):
```

### Data Flow Per Epoch

1. `__init__`: Walk directory, create `FileItemDTO` per file, read dimensions
2. `setup_epoch()`:
   - `setup_buckets()` — assign files to aspect-ratio buckets
   - `cache_latents_all_latents()` — VAE-encode all, save to disk
   - `cache_text_embeddings()` — T5/CLIP encode all, save to disk
   - `setup_controls()` — generate depth/pose/line maps if requested
3. `__getitem__(item)`: Returns pre-collated batch from one bucket

**Key pattern**: The dataset returns pre-collated batches (not individual samples). PyTorch DataLoader configured with `batch_size=None`.

---

## Video vs Image Distinction

Controlled by a single flag:

```python
self.is_video = dataset_config.num_frames > 1
```

Different file extensions discovered based on this flag. **No mixed dataset support** — a folder is either all images or all videos.

### Frame Extraction

Two modes:
1. **`shrink_video_to_frames: true`** (default): Evenly distribute `num_frames` across entire video (subsampling)
2. **`shrink_video_to_frames: false`**: Use `fps` config for frame interval, random start position, consecutive extraction

Uses OpenCV `cv2.VideoCapture` with fallback logic for failed reads.

**Video augmentations explicitly blocked**: raises exception if augments configured for video.

---

## Encoder Routing

Driven by the **model class**, not the dataset:

```python
# Model provides encode methods:
model.encode_images(imgs)    # VAE encoding
model.encode_prompt(text)    # Text encoding
model.encode_audio(audio)    # Audio encoding (if supported)
```

**Device state management**: `set_device_state_preset('cache_latents')` moves VAE to GPU, everything else to CPU. After caching, `restore_device_state()` reverses.

**I2V first frame**: Encoded at cache time, stored as `first_frame_latent` key in the safetensors cache file.

---

## Latent Caching

### Format
Per-sample safetensors files in `_latent_cache/` subdirectory:

```python
state_dict = OrderedDict()
state_dict['latent'] = latent
state_dict['first_frame_latent'] = first_frame_latent  # I2V
state_dict['audio_latent'] = audio_latent  # audio models
```

Text embeddings in `_t_e_cache/`, CLIP vision in `_clip_vision_cache/`.

### Hash-Based Invalidation

Cache filenames include MD5 hash of encoding parameters:

```python
def get_latent_info_dict(self):
    return OrderedDict([
        ("filename", os.path.basename(self.path)),
        ("scale_to_width", self.scale_to_width),
        ("scale_to_height", self.scale_to_height),
        ("crop_x", self.crop_x), ("crop_y", self.crop_y),
        ("latent_space_version", self.latent_space_version),
    ])
# Hash = MD5(JSON(info_dict)), base64-encoded
```

Changing resolution/crop/model automatically invalidates cache.

### Loading Modes

1. **`cache_latents: true`**: Encode + keep in CPU memory (fastest, most RAM)
2. **`cache_latents_to_disk: true`**: Encode + save to disk, load on-demand
3. Both true: Save to disk AND keep in memory

Augmentations disable caching (must apply in pixel space).

---

## Control Signal Handling

**Four separate, non-unified pathways**:

### a) Control Images (depth/edge/pose)
- Config: `control_path` or `controls: ['depth', 'pose']`
- Auto-generation: runs Depth-Anything-V2, DWPose, TEED
- Cached in `_controls/` subfolder
- Multiple control paths supported (stacked tensors)

### b) CLIP Image (IP-Adapter)
- Config: `clip_image_path` or `clip_image_from_same_folder: true`
- Separate from control images
- Cached in `_clip_vision_cache/`
- Supports quad-image layouts (2x2 grids)

### c) Inpaint Images
- Config: `inpaint_path`
- RGBA images, alpha=0 = inpaint region
- Completely separate from control images

### d) Mask Images
- Config: `mask_path` or `alpha_mask: true`
- Black/white, white = higher loss weight

### e) First Frame (I2V)
- Config: `do_i2v: true`
- First frame VAE-encoded at cache time
- At training: `[noisy_latent(16ch) | mask(1ch) | first_frame(16ch)]` = 33ch

### f) Audio
- Extracted from video via torchaudio
- Cached as `audio_latent`

**No unified abstraction** — each is a hard-coded mixin. Adding new control type requires touching 5+ files.

---

## Wan 2.2 MoE Implementation

`DualWanTransformer3DModel` wraps two transformers, routes by timestep:

```python
def forward(self, hidden_states, timestep, ...):
    if timestep.float().mean().item() > self.boundary:
        t_name = "transformer_1"  # high noise
    else:
        t_name = "transformer_2"  # low noise
```

With `low_vram: true`, inactive transformer moved to CPU.

`switch_boundary_every: 10` alternates which expert trains every N steps. `trainable_multistage_boundaries` selects which experts to train. But NO per-expert hyperparameters.

---

## Key Takeaways for Dimljus

### What Works
- Hash-based cache invalidation (content-aware, no manual deletion)
- Device state presets (clean VRAM management during caching)
- Bucketed batching inside dataset (pre-collated batches)

### What Doesn't Work
- Mixin explosion (12-class inheritance, fragile cooperative init)
- No config validation (silent failures)
- No unified control signal abstraction
- No mixed image/video datasets
- Video augmentations blocked
- No differential MoE training support
