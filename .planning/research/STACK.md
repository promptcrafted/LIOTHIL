# Technology Stack: Video LoRA Inference Pipeline

**Project:** Dimljus - Inference Fix
**Researched:** 2026-02-26
**Focus:** How inference/sampling works in video LoRA trainers for Wan 2.2 dual-expert MoE models

## How the Inference Stack Works

This document maps the exact call sequences and state management that working inference pipelines use, organized by component. The goal is not to recommend a new stack -- Dimljus already has its stack. The goal is to document the precise mechanics so the noisy-grid inference bug can be diagnosed.

---

## 1. The Diffusers WanPipeline Denoising Loop (Step by Step)

**Source:** [diffusers pipeline_wan.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py)
**Confidence:** HIGH (direct source code)

Dimljus calls `pipeline(**kwargs)` which invokes `WanPipeline.__call__()`. Here is the exact sequence:

### Step 1: Prompt Encoding

```python
# If prompt_embeds are provided (pre-encoded), skip encoding.
# If raw text, encode via T5:
prompt_embeds, negative_prompt_embeds = self.encode_prompt(
    prompt=prompt,
    negative_prompt=negative_prompt,
    do_classifier_free_guidance=self.do_classifier_free_guidance,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
)
# Cast to transformer dtype
transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype
prompt_embeds = prompt_embeds.to(transformer_dtype)
if negative_prompt_embeds is not None:
    negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)
```

**Critical detail:** When `text_encoder=None` (as in Dimljus), you MUST pass `prompt_embeds` and `negative_prompt_embeds`. The pipeline casts them to the transformer's dtype.

### Step 2: Latent Initialization

```python
num_channels_latents = (
    self.transformer.config.in_channels
    if self.transformer is not None
    else self.transformer_2.config.in_channels
)
# Shape: [batch, channels, latent_frames, latent_h, latent_w]
num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
shape = (
    batch_size,
    num_channels_latents,
    num_latent_frames,
    int(height) // self.vae_scale_factor_spatial,
    int(width) // self.vae_scale_factor_spatial,
)
latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
```

**Critical detail:** Latents are initialized in **float32**, not in the model's dtype. The WanPipeline converts to transformer dtype only when feeding into the model.

### Step 3: Boundary Timestep Computation

```python
if self.config.boundary_ratio is not None:
    boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
else:
    boundary_timestep = None
```

With `boundary_ratio=0.5` and `num_train_timesteps=1000`, `boundary_timestep=500.0`. Scheduler timesteps are compared directly against this value.

### Step 4: Scheduler Timestep Setup

```python
self.scheduler.set_timesteps(num_inference_steps, device=device)
timesteps = self.scheduler.timesteps
```

With `FlowMatchEulerDiscreteScheduler(shift=5.0)` and 30 steps, this produces 30 timestep values descending from ~1000 to ~0. The shift parameter applies: `sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)`, which concentrates more steps at higher noise levels.

### Step 5: The Denoising Loop

```python
for i, t in enumerate(timesteps):
    # 5a. Select expert based on timestep vs boundary
    if boundary_timestep is None or t >= boundary_timestep:
        current_model = self.transformer        # HIGH-noise expert
        current_guidance_scale = guidance_scale
    else:
        current_model = self.transformer_2      # LOW-noise expert
        current_guidance_scale = guidance_scale_2

    # 5b. Cast latents to transformer dtype for model input
    latent_model_input = latents.to(transformer_dtype)

    # 5c. Expand timestep for batch
    timestep = t.expand(latents.shape[0])

    # 5d. Conditional prediction (positive prompt)
    with current_model.cache_context("cond"):
        noise_pred = current_model(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

    # 5e. Unconditional prediction (negative prompt) for CFG
    if self.do_classifier_free_guidance:
        with current_model.cache_context("uncond"):
            noise_uncond = current_model(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=negative_prompt_embeds,
                return_dict=False,
            )[0]
        # CFG formula
        noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

    # 5f. Scheduler step (Euler update: latents = latents + dt * model_output)
    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
```

**Critical details:**
- `cache_context` manages attention KV caching. Conditional and unconditional passes are separate.
- `do_classifier_free_guidance` is True when `guidance_scale > 1.0`.
- CFG requires TWO forward passes per step (conditional + unconditional).
- The scheduler operates on latents in float32 (the latents variable stays float32 throughout).

### Step 6: VAE Decoding with Latent Denormalization

```python
latents = latents.to(self.vae.dtype)
latents_mean = (
    torch.tensor(self.vae.config.latents_mean)
    .view(1, self.vae.config.z_dim, 1, 1, 1)
    .to(latents.device, latents.dtype)
)
latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
    1, self.vae.config.z_dim, 1, 1, 1
).to(latents.device, latents.dtype)
# Denormalize: inverse of (latents - mean) * (1/std)
latents = latents / latents_std + latents_mean
video = self.vae.decode(latents, return_dict=False)[0]
video = self.video_processor.postprocess_video(video, output_type=output_type)
```

**Critical detail:** The WanPipeline applies per-channel latent denormalization before VAE decode. This uses `latents_mean` and `latents_std` from the VAE config. If these values are missing or wrong (e.g., due to from_single_file loading wrong config), the VAE decode produces garbage.

---

## 2. How musubi-tuner Handles Inference

**Source:** [musubi-tuner wan_generate_video.py](https://github.com/kohya-ss/musubi-tuner/blob/main/src/musubi_tuner/wan_generate_video.py)
**Confidence:** MEDIUM (WebFetch + search, not direct code reading)

### Model Loading

- Two separate model instances for Wan 2.2: `models[0]` = high-noise, `models[-1]` = low-noise
- Loaded via custom `load_wan_model()`, not diffusers `from_single_file`
- LoRA **merged into base weights** before inference via `network.merge_to()`

### Scheduler

- `FlowUniPCMultistepScheduler(shift=1, use_dynamic_shifting=False)`
- Actual shift passed via `scheduler.set_timesteps(steps, shift=discrete_flow_shift)`
- T2V 480p uses shift=12.0 (much higher than diffusers default)

### Expert Switching

- Boundary check: `(t / 1000.0) >= timestep_boundary`
- When boundary crossed, swap the entire model object
- `--offload_inactive_dit`: moves inactive model to CPU between switches
- Single switch point during denoising (not per-step)

### CFG (Guidance Scale)

- Separate guidance per expert: `guidance_scale_high_noise` and `guidance_scale` (low noise)
- T2V defaults: high=4.0, low=3.0

### Training-Time Sampling

- Uses **low-noise expert only** (forces `self.next_model_is_high_noise = False`)
- No dual-expert switching during training samples
- Uses `sample_guide_scale[0]` (low noise guidance only)
- LoRA kept as wrapper (not merged) during training-time sampling

---

## 3. How ai-toolkit Handles Inference

**Source:** [ai-toolkit wan22](https://deepwiki.com/ostris/ai-toolkit/13-wan-2.2-advanced-features)
**Confidence:** MEDIUM (DeepWiki analysis + grep of local source copy)

### Model Structure

- `DualWanTransformer3DModel` wrapping two `WanTransformer3DModel` instances
- `transformer_1` = high noise, `transformer_2` = low noise
- Forward pass selects model based on mean timestep vs boundary

### LoRA for Inference

- **Merges LoRA into base weights** when all samples use same multiplier
- Unmerges after inference
- Split LoRA files: separate high-noise and low-noise checkpoints

### VAE Latent Normalization (Training Side)

```python
# After VAE encoding, ai-toolkit normalizes latents:
latents = vae.encode(images).latent_dist.sample()
latents_mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1)
latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1)
latents = (latents - latents_mean) * latents_std
```

This is the **inverse** of what WanPipeline does at decode time. Training with normalized latents means the diffusion model learns in normalized space. The pipeline denormalizes before VAE decode.

### Memory Management

- Aggressive component offloading: one transformer on GPU at a time
- Asynchronous CUDA streams for pipelined CPU/GPU transfer

---

## 4. The from_single_file Config Bug

**Source:** [diffusers#12329](https://github.com/huggingface/diffusers/issues/12329)
**Confidence:** HIGH (confirmed bug report, closed as resolved)

### The Problem

`WanTransformer3DModel.from_single_file()` can load the wrong config, inferring Wan 2.1 instead of Wan 2.2. This causes `image_dim` to be set when it should be `None`.

### The Fix

Pass explicit config:
```python
model = WanTransformer3DModel.from_single_file(
    "model.safetensors",
    config="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)
```

### Impact on Dimljus

The test3-inference.py script DOES pass explicit config for the high-noise expert but uses `subfolder="transformer"` for low-noise (should be `subfolder="transformer_2"`). The test_base_inference.py does NOT pass explicit config at all. The main inference.py (WanInferencePipeline) calls `from_single_file` without explicit config in its `_load_vae` path.

However, this bug primarily affects I2V models (image_dim setting) and the VAE. For T2V models, the transformer config differences between 2.1 and 2.2 are minimal -- but the VAE config could be affected, potentially missing latents_mean/latents_std values.

---

## 5. FlowMatchEulerDiscreteScheduler Shift Mechanics

**Source:** [diffusers scheduler docs](https://huggingface.co/docs/diffusers/en/api/schedulers/flow_match_euler_discrete)
**Confidence:** HIGH (official docs + source code)

### How Shift Works

```python
# Base sigmas: linearly spaced from 1.0 to 0.0
sigmas = np.linspace(1.0, 0.0, num_inference_steps)

# Apply shift transformation:
sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

# Convert to timesteps:
timesteps = sigmas * num_train_timesteps  # num_train_timesteps = 1000
```

With `shift=5.0`:
- At sigma=1.0: `5.0 * 1.0 / (1 + 4.0 * 1.0) = 1.0` (timestep 1000)
- At sigma=0.5: `5.0 * 0.5 / (1 + 4.0 * 0.5) = 0.833` (timestep 833)
- At sigma=0.1: `5.0 * 0.1 / (1 + 4.0 * 0.1) = 0.357` (timestep 357)

Higher shift concentrates steps at higher noise levels (early denoising). This matters for expert switching -- with shift=5.0 and boundary=500, fewer steps fall in the low-noise range.

### Euler Update Rule

```python
# Deterministic:
dt = sigma_next - sigma  # negative (moving toward clean)
prev_sample = sample + dt * model_output

# The model predicts velocity (noise - clean), the scheduler
# integrates along the ODE trajectory.
```

---

## 6. LoRA Handling: Merge vs Wrapper

| Approach | When Used | How It Works | Implications |
|----------|-----------|--------------|--------------|
| **Merge** | musubi/ai-toolkit standalone inference | `merge_to()` or `merge_and_unload()` adds LoRA deltas to base weights | Model behaves as if fine-tuned. No extra memory. Cannot easily unload. |
| **PEFT Wrapper** | Dimljus training-time sampling | PEFT wraps each target module with LoRA layers | Forward pass goes through wrapper. LoRA can be easily removed. Slightly different numerics due to dtype handling in wrapper. |
| **pipeline.load_lora_weights()** | Dimljus standalone inference | Diffusers API creates PEFT adapters | Standard API path. Should be equivalent to PEFT wrapper. |

**Key difference:** During training, Dimljus's model already has PEFT wrappers applied. The `WanInferencePipeline.generate()` calls `pipeline(**kwargs)` with the PEFT-wrapped model as the transformer. The diffusers `WanPipeline.__call__` then runs the denoising loop using this model.

For standalone inference (loading from checkpoint), Dimljus uses `pipeline.load_lora_weights()` which is the standard diffusers API. This should work correctly.

---

## 7. Training-Time vs Standalone Inference

| Aspect | Training-Time | Standalone |
|--------|---------------|------------|
| **Model source** | Already loaded on GPU with PEFT LoRA | Loaded fresh from disk |
| **LoRA state** | Live PEFT wrappers (training weights) | Merged or loaded via API |
| **T5 encoder** | Loaded temporarily, then freed | Same |
| **VAE** | Loaded fresh for decode, then freed | Same |
| **Expert switching** | musubi: low-noise only; Dimljus: attempts dual | Full dual-expert |
| **Model mode** | Switched from train() to eval() temporarily | Always eval() |
| **Gradient state** | torch.no_grad() context | Same |
| **Numerical differences** | PEFT wrapper dtype handling | Clean merged weights |

**Why musubi uses low-noise only for training samples:** Simpler (no expert switching mid-denoising), and the low-noise expert handles fine detail -- training samples at low resolution are primarily about verifying detail/identity, not composition. This is pragmatic, not principled.

---

## 8. What Dimljus Does vs What Working Pipelines Do

### Things Dimljus Gets Right

| Component | Status |
|-----------|--------|
| FlowMatchEulerDiscreteScheduler | Correct (Minta-validated) |
| shift=5.0 | Correct (matches ComfyUI quality path) |
| boundary_ratio=0.5 for inference | Correct (Minta-validated) |
| VAE in float32 | Correct (prevents gridded artifacts) |
| T5 embed_tokens fix | Correct (addresses weight tying bug) |
| Expert assignment (transformer=high, transformer_2=low) | Correct (matches diffusers convention) |
| text_encoder=None with prompt_embeds | Correct (standard pre-encoded path) |

### Potential Issues to Investigate

| Issue | Risk | What to Check |
|-------|------|---------------|
| **from_single_file without explicit config** | HIGH | VAE loaded via from_single_file may lack latents_mean/latents_std. Check `vae.config.latents_mean` after loading. |
| **Transformer config mismatch** | HIGH | from_single_file may load Wan 2.1 config instead of 2.2. Test_base_inference.py does NOT pass `config=` parameter. |
| **VAE latent normalization during training** | MEDIUM | Dimljus does NOT normalize latents after VAE encoding (no (latents-mean)*std). ai-toolkit DOES. This affects whether training and inference are in the same space. Not the cause of base model noise (no LoRA), but affects LoRA quality. |
| **Latents dtype** | MEDIUM | WanPipeline initializes latents in float32 and keeps them float32 throughout denoising. Verify Dimljus does the same. |
| **cache_context** | LOW | WanPipeline wraps model calls in `cache_context("cond")`/`cache_context("uncond")`. If the model object doesn't support this (e.g., PEFT wrapper), it might silently fail or cause issues. |
| **Prompt embedding dtype** | MEDIUM | WanPipeline casts prompt_embeds to transformer_dtype. If Dimljus passes embeddings in a different dtype, the model may produce garbage. |

---

## Sources

- [diffusers WanPipeline source](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py)
- [diffusers FlowMatchEulerDiscreteScheduler docs](https://huggingface.co/docs/diffusers/en/api/schedulers/flow_match_euler_discrete)
- [diffusers#12329 from_single_file config bug](https://github.com/huggingface/diffusers/issues/12329)
- [musubi-tuner GitHub](https://github.com/kohya-ss/musubi-tuner)
- [musubi-tuner sampling docs](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/sampling_during_training.md)
- [ai-toolkit GitHub](https://github.com/ostris/ai-toolkit)
- [ai-toolkit Wan 2.2 DeepWiki](https://deepwiki.com/ostris/ai-toolkit/13-wan-2.2-advanced-features)
- [musubi-tuner wan.md docs](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/wan.md)
