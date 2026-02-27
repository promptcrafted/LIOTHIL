# Domain Pitfalls: Video Diffusion Inference Debugging

**Domain:** Wan 2.2 dual-expert video diffusion model inference
**Researched:** 2026-02-26
**Overall confidence:** HIGH (multiple sources corroborate each finding)

---

## Critical Pitfalls

Mistakes that cause complete inference failure (noise output, gridded artifacts, or silent wrong results).

---

### Pitfall 1: from_single_file Loads Wrong Model Config (Wan 2.1 vs 2.2)

**Severity:** CRITICAL -- silent corruption, produces noise or wrong output
**Confidence:** HIGH (documented in diffusers#12329, verified against HF model card)

**What goes wrong:** `WanTransformer3DModel.from_single_file()` infers the model
architecture from the checkpoint file's keys. When the checkpoint does not contain
sufficient version-discriminating keys, diffusers defaults to Wan 2.1 configuration
instead of Wan 2.2. The two architectures share identical weight shapes (same
num_layers=40, same attention_head_dim=128) but differ in config fields like
`image_dim`. The model loads without error but has wrong internal configuration.

**Why it happens:** The `from_single_file` auto-detection scans checkpoint keys
against known patterns in `single_file_utils.py`. Wan 2.1 and 2.2 T2V models have
identical weight key names and tensor shapes, making version detection ambiguous.
The fallback is Wan 2.1 config.

**Symptoms:**
- Model loads without any error or warning
- Inference produces noise, static, or completely incoherent video
- No obvious error in logs -- everything LOOKS correct
- Same model works fine in ComfyUI or official Wan code (which don't use diffusers loading)

**Diagnostic:**
```python
# After loading, check the config:
model = WanTransformer3DModel.from_single_file("model.safetensors", torch_dtype=torch.bfloat16)
print(model.config)
# Check image_dim -- for T2V it should be None/null
# For I2V it should be a specific integer
# If T2V model shows image_dim != None, wrong config was loaded
```

**Fix:**
```python
# ALWAYS provide explicit config when using from_single_file:
model = WanTransformer3DModel.from_single_file(
    "model.safetensors",
    torch_dtype=torch.bfloat16,
    config="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    subfolder="transformer",       # for high-noise expert
    # subfolder="transformer_2",   # for low-noise expert
)
```

**Prevention:** Never use `from_single_file()` without explicit `config=` and
`subfolder=` parameters. This is a hard rule -- there is no reliable auto-detection
for Wan 2.1 vs 2.2.

**Sources:**
- [diffusers#12329: WanImageToVideoPipeline loads single-file transformer as Wan2.1 instead of Wan2.2](https://github.com/huggingface/diffusers/issues/12329)
- [Wan2.2-T2V-A14B-Diffusers transformer/config.json](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers/blob/main/transformer/config.json)

---

### Pitfall 2: VAE Dtype Causes Grid Artifacts

**Severity:** CRITICAL -- produces visible grid patterns overlaid on output
**Confidence:** HIGH (documented in diffusers#12141, #211, confirmed in Dimljus testing)

**What goes wrong:** The Wan VAE (AutoencoderKLWan) produces structured grid
artifacts and noise patterns when run in bfloat16 or float16. The VAE's 3D causal
convolutions accumulate numerical errors in reduced precision, causing visible grid
patterns aligned with the temporal compression boundaries (every ~4 frames) and
spatial patch boundaries.

**Why it happens:** VAE decoders use iterative upsampling with residual connections.
In bf16/fp16, the limited mantissa precision causes small errors in each convolution
layer to compound through the decoder stack. The Wan VAE's 3D architecture (temporal +
spatial) makes this worse than 2D image VAEs because errors propagate along three axes.

**Symptoms:**
- Output video has a visible grid pattern overlaid on content
- Grid aligns with spatial boundaries (every ~2 or ~8 pixels depending on resolution)
- Pattern is consistent across different prompts and seeds
- Content may be partially visible BENEATH the grid
- Worsens at higher resolutions

**Diagnostic:**
```python
# Check VAE dtype:
print(f"VAE dtype: {vae.dtype}")
# If it shows bfloat16 or float16, this is the problem

# Definitive test: decode the same latent in both dtypes
latent = torch.randn(1, 16, 5, 30, 52, dtype=torch.bfloat16, device="cuda")
vae_fp32 = vae.to(torch.float32)
vae_bf16 = vae.to(torch.bfloat16)
out_fp32 = vae_fp32.decode(latent.float()).sample
out_bf16 = vae_bf16.decode(latent).sample
# If out_bf16 has grid artifacts and out_fp32 does not, confirmed
```

**Fix:**
```python
# ALWAYS load and run VAE in float32:
vae = AutoencoderKLWan.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    subfolder="vae",
    torch_dtype=torch.float32,  # MUST be float32
)
# or:
vae = AutoencoderKLWan.from_single_file(
    "wan_2.1_vae.safetensors",
    torch_dtype=torch.float32,
)
```

**Prevention:** Hard-code VAE dtype to float32. Never inherit dtype from training
config or pipeline default. The official Wan model card example explicitly loads
VAE in float32 while using bfloat16 for everything else.

**Sources:**
- [diffusers#211: Poor generation quality with Wan2.2](https://github.com/Wan-Video/Wan2.2/issues/211)
- [diffusers#12141: WanVACEPipeline doesn't work with bfloat16/float16 vae](https://github.com/huggingface/diffusers/issues/12141)
- [Wan2.2-T2V-A14B-Diffusers model card](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) (example code loads VAE in float32)
- Dimljus testing confirmed this fix (MEMORY.md: "VAE always float32")

---

### Pitfall 3: T5 embed_tokens Weight Tying Bug -- All-Zero Embeddings

**Severity:** CRITICAL -- model ignores prompt entirely, generates unconditionally
**Confidence:** HIGH (reproduced in Dimljus, root cause traced in transformers source)

**What goes wrong:** When loading UMT5EncoderModel from a Wan checkpoint file
(not from_pretrained with a full directory), the `encoder.embed_tokens.weight`
tensor remains as all zeros. The T5 produces zero embeddings for all inputs,
causing the diffusion model to generate as if no prompt was given (unconditional
generation -- random content, no semantic control).

**Why it happens:** The Wan T5 checkpoint stores the token embedding table under
the key `shared.weight`. UMT5EncoderModel expects this to be tied to
`encoder.embed_tokens.weight`, but `load_state_dict(strict=False)` does not
perform weight tying -- it only loads weights that match existing key names.
The `shared` module gets loaded correctly, but `encoder.embed_tokens` stays
at its initialization values (zeros from `torch.empty` or random init).

The `_tied_weights_keys` mechanism in transformers is designed to handle this
during `from_pretrained()`, but it does NOT activate during manual
`load_state_dict()` calls. Additionally, transformers v5.0 changed the tying
behavior further (see Pitfall 7).

**Symptoms:**
- Model generates video that has no relation to the prompt
- Different prompts produce similar-looking random output
- Negative prompt has no effect
- Output may still have some visual coherence (model generates "something" but
  not what was asked)
- No error or warning during loading

**Diagnostic:**
```python
# Check embed_tokens after loading:
embed_weight = text_encoder.encoder.embed_tokens.weight
shared_weight = text_encoder.shared.weight

print(f"embed_tokens all zeros: {(embed_weight == 0).all()}")
print(f"shared all zeros: {(shared_weight == 0).all()}")
print(f"embed_tokens mean: {embed_weight.float().mean():.6f}")
print(f"shared mean: {shared_weight.float().mean():.6f}")

# If embed_tokens is all zeros but shared is not, this is the bug

# Also verify actual output:
tokens = tokenizer("test prompt", return_tensors="pt", padding="max_length", max_length=512)
with torch.no_grad():
    output = text_encoder(input_ids=tokens.input_ids.to("cuda"))
    embeds = output.last_hidden_state
print(f"Embedding output all zeros: {(embeds == 0).all()}")
print(f"Embedding output mean: {embeds.float().mean():.6f}")
```

**Fix:**
```python
# After load_state_dict, manually tie the weights:
if hasattr(text_encoder, "shared") and hasattr(text_encoder.encoder, "embed_tokens"):
    shared = text_encoder.shared.weight
    embed = text_encoder.encoder.embed_tokens.weight
    if (embed == 0).all() and not (shared == 0).all():
        text_encoder.encoder.embed_tokens.weight = shared
```

**Prevention:** Always call the embed_tokens fix after any manual T5 loading.
The `from_pretrained()` path with a full diffusers directory handles this
automatically (via tie_weights), but manual loading from .pth or .safetensors
files always requires this fix.

**Sources:**
- [transformers UMT5 modeling source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/umt5/modeling_umt5.py) -- `_tied_weights_keys = ["encoder.embed_tokens.weight"]`
- [transformers#42832: Question about tie_weights](https://github.com/huggingface/transformers/issues/42832)
- Dimljus inference.py `_fix_t5_embed_tokens()` method (working fix)

---

### Pitfall 4: Wrong Scheduler and Shift Value -- Most Common Cause of Noise

**Severity:** CRITICAL -- produces pure noise or heavily degraded output
**Confidence:** HIGH (official config verified, multiple community reports)

**What goes wrong:** Using the wrong scheduler type or shift value for inference.
The official Wan 2.2 T2V configuration uses specific scheduler settings that differ
significantly from common defaults, and these values are NOT interchangeable between
scheduler types.

**Why it happens:** There are THREE different "correct" configurations depending on
which scheduler you use, and the values are NOT portable between them:

| Setting | Official Wan Code | Diffusers HF Default | ComfyUI | musubi-tuner |
|---------|-------------------|----------------------|---------|-------------|
| Scheduler | custom flow matching | UniPCMultistepScheduler | euler/beta | FlowUniPCMultistep |
| Shift | sample_shift=12.0 | flow_shift=3.0 | shift=5.0 | shift=12.0 |
| Boundary | 0.875 | 0.875 | varies | 0.875 |
| Steps | 40 | 40 | 6+4 (accelerated) | 20-40 |

The shift parameter transforms the noise schedule via:
`sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)`

A shift of 12.0 vs 3.0 vs 5.0 produces COMPLETELY different sigma schedules.
Using shift=12.0 with a scheduler expecting shift=3.0 (or vice versa) will produce
noise because the denoising trajectory is wrong.

**Symptoms:**
- Pure noise output (model never denoises properly)
- Extremely dark or extremely bright output
- Partially denoised output (recognizable shapes but heavy noise)
- Output quality varies dramatically with small shift changes

**Diagnostic:**
```python
# Print the actual sigma schedule being used:
scheduler.set_timesteps(30)
print(f"Timesteps: {scheduler.timesteps[:5]}...{scheduler.timesteps[-5:]}")
print(f"Sigmas: {scheduler.sigmas[:5]}...{scheduler.sigmas[-5:]}")
print(f"Shift: {getattr(scheduler.config, 'shift', 'N/A')}")
print(f"Flow shift: {getattr(scheduler.config, 'flow_shift', 'N/A')}")

# Compare against known-good sigma schedules
# For UniPCMultistep with flow_shift=3.0 and 40 steps,
# first sigma should be ~1.0 and last should be ~0.0
```

**Fix:** Match scheduler type to shift value:

```python
# Option A: Diffusers default (from official HF model card) -- RECOMMENDED
from diffusers import UniPCMultistepScheduler
scheduler = UniPCMultistepScheduler(
    flow_shift=3.0,
    prediction_type="flow_prediction",
)

# Option B: FlowMatchEuler (ComfyUI path)
from diffusers import FlowMatchEulerDiscreteScheduler
scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
# NOTE: shift=5.0 with FlowMatchEuler is NOT the same as flow_shift=3.0 with UniPC

# Option C: Match official Wan code -- requires custom scheduler
# Use shift=12.0 ONLY with the official Wan sampler or musubi's FlowUniPCMultistep
```

**Prevention:**
1. Use `WanPipeline.from_pretrained()` which loads the correct scheduler config
2. If building manually, use UniPCMultistepScheduler with flow_shift=3.0 (the HF default)
3. If using FlowMatchEulerDiscreteScheduler, test with shift=5.0 (ComfyUI path)
4. NEVER mix shift values between scheduler types

**Sources:**
- [Wan2.2-T2V-A14B-Diffusers scheduler_config.json](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers/blob/main/scheduler/scheduler_config.json) -- UniPCMultistepScheduler, flow_shift=3.0
- [Wan2.2-T2V-A14B-Diffusers model_index.json](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers/blob/main/model_index.json) -- boundary_ratio=0.875
- [Official Wan 2.2 config](https://github.com/Wan-Video/Wan2.2) -- sample_shift=12.0, boundary=0.875
- [diffusers#12094: pipeline_wan miss the shift parameter](https://github.com/huggingface/diffusers/issues/12094)
- [diffusers#11160: Inconsistent inference results with Wan 2.1](https://github.com/huggingface/diffusers/issues/11160)
- [musubi-tuner docs](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/wan.md) -- shift=12.0 for T2V with FlowUniPCMultistep
- [Civitai best parameters article](https://civitai.com/articles/18517/wan22best-parameters-for-t2it2vi2vbased-on-the-org-workflow)

---

### Pitfall 5: Expert Assignment Swapped (High-Noise / Low-Noise Reversed)

**Severity:** CRITICAL -- produces noise or extremely degraded output
**Confidence:** HIGH (verified against diffusers source and model card)

**What goes wrong:** The high-noise and low-noise expert models are assigned to
the wrong pipeline slots. In diffusers WanPipeline:
- `transformer` = HIGH-noise expert (handles timesteps >= boundary)
- `transformer_2` = LOW-noise expert (handles timesteps < boundary)

If these are swapped, each expert processes noise levels it was never trained for,
producing garbage output.

**Why it happens:**
1. The naming is counterintuitive -- `transformer` (no suffix) is the FIRST to run
   but handles HIGH noise, while `transformer_2` handles LOW noise
2. The Wan checkpoint files are named `wan2.2_t2v_high_noise_14B_fp16.safetensors`
   and `wan2.2_t2v_low_noise_14B_fp16.safetensors` -- easy to mix up which goes where
3. When building pipelines manually (not from_pretrained), you must get the mapping right

**Symptoms:**
- Complete noise output
- Each expert produces reasonable partial results alone, but dual-expert produces garbage
- Swapping the models "fixes" the output

**Diagnostic:**
```python
# Verify assignment:
print(f"transformer (high-noise): {type(pipeline.transformer)}")
print(f"transformer_2 (low-noise): {type(pipeline.transformer_2)}")

# Run single-expert test: set boundary_ratio=1.0 to use only transformer
# If this produces recognizable output, transformer is correct
# Then set boundary_ratio=0.0 to test transformer_2 alone
```

**Fix:**
```python
# Correct assignment:
pipeline = WanPipeline(
    transformer=model_high,       # HIGH-noise first (large timesteps)
    transformer_2=model_low,      # LOW-noise second (small timesteps)
    vae=vae,
    scheduler=scheduler,
    boundary_ratio=0.875,
    text_encoder=None,
    tokenizer=None,
)
```

**Prevention:** Always verify expert assignment by checking the source file name
against the pipeline slot. When using from_pretrained, this is handled automatically.

**Sources:**
- [diffusers WanPipeline source](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py)
- [Wan2.2-T2V-A14B-Diffusers model_index.json](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers/blob/main/model_index.json)
- Dimljus commit d2c236d fixed this exact issue

---

### Pitfall 6: Boundary Ratio Mismatch Between Design and Inference

**Severity:** CRITICAL for inference quality -- may produce noise or degraded output
**Confidence:** HIGH (official config verified, multiple sources agree)

**What goes wrong:** The boundary_ratio determines WHEN the pipeline switches from
the high-noise expert to the low-noise expert. If the inference boundary differs
from what the model expects, one expert processes timesteps outside its training
distribution.

**Why it happens:** Different tools use different boundary values:
- Official Wan 2.2 T2V: boundary=0.875 (87.5% of timesteps use high-noise expert)
- Diffusers HF config: boundary=0.875
- musubi-tuner T2V: boundary=0.875
- Dimljus training: boundary=0.875
- Dimljus inference: boundary=0.5 (**MISMATCH**)
- Official Wan 2.2 I2V: boundary=0.9

The boundary_ratio=0.5 in Dimljus inference means the low-noise expert runs for
50% of timesteps instead of 12.5%. This forces the low-noise expert to process
moderate-noise timesteps it was never trained on.

With num_train_timesteps=1000 and 30 inference steps:
- boundary=0.875 -> boundary_timestep=875 -> high-noise expert runs ~26 of 30 steps
- boundary=0.5 -> boundary_timestep=500 -> high-noise expert runs ~15 of 30 steps

**Symptoms:**
- Partially denoised output (noise mixed with content)
- Output looks "wrong" but not pure noise
- Different boundary values produce wildly different quality levels
- Low-noise expert produces artifacts when forced to handle high-noise timesteps

**Diagnostic:**
```python
# Check which expert runs at each timestep:
boundary_timestep = boundary_ratio * scheduler.config.num_train_timesteps
scheduler.set_timesteps(num_inference_steps)
for i, t in enumerate(scheduler.timesteps):
    expert = "HIGH" if t >= boundary_timestep else "LOW"
    print(f"  Step {i}: t={t:.1f} -> {expert}-noise expert")
```

**Fix:**
```python
# Use the official boundary ratio:
boundary_ratio = 0.875  # for T2V
# boundary_ratio = 0.9  # for I2V
```

**Prevention:** Use the same boundary_ratio for inference as the official model config.
The training boundary_ratio can differ (training selects timesteps per-batch), but
inference must match the model's design.

**Sources:**
- [Wan2.2-T2V-A14B-Diffusers model_index.json](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers/blob/main/model_index.json) -- boundary_ratio=0.875
- [Official Wan 2.2 T2V config](https://github.com/Wan-Video/Wan2.2) -- boundary=0.875
- Dimljus MEMORY.md documents the intentional difference at boundary=0.5

---

## Moderate Pitfalls

Issues that degrade quality significantly but may not cause complete failure.

---

### Pitfall 7: transformers Library Version Incompatibility

**Severity:** MODERATE to CRITICAL
**Confidence:** HIGH (documented in diffusers#12878)

**What goes wrong:** Upgrading the `transformers` library to version 5.0 (release
candidate) causes Wan 2.2 inference to fail. The T5 encoder ignores prompts entirely,
and when combined with SageAttention, produces black frames from NaN values in the
Query tensor.

**Why it happens:** transformers v5.0 changed internal weight tying behavior and
model loading logic. The `tie_weights` function no longer ties weights when both
`embed_tokens.weight` and `lm_head.weight` are explicitly present in a checkpoint.
This interacts badly with UMT5EncoderModel loading.

**Symptoms:**
- Model/transformer ignores the entire prompt
- Black image/video output
- NaN values in attention Query tensor (with SageAttention)
- No error during loading

**Diagnostic:**
```python
import transformers
print(f"transformers version: {transformers.__version__}")
# Problematic: 5.0.0, 5.0.0rc*
# Safe: 4.57.3 and earlier
```

**Fix:**
```bash
pip install transformers==4.57.3
```

**Prevention:** Pin transformers version in requirements. Do not use release
candidates in production.

**Sources:**
- [diffusers#12878: This new version of diffusers causes problems with Wan2.2](https://github.com/huggingface/diffusers/issues/12878)

---

### Pitfall 8: Prompt Embeddings Shape/Dtype Mismatch

**Severity:** MODERATE -- may produce degraded output or errors
**Confidence:** MEDIUM (inferred from pipeline source analysis)

**What goes wrong:** When passing pre-computed `prompt_embeds` directly to the
WanPipeline, the embeddings must match the expected shape and dtype. Mismatches
can cause silent quality degradation or errors.

**Why it happens:**
1. The pipeline casts prompt_embeds to the transformer's dtype, but if they
   were encoded in float32 and the transformer is bfloat16, precision is lost
2. The WanPipeline does NOT pass attention masks to the transformer when using
   pre-computed embeddings. This means the T5 padding tokens are treated as
   real content, which can slightly degrade quality
3. Shape must be [batch, seq_len, hidden_dim] where hidden_dim=4096 for UMT5-XXL

**Symptoms:**
- Slightly degraded output quality (compared to using prompt= directly)
- Prompt seems partially followed
- No error during generation

**Diagnostic:**
```python
print(f"prompt_embeds shape: {prompt_embeds.shape}")
print(f"prompt_embeds dtype: {prompt_embeds.dtype}")
print(f"transformer dtype: {pipeline.transformer.dtype}")
# Shape should be [1, 512, 4096] for max_length=512
# dtype should match transformer dtype
```

**Fix:**
```python
# Ensure correct dtype before passing:
prompt_embeds = prompt_embeds.to(device=device, dtype=pipeline.transformer.dtype)

# Verify non-zero:
assert not (prompt_embeds == 0).all(), "Embeddings are all zeros -- T5 loading failed"
```

**Prevention:** Always verify embeddings are non-zero after encoding. Cast to
the correct dtype explicitly.

---

### Pitfall 9: cache_context NoneType Error During Expert Switching

**Severity:** MODERATE -- causes hard crash during inference
**Confidence:** HIGH (documented in diffusers#12126)

**What goes wrong:** During the denoising loop, when the pipeline switches from
the high-noise transformer to the low-noise transformer_2, it crashes with
`AttributeError: 'NoneType' object has no attribute 'cache_context'`.

**Why it happens:** The `current_model` variable becomes None when:
1. `transformer_2` was not properly loaded or assigned
2. The boundary condition logic selects a model that is None
3. Memory offloading moved the model to CPU and it wasn't moved back

**Symptoms:**
- Hard crash mid-inference with NoneType AttributeError
- Crash happens partway through denoising (at the boundary timestep)
- Single-expert inference works fine

**Diagnostic:**
```python
# Before building pipeline, verify both models:
assert pipeline.transformer is not None, "transformer (high-noise) is None!"
assert pipeline.transformer_2 is not None, "transformer_2 (low-noise) is None!"

# Check they're on the right device:
print(f"transformer device: {next(pipeline.transformer.parameters()).device}")
print(f"transformer_2 device: {next(pipeline.transformer_2.parameters()).device}")
```

**Fix:** Ensure both transformers are loaded, non-None, and on the correct device
before calling the pipeline. If using memory offloading, use
`pipeline.enable_model_cpu_offload()` which handles the switching automatically.

**Sources:**
- [diffusers#12126: NoneType cache_context when switching transformers](https://github.com/huggingface/diffusers/issues/12126)

---

### Pitfall 10: Guidance Scale Configuration for Dual-Expert

**Severity:** MODERATE -- degrades output quality
**Confidence:** HIGH (official config verified)

**What goes wrong:** The high-noise and low-noise experts require different
classifier-free guidance (CFG) scales. Using the same scale for both degrades
quality, particularly in the low-noise (detail refinement) stage.

**Why it happens:** The high-noise expert handles global composition and motion,
benefiting from stronger guidance (4.0). The low-noise expert handles fine details
and textures, where excessive guidance causes over-saturation and artifacts. The
official config uses guidance_scale=4.0 (high) and guidance_scale_2=3.0 (low).

**Symptoms:**
- Over-saturated colors or washed-out details
- Artifacts in fine details (faces, textures, text)
- Output quality is "okay but not as good as ComfyUI/official"

**Diagnostic:**
```python
# Check what guidance scales are being used:
print(f"guidance_scale (high-noise): {guidance_scale}")
print(f"guidance_scale_2 (low-noise): {guidance_scale_2}")
# Official T2V values: 4.0 and 3.0
```

**Fix:**
```python
output = pipeline(
    prompt_embeds=embeds,
    guidance_scale=4.0,      # high-noise expert
    guidance_scale_2=3.0,    # low-noise expert
    # ...
)
```

**Sources:**
- [Official Wan 2.2 T2V config](https://github.com/Wan-Video/Wan2.2) -- guidance_scales=(3.0, 4.0)
- [Wan2.2-T2V-A14B-Diffusers model card](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) -- example uses guidance_scale=4.0, guidance_scale_2=3.0

---

## Minor Pitfalls

Issues that cause subtle quality differences or edge-case problems.

---

### Pitfall 11: Insufficient Inference Steps

**Severity:** MINOR -- degraded quality, not complete failure
**Confidence:** HIGH

**What goes wrong:** Too few inference steps leaves residual noise in the output.
The official Wan 2.2 T2V config uses 40 steps. Using fewer than ~25 steps with
Euler-type schedulers produces visibly noisy output.

**Prevention:** Use 30-40 steps minimum for FlowMatchEuler. UniPC can produce
acceptable results in 20-25 steps due to its higher-order solver.

---

### Pitfall 12: FlowMatchEulerDiscreteScheduler Floating-Point Timestep Comparison

**Severity:** MINOR -- can cause off-by-one step errors
**Confidence:** MEDIUM (documented in diffusers#9331)

**What goes wrong:** The scheduler's `index_for_timestep()` method uses `==`
comparison on floating-point timestep values. Due to floating-point precision,
the provided timestep may not exactly match any scheduler timestep, causing
incorrect sigma selection.

**Prevention:** This is a diffusers-internal issue. If you see inconsistent
results between runs or unexpected step counts, this may be the cause. Using
UniPCMultistepScheduler avoids this specific issue.

**Sources:**
- [diffusers#9331: Problem in FlowMatchEulerDiscreteScheduler's index_for_timestep method](https://github.com/huggingface/diffusers/issues/9331)

---

### Pitfall 13: num_frames Must Match Wan VAE Temporal Compression

**Severity:** MINOR -- may cause dimension errors
**Confidence:** MEDIUM

**What goes wrong:** The Wan VAE has ~4x temporal compression. The number of
frames must satisfy the VAE's temporal stride requirements. Specifically:
`(num_frames - 1) % 4 == 0` for the causal 3D VAE (81 -> 21 latent frames,
17 -> 5 latent frames, etc.).

Using non-conforming frame counts may cause dimension mismatch errors or
force padding that produces artifacts in the last frames.

**Fix:** Use frame counts of 5, 9, 13, 17, 21, 25, ..., 81.

---

## Dimljus-Specific Analysis: Current Inference Bug

Based on the research findings and the current Dimljus inference code, the most
likely causes of the "base model produces noise" bug are (in order of probability):

### 1. MOST LIKELY: Scheduler + Shift + Boundary Triple Mismatch (Pitfalls 4 + 6)

Current Dimljus inference uses:
- `FlowMatchEulerDiscreteScheduler(shift=5.0)`
- `boundary_ratio=0.5`

The official Wan 2.2 T2V HF config uses:
- `UniPCMultistepScheduler(flow_shift=3.0)`
- `boundary_ratio=0.875`

These are fundamentally different denoising trajectories. The boundary=0.5 alone
means the low-noise expert runs for 15 of 30 steps instead of 4 of 30 steps.
Combined with a different scheduler type and shift, this produces a completely
wrong sigma schedule.

**Recommended test:** Replace the scheduler and boundary in inference.py:
```python
from diffusers import UniPCMultistepScheduler
scheduler = UniPCMultistepScheduler(flow_shift=3.0, prediction_type="flow_prediction")
# ... with boundary_ratio=0.875
```

### 2. LIKELY: from_single_file Without Explicit Config (Pitfall 1)

The `test_base_inference.py` loads models without explicit config:
```python
model_high = WanTransformer3DModel.from_single_file(
    MODEL_HIGH, torch_dtype=torch.bfloat16
)  # NO config= parameter!
```

The `test3-inference.py` DOES add explicit config. If the noisy output was from
`test_base_inference.py`, this is the cause.

The WanInferencePipeline class receives already-loaded models from the training
loop. Check how the training backend loads models -- if it uses from_single_file
without explicit config, the models may have wrong config from the start.

### 3. POSSIBLE: T5 embed_tokens Not Verified on Pod (Pitfall 3)

The fix exists in inference.py but needs verification that it actually runs
correctly on the pod. Print embed_tokens values after loading to confirm.

### Recommended Debugging Sequence

1. **Simplest first:** Try `WanPipeline.from_pretrained("Wan-AI/Wan2.2-T2V-A14B-Diffusers")`
   with the official example code from the model card. If this works, the issue
   is in Dimljus's manual pipeline construction. If this also produces noise,
   the issue is environment-level (diffusers/transformers version).

2. **Check versions:**
   ```python
   import diffusers, transformers, torch
   print(f"diffusers: {diffusers.__version__}")
   print(f"transformers: {transformers.__version__}")
   print(f"torch: {torch.__version__}")
   ```
   transformers must be <5.0.

3. **If from_pretrained works:** Compare the loaded scheduler config,
   boundary_ratio, and transformer configs against what Dimljus constructs
   manually. Print everything and diff.

4. **If from_pretrained also fails:** Environment issue. Reinstall diffusers
   and transformers from stable versions.

---

## Phase-Specific Warnings

| Issue | Likely Pitfall | Priority | Mitigation |
|-------|---------------|----------|------------|
| Base model produces noise | Pitfall 4 (scheduler/shift) + Pitfall 6 (boundary) + Pitfall 1 (config) | P0 | Match official HF config exactly |
| Grid overlay on output | Pitfall 2 (VAE dtype) | P0 | Force VAE to float32 (already done) |
| Prompt ignored | Pitfall 3 (T5 embed_tokens) OR Pitfall 7 (transformers version) | P0 | Check embed_tokens, check transformers version |
| Partially denoised output | Pitfall 4 (shift value) OR Pitfall 6 (boundary) | P1 | Match scheduler+shift+boundary to official config |
| Crash during dual-expert | Pitfall 5 (expert swap) OR Pitfall 9 (cache_context) | P1 | Verify expert assignment, check both models non-None |
| Quality worse than ComfyUI | Pitfall 10 (guidance scales) OR Pitfall 11 (step count) | P2 | Use 4.0/3.0 guidance, 40 steps |

---

## Sources Summary

| Source | Type | Used For |
|--------|------|----------|
| [diffusers#12329](https://github.com/huggingface/diffusers/issues/12329) | GitHub Issue | from_single_file Wan 2.1 vs 2.2 bug |
| [diffusers#12878](https://github.com/huggingface/diffusers/issues/12878) | GitHub Issue | transformers v5.0 incompatibility |
| [diffusers#11160](https://github.com/huggingface/diffusers/issues/11160) | GitHub Issue | Wrong scheduler in model_index |
| [diffusers#12126](https://github.com/huggingface/diffusers/issues/12126) | GitHub Issue | cache_context NoneType error |
| [diffusers#12141](https://github.com/huggingface/diffusers/issues/12141) | GitHub Issue | VAE bf16/fp16 dtype issue |
| [diffusers#12094](https://github.com/huggingface/diffusers/issues/12094) | GitHub Issue | Missing shift parameter |
| [diffusers#9331](https://github.com/huggingface/diffusers/issues/9331) | GitHub Issue | Timestep float comparison |
| [Wan2.2#211](https://github.com/Wan-Video/Wan2.2/issues/211) | GitHub Issue | Poor diffusers quality (VAE dtype) |
| [diffusion-pipe#372](https://github.com/tdrussell/diffusion-pipe/issues/372) | GitHub Issue | Blurry LoRA training output |
| [Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | Model Card | Official config values |
| [Official Wan 2.2 repo](https://github.com/Wan-Video/Wan2.2) | Source Code | sample_shift=12.0, boundary=0.875 |
| [musubi-tuner docs](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/wan.md) | Documentation | Shift=12.0 for T2V inference |
| [Civitai best parameters](https://civitai.com/articles/18517/wan22best-parameters-for-t2it2vi2vbased-on-the-org-workflow) | Community | Scheduler comparison testing |
| [diffusers FlowMatchEuler docs](https://huggingface.co/docs/diffusers/en/api/schedulers/flow_match_euler_discrete) | Documentation | Scheduler parameters |
| [transformers UMT5 source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/umt5/modeling_umt5.py) | Source Code | Weight tying mechanism |
