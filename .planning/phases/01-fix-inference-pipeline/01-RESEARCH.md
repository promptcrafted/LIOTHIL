# Phase 1: Fix Inference Pipeline - Research

**Researched:** 2026-02-26
**Domain:** Diffusers WanPipeline inference, model loading, dual-expert denoising
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Start by diffing current code against last known-good commit (d2c236d) to find what diverged
- Prioritize investigating the model loading path first -- known diffusers bug (#12329) where `from_single_file()` loads wrong config (Wan 2.1 vs 2.2) is the most likely root cause
- If diff + model loading fix doesn't resolve: try `from_pretrained` with full HF repo "Wan-AI/Wan2.2-T2V-A14B-Diffusers" as quick test
- If `from_pretrained` also fails: build minimal stock diffusers reference pipeline as ground truth
- Escalation order: diff -> model loading -> from_pretrained -> fresh reference
- shift=5.0, boundary_ratio=0.5 for inference -- validated by Minta, treat as hard constraints
- FlowMatchEulerDiscreteScheduler -- not UniPC
- These are non-negotiable. If something doesn't work, the bug is in the code plumbing, not the parameters
- Training boundary_ratio=0.875 is intentionally different from inference boundary_ratio=0.5
- Primary reference: ai-toolkit (source available locally in `ai-toolkit-source/`)
- Match ai-toolkit's code plumbing: model loading sequence, scheduler initialization, denoising loop structure
- DO NOT override Minta's hyperparameters with ai-toolkit's values
- Only flag if our code is MISSING a parameter that ai-toolkit sets (a gap, not a difference)
- Use existing test3 checkpoints (expert-only training on Holly) for LoRA tests
- Fix standalone inference first, then verify training-pipeline integration
- Output spec: 17 frames at 480x832, save .mp4 + PNG keyframe grid (frames 1, 5, 9, 13, 17)

### Claude's Discretion
- Specific test prompts to use (2-3 diverse prompts)
- Keyframe grid layout and formatting
- Order of investigation within the model loading path
- How to structure the ai-toolkit comparison documentation
- Engineering decisions about code structure and implementation approach

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| INFER-01 | Base model inference produces recognizable video (not noise) with Wan 2.2 T2V | Root cause identified: `from_single_file()` config mismatch (diffusers#12329). Fix: pass explicit `config=` parameter. Code pattern documented below. |
| INFER-02 | LoRA inference modifies base model output in a visually detectable way | Existing test3 checkpoints (merged 586MB, 800 transformer. + 800 transformer_2. keys) are ready. Pipeline `load_lora_weights()` API and key prefix routing documented. |
| INFER-03 | Inference works both standalone (separate script) and integrated (from training pipeline) | Training loop calls `WanInferencePipeline.generate()` via `SamplingEngine.generate_samples()`. Same code path, different model source (training model vs loaded model). Integration points documented. |
</phase_requirements>

## Summary

The inference pipeline produces noise because `WanTransformer3DModel.from_single_file()` silently loads Wan 2.1 configuration instead of Wan 2.2 configuration. This is a confirmed diffusers bug (#12329, closed as COMPLETED Jan 2026). The fix is a one-line addition: pass `config="Wan-AI/Wan2.2-T2V-A14B-Diffusers"` with the appropriate `subfolder` parameter.

The test3-inference.py script already applies this fix for its local test, but the main codebase files (`backend.py` line 200 and `inference.py` line 324 for VAE) do not pass the `config` parameter. The ai-toolkit reference implementation uses `from_pretrained()` exclusively (never `from_single_file()`), which does not have this bug -- confirming that model loading method is the root cause.

Secondary finding: the ai-toolkit custom pipeline (Wan22Pipeline) performs explicit latent denormalization before VAE decode (`latents / latents_std + latents_mean`), but this is already handled automatically by the stock diffusers WanPipeline, which dimljus uses. No action needed on this front. The stock WanPipeline also handles dual-expert denoising (boundary_ratio switching) internally.

**Primary recommendation:** Add explicit `config=` parameter to all `from_single_file()` calls, then run the escalation sequence to verify. The test3-inference.py already has the fix pattern -- it just needs to be ported into the main codebase.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| diffusers | Latest (0.34+) | WanPipeline, WanTransformer3DModel, FlowMatchEulerDiscreteScheduler, AutoencoderKLWan | Official HF pipeline for Wan 2.2 inference, handles dual-expert routing, latent normalization, CFG |
| transformers | Latest | UMT5EncoderModel, AutoTokenizer for T5 text encoding | Official HF library for T5 text encoder |
| torch | 2.1+ | GPU compute, tensor operations | Required for all model operations |
| safetensors | Latest | Fast model weight loading/saving | Standard format for LoRA checkpoints |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Pillow (PIL) | Latest | Image processing for keyframe grids | For PNG keyframe extraction and grid assembly |
| imageio | Latest | Video export (used by diffusers export_to_video) | For saving .mp4 sample videos |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `from_single_file()` | `from_pretrained()` | from_pretrained is more reliable (no config guessing) but requires Diffusers directory structure on disk; from_single_file works with raw safetensors |
| Stock WanPipeline | Custom pipeline (ai-toolkit pattern) | Custom gives more control over denoising loop but duplicates diffusers logic; stock pipeline already handles latent normalization, dual experts, CFG correctly |
| FlowMatchEulerDiscreteScheduler | UniPCMultistepScheduler | UniPC is what ai-toolkit uses for generation; Minta validated FlowMatchEuler with shift=5.0. LOCKED to FlowMatchEuler. |

## Architecture Patterns

### Relevant File Structure
```
dimljus/training/wan/
├── inference.py      # WanInferencePipeline — builds pipeline, generates samples
├── backend.py        # WanModelBackend — loads model (from_single_file bug is HERE)
├── registry.py       # Variant config lookup (boundary ratios, flow_shift values)
├── checkpoint_io.py  # LoRA key prefix add/strip for diffusers compatibility
└── constants.py      # Architecture constants (block count, channel counts, targets)

dimljus/training/
├── loop.py           # TrainingOrchestrator — calls _generate_samples()
├── sampler.py        # SamplingEngine — manages when/what to sample
├── __main__.py       # _resolve_inference_pipeline() — creates WanInferencePipeline
└── protocols.py      # InferencePipeline protocol definition

runpod/
├── test_base_inference.py   # Standalone base model test (without config= fix)
├── test3-inference.py       # Post-training test (HAS config= fix already)
└── train.py                 # Training runner (calls dimljus training loop)
```

### Pattern 1: from_single_file with Explicit Config (THE FIX)
**What:** Force correct model configuration when loading from raw safetensors files
**When to use:** Every call to `WanTransformer3DModel.from_single_file()` for Wan 2.2 models
**Example (from test3-inference.py, already validated):**
```python
# Source: runpod/test3-inference.py lines 108-116
# HIGH-noise expert
model_high = WanTransformer3DModel.from_single_file(
    MODEL_HIGH, torch_dtype=torch.bfloat16,
    config="Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="transformer",
).to("cuda").eval()

# LOW-noise expert
model_low = WanTransformer3DModel.from_single_file(
    MODEL_LOW, torch_dtype=torch.bfloat16,
    config="Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="transformer_2",
).to("cuda").eval()
```

### Pattern 2: ai-toolkit Model Loading (Reference)
**What:** ai-toolkit always uses from_pretrained, never from_single_file
**Key observation:** This avoids the config detection bug entirely
```python
# Source: ai-toolkit-source/extensions_built_in/diffusion_models/wan22/wan22_14b_model.py lines 284-318
transformer_1 = WanTransformer3DModel.from_pretrained(
    transformer_path_1, subfolder=subfolder_1, torch_dtype=dtype,
).to(dtype=dtype)
transformer_2 = WanTransformer3DModel.from_pretrained(
    transformer_path_2, subfolder=subfolder_2, torch_dtype=dtype,
).to(dtype=dtype)
```

### Pattern 3: Dual-Expert Pipeline Construction
**What:** Build WanPipeline with both experts and correct boundary
**When to use:** All dual-expert inference (T2V with Wan 2.2)
```python
# Source: dimljus inference.py _build_pipeline() + test scripts
scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)  # LOCKED
pipeline = WanPipeline(
    transformer=model_high,      # HIGH-noise expert
    transformer_2=model_low,     # LOW-noise expert
    vae=vae,                     # Always float32
    text_encoder=None,           # Pre-encoded embeddings
    tokenizer=None,
    scheduler=scheduler,
    boundary_ratio=0.5,          # LOCKED for T2V inference
)
```

### Pattern 4: T5 Embed Tokens Fix
**What:** Fix weight tying bug where embed_tokens stays all zeros
**When to use:** Every time T5 is loaded (from .pth file or from_pretrained)
```python
# Source: dimljus inference.py _fix_t5_embed_tokens()
if (embed == 0).all() and not (shared == 0).all():
    text_encoder.encoder.embed_tokens.weight = shared
```

### Anti-Patterns to Avoid
- **Calling from_single_file() without config= for Wan 2.2:** Silently loads Wan 2.1 config, producing garbage output. Always pass explicit config.
- **Running VAE in bf16/fp16:** Produces gridded artifacts. Always use float32 for AutoencoderKLWan.
- **Changing shift or boundary_ratio:** These are Minta's validated parameters. Debug code plumbing, not hyperparameters.
- **Building custom denoising loops:** Stock WanPipeline handles latent normalization, dual-expert switching, and CFG correctly. Do not reimplement.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Dual-expert denoising | Custom timestep routing loop | `WanPipeline(boundary_ratio=0.5)` | Pipeline handles expert switching, per-expert CFG scales, and timestep boundary internally |
| Latent denormalization | Manual `latents / std + mean` before VAE decode | Stock WanPipeline `__call__()` | The denormalization is automatic in WanPipeline; ai-toolkit's Wan22Pipeline only does it manually because they wrote a custom pipeline |
| CFG (classifier-free guidance) | Two forward passes + manual combination | WanPipeline `guidance_scale` parameter | Pipeline handles CFG internally with proper batching |
| LoRA loading for inference | Manual weight merging | `pipeline.load_lora_weights(state_dict, adapter_name="dimljus")` | Diffusers routes keys to correct transformer via prefix (transformer./transformer_2.) |
| Keyframe grid assembly | Frame-by-frame custom layout | PIL Image.new + paste grid | Simple concatenation, no library needed beyond PIL |

**Key insight:** The stock diffusers WanPipeline already does everything the ai-toolkit custom Wan22Pipeline does (latent normalization, dual-expert switching, CFG). The dimljus pipeline correctly delegates to the stock pipeline. The only bug is in model LOADING, not in the denoising/decoding path.

## Common Pitfalls

### Pitfall 1: from_single_file Config Mismatch (THE ROOT CAUSE)
**What goes wrong:** `WanTransformer3DModel.from_single_file()` auto-detects the wrong model config. For Wan 2.2 T2V safetensors files, it infers Wan 2.1 config, loading with incorrect architecture parameters.
**Why it happens:** Diffusers' auto-detection logic uses checkpoint metadata and weight shapes to guess the config. Wan 2.1 and 2.2 share the same transformer architecture (same shapes), so the heuristic picks the wrong one.
**How to avoid:** Always pass `config="Wan-AI/Wan2.2-T2V-A14B-Diffusers"` with `subfolder="transformer"` (or `"transformer_2"`) to every `from_single_file()` call.
**Warning signs:** Model loads without error but produces pure noise/garbage at inference time. The failure is silent.
**Source:** [diffusers#12329](https://github.com/huggingface/diffusers/issues/12329) -- closed COMPLETED Jan 2026.

### Pitfall 2: T5 embed_tokens Weight Tying
**What goes wrong:** UMT5EncoderModel's embed_tokens layer stays all zeros after loading Wan-AI checkpoint weights.
**Why it happens:** The checkpoint stores token embeddings as "shared.weight" but the model keeps a separate "encoder.embed_tokens.weight". Neither `load_state_dict(strict=False)` nor `from_pretrained()` ties these automatically.
**How to avoid:** Call `_fix_t5_embed_tokens()` after every T5 load. Already implemented in inference.py.
**Warning signs:** T5 produces all-zero embeddings; model generates unconditionally (ignores prompts but may still produce recognizable shapes if base model config is correct).

### Pitfall 3: VAE Precision
**What goes wrong:** VAE decode produces gridded artifacts, banding, color distortion.
**Why it happens:** Wan-VAE's 3D causal decoder has numerical precision requirements that bf16/fp16 cannot meet.
**How to avoid:** Always load AutoencoderKLWan with `torch_dtype=torch.float32`.
**Warning signs:** Output has visible grid patterns overlaid on otherwise recognizable content.
**Note:** Already fixed in current code (inference.py always uses float32 for VAE).

### Pitfall 4: Boundary Ratio Confusion (Training vs Inference)
**What goes wrong:** Using training boundary (0.875) for inference, or inference boundary (0.5) for training.
**Why it happens:** Two different values exist for a reason: training uses 0.875 to match the model's intended noise-level split, while inference uses 0.5 for balanced dual-expert sampling.
**How to avoid:** Registry stores training values (0.875). Inference pipeline hardcodes 0.5. Keep them separate.
**Warning signs:** Inference quality changes unexpectedly; one expert dominates output.

### Pitfall 5: Stale Test Scripts Diverging from Main Code
**What goes wrong:** test_base_inference.py does NOT have the `config=` fix, while test3-inference.py does. Running the older script gives misleading results.
**Why it happens:** Test scripts on the pod were edited independently of the main codebase during debugging sessions.
**How to avoid:** All inference code paths must flow through `WanInferencePipeline` or be updated in sync. Update test_base_inference.py to match test3-inference.py's fix.

## Code Examples

### Example 1: Fixed Model Loading in backend.py
```python
# CURRENT (broken) — backend.py line 200
model = WanTransformer3DModel.from_single_file(
    single_file,
    torch_dtype=dtype,
    device="cpu",
)

# FIXED — add explicit config and subfolder
model = WanTransformer3DModel.from_single_file(
    single_file,
    torch_dtype=dtype,
    device="cpu",
    config="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    subfolder=self._resolve_config_subfolder(expert),
)

# Helper to determine which subfolder config to use
# transformer = high-noise expert, transformer_2 = low-noise expert
def _resolve_config_subfolder(self, expert: str | None) -> str:
    if expert == "low_noise":
        return "transformer_2"
    return "transformer"  # high_noise or None (unified)
```

### Example 2: Keyframe Grid Generation
```python
from PIL import Image

def save_keyframe_grid(frames, output_path, frame_indices=(0, 4, 8, 12, 16)):
    """Save selected frames as a PNG grid for quick visual review.

    Args:
        frames: List of PIL Images or numpy arrays (H, W, C).
        output_path: Path to save the grid PNG.
        frame_indices: Which frame indices to include (0-based).
    """
    import numpy as np

    images = []
    for idx in frame_indices:
        if idx >= len(frames):
            break
        frame = frames[idx]
        if isinstance(frame, np.ndarray):
            if frame.dtype != np.uint8:
                frame = (frame * 255).clip(0, 255).astype(np.uint8)
            frame = Image.fromarray(frame)
        images.append(frame)

    if not images:
        return

    # Horizontal grid: all frames side by side
    widths = [img.width for img in images]
    height = max(img.height for img in images)
    grid = Image.new("RGB", (sum(widths), height))
    x_offset = 0
    for img in images:
        grid.paste(img, (x_offset, 0))
        x_offset += img.width
    grid.save(output_path)
```

### Example 3: Side-by-Side LoRA Comparison
```python
# Generate with identical prompt + seed, once without LoRA, once with
def compare_base_vs_lora(pipeline, prompt_embeds, neg_embeds, lora_sd, seed=42):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Base model output
    base_output = pipeline(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=neg_embeds,
        num_inference_steps=30,
        guidance_scale=4.0,
        guidance_scale_2=3.0,
        height=480, width=832, num_frames=17,
        generator=generator,
    )

    # Apply LoRA
    pipeline.load_lora_weights(lora_sd, adapter_name="test")

    # Same seed for direct comparison
    generator = torch.Generator(device="cuda").manual_seed(seed)
    lora_output = pipeline(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=neg_embeds,
        num_inference_steps=30,
        guidance_scale=4.0,
        guidance_scale_2=3.0,
        height=480, width=832, num_frames=17,
        generator=generator,
    )

    pipeline.unload_lora_weights()
    return base_output, lora_output
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `from_single_file()` without config | `from_single_file(config="Wan-AI/...", subfolder="...")` | Jan 2026 (diffusers#12329 resolved) | Explicit config prevents silent Wan 2.1/2.2 misdetection |
| ai-toolkit: UniPC scheduler for generation | Minta: FlowMatchEuler with shift=5.0 | Validated by Minta | Different scheduler choice; both work but parameters differ |
| ai-toolkit: boundary_ratio=0.875 for both train+infer | Dimljus: 0.875 train, 0.5 infer | Minta's methodology | Intentional split for different purposes |
| Manual latent denormalization | Stock WanPipeline handles it internally | diffusers 0.33+ | No need for custom `latents / std + mean` code |

**Key divergence from ai-toolkit:**
- ai-toolkit uses `from_pretrained()` exclusively (avoids the bug entirely)
- ai-toolkit uses UniPCMultistepScheduler with flow_shift=3.0 for generation
- ai-toolkit uses boundary_ratio=0.875 for both training AND generation
- ai-toolkit has a custom Wan22Pipeline that reimplements the denoising loop
- Dimljus uses stock WanPipeline, FlowMatchEuler with shift=5.0, boundary_ratio=0.5 for inference
- These are hyperparameter differences (Minta's domain), not code bugs

## Open Questions

1. **Does the config= fix require network access on first use?**
   - What we know: `config="Wan-AI/Wan2.2-T2V-A14B-Diffusers"` triggers a HuggingFace Hub config file download the first time. After that, it is cached.
   - What's unclear: Whether the RunPod pod already has this config cached from previous `from_pretrained` calls, or if it needs HF_TOKEN + network access.
   - Recommendation: The pod has HF_TOKEN set and network access. If caching is an issue, the config JSON can be downloaded once during setup.sh. LOW risk.

2. **Is the from_single_file config= fix sufficient alone, or are there additional issues?**
   - What we know: test3-inference.py has the fix and was committed, but MEMORY.md says "inference still produces noise" even after that commit. However, MEMORY.md also says "inference was working before this session" and changes got lost during code edits.
   - What's unclear: Whether the noise in test3 was because the pod was running OLD code (before the fix was pushed) or because the fix itself is insufficient.
   - Recommendation: Follow the escalation order. Apply the fix to ALL from_single_file calls in the main codebase (backend.py + inference.py for VAE), push to pod, test. If still noise, escalate to from_pretrained.

3. **Should backend.py's load_model() also get the config= fix?**
   - What we know: backend.py uses from_single_file at line 200 without config=. This is used during TRAINING model loading, not inference. But it affects training-time sample generation because the same model is used.
   - What's unclear: Whether training itself works despite the wrong config (loss decreases suggest it does -- the forward pass may work even with wrong config metadata if the weights are identical).
   - Recommendation: Fix it everywhere. Even if training works by accident, wrong config metadata could cause subtle issues.

## Sources

### Primary (HIGH confidence)
- [diffusers#12329](https://github.com/huggingface/diffusers/issues/12329) - from_single_file config mismatch bug. Closed COMPLETED Jan 2026. Explicit `config=` parameter is the confirmed fix.
- [diffusers WanPipeline source](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py) - Stock pipeline handles latent normalization, dual-expert switching, CFG automatically.
- [diffusers official docs](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/wan.md) - Confirms VAE float32 requirement, shift value guidance, LoRA loading for Wan 2.2 dual denoisers.
- ai-toolkit source code (local: `ai-toolkit-source/extensions_built_in/diffusion_models/wan22/`) - Reference implementation using from_pretrained exclusively.

### Secondary (MEDIUM confidence)
- Dimljus codebase diff (d2c236d vs current) - Shows all changes since last known-good state. Changes are well-understood.
- MEMORY.md session history - Documents what was tried and what produced noise. Useful context but written during active debugging (may have gaps).

### Tertiary (LOW confidence)
- Whether the RunPod pod was running the latest code when tests produced noise (need to verify by checking git state on pod).

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Using official diffusers WanPipeline, well-documented API
- Architecture: HIGH - Root cause identified (from_single_file bug), fix pattern validated in test3-inference.py
- Pitfalls: HIGH - All pitfalls documented from direct code analysis and confirmed bugs

**Research date:** 2026-02-26
**Valid until:** 2026-03-26 (stable domain, diffusers API unlikely to change significantly)
