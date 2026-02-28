"""Diagnostic: isolate WHY test4-inference.py produces noise.

Hypothesis: pipeline.to("cuda") breaks inference. In every working path
(test3, WanInferencePipeline), components are moved to cuda INDIVIDUALLY
before constructing WanPipeline. test4 is the ONLY path that calls
pipeline.to("cuda"), and it's the ONLY path that produces noise.

This script tests both patterns with the SAME prompt, seed, and settings.
If Method A (test3 pattern) produces clean output and Method B (test4 pattern)
produces noise, we've found the bug.

Usage:
    python /workspace/dimljus/runpod/diagnose_inference.py
"""
import torch
import gc
import numpy as np
from pathlib import Path

MODEL_HIGH = "/workspace/models/wan2.2_t2v_high_noise_14B_fp16.safetensors"
MODEL_LOW = "/workspace/models/wan2.2_t2v_low_noise_14B_fp16.safetensors"
VAE_PATH = "/workspace/models/wan_2.1_vae.safetensors"
T5_PATH = "/workspace/models/models_t5_umt5-xxl-enc-bf16.pth"
HF_REPO = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

OUT_DIR = Path("/workspace/outputs/inference_diagnosis")

PROMPT = "Medium shot, Holly Golightly walks up an indoor staircase, looking back over her shoulder, morning light."
NEG = "blurry, low quality, distorted"
SEED = 42
STEPS = 30
GUIDANCE = 4.0
GUIDANCE_2 = 3.0
SHIFT = 5.0


def clean():
    gc.collect()
    torch.cuda.empty_cache()


def check_tensor_stats(name, tensor):
    """Print diagnostic stats for a tensor."""
    print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}, "
          f"device={tensor.device}, min={tensor.min().item():.4f}, "
          f"max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}")


def save_grid(pil_frames, path):
    """Save keyframe grid from PIL frames."""
    from PIL import Image
    n = len(pil_frames)
    indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    selected = [pil_frames[min(i, n - 1)] for i in indices]
    w, h = selected[0].size
    grid = Image.new("RGB", (w * len(selected), h))
    for i, img in enumerate(selected):
        grid.paste(img, (i * w, 0))
    grid.save(path)
    arr = np.array(grid).astype(np.float32) / 255.0
    print(f"  Grid saved: {path.name} | mean={arr.mean():.4f}, std={arr.std():.4f}, "
          f"min={arr.min():.4f}, max={arr.max():.4f}")


def extract_pil(output):
    """Extract PIL frames from pipeline output.

    WanPipeline returns various formats depending on version:
      - list[list[PIL.Image]]  (batch of videos)
      - list[PIL.Image]
      - WanPipelineOutput with .frames attribute
    We always unwrap to a flat list of PIL Images.
    """
    frames = output.frames if hasattr(output, "frames") else output

    # Debug: print what we got
    print(f"  [debug] output type: {type(output)}")
    print(f"  [debug] frames type: {type(frames)}, len: {len(frames) if hasattr(frames, '__len__') else 'N/A'}")
    if isinstance(frames, (list, tuple)) and len(frames) > 0:
        print(f"  [debug] frames[0] type: {type(frames[0])}")
        if isinstance(frames[0], (list, tuple)) and len(frames[0]) > 0:
            print(f"  [debug] frames[0][0] type: {type(frames[0][0])}")
            print(f"  [debug] frames[0] len: {len(frames[0])}")

    # Unwrap nested lists until we get PIL Images
    if isinstance(frames, (list, tuple)) and len(frames) > 0:
        if isinstance(frames[0], (list, tuple)):
            frames = frames[0]

    # Handle numpy arrays
    if hasattr(frames, 'shape'):
        import numpy as _np
        from PIL import Image as _Img
        arr = frames
        while arr.ndim > 4:
            arr = arr[0]
        pil_list = []
        for i in range(arr.shape[0]):
            f = arr[i]
            if f.dtype != _np.uint8:
                if f.max() <= 1.0:
                    f = (f * 255).clip(0, 255).astype(_np.uint8)
                else:
                    f = f.clip(0, 255).astype(_np.uint8)
            pil_list.append(_Img.fromarray(f))
        return pil_list

    return frames


def main():
    print("=" * 70)
    print("  INFERENCE DIAGNOSIS: pipeline.to('cuda') hypothesis")
    print("=" * 70)

    import diffusers
    print(f"\n  diffusers: {diffusers.__version__}")
    print(f"  torch: {torch.__version__}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load T5, encode prompts, free T5 ──────────────────
    # (identical in both test3 and test4)
    print("\n--- Step 1: Encode prompts ---")
    from transformers import AutoTokenizer, UMT5EncoderModel, UMT5Config
    from dimljus.training.wan.inference import WanInferencePipeline

    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
    config = UMT5Config.from_pretrained("google/umt5-xxl")
    text_encoder = UMT5EncoderModel(config)
    t5_sd = torch.load(T5_PATH, map_location="cpu", weights_only=True)
    text_encoder.load_state_dict(t5_sd, strict=False)
    del t5_sd
    WanInferencePipeline._fix_t5_embed_tokens(text_encoder)
    text_encoder = text_encoder.to("cuda", dtype=torch.bfloat16).eval()

    tokens = tokenizer(
        PROMPT, max_length=512, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        prompt_embeds = text_encoder(
            input_ids=tokens.input_ids.to("cuda"),
            attention_mask=tokens.attention_mask.to("cuda"),
        ).last_hidden_state

    tokens_neg = tokenizer(
        NEG, max_length=512, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        neg_embeds = text_encoder(
            input_ids=tokens_neg.input_ids.to("cuda"),
            attention_mask=tokens_neg.attention_mask.to("cuda"),
        ).last_hidden_state

    # Store on CPU like test4 does (we'll .to("cuda") when needed)
    prompt_embeds_cpu = prompt_embeds.cpu()
    neg_embeds_cpu = neg_embeds.cpu()

    check_tensor_stats("prompt_embeds (cuda)", prompt_embeds)
    check_tensor_stats("prompt_embeds (cpu)", prompt_embeds_cpu)
    check_tensor_stats("neg_embeds (cuda)", neg_embeds)

    del text_encoder, tokenizer
    clean()
    print(f"  T5 freed. VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # ── Step 2: Load models ────────────────────────────────────────
    print("\n--- Step 2: Load expert models ---")
    from diffusers.models import WanTransformer3DModel, AutoencoderKLWan
    from diffusers import WanPipeline, FlowMatchEulerDiscreteScheduler

    model_high = WanTransformer3DModel.from_single_file(
        MODEL_HIGH, torch_dtype=torch.bfloat16,
        config=HF_REPO, subfolder="transformer",
    ).to("cuda").eval()

    model_low = WanTransformer3DModel.from_single_file(
        MODEL_LOW, torch_dtype=torch.bfloat16,
        config=HF_REPO, subfolder="transformer_2",
    ).to("cuda").eval()

    print(f"  Both experts loaded. VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # ════════════════════════════════════════════════════════════════
    # METHOD A: test3 pattern (KNOWN WORKING)
    #   - VAE moved to cuda BEFORE pipeline construction
    #   - NO pipeline.to("cuda")
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  METHOD A: test3 pattern — components pre-moved, no pipeline.to()")
    print("=" * 70)

    vae_a = AutoencoderKLWan.from_single_file(
        VAE_PATH, torch_dtype=torch.float32
    ).to("cuda")

    print(f"  VAE dtype: {next(vae_a.parameters()).dtype}")
    print(f"  VAE device: {next(vae_a.parameters()).device}")

    scheduler_a = FlowMatchEulerDiscreteScheduler(shift=SHIFT)
    pipeline_a = WanPipeline(
        transformer=model_high,
        transformer_2=model_low,
        vae=vae_a,
        text_encoder=None,
        tokenizer=None,
        scheduler=scheduler_a,
        boundary_ratio=0.6,
    )

    # Check pipeline state
    print(f"  Pipeline VAE dtype after construction: {next(pipeline_a.vae.parameters()).dtype}")
    print(f"  Pipeline VAE device after construction: {next(pipeline_a.vae.parameters()).device}")

    # Use prompt_embeds that stayed on cuda (test3 pattern)
    generator_a = torch.Generator(device="cuda").manual_seed(SEED)

    print("  Generating (Method A)...")
    with torch.no_grad():
        output_a = pipeline_a(
            prompt_embeds=prompt_embeds_cpu.to("cuda", dtype=torch.bfloat16),
            negative_prompt_embeds=neg_embeds_cpu.to("cuda", dtype=torch.bfloat16),
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            guidance_scale_2=GUIDANCE_2,
            height=480, width=832, num_frames=17,
            generator=generator_a,
        )

    frames_a = extract_pil(output_a)
    print(f"  Got {len(frames_a)} frames")
    save_grid(frames_a, OUT_DIR / "method_a_test3_pattern.png")

    # Clean up VAE (keep transformers for Method B)
    del vae_a, pipeline_a, output_a
    clean()

    # ════════════════════════════════════════════════════════════════
    # METHOD B: test4 pattern (SUSPECTED BROKEN)
    #   - VAE NOT moved to cuda before pipeline construction
    #   - pipeline.to("cuda") after construction
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  METHOD B: test4 pattern — VAE on CPU, pipeline.to('cuda')")
    print("=" * 70)

    vae_b = AutoencoderKLWan.from_single_file(
        VAE_PATH, torch_dtype=torch.float32
    )
    # NOTE: NOT calling .to("cuda") on VAE — test4 pattern

    print(f"  VAE dtype BEFORE pipeline.to(): {next(vae_b.parameters()).dtype}")
    print(f"  VAE device BEFORE pipeline.to(): {next(vae_b.parameters()).device}")

    scheduler_b = FlowMatchEulerDiscreteScheduler(shift=SHIFT)
    pipeline_b = WanPipeline(
        transformer=model_high,
        transformer_2=model_low,
        vae=vae_b,
        text_encoder=None,
        tokenizer=None,
        scheduler=scheduler_b,
        boundary_ratio=0.6,
    )

    # This is the suspected culprit
    print("  Calling pipeline.to('cuda')...")
    pipeline_b = pipeline_b.to("cuda")

    # Check what pipeline.to() did to the VAE
    print(f"  VAE dtype AFTER pipeline.to(): {next(pipeline_b.vae.parameters()).dtype}")
    print(f"  VAE device AFTER pipeline.to(): {next(pipeline_b.vae.parameters()).device}")

    # Also check transformer dtype (should still be bfloat16)
    print(f"  Transformer dtype AFTER pipeline.to(): {next(pipeline_b.transformer.parameters()).dtype}")

    generator_b = torch.Generator(device="cuda").manual_seed(SEED)

    print("  Generating (Method B)...")
    with torch.no_grad():
        output_b = pipeline_b(
            prompt_embeds=prompt_embeds_cpu.to("cuda", dtype=torch.bfloat16),
            negative_prompt_embeds=neg_embeds_cpu.to("cuda", dtype=torch.bfloat16),
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            guidance_scale_2=GUIDANCE_2,
            height=480, width=832, num_frames=17,
            generator=generator_b,
        )

    frames_b = extract_pil(output_b)
    print(f"  Got {len(frames_b)} frames")
    save_grid(frames_b, OUT_DIR / "method_b_test4_pattern.png")

    del vae_b, pipeline_b, output_b
    clean()

    # ════════════════════════════════════════════════════════════════
    # METHOD C: test4 pattern with manual VAE dtype preservation
    #   - Same as B but force VAE back to float32 after pipeline.to()
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  METHOD C: test4 pattern + VAE dtype forced back to float32")
    print("=" * 70)

    vae_c = AutoencoderKLWan.from_single_file(
        VAE_PATH, torch_dtype=torch.float32
    )

    scheduler_c = FlowMatchEulerDiscreteScheduler(shift=SHIFT)
    pipeline_c = WanPipeline(
        transformer=model_high,
        transformer_2=model_low,
        vae=vae_c,
        text_encoder=None,
        tokenizer=None,
        scheduler=scheduler_c,
        boundary_ratio=0.6,
    )

    pipeline_c = pipeline_c.to("cuda")

    # Force VAE back to float32 (in case pipeline.to changed it)
    pipeline_c.vae = pipeline_c.vae.to(dtype=torch.float32)
    print(f"  VAE dtype after force: {next(pipeline_c.vae.parameters()).dtype}")
    print(f"  VAE device after force: {next(pipeline_c.vae.parameters()).device}")

    generator_c = torch.Generator(device="cuda").manual_seed(SEED)

    print("  Generating (Method C)...")
    with torch.no_grad():
        output_c = pipeline_c(
            prompt_embeds=prompt_embeds_cpu.to("cuda", dtype=torch.bfloat16),
            negative_prompt_embeds=neg_embeds_cpu.to("cuda", dtype=torch.bfloat16),
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            guidance_scale_2=GUIDANCE_2,
            height=480, width=832, num_frames=17,
            generator=generator_c,
        )

    frames_c = extract_pil(output_c)
    print(f"  Got {len(frames_c)} frames")
    save_grid(frames_c, OUT_DIR / "method_c_forced_float32.png")

    del vae_c, pipeline_c, output_c
    clean()

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  DIAGNOSIS COMPLETE")
    print("=" * 70)
    print(f"\n  Output directory: {OUT_DIR}")
    print(f"  method_a_test3_pattern.png  — components pre-moved (test3 pattern)")
    print(f"  method_b_test4_pattern.png  — pipeline.to('cuda') (test4 pattern)")
    print(f"  method_c_forced_float32.png — pipeline.to + force VAE float32")
    print()
    print("  If A is clean and B is noise → pipeline.to('cuda') is the bug")
    print("  If A is clean and C is clean → pipeline.to() changed VAE dtype")
    print("  If A is also noise → pod environment issue")
    print("=" * 70)


if __name__ == "__main__":
    main()
