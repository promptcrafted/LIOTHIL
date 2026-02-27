"""Diagnostic inference test -- dual expert, proper Wan 2.2 pipeline.

Tests:
  A. Both experts, no LoRA, 30 steps -- verify base model generates coherently
  B. Both experts + LoRA via load_lora_weights() -- verify the LoRA loading path

Approach:
  Load the full pipeline via WanPipeline.from_pretrained() (which gets the
  correct scheduler, T5, and config), then swap the transformer weights to
  our local from_single_file copies. This ensures all pipeline plumbing
  matches the HF defaults while using our local model files.

KEY FIXES:
  1. (diffusers#12329) Every WanTransformer3DModel.from_single_file() call MUST
     include config= and subfolder= to prevent silent Wan 2.1 misdetection.
  2. T5 loaded from HF repo (Wan-AI/Wan2.2-T2V-A14B-Diffusers), NOT from the
     standalone .pth file which contains wrong/uninitialized weights.
  3. Pipeline handles T5 encoding internally -- avoids embed_tokens weight
     tying issues that occur with separate T5 loading.

Usage:
    HF_TOKEN=hf_xxx python /workspace/dimljus/runpod/test_base_inference.py
"""
import torch
import gc
import os
import numpy as np
from pathlib import Path

# Ensure HF token is available for gated repos.
# Set HF_TOKEN in your environment (RunPod pod settings or export HF_TOKEN=...).
if not os.environ.get("HF_TOKEN"):
    print("WARNING: HF_TOKEN not set. Downloads from gated repos may fail.")
    print("  Fix: export HF_TOKEN=hf_your_token_here")
# Use workspace cache to avoid filling root partition
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")


def clean():
    gc.collect()
    torch.cuda.empty_cache()


def inspect_output(frames, label):
    """Print diagnostic info about pipeline output."""
    print(f"\n  [{label}] Output type: {type(frames)}")
    if isinstance(frames, np.ndarray):
        print(f"  [{label}] Shape: {frames.shape}, dtype: {frames.dtype}")
        print(f"  [{label}] Value range: min={frames.min():.3f}, max={frames.max():.3f}, mean={frames.mean():.3f}")
    elif isinstance(frames, (list, tuple)):
        print(f"  [{label}] List length: {len(frames)}")
        if frames and isinstance(frames[0], (list, tuple)):
            print(f"  [{label}] Inner length: {len(frames[0])}")


def extract_pil_frames(frames):
    """Convert pipeline output to a list of PIL Images."""
    from PIL import Image

    if isinstance(frames, np.ndarray):
        arr = frames
        while arr.ndim > 4:
            arr = arr[0]
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        return [Image.fromarray(arr[i]) for i in range(arr.shape[0])]
    elif isinstance(frames, (list, tuple)) and frames:
        inner = frames[0]
        if isinstance(inner, (list, tuple)):
            return list(inner)
        elif hasattr(inner, "save"):
            return list(frames)
    return None


def save_outputs(frames, base_path, label):
    """Save first frame, keyframe grid, and video from pipeline output."""
    pil_frames = extract_pil_frames(frames)
    if not pil_frames:
        print(f"  [{label}] Could not extract frames")
        return

    from PIL import Image

    # First frame
    png_path = base_path.with_suffix(".png")
    pil_frames[0].save(png_path)
    print(f"  [{label}] First frame: {png_path} ({png_path.stat().st_size / 1024:.0f} KB)")

    # Keyframe grid (5 evenly spaced frames)
    n = len(pil_frames)
    indices = [int(i * (n - 1) / min(4, n - 1)) for i in range(min(5, n))]
    selected = [pil_frames[i] for i in indices]
    w, h = selected[0].size
    grid = Image.new("RGB", (w * len(selected), h))
    for i, img in enumerate(selected):
        grid.paste(img, (i * w, 0))
    grid_path = base_path.with_suffix(".grid.png")
    grid.save(grid_path)
    print(f"  [{label}] Grid: {grid_path} ({grid_path.stat().st_size / 1024:.0f} KB)")

    # Video
    try:
        from diffusers.utils import export_to_video
        mp4_path = base_path.with_suffix(".mp4")
        export_to_video(pil_frames, str(mp4_path), fps=16)
        print(f"  [{label}] Video: {mp4_path} ({mp4_path.stat().st_size / 1024:.0f} KB)")
    except Exception as e:
        print(f"  [{label}] Warning: Could not save video: {e}")


def main():
    PROMPT = "A woman with dark hair walks down a city street, morning light"
    NEG = "blurry, low quality, distorted"
    LORA_PATH = "/workspace/outputs/test2-unified-only/unified/test_lora_unified_epoch003.safetensors"

    MODEL_HIGH = "/workspace/models/wan2.2_t2v_high_noise_14B_fp16.safetensors"
    MODEL_LOW = "/workspace/models/wan2.2_t2v_low_noise_14B_fp16.safetensors"
    VAE_PATH = "/workspace/models/wan_2.1_vae.safetensors"

    # HF repo for from_pretrained pipeline (gets correct T5, scheduler, config)
    HF_REPO = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    # Generation settings
    HEIGHT = 480
    WIDTH = 832
    NUM_FRAMES = 17
    STEPS = 30
    GUIDANCE_SCALE = 4.0

    # ── Step 1: Load full pipeline from HF ─────────────────────────
    # The from_pretrained() approach gets the correct T5, scheduler (UniPC),
    # boundary_ratio (0.875), and text encoding pipeline. We then swap the
    # transformers to our local from_single_file copies.
    print(f"Loading pipeline from {HF_REPO}...")
    from diffusers import WanPipeline
    from diffusers.models import WanTransformer3DModel, AutoencoderKLWan

    pipeline = WanPipeline.from_pretrained(
        HF_REPO, torch_dtype=torch.bfloat16,
    )
    # VAE in float32 for quality (official recommendation)
    pipeline.vae = pipeline.vae.to(dtype=torch.float32)

    print(f"  Pipeline loaded (from_pretrained)")
    print(f"  Scheduler: {type(pipeline.scheduler).__name__}")
    if hasattr(pipeline, 'boundary_ratio'):
        print(f"  boundary_ratio: {pipeline.config.boundary_ratio}")

    # ── Step 2: Swap transformers to local from_single_file copies ──
    # This proves our local model files produce the same output as the
    # HF-hosted weights. If output differs, the from_single_file loading
    # or the model weights themselves are the issue.
    print("\nSwapping transformers to local from_single_file copies...")
    # Free the from_pretrained transformers first to save VRAM
    old_t = pipeline.transformer
    old_t2 = pipeline.transformer_2
    del old_t, old_t2
    clean()

    model_high = WanTransformer3DModel.from_single_file(
        MODEL_HIGH, torch_dtype=torch.bfloat16,
        config=HF_REPO, subfolder="transformer",
    )
    model_low = WanTransformer3DModel.from_single_file(
        MODEL_LOW, torch_dtype=torch.bfloat16,
        config=HF_REPO, subfolder="transformer_2",
    )
    pipeline.transformer = model_high
    pipeline.transformer_2 = model_low

    # Also swap VAE to our local copy (optional but proves it works)
    if Path(VAE_PATH).exists():
        pipeline.vae = AutoencoderKLWan.from_single_file(
            VAE_PATH, torch_dtype=torch.float32
        )

    pipeline = pipeline.to("cuda")
    print(f"  Swapped. VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # ── Test A: Both experts, no LoRA ─────────────────────────────
    print("\n" + "=" * 60)
    print(f"  TEST A: Both experts, no LoRA, {STEPS} steps, {HEIGHT}x{WIDTH}, {NUM_FRAMES} frames")
    print(f"  Using pipeline's native prompt encoding (T5 from HF)")
    print("=" * 60)

    generator = torch.Generator(device="cuda").manual_seed(42)
    with torch.no_grad():
        output = pipeline(
            prompt=PROMPT,
            negative_prompt=NEG,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE_SCALE,
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            generator=generator,
        )

    frames = output.frames if hasattr(output, "frames") else output
    inspect_output(frames, "DUAL-BASE")

    out_a = Path("/workspace/test_A_dual_base")
    save_outputs(frames, out_a, "DUAL-BASE")

    # ── Test B: LoRA via pipeline.load_lora_weights() ─────────────
    print("\n" + "=" * 60)
    print(f"  TEST B: Both experts + LoRA via load_lora_weights(), {STEPS} steps")
    print("=" * 60)

    if not Path(LORA_PATH).exists():
        print(f"  SKIPPED: LoRA not found at {LORA_PATH}")
        print("  Run test2-unified-only training first to generate this checkpoint")
    else:
        from safetensors.torch import load_file
        from dimljus.training.wan.checkpoint_io import (
            has_diffusers_prefix,
            add_diffusers_prefix,
        )

        lora_sd = load_file(LORA_PATH)
        print(f"  LoRA: {len(lora_sd)} keys, {sum(v.numel() for v in lora_sd.values()) / 1e6:.1f}M params")

        # Auto-prefix if needed (old checkpoints may not have transformer. prefix)
        if not has_diffusers_prefix(lora_sd):
            print("  Adding transformer. prefix for diffusers compatibility...")
            lora_sd = add_diffusers_prefix(lora_sd, prefix="transformer")

        # Apply LoRA via the standard diffusers API
        print(f"  Loading LoRA via pipeline.load_lora_weights()...")
        pipeline.load_lora_weights(lora_sd, adapter_name="dimljus")
        print(f"  LoRA applied. VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

        generator = torch.Generator(device="cuda").manual_seed(42)
        with torch.no_grad():
            output = pipeline(
                prompt=PROMPT,
                negative_prompt=NEG,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE_SCALE,
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
                generator=generator,
            )

        frames = output.frames if hasattr(output, "frames") else output
        inspect_output(frames, "LORA")

        out_b = Path("/workspace/test_B_lora_diffusers")
        save_outputs(frames, out_b, "LORA")

        # Unload LoRA for next test
        pipeline.unload_lora_weights()
        print("  LoRA unloaded")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print("  Check .grid.png files -- quickest way to see if output is coherent.")
    print()
    for label, path_str in [("Dual-Base", "/workspace/test_A_dual_base"),
                            ("LoRA", "/workspace/test_B_lora_diffusers")]:
        grid = Path(f"{path_str}.grid.png")
        mp4 = Path(f"{path_str}.mp4")
        png = Path(f"{path_str}.png")
        if grid.exists():
            print(f"  {label} GRID: {grid} ({grid.stat().st_size / 1024:.0f} KB)")
        if mp4.exists():
            print(f"  {label} VIDEO: {mp4} ({mp4.stat().st_size / 1024:.0f} KB)")
        if png.exists():
            print(f"  {label} FRAME: {png} ({png.stat().st_size / 1024:.0f} KB)")
        if not grid.exists() and not mp4.exists() and not png.exists():
            print(f"  {label}: FAILED or SKIPPED")

    clean()
    print("\nDone!")


if __name__ == "__main__":
    main()
