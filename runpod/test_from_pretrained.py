"""Ground truth test: fully stock WanPipeline.from_pretrained().

NO manual model loading. NO dimljus code. Pure diffusers pipeline from HF repo.
This isolates whether the issue is in our manual model assembly or something else.

If this produces coherent video → the bug is in how we load/assemble models.
If this also produces noise → the bug is in diffusers version, model weights, or environment.

Usage:
    HF_TOKEN=hf_xxx python /workspace/dimljus/runpod/test_from_pretrained.py
"""
import torch
import gc
import os
from pathlib import Path

# Ensure HF token is available for gated repos.
# Set HF_TOKEN in your environment (RunPod pod settings or export HF_TOKEN=...).
if not os.environ.get("HF_TOKEN"):
    print("WARNING: HF_TOKEN not set. Downloads from gated repos may fail.")
    print("  Fix: export HF_TOKEN=hf_your_token_here")


def clean():
    gc.collect()
    torch.cuda.empty_cache()


def save_grid(frames, path):
    """Save a keyframe grid from pipeline output for quick visual check."""
    try:
        import numpy as np
        from PIL import Image

        pil_frames = None

        # Handle numpy array output: (batch, frames, H, W, C) or (frames, H, W, C)
        if isinstance(frames, np.ndarray):
            print(f"  Numpy output shape: {frames.shape}, dtype: {frames.dtype}")
            print(f"  Value range: min={frames.min():.3f}, max={frames.max():.3f}, mean={frames.mean():.3f}")
            arr = frames
            while arr.ndim > 4:
                arr = arr[0]  # Remove batch dims
            # arr is now (frames, H, W, C)
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(np.uint8)
            pil_frames = [Image.fromarray(arr[i]) for i in range(arr.shape[0])]

        # Handle list[list[PIL.Image]] format
        elif isinstance(frames, (list, tuple)) and frames:
            inner = frames[0]
            if isinstance(inner, (list, tuple)) and inner:
                pil_frames = inner
            elif hasattr(inner, "save"):
                pil_frames = frames
            else:
                print(f"  Could not parse frames: inner type={type(inner)}")
                return
        else:
            print(f"  Could not parse frames: type={type(frames)}")
            return

        if not pil_frames:
            print("  No frames to save")
            return

        print(f"  Extracted {len(pil_frames)} frames, size: {pil_frames[0].size}")

        # Pick up to 5 evenly-spaced frames for the grid
        n = len(pil_frames)
        indices = [int(i * (n - 1) / min(4, n - 1)) for i in range(min(5, n))]
        selected = [pil_frames[i] for i in indices]

        # Build horizontal grid
        w, h = selected[0].size
        grid = Image.new("RGB", (w * len(selected), h))
        for i, img in enumerate(selected):
            grid.paste(img, (i * w, 0))

        grid_path = path.with_suffix(".grid.png")
        grid.save(grid_path)
        print(f"  Grid saved: {grid_path} ({grid_path.stat().st_size / 1024:.0f} KB)")

        # Also save first frame alone
        png_path = path.with_suffix(".png")
        selected[0].save(png_path)
        print(f"  First frame: {png_path} ({png_path.stat().st_size / 1024:.0f} KB)")

    except Exception as e:
        print(f"  Warning: Could not save grid: {e}")
        import traceback
        traceback.print_exc()


def save_video(frames, path, fps=16):
    """Save frames as MP4 video."""
    try:
        import numpy as np
        from PIL import Image
        from diffusers.utils import export_to_video

        # Convert numpy to list of PIL images first
        if isinstance(frames, np.ndarray):
            arr = frames
            while arr.ndim > 4:
                arr = arr[0]
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(np.uint8)
            video_frames = [Image.fromarray(arr[i]) for i in range(arr.shape[0])]
        elif isinstance(frames, (list, tuple)) and frames:
            inner = frames[0]
            if isinstance(inner, (list, tuple)):
                video_frames = inner
            else:
                video_frames = frames
        else:
            video_frames = frames

        export_to_video(video_frames, str(path), fps=fps)
        print(f"  Video saved: {path} ({path.stat().st_size / 1024:.0f} KB)")
    except Exception as e:
        print(f"  Warning: Could not save video: {e}")
        import traceback
        traceback.print_exc()


def main():
    PROMPT = "A woman with dark hair walks down a city street, morning light"
    NEG = "blurry, low quality, distorted"
    REPO = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    HEIGHT = 480
    WIDTH = 832
    NUM_FRAMES = 17
    STEPS = 30
    GUIDANCE_SCALE = 4.0

    print("=" * 60)
    print("  GROUND TRUTH: WanPipeline.from_pretrained()")
    print(f"  Model: {REPO}")
    print(f"  {HEIGHT}x{WIDTH}, {NUM_FRAMES} frames, {STEPS} steps, CFG={GUIDANCE_SCALE}")
    print("=" * 60)

    # ── Load entire pipeline from HF ──────────────────────────────
    print("\nLoading full pipeline from HF (this may download ~28GB on first run)...")
    from diffusers import WanPipeline

    pipeline = WanPipeline.from_pretrained(
        REPO,
        torch_dtype=torch.bfloat16,
    )

    # Move VAE to float32 for quality (official recommendation)
    pipeline.vae = pipeline.vae.to(dtype=torch.float32)
    pipeline = pipeline.to("cuda")

    print(f"  Pipeline loaded. VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # Print pipeline config for diagnosis
    print(f"\n  Pipeline type: {type(pipeline)}")
    print(f"  Has transformer: {pipeline.transformer is not None}")
    print(f"  Has transformer_2: {hasattr(pipeline, 'transformer_2') and pipeline.transformer_2 is not None}")
    if hasattr(pipeline, 'boundary_ratio'):
        print(f"  boundary_ratio: {pipeline.boundary_ratio}")
    print(f"  Scheduler: {type(pipeline.scheduler).__name__}")
    if hasattr(pipeline.scheduler, 'config'):
        sched_cfg = pipeline.scheduler.config
        print(f"  Scheduler shift: {getattr(sched_cfg, 'shift', 'N/A')}")

    # ── Generate ───────────────────────────────────────────────────
    print(f"\nGenerating...")
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
    print(f"\n  Output type: {type(frames)}")
    if isinstance(frames, (list, tuple)):
        print(f"  List length: {len(frames)}")
        if frames and isinstance(frames[0], (list, tuple)):
            print(f"  Inner length: {len(frames[0])}")

    # ── Save outputs ──────────────────────────────────────────────
    out_path = Path("/workspace/test_C_from_pretrained.mp4")
    save_grid(frames, out_path)
    save_video(frames, out_path)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULT")
    print("=" * 60)
    grid = Path("/workspace/test_C_from_pretrained.grid.png")
    if grid.exists():
        print(f"  GRID: {grid} ({grid.stat().st_size / 1024:.0f} KB)")
        print(f"  → If this grid shows coherent video: bug is in our manual loading")
        print(f"  → If this grid shows noise: bug is in diffusers/weights/environment")
    else:
        print("  GRID: NOT CREATED — check errors above")

    clean()
    print("\nDone!")


if __name__ == "__main__":
    main()
