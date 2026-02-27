"""Scheduler comparison test -- FlowMatchEuler vs UniPC with different boundary ratios.

Tests the scheduler settings that match Minta's ComfyUI workflow:
  - ComfyUI uses: euler sampler, simple scheduler, shift=5.0, boundary at step 15/25 = 0.6
  - HF defaults: UniPC, boundary=0.875

This script loads the pipeline ONCE, then runs multiple configs by swapping
the scheduler and boundary_ratio between runs. Same seed for all runs so
differences are purely from scheduler/boundary settings.

Usage:
    HF_TOKEN=hf_xxx python /workspace/dimljus/runpod/test_scheduler_comparison.py
"""
import torch
import gc
import os
import numpy as np
from pathlib import Path

if not os.environ.get("HF_TOKEN"):
    print("WARNING: HF_TOKEN not set.")
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")


def clean():
    gc.collect()
    torch.cuda.empty_cache()


def extract_pil_frames(frames):
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
    from PIL import Image
    pil_frames = extract_pil_frames(frames)
    if not pil_frames:
        print(f"  [{label}] Could not extract frames")
        return

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


def run_inference(pipeline, label, out_path, steps=30):
    """Run inference with current pipeline settings and save output."""
    PROMPT = "A woman with dark hair walks down a city street, morning light"
    NEG = "blurry, low quality, distorted"

    sched_name = type(pipeline.scheduler).__name__
    boundary = pipeline.config.get("boundary_ratio", "N/A")
    shift = getattr(pipeline.scheduler.config, "shift", "N/A")

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  Scheduler: {sched_name}, boundary: {boundary}, shift: {shift}")
    print(f"  Steps: {steps}, Size: 480x832, Frames: 17")
    print(f"{'=' * 60}")

    generator = torch.Generator(device="cuda").manual_seed(42)
    with torch.no_grad():
        output = pipeline(
            prompt=PROMPT,
            negative_prompt=NEG,
            num_inference_steps=steps,
            guidance_scale=4.0,
            height=480,
            width=832,
            num_frames=17,
            generator=generator,
        )

    frames = output.frames if hasattr(output, "frames") else output
    save_outputs(frames, out_path, label)
    clean()


def main():
    MODEL_HIGH = "/workspace/models/wan2.2_t2v_high_noise_14B_fp16.safetensors"
    MODEL_LOW = "/workspace/models/wan2.2_t2v_low_noise_14B_fp16.safetensors"
    VAE_PATH = "/workspace/models/wan_2.1_vae.safetensors"
    HF_REPO = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    OUT_DIR = Path("/workspace/outputs/scheduler_test")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load pipeline from HF ────────────────────────────────────
    print(f"Loading pipeline from {HF_REPO}...")
    from diffusers import WanPipeline
    from diffusers.models import WanTransformer3DModel, AutoencoderKLWan

    pipeline = WanPipeline.from_pretrained(HF_REPO, torch_dtype=torch.bfloat16)
    pipeline.vae = pipeline.vae.to(dtype=torch.float32)

    # Fix T5 embed_tokens weight tying
    t5 = pipeline.text_encoder
    if hasattr(t5, "shared") and hasattr(t5, "encoder") and hasattr(t5.encoder, "embed_tokens"):
        if not torch.equal(t5.shared.weight, t5.encoder.embed_tokens.weight):
            t5.encoder.embed_tokens.weight = t5.shared.weight
            print("  Fixed T5 embed_tokens weight tying")

    # ── Swap transformers to local copies ────────────────────────
    print("Swapping transformers to local from_single_file copies...")
    del pipeline.transformer, pipeline.transformer_2
    clean()

    pipeline.transformer = WanTransformer3DModel.from_single_file(
        MODEL_HIGH, torch_dtype=torch.bfloat16,
        config=HF_REPO, subfolder="transformer",
    )
    pipeline.transformer_2 = WanTransformer3DModel.from_single_file(
        MODEL_LOW, torch_dtype=torch.bfloat16,
        config=HF_REPO, subfolder="transformer_2",
    )

    if Path(VAE_PATH).exists():
        pipeline.vae = AutoencoderKLWan.from_single_file(
            VAE_PATH, torch_dtype=torch.float32
        )

    pipeline = pipeline.to("cuda")
    print(f"  Ready. VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # ── Save the original scheduler for restoration ──────────────
    original_scheduler = pipeline.scheduler

    # ──────────────────────────────────────────────────────────────
    # TEST 1: HF defaults -- UniPC, boundary=0.875 (known good)
    # ──────────────────────────────────────────────────────────────
    # Pipeline already has these settings from from_pretrained
    run_inference(pipeline, "TEST 1: UniPC boundary=0.875 (HF default)",
                  OUT_DIR / "test1_unipc_0.875")

    # ──────────────────────────────────────────────────────────────
    # TEST 2: FlowMatchEuler shift=5.0, boundary=0.6 (ComfyUI-aligned)
    # ──────────────────────────────────────────────────────────────
    from diffusers import FlowMatchEulerDiscreteScheduler
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    # Update pipeline config for boundary_ratio
    pipeline._internal_dict["boundary_ratio"] = 0.6
    run_inference(pipeline, "TEST 2: FlowMatchEuler shift=5.0 boundary=0.6",
                  OUT_DIR / "test2_euler_0.6")

    # ──────────────────────────────────────────────────────────────
    # TEST 3: FlowMatchEuler shift=5.0, boundary=0.875 (euler + HF boundary)
    # ──────────────────────────────────────────────────────────────
    pipeline._internal_dict["boundary_ratio"] = 0.875
    run_inference(pipeline, "TEST 3: FlowMatchEuler shift=5.0 boundary=0.875",
                  OUT_DIR / "test3_euler_0.875")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SCHEDULER COMPARISON SUMMARY")
    print("=" * 60)
    print()
    print("  Test 1: UniPC + boundary=0.875 (HF default, known good)")
    print("  Test 2: FlowMatchEuler shift=5.0 + boundary=0.6 (ComfyUI-aligned)")
    print("  Test 3: FlowMatchEuler shift=5.0 + boundary=0.875 (euler + HF boundary)")
    print()
    print(f"  All outputs in: {OUT_DIR}")
    print()

    for i, name in enumerate(["test1_unipc_0.875", "test2_euler_0.6", "test3_euler_0.875"], 1):
        grid = OUT_DIR / f"{name}.grid.png"
        if grid.exists():
            print(f"  Test {i} grid: {grid} ({grid.stat().st_size / 1024:.0f} KB)")
        else:
            print(f"  Test {i} grid: MISSING")

    clean()
    print("\nDone!")


if __name__ == "__main__":
    main()
