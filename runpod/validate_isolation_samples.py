"""Validate expert isolation sampling with dual-expert pipeline.

Quick validation that the sampling fixes (partner model loading +
diffusers <0.36 pin) produce recognizable output, not noise.

Loads the test4 low-noise LoRA checkpoint, builds the dual-expert
pipeline (trained expert + base partner), and generates one sample.
Then runs basic quality checks (not black, not uniform, has variance).

Usage:
    python /workspace/dimljus/runpod/validate_isolation_samples.py
"""
import sys
import gc
import time
import numpy as np
from pathlib import Path

# Paths
MODEL_HIGH = "/workspace/models/wan2.2_t2v_high_noise_14B_fp16.safetensors"
MODEL_LOW = "/workspace/models/wan2.2_t2v_low_noise_14B_fp16.safetensors"
HF_REPO = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

# Use test4 low-noise epoch 5 checkpoint for validation
LORA_PATH = "/workspace/outputs/test4-low-only/low_noise/test_lora_low_epoch005.safetensors"
OUT_DIR = Path("/workspace/outputs/validation")

PROMPT = "Medium shot, Holly Golightly walks up an indoor staircase, looking back over her shoulder, morning light."
NEG = "blurry, low quality, distorted"
STEPS = 20  # Fewer steps for quick validation (not full quality)
SEED = 42


def clean():
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass


def check_frame_quality(frames_pil):
    """Basic quality checks on generated frames.

    Returns (passed: bool, report: str) tuple.
    A frame that is all black, all white, uniform noise, or has very
    low variance is considered a failure.
    """
    checks = []

    if not frames_pil:
        return False, "No frames generated"

    # Check first and last frames
    for label, frame in [("first", frames_pil[0]), ("last", frames_pil[-1])]:
        arr = np.array(frame).astype(np.float32) / 255.0

        mean = arr.mean()
        std = arr.std()

        # Check for all black
        if mean < 0.05:
            checks.append(f"FAIL: {label} frame is nearly black (mean={mean:.4f})")
            continue

        # Check for all white / washed out
        if mean > 0.90:
            checks.append(f"FAIL: {label} frame is nearly white/washed out (mean={mean:.4f})")
            continue

        # Check for very low variance (uniform color or noise)
        if std < 0.05:
            checks.append(f"FAIL: {label} frame has very low variance (std={std:.4f})")
            continue

        checks.append(f"PASS: {label} frame looks valid (mean={mean:.4f}, std={std:.4f})")

    # Check inter-frame difference (motion detection)
    if len(frames_pil) >= 2:
        first = np.array(frames_pil[0]).astype(np.float32)
        last = np.array(frames_pil[-1]).astype(np.float32)
        diff = np.abs(first - last).mean() / 255.0
        if diff < 0.01:
            checks.append(f"WARN: Very little motion between first and last frame (diff={diff:.4f})")
        else:
            checks.append(f"PASS: Motion detected between frames (diff={diff:.4f})")

    passed = all("FAIL" not in c for c in checks)
    report = "\n".join(f"  {c}" for c in checks)
    return passed, report


def main():
    import torch
    from PIL import Image

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  VALIDATION: Expert Isolation Sampling Fixes")
    print("=" * 60)
    print(f"  LoRA: {LORA_PATH}")
    print(f"  Prompt: {PROMPT[:60]}...")
    print(f"  Steps: {STEPS}, Seed: {SEED}")
    print(f"  diffusers version: ", end="")
    import diffusers
    print(diffusers.__version__)

    if not Path(LORA_PATH).exists():
        print(f"\nERROR: LoRA checkpoint not found: {LORA_PATH}")
        sys.exit(1)

    # -- Load pipeline from HF (gets tokenizer + T5 + VAE + scheduler) --
    print("\nStep 1: Loading pipeline from HF repo...")
    t0 = time.time()

    from diffusers import WanPipeline, FlowMatchEulerDiscreteScheduler
    from diffusers.models import WanTransformer3DModel, AutoencoderKLWan

    pipeline = WanPipeline.from_pretrained(HF_REPO, torch_dtype=torch.bfloat16)
    pipeline.vae = pipeline.vae.to(dtype=torch.float32)

    # Fix T5 embed_tokens weight tying (required for all loading paths)
    t5 = pipeline.text_encoder
    if hasattr(t5, "shared") and hasattr(t5, "encoder") and hasattr(t5.encoder, "embed_tokens"):
        if not torch.equal(t5.shared.weight, t5.encoder.embed_tokens.weight):
            t5.encoder.embed_tokens.weight = t5.shared.weight
            print("  Fixed T5 embed_tokens weight tying")

    print(f"  Pipeline loaded in {time.time() - t0:.0f}s")

    # -- Swap in local expert models (single-file weights) --
    print("\nStep 2: Loading local expert models...")
    t0 = time.time()

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

    # Set validated scheduler: FlowMatchEuler shift=5.0, boundary=0.6
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    pipeline._internal_dict["boundary_ratio"] = 0.6

    pipeline = pipeline.to("cuda")
    print(f"  Both experts loaded in {time.time() - t0:.0f}s")
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # -- Load LoRA onto the low-noise expert (transformer_2) --
    print("\nStep 3: Loading LoRA checkpoint...")
    t0 = time.time()

    from safetensors.torch import load_file
    lora_sd = load_file(LORA_PATH)
    n_keys = len(lora_sd)
    n_params = sum(v.numel() for v in lora_sd.values())
    print(f"  LoRA: {n_keys} keys, {n_params / 1e6:.1f}M params")

    # Check key format — isolation LoRA should NOT have diffusers prefix
    sample_keys = list(lora_sd.keys())[:3]
    print(f"  Sample keys: {sample_keys}")

    # The isolation LoRA is for low_noise expert = transformer_2 in diffusers
    # Need to add "transformer_2." prefix for load_lora_weights()
    from dimljus.training.wan.checkpoint_io import has_diffusers_prefix, add_diffusers_prefix
    if not has_diffusers_prefix(lora_sd):
        print("  Adding transformer_2. prefix (low-noise expert)...")
        lora_sd = add_diffusers_prefix(lora_sd, prefix="transformer_2")

    pipeline.load_lora_weights(lora_sd, adapter_name="dimljus")
    print(f"  LoRA applied in {time.time() - t0:.0f}s")
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # -- Generate sample --
    print("\nStep 4: Generating validation sample...")
    t0 = time.time()

    generator = torch.Generator(device="cuda").manual_seed(SEED)
    with torch.no_grad():
        output = pipeline(
            prompt=PROMPT,
            negative_prompt=NEG,
            num_inference_steps=STEPS,
            guidance_scale=4.0,
            height=480, width=832, num_frames=17,
            generator=generator,
        )

    gen_time = time.time() - t0
    print(f"  Generated in {gen_time:.0f}s")

    # Extract frames -- debug the output format
    print(f"  Output type: {type(output)}")
    if hasattr(output, "frames"):
        raw = output.frames
        print(f"  output.frames type: {type(raw)}")
        if isinstance(raw, (list, tuple)):
            print(f"  output.frames length: {len(raw)}")
            if len(raw) > 0:
                print(f"  output.frames[0] type: {type(raw[0])}")
                if isinstance(raw[0], (list, tuple)):
                    print(f"  output.frames[0] length: {len(raw[0])}")
                    if len(raw[0]) > 0:
                        print(f"  output.frames[0][0] type: {type(raw[0][0])}")
        elif isinstance(raw, np.ndarray):
            print(f"  output.frames shape: {raw.shape}, dtype: {raw.dtype}")
    else:
        raw = output
        print(f"  No .frames attribute")

    # Flatten nested list structures from diffusers WanPipeline
    pil_frames = []
    frames = output.frames if hasattr(output, "frames") else output

    if isinstance(frames, np.ndarray):
        # Numpy array -- squeeze batch dim and convert to PIL
        while frames.ndim > 4:
            frames = frames[0]
        if frames.dtype != np.uint8:
            if frames.max() <= 1.0:
                frames = (frames * 255).clip(0, 255).astype(np.uint8)
            else:
                frames = frames.clip(0, 255).astype(np.uint8)
        from PIL import Image as PILImage
        for i in range(frames.shape[0]):
            pil_frames.append(PILImage.fromarray(frames[i]))
    elif isinstance(frames, (list, tuple)):
        # Could be list[list[PIL]] or list[PIL] or list[ndarray]
        flat = frames
        # Unwrap one level if nested
        if len(flat) > 0 and isinstance(flat[0], (list, tuple)):
            flat = flat[0]
        for item in flat:
            if hasattr(item, "save"):
                pil_frames.append(item)
            elif isinstance(item, np.ndarray):
                if item.dtype != np.uint8:
                    item = (item * 255).clip(0, 255).astype(np.uint8)
                from PIL import Image as PILImage
                pil_frames.append(PILImage.fromarray(item))

    print(f"  Got {len(pil_frames)} frames")

    # -- Save outputs --
    print("\nStep 5: Saving outputs...")
    if pil_frames:
        # Save first frame
        first_path = OUT_DIR / "validation_first_frame.png"
        pil_frames[0].save(first_path)
        print(f"  First frame: {first_path} ({first_path.stat().st_size / 1024:.0f} KB)")

        # Save keyframe grid
        n = len(pil_frames)
        indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        selected = [pil_frames[min(i, n - 1)] for i in indices]
        w, h = selected[0].size
        grid = Image.new("RGB", (w * len(selected), h))
        for i, img in enumerate(selected):
            grid.paste(img, (i * w, 0))
        grid_path = OUT_DIR / "validation_grid.png"
        grid.save(grid_path)
        print(f"  Grid: {grid_path} ({grid_path.stat().st_size / 1024:.0f} KB)")

        # Save video
        try:
            from diffusers.utils import export_to_video
            vid_path = OUT_DIR / "validation_sample.mp4"
            export_to_video(pil_frames, str(vid_path), fps=16)
            print(f"  Video: {vid_path} ({vid_path.stat().st_size / 1024:.0f} KB)")
        except Exception as e:
            print(f"  Warning: Could not save video: {e}")

    # -- Quality check --
    print("\nStep 6: Quality checks...")
    passed, report = check_frame_quality(pil_frames)
    print(report)

    print("\n" + "=" * 60)
    if passed:
        print("  VALIDATION PASSED - Sampling fixes produce recognizable output")
    else:
        print("  VALIDATION FAILED - Output still has quality issues")
    print("=" * 60)

    clean()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
