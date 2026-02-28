"""Post-training inference validation for expert isolation tests (test4/test5).

Generates 6 comparison videos:
  - 2 base model (no LoRA) -- baseline reference
  - 2 low-noise LoRA only (test4 checkpoint) -- low-noise expert trained
  - 2 high-noise LoRA only (test5 checkpoint) -- high-noise expert trained

Each condition uses 2 prompts: one from training data, one unseen.
Same seed (42, 43) across all conditions for direct comparison.

Automated quality checks verify each video is not black, has motion,
and has pixel variance -- catching broken generation before visual review.

Usage:
    python /workspace/dimljus/runpod/test4-inference.py

Requirements:
    - diffusers<0.36 (0.36.0 has WanPipeline regression)
    - Both expert base models in /workspace/models/
    - T5 encoder .pth and VAE .safetensors in /workspace/models/
    - Both isolation checkpoints in /workspace/outputs/

NOTE: This script loads ALL models from local files (no HF downloads).
      The pod's root disk is full so from_pretrained() fails. Instead we
      load T5, VAE, and both expert transformers from /workspace/models/
      and construct WanPipeline manually. Text encoding uses pre-computed
      prompt_embeds (same approach as test3-inference.py).
"""
import sys
import gc
import time
import json
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
MODEL_HIGH = "/workspace/models/wan2.2_t2v_high_noise_14B_fp16.safetensors"
MODEL_LOW = "/workspace/models/wan2.2_t2v_low_noise_14B_fp16.safetensors"
VAE_PATH = "/workspace/models/wan_2.1_vae.safetensors"
T5_PATH = "/workspace/models/models_t5_umt5-xxl-enc-bf16.pth"
HF_REPO = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

# Isolation checkpoints from Plan 03-01
LOW_LORA = "/workspace/outputs/test4-low-only/low_noise/test_lora_low_epoch005.safetensors"
HIGH_LORA = "/workspace/outputs/test5-high-only/high_noise/test_lora_high_epoch005.safetensors"

OUT_DIR = Path("/workspace/outputs/test4-low-only/inference_test")

# ── Inference settings (validated in Phase 1) ──────────────────────
PROMPTS = [
    # Prompt 0: Same as training data -- tests memorization
    "Medium shot, Holly Golightly walks up an indoor staircase, looking back over her shoulder, morning light.",
    # Prompt 1: Unseen prompt -- tests generalization
    "Close up, Holly Golightly sits at a vanity mirror, adjusting a pearl necklace, soft warm lighting.",
]
NEG_PROMPT = "blurry, low quality, distorted"
STEPS = 30
CFG = 4.0
CFG_2 = 3.0       # Low-noise expert uses slightly lower CFG (from test3)
SHIFT = 5.0       # FlowMatchEulerDiscreteScheduler shift
BOUNDARY = 0.6    # Inference boundary (NOT 0.875 training boundary)
HEIGHT = 480
WIDTH = 832
NUM_FRAMES = 17
SEEDS = [42, 43]  # One per prompt, consistent across conditions


def clean():
    """Free GPU memory between pipeline operations."""
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass


def extract_pil_frames(output):
    """Extract PIL frames from WanPipeline output.

    WanPipeline output.frames can be:
      - list[list[PIL.Image]] (batch of videos)
      - list[PIL.Image]
      - numpy array

    We always unwrap to a flat list of PIL Images.
    """
    from PIL import Image as PILImage

    frames = output.frames if hasattr(output, "frames") else output
    pil_frames = []

    if isinstance(frames, np.ndarray):
        # Squeeze batch dimensions
        while frames.ndim > 4:
            frames = frames[0]
        if frames.dtype != np.uint8:
            if frames.max() <= 1.0:
                frames = (frames * 255).clip(0, 255).astype(np.uint8)
            else:
                frames = frames.clip(0, 255).astype(np.uint8)
        for i in range(frames.shape[0]):
            pil_frames.append(PILImage.fromarray(frames[i]))
    elif isinstance(frames, (list, tuple)):
        flat = frames
        # Unwrap one nesting level if needed (list[list[PIL]])
        if len(flat) > 0 and isinstance(flat[0], (list, tuple)):
            flat = flat[0]
        for item in flat:
            if hasattr(item, "save"):
                pil_frames.append(item)
            elif isinstance(item, np.ndarray):
                if item.dtype != np.uint8:
                    item = (item * 255).clip(0, 255).astype(np.uint8)
                pil_frames.append(PILImage.fromarray(item))

    return pil_frames


def quality_check(pil_frames, label):
    """Run automated quality checks on generated frames.

    Checks:
      1. Not black: mean pixel value > 10/255 (catches failed generation)
      2. Has motion: mean frame-to-frame diff > 1.0/255 (catches frozen output)
      3. Has variance: pixel std > sqrt(100)/255 ~ 0.039 (catches flat solid color)

    Returns (passed: bool, results: dict) where results has numeric values.
    """
    results = {"label": label, "checks": {}}

    if not pil_frames or len(pil_frames) < 2:
        results["checks"]["frames"] = {"pass": False, "value": len(pil_frames) if pil_frames else 0, "threshold": 2}
        return False, results

    # Use first frame for static checks
    first_arr = np.array(pil_frames[0]).astype(np.float32) / 255.0
    last_arr = np.array(pil_frames[-1]).astype(np.float32) / 255.0

    # Check 1: Not black (mean > 10/255 = 0.039)
    mean_val = first_arr.mean()
    not_black = mean_val > 0.039
    results["checks"]["not_black"] = {
        "pass": bool(not_black),
        "value": round(float(mean_val), 4),
        "threshold": 0.039,
    }

    # Check 2: Has motion (mean absolute frame diff > 1/255 = 0.004)
    diff = np.abs(first_arr - last_arr).mean()
    has_motion = diff > 0.004
    results["checks"]["has_motion"] = {
        "pass": bool(has_motion),
        "value": round(float(diff), 4),
        "threshold": 0.004,
    }

    # Check 3: Has variance (pixel std > sqrt(100)/255 ~ 0.039)
    variance = first_arr.var()
    has_variance = variance > (100.0 / (255.0 * 255.0))  # 100 in uint8 space
    results["checks"]["has_variance"] = {
        "pass": bool(has_variance),
        "value": round(float(variance), 6),
        "threshold": round(100.0 / (255.0 * 255.0), 6),
    }

    # Also check for washed-out white (mean > 0.9)
    not_white = mean_val < 0.90
    results["checks"]["not_white"] = {
        "pass": bool(not_white),
        "value": round(float(mean_val), 4),
        "threshold": 0.90,
    }

    all_passed = all(c["pass"] for c in results["checks"].values())
    return all_passed, results


def print_quality_report(results):
    """Pretty-print quality check results for one video."""
    label = results["label"]
    checks = results["checks"]
    all_passed = all(c["pass"] for c in checks.values())

    status = "PASS" if all_passed else "FAIL"
    print(f"\n  [{status}] {label}")
    for name, check in checks.items():
        mark = "OK" if check["pass"] else "FAIL"
        print(f"    {mark}: {name} = {check['value']} (threshold: {check['threshold']})")


def save_video(pil_frames, out_path):
    """Save PIL frames as MP4 with keyframe grid."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Try export_to_video first
    try:
        from diffusers.utils import export_to_video
        export_to_video(pil_frames, str(out_path), fps=16)
        size_kb = out_path.stat().st_size / 1024
        print(f"    Saved: {out_path.name} ({size_kb:.0f} KB)")
        return True
    except Exception as e:
        print(f"    Warning: export_to_video failed ({e})")

    # Fallback: save individual frames as PNG
    from PIL import Image
    png_dir = out_path.with_suffix("")
    png_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(pil_frames):
        frame.save(png_dir / f"frame_{idx:04d}.png")
    print(f"    Saved: {len(pil_frames)} frames to {png_dir.name}/")
    return True


def save_keyframe_grid(pil_frames, out_path):
    """Save a horizontal strip of evenly-spaced keyframes."""
    from PIL import Image

    n = len(pil_frames)
    indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    selected = [pil_frames[min(i, n - 1)] for i in indices]
    w, h = selected[0].size
    grid = Image.new("RGB", (w * len(selected), h))
    for i, img in enumerate(selected):
        grid.paste(img, (i * w, 0))
    grid.save(out_path)
    size_kb = out_path.stat().st_size / 1024
    print(f"    Grid: {out_path.name} ({size_kb:.0f} KB)")


def main():
    import torch
    from PIL import Image

    print("=" * 70)
    print("  INFERENCE VALIDATION: Expert Isolation Tests (test4 / test5)")
    print("=" * 70)

    # ── Check prerequisites ────────────────────────────────────────
    print("\n--- Prerequisites ---")

    import diffusers
    print(f"  diffusers: {diffusers.__version__}")

    missing = []
    for label, path in [
        ("High-noise base model", MODEL_HIGH),
        ("Low-noise base model", MODEL_LOW),
        ("VAE", VAE_PATH),
        ("T5 encoder", T5_PATH),
        ("Low-noise LoRA (test4)", LOW_LORA),
        ("High-noise LoRA (test5)", HIGH_LORA),
    ]:
        exists = Path(path).exists()
        size_str = ""
        if exists:
            size_mb = Path(path).stat().st_size / 1024 / 1024
            size_str = f"({size_mb:.0f} MB)"
        status = "FOUND" if exists else "MISSING"
        print(f"  {status}: {label} {size_str}")
        if not exists:
            missing.append(path)

    if missing:
        print(f"\n  ERROR: {len(missing)} required files not found. Cannot proceed.")
        sys.exit(1)

    # ── Step 1: Load T5 encoder and pre-encode all prompts ─────────
    # We load T5 from local .pth file and encode prompts, then free
    # T5 from GPU before loading the transformer models. This avoids
    # needing T5 in memory during inference (saves ~10GB VRAM).
    print("\n--- Step 1: Loading T5 encoder and encoding prompts ---")
    t0 = time.time()

    from transformers import AutoTokenizer, UMT5EncoderModel, UMT5Config
    from dimljus.training.wan.inference import WanInferencePipeline

    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
    config = UMT5Config.from_pretrained("google/umt5-xxl")
    text_encoder = UMT5EncoderModel(config)
    t5_sd = torch.load(T5_PATH, map_location="cpu", weights_only=True)
    text_encoder.load_state_dict(t5_sd, strict=False)
    del t5_sd

    # Fix T5 embed_tokens weight tying (required for all loading paths)
    WanInferencePipeline._fix_t5_embed_tokens(text_encoder)
    text_encoder = text_encoder.to("cuda", dtype=torch.bfloat16)
    text_encoder.eval()

    # Encode all prompts + negative
    all_embeds = {}
    for text in PROMPTS + [NEG_PROMPT]:
        tokens = tokenizer(
            text, max_length=512, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            output = text_encoder(
                input_ids=tokens.input_ids.to("cuda"),
                attention_mask=tokens.attention_mask.to("cuda"),
            )
        all_embeds[text] = output.last_hidden_state.cpu()

    del text_encoder, tokenizer
    clean()
    print(f"  Encoded {len(PROMPTS)} prompts + neg in {time.time() - t0:.0f}s")
    print(f"  VRAM after T5 unload: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # ── Step 2: Load both expert models + VAE, build pipeline ──────
    # All models loaded from local single-file weights (no HF download).
    print("\n--- Step 2: Loading expert models + VAE ---")
    t0 = time.time()

    from diffusers import WanPipeline, FlowMatchEulerDiscreteScheduler
    from diffusers.models import WanTransformer3DModel, AutoencoderKLWan

    # High-noise expert = transformer (handles early denoising steps)
    model_high = WanTransformer3DModel.from_single_file(
        MODEL_HIGH, torch_dtype=torch.bfloat16,
        config=HF_REPO, subfolder="transformer",
    ).to("cuda").eval()
    # Low-noise expert = transformer_2 (handles late denoising steps)
    model_low = WanTransformer3DModel.from_single_file(
        MODEL_LOW, torch_dtype=torch.bfloat16,
        config=HF_REPO, subfolder="transformer_2",
    ).to("cuda").eval()
    # VAE (must be float32 for stable decoding)
    vae = AutoencoderKLWan.from_single_file(
        VAE_PATH, torch_dtype=torch.float32
    )

    # Build pipeline with validated scheduler settings
    scheduler = FlowMatchEulerDiscreteScheduler(shift=SHIFT)
    pipeline = WanPipeline(
        transformer=model_high,
        transformer_2=model_low,
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        scheduler=scheduler,
        boundary_ratio=BOUNDARY,
    )
    pipeline = pipeline.to("cuda")

    print(f"  Pipeline built in {time.time() - t0:.0f}s")
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # ── Create output directory ────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Pre-move negative prompt embeds to GPU (reused for all generations)
    neg_embeds = all_embeds[NEG_PROMPT].to("cuda", dtype=torch.bfloat16)

    # ── Test A: Base model (no LoRA) ───────────────────────────────
    print("\n" + "=" * 70)
    print("  TEST A: Base model (no LoRA) -- baseline reference")
    print("=" * 70)

    for i, prompt in enumerate(PROMPTS):
        print(f"\n  Prompt {i}: {prompt[:70]}...")
        generator = torch.Generator(device="cuda").manual_seed(SEEDS[i])
        prompt_embeds = all_embeds[prompt].to("cuda", dtype=torch.bfloat16)
        t0 = time.time()

        with torch.no_grad():
            output = pipeline(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=neg_embeds,
                num_inference_steps=STEPS,
                guidance_scale=CFG,
                guidance_scale_2=CFG_2,
                height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
                generator=generator,
            )

        elapsed = time.time() - t0
        print(f"    Generated in {elapsed:.0f}s")

        pil_frames = extract_pil_frames(output)
        print(f"    Got {len(pil_frames)} frames")

        out_path = OUT_DIR / f"base_prompt{i}.mp4"
        save_video(pil_frames, out_path)
        save_keyframe_grid(pil_frames, OUT_DIR / f"base_prompt{i}.grid.png")

        passed, results = quality_check(pil_frames, f"base_prompt{i}")
        print_quality_report(results)
        all_results.append(results)

        del output
        clean()

    # ── Test B: Low-noise LoRA only (test4 checkpoint) ─────────────
    print("\n" + "=" * 70)
    print("  TEST B: Low-noise LoRA only (test4 checkpoint)")
    print("  LoRA applies to transformer_2 (low-noise expert)")
    print("  transformer (high-noise expert) runs with base weights")
    print("=" * 70)

    t0 = time.time()
    from safetensors.torch import load_file
    from dimljus.training.wan.checkpoint_io import has_diffusers_prefix, add_diffusers_prefix

    low_sd = load_file(LOW_LORA)
    n_keys = len(low_sd)
    n_params = sum(v.numel() for v in low_sd.values())
    print(f"  LoRA: {n_keys} keys, {n_params / 1e6:.1f}M params")

    # Inspect key format
    sample_keys = list(low_sd.keys())[:3]
    has_prefix = has_diffusers_prefix(low_sd)
    print(f"  Sample keys: {sample_keys}")
    print(f"  Has diffusers prefix: {has_prefix}")

    # Low-noise isolation LoRA targets transformer_2 (low-noise expert).
    #
    # CRITICAL DIFFUSERS QUIRK: WanPipeline.load_lora_weights() always looks
    # for "transformer." as the key prefix to strip, even when loading into
    # transformer_2. So we must:
    #   1. Ensure keys have "transformer." prefix (not "transformer_2.")
    #   2. Pass load_into_transformer_2=True to route to the right component
    #
    # Without this, the LoRA silently fails to load (0 modules injected)
    # and output is identical to base model.
    if has_prefix:
        # Replace transformer_2. prefix with transformer. prefix
        # (diffusers expects transformer. prefix for lora_state_dict parsing)
        fixed_sd = {}
        for k, v in low_sd.items():
            new_k = k.replace("transformer_2.", "transformer.", 1)
            fixed_sd[new_k] = v
        low_sd = fixed_sd
        print("  Replaced transformer_2. prefix with transformer. (diffusers quirk)")
    else:
        # No prefix -- add transformer. prefix
        print("  Adding transformer. prefix...")
        low_sd = add_diffusers_prefix(low_sd, prefix="transformer")

    pipeline.load_lora_weights(low_sd, adapter_name="low_noise_lora",
                               load_into_transformer_2=True)
    print(f"  Low-noise LoRA applied in {time.time() - t0:.0f}s")
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    for i, prompt in enumerate(PROMPTS):
        print(f"\n  Prompt {i}: {prompt[:70]}...")
        generator = torch.Generator(device="cuda").manual_seed(SEEDS[i])
        prompt_embeds = all_embeds[prompt].to("cuda", dtype=torch.bfloat16)
        t0 = time.time()

        with torch.no_grad():
            output = pipeline(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=neg_embeds,
                num_inference_steps=STEPS,
                guidance_scale=CFG,
                guidance_scale_2=CFG_2,
                height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
                generator=generator,
            )

        elapsed = time.time() - t0
        print(f"    Generated in {elapsed:.0f}s")

        pil_frames = extract_pil_frames(output)
        print(f"    Got {len(pil_frames)} frames")

        out_path = OUT_DIR / f"low_lora_prompt{i}.mp4"
        save_video(pil_frames, out_path)
        save_keyframe_grid(pil_frames, OUT_DIR / f"low_lora_prompt{i}.grid.png")

        passed, results = quality_check(pil_frames, f"low_lora_prompt{i}")
        print_quality_report(results)
        all_results.append(results)

        del output
        clean()

    # Unload low-noise LoRA before loading high-noise
    pipeline.unload_lora_weights()
    clean()
    print("\n  Low-noise LoRA unloaded")

    # ── Test C: High-noise LoRA only (test5 checkpoint) ────────────
    print("\n" + "=" * 70)
    print("  TEST C: High-noise LoRA only (test5 checkpoint)")
    print("  LoRA applies to transformer (high-noise expert)")
    print("  transformer_2 (low-noise expert) runs with base weights")
    print("=" * 70)

    t0 = time.time()
    high_sd = load_file(HIGH_LORA)
    n_keys = len(high_sd)
    n_params = sum(v.numel() for v in high_sd.values())
    print(f"  LoRA: {n_keys} keys, {n_params / 1e6:.1f}M params")

    # Inspect key format
    sample_keys = list(high_sd.keys())[:3]
    has_prefix = has_diffusers_prefix(high_sd)
    print(f"  Sample keys: {sample_keys}")
    print(f"  Has diffusers prefix: {has_prefix}")

    # High-noise isolation LoRA should have transformer. prefix
    # If not, add it so load_lora_weights() routes to the right expert
    if not has_prefix:
        print("  Adding transformer. prefix (high-noise expert)...")
        high_sd = add_diffusers_prefix(high_sd, prefix="transformer")

    pipeline.load_lora_weights(high_sd, adapter_name="high_noise_lora")
    print(f"  High-noise LoRA applied in {time.time() - t0:.0f}s")
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    for i, prompt in enumerate(PROMPTS):
        print(f"\n  Prompt {i}: {prompt[:70]}...")
        generator = torch.Generator(device="cuda").manual_seed(SEEDS[i])
        prompt_embeds = all_embeds[prompt].to("cuda", dtype=torch.bfloat16)
        t0 = time.time()

        with torch.no_grad():
            output = pipeline(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=neg_embeds,
                num_inference_steps=STEPS,
                guidance_scale=CFG,
                guidance_scale_2=CFG_2,
                height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
                generator=generator,
            )

        elapsed = time.time() - t0
        print(f"    Generated in {elapsed:.0f}s")

        pil_frames = extract_pil_frames(output)
        print(f"    Got {len(pil_frames)} frames")

        out_path = OUT_DIR / f"high_lora_prompt{i}.mp4"
        save_video(pil_frames, out_path)
        save_keyframe_grid(pil_frames, OUT_DIR / f"high_lora_prompt{i}.grid.png")

        passed, results = quality_check(pil_frames, f"high_lora_prompt{i}")
        print_quality_report(results)
        all_results.append(results)

        del output
        clean()

    pipeline.unload_lora_weights()
    clean()
    print("\n  High-noise LoRA unloaded")

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  INFERENCE VALIDATION SUMMARY")
    print("=" * 70)

    # List generated files
    print("\n  Generated files:")
    if OUT_DIR.exists():
        for f in sorted(OUT_DIR.glob("*.mp4")):
            size_kb = f.stat().st_size / 1024
            print(f"    {f.name} ({size_kb:.0f} KB)")
        for f in sorted(OUT_DIR.glob("*.grid.png")):
            size_kb = f.stat().st_size / 1024
            print(f"    {f.name} ({size_kb:.0f} KB)")

    # Quality check summary
    print("\n  Quality check results:")
    n_passed = 0
    n_total = len(all_results)
    for r in all_results:
        label = r["label"]
        checks = r["checks"]
        all_ok = all(c["pass"] for c in checks.values())
        status = "PASS" if all_ok else "FAIL"
        if all_ok:
            n_passed += 1
        print(f"    [{status}] {label}")

    print(f"\n  Overall: {n_passed}/{n_total} videos passed all quality checks")

    # Save results as JSON for later analysis
    report_path = OUT_DIR / "quality_report.json"
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "settings": {
            "steps": STEPS,
            "cfg": CFG,
            "shift": SHIFT,
            "boundary": BOUNDARY,
            "resolution": f"{HEIGHT}x{WIDTH}",
            "num_frames": NUM_FRAMES,
            "seeds": SEEDS,
        },
        "checkpoints": {
            "low_lora": LOW_LORA,
            "high_lora": HIGH_LORA,
        },
        "results": all_results,
        "summary": {
            "passed": n_passed,
            "total": n_total,
            "all_passed": n_passed == n_total,
        },
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Quality report saved: {report_path}")

    # ── Final verdict ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    if n_passed == n_total:
        print("  VALIDATION PASSED -- All videos pass automated quality checks")
        print("  Ready for Minta's visual review (compare base vs low-LoRA vs high-LoRA)")
    else:
        print(f"  VALIDATION PARTIAL -- {n_total - n_passed}/{n_total} videos failed quality checks")
        print("  Review quality_report.json for details")
    print("=" * 70)

    clean()
    return 0 if n_passed == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
