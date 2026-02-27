"""Diagnostic inference test — dual expert, proper Wan 2.2 pipeline.

Tests:
  A. Both experts, no LoRA, 30 steps — verify base model generates coherently
  B. Both experts + LoRA via load_lora_weights() — verify the LoRA loading path

Diffusers WanPipeline convention:
  transformer   = HIGH-noise expert (runs first, handles large timesteps)
  transformer_2 = LOW-noise expert  (runs second, handles small timesteps)
  boundary_ratio = 0.5 for T2V (50/50 split between experts)

KEY FIXES:
  1. (diffusers#12329) Every WanTransformer3DModel.from_single_file() call MUST
     include config= and subfolder= to prevent silent Wan 2.1 misdetection.
  2. T5 loaded from HF repo (Wan-AI/Wan2.2-T2V-A14B-Diffusers, subfolder
     text_encoder), NOT from the standalone .pth file — the .pth contains
     wrong/uninitialized weights (all 243 keys differ from HF, layer_norm=1.0).
  3. embed_tokens weight tying fix applied after loading — from_pretrained()
     does not tie shared.weight to encoder.embed_tokens.weight automatically,
     causing T5 to ignore prompt content.

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


def encode_prompt(tokenizer, text_encoder, text):
    tokens = tokenizer(
        text, max_length=512, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        output = text_encoder(
            input_ids=tokens.input_ids.to("cuda"),
            attention_mask=tokens.attention_mask.to("cuda"),
        )
    return output.last_hidden_state


def inspect_output(frames, label):
    print(f"\n  [{label}] Output type: {type(frames)}")
    if isinstance(frames, np.ndarray):
        print(f"  [{label}] Shape: {frames.shape}, dtype: {frames.dtype}")
        print(f"  [{label}] Value range: min={frames.min():.3f}, max={frames.max():.3f}, mean={frames.mean():.3f}")
    elif isinstance(frames, (list, tuple)):
        print(f"  [{label}] List length: {len(frames)}")
        if frames and isinstance(frames[0], (list, tuple)):
            print(f"  [{label}] Inner length: {len(frames[0])}")
            if frames[0]:
                elem = frames[0][0]
                print(f"  [{label}] Element type: {type(elem)}")
                if hasattr(elem, "size"):
                    print(f"  [{label}] Element size: {elem.size}")


def save_first_frame(frames, path):
    """Extract and save the first frame as PNG for quick visual check."""
    try:
        from PIL import Image

        # Navigate to the first frame
        frame = None
        if isinstance(frames, (list, tuple)) and frames:
            inner = frames[0]
            if isinstance(inner, (list, tuple)) and inner:
                frame = inner[0]  # list[list[PIL]] format
            elif hasattr(inner, "save"):
                frame = inner  # list[PIL] format
            elif isinstance(inner, np.ndarray):
                frame = inner
        elif isinstance(frames, np.ndarray):
            while frames.ndim > 3:
                frames = frames[0]
            frame = frames

        if frame is None:
            print(f"  Could not extract first frame")
            return

        # Convert numpy to PIL if needed
        if isinstance(frame, np.ndarray):
            if frame.dtype != np.uint8:
                frame = (frame * 255).clip(0, 255).astype(np.uint8)
            frame = Image.fromarray(frame)

        png_path = path.with_suffix(".png")
        frame.save(png_path)
        print(f"  First frame saved: {png_path} ({png_path.stat().st_size / 1024:.0f} KB, {frame.size[0]}x{frame.size[1]})")
    except Exception as e:
        print(f"  Warning: Could not save first frame: {e}")


def main():
    PROMPT = "A woman with dark hair walks down a city street, morning light"
    NEG = "blurry, low quality, distorted"
    LORA_PATH = "/workspace/outputs/test2-unified-only/unified/test_lora_unified_epoch003.safetensors"

    MODEL_HIGH = "/workspace/models/wan2.2_t2v_high_noise_14B_fp16.safetensors"
    MODEL_LOW = "/workspace/models/wan2.2_t2v_low_noise_14B_fp16.safetensors"
    VAE_PATH = "/workspace/models/wan_2.1_vae.safetensors"
    # T5 loaded from HF repo (correct weights), NOT from .pth file (wrong weights)
    T5_REPO = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    # Validated generation settings (from ComfyUI quality path + practitioner advice)
    HEIGHT = 480
    WIDTH = 832
    NUM_FRAMES = 17
    STEPS = 30
    BOUNDARY_RATIO = 0.5        # 50/50 split: 15 steps high-noise, 15 steps low-noise
    GUIDANCE_SCALE = 4.0        # Same CFG for both experts
    SHIFT = 5.0                 # FlowMatchEulerDiscreteScheduler shift

    # ── Load T5 and encode prompts ────────────────────────────────
    # T5 is loaded from the HF repo (correct weights) instead of the .pth file
    # which was identified as containing wrong/uninitialized weights.
    print(f"Loading T5 encoder from {T5_REPO}...")
    from transformers import AutoTokenizer, UMT5EncoderModel
    from dimljus.training.wan.inference import WanInferencePipeline

    tokenizer = AutoTokenizer.from_pretrained(T5_REPO, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        T5_REPO, subfolder="text_encoder", torch_dtype=torch.bfloat16,
    )
    # Fix embed_tokens weight tying: from_pretrained() does not tie
    # shared.weight to encoder.embed_tokens.weight automatically.
    WanInferencePipeline._fix_t5_embed_tokens(text_encoder)
    text_encoder = text_encoder.to("cuda")
    text_encoder.eval()

    prompt_embeds = encode_prompt(tokenizer, text_encoder, PROMPT)
    neg_embeds = encode_prompt(tokenizer, text_encoder, NEG)
    print(f"  Prompt embeds: {prompt_embeds.shape}")

    del text_encoder, tokenizer
    clean()
    print(f"  T5 freed. VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # ── Load BOTH expert models ───────────────────────────────────
    print("\nLoading both expert models...")
    from diffusers.models import WanTransformer3DModel

    print("  Loading high-noise expert...")
    # config= prevents diffusers from guessing wrong model config (diffusers#12329).
    # Without it, from_single_file auto-detects Wan 2.1 config instead of Wan 2.2
    # because the transformer weight shapes are identical — silent garbage output.
    model_high = WanTransformer3DModel.from_single_file(
        MODEL_HIGH, torch_dtype=torch.bfloat16,
        config="Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="transformer",
    ).to("cuda").eval()
    print(f"  High loaded. VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    print("  Loading low-noise expert...")
    model_low = WanTransformer3DModel.from_single_file(
        MODEL_LOW, torch_dtype=torch.bfloat16,
        config="Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="transformer_2",
    ).to("cuda").eval()
    print(f"  Both loaded. VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # ── Build dual-expert pipeline ────────────────────────────────
    print("\nBuilding dual-expert pipeline...")
    from diffusers import WanPipeline, FlowMatchEulerDiscreteScheduler
    from diffusers.models import AutoencoderKLWan

    # VAE in float32 for better decoding quality (official recommendation)
    vae = AutoencoderKLWan.from_single_file(
        VAE_PATH, torch_dtype=torch.float32
    ).to("cuda")

    scheduler = FlowMatchEulerDiscreteScheduler(shift=SHIFT)

    pipeline = WanPipeline(
        transformer=model_high,
        transformer_2=model_low,
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        scheduler=scheduler,
        boundary_ratio=BOUNDARY_RATIO,
    )
    print(f"  Pipeline built. VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # ── Test A: Both experts, no LoRA ─────────────────────────────
    print("\n" + "=" * 60)
    print(f"  TEST A: Both experts, no LoRA, {STEPS} steps, {HEIGHT}x{WIDTH}, {NUM_FRAMES} frames")
    print(f"  boundary_ratio={BOUNDARY_RATIO}, shift={SHIFT}, CFG={GUIDANCE_SCALE}")
    print("=" * 60)

    generator = torch.Generator(device="cuda").manual_seed(42)
    with torch.no_grad():
        output = pipeline(
            prompt_embeds=prompt_embeds.to("cuda", dtype=torch.bfloat16),
            negative_prompt_embeds=neg_embeds.to("cuda", dtype=torch.bfloat16),
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE_SCALE,
            guidance_scale_2=GUIDANCE_SCALE,
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            generator=generator,
        )

    frames = output.frames if hasattr(output, "frames") else output
    inspect_output(frames, "DUAL-BASE")

    out_a = Path("/workspace/test_A_dual_base.mp4")
    save_first_frame(frames, out_a)

    from dimljus.training.sampler import _save_frames_to_video
    _save_frames_to_video(frames, out_a, fps=16)

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
                prompt_embeds=prompt_embeds.to("cuda", dtype=torch.bfloat16),
                negative_prompt_embeds=neg_embeds.to("cuda", dtype=torch.bfloat16),
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE_SCALE,
                guidance_scale_2=GUIDANCE_SCALE,
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
                generator=generator,
            )

        frames = output.frames if hasattr(output, "frames") else output
        inspect_output(frames, "LORA-DIFFUSERS")

        out_b = Path("/workspace/test_B_lora_diffusers.mp4")
        save_first_frame(frames, out_b)
        _save_frames_to_video(frames, out_b, fps=16)

        # Unload LoRA for next test
        pipeline.unload_lora_weights()
        print("  LoRA unloaded for next test")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print("  Check .grid.png files first — quickest way to see if output is coherent.")
    print()
    for label, path_str in [("Dual-Base", "/workspace/test_A_dual_base"),
                            ("LoRA-Diffusers", "/workspace/test_B_lora_diffusers")]:
        mp4 = Path(f"{path_str}.mp4")
        png = Path(f"{path_str}.png")
        grid = Path(f"{path_str}.grid.png")
        if grid.exists():
            print(f"  {label} GRID: {grid} ({grid.stat().st_size / 1024:.0f} KB)")
        if mp4.exists():
            print(f"  {label} VIDEO: {mp4} ({mp4.stat().st_size / 1024:.0f} KB)")
        if png.exists():
            print(f"  {label} first frame: {png} ({png.stat().st_size / 1024:.0f} KB)")
        if not mp4.exists() and not png.exists() and not grid.exists():
            png_dir = mp4.with_suffix("")
            if png_dir.exists():
                pngs = list(png_dir.glob("*.png"))
                print(f"  {label}: {len(pngs)} PNGs in {png_dir}")
            else:
                print(f"  {label}: FAILED or SKIPPED")

    # Cleanup
    clean()
    print("\nDone!")


if __name__ == "__main__":
    main()
