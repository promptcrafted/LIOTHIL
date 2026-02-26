"""Diagnostic inference test — dual expert, proper Wan 2.2 pipeline.

Tests:
  A. Both experts, no LoRA, 20 steps — verify base model generates coherently
  B. Both experts, unified LoRA on both, 20 steps — verify LoRA adds something

Diffusers WanPipeline convention:
  transformer   = HIGH-noise expert (runs first, handles large timesteps)
  transformer_2 = LOW-noise expert  (runs second, handles small timesteps)
  boundary_ratio = 0.875 for T2V (switch at timestep 875/1000)

Usage:
    python /workspace/dimljus/runpod/test_base_inference.py
"""
import torch
import gc
import numpy as np
from pathlib import Path


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
    T5_PATH = "/workspace/models/models_t5_umt5-xxl-enc-bf16.pth"

    # Generation settings — 480x832 is the WanPipeline default for 480p
    HEIGHT = 480
    WIDTH = 832
    NUM_FRAMES = 17
    STEPS = 20
    # Wan 2.2 T2V: boundary at 87.5% of noise schedule
    BOUNDARY_RATIO = 0.875
    # CFG: 4.0 for high-noise expert, 3.0 for low-noise expert (from official example)
    GUIDANCE_SCALE = 4.0
    GUIDANCE_SCALE_2 = 3.0

    # ── Load T5 and encode prompts ────────────────────────────────
    print("Loading T5 encoder...")
    from transformers import AutoTokenizer, UMT5EncoderModel, UMT5Config

    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
    config = UMT5Config.from_pretrained("google/umt5-xxl")
    text_encoder = UMT5EncoderModel(config)
    t5_sd = torch.load(T5_PATH, map_location="cpu", weights_only=True)
    text_encoder.load_state_dict(t5_sd, strict=False)
    del t5_sd
    text_encoder = text_encoder.to("cuda", dtype=torch.bfloat16)
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
    model_high = WanTransformer3DModel.from_single_file(
        MODEL_HIGH, torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    print(f"  High loaded. VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    print("  Loading low-noise expert...")
    model_low = WanTransformer3DModel.from_single_file(
        MODEL_LOW, torch_dtype=torch.bfloat16
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

    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)

    # Diffusers convention:
    #   transformer   = HIGH-noise expert (large timesteps, runs first)
    #   transformer_2 = LOW-noise expert  (small timesteps, runs second)
    #   boundary_ratio = 0.875 means switch at timestep 875/1000
    pipeline = WanPipeline(
        transformer=model_high,
        transformer_2=model_low,
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        scheduler=scheduler,
    )
    print(f"  Pipeline built. VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # ── Test A: Both experts, no LoRA ─────────────────────────────
    print("\n" + "=" * 60)
    print(f"  TEST A: Both experts, no LoRA, {STEPS} steps, {HEIGHT}x{WIDTH}, {NUM_FRAMES} frames")
    print(f"  boundary_ratio={BOUNDARY_RATIO}, CFG={GUIDANCE_SCALE}/{GUIDANCE_SCALE_2}")
    print("=" * 60)

    generator = torch.Generator(device="cuda").manual_seed(42)
    with torch.no_grad():
        output = pipeline(
            prompt_embeds=prompt_embeds.to("cuda", dtype=torch.bfloat16),
            negative_prompt_embeds=neg_embeds.to("cuda", dtype=torch.bfloat16),
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE_SCALE,
            guidance_scale_2=GUIDANCE_SCALE_2,
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            boundary_ratio=BOUNDARY_RATIO,
            generator=generator,
        )

    frames = output.frames if hasattr(output, "frames") else output
    inspect_output(frames, "DUAL-BASE")

    out_a = Path("/workspace/test_A_dual_base.mp4")
    save_first_frame(frames, out_a)

    from dimljus.training.sampler import _save_frames_to_video
    _save_frames_to_video(frames, out_a, fps=16)

    # ── Test B: Both experts + unified LoRA on both ───────────────
    print("\n" + "=" * 60)
    print(f"  TEST B: Both experts + unified LoRA, {STEPS} steps")
    print("=" * 60)

    # Check if LoRA checkpoint exists
    if not Path(LORA_PATH).exists():
        print(f"  SKIPPED: LoRA not found at {LORA_PATH}")
        print("  Run test2-unified-only training first to generate this checkpoint")
    else:
        # Remove models from pipeline before modifying them
        pipeline.transformer = None
        pipeline.transformer_2 = None
        del pipeline
        clean()

        # Apply LoRA to both experts
        print("  Applying LoRA to both experts...")
        from dimljus.training.wan.modules import create_lora_on_model, inject_lora_state_dict
        from dimljus.training.wan.constants import T2V_LORA_TARGETS
        from safetensors.torch import load_file

        lora_sd = load_file(LORA_PATH)
        print(f"  LoRA: {len(lora_sd)} keys, {sum(v.numel() for v in lora_sd.values()) / 1e6:.1f}M params")

        # Apply same unified LoRA to high-noise expert
        model_high = create_lora_on_model(
            model=model_high,
            target_modules=T2V_LORA_TARGETS,
            rank=16, alpha=16, dropout=0.0,
        )
        inject_lora_state_dict(model_high, lora_sd)
        model_high.eval()

        # Apply same unified LoRA to low-noise expert
        model_low = create_lora_on_model(
            model=model_low,
            target_modules=T2V_LORA_TARGETS,
            rank=16, alpha=16, dropout=0.0,
        )
        inject_lora_state_dict(model_low, lora_sd)
        model_low.eval()

        print(f"  LoRA applied to both. VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

        # Rebuild dual pipeline (same expert assignment)
        pipeline = WanPipeline(
            transformer=model_high,
            transformer_2=model_low,
            vae=vae,
            text_encoder=None,
            tokenizer=None,
            scheduler=FlowMatchEulerDiscreteScheduler(shift=3.0),
        )

        generator = torch.Generator(device="cuda").manual_seed(42)
        with torch.no_grad():
            output = pipeline(
                prompt_embeds=prompt_embeds.to("cuda", dtype=torch.bfloat16),
                negative_prompt_embeds=neg_embeds.to("cuda", dtype=torch.bfloat16),
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE_SCALE,
                guidance_scale_2=GUIDANCE_SCALE_2,
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
                boundary_ratio=BOUNDARY_RATIO,
                generator=generator,
            )

        frames = output.frames if hasattr(output, "frames") else output
        inspect_output(frames, "DUAL-LORA")

        out_b = Path("/workspace/test_B_dual_lora.mp4")
        save_first_frame(frames, out_b)
        _save_frames_to_video(frames, out_b, fps=16)

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for label, path_str in [("Dual-Base", "/workspace/test_A_dual_base"),
                            ("Dual-LoRA", "/workspace/test_B_dual_lora")]:
        mp4 = Path(f"{path_str}.mp4")
        png = Path(f"{path_str}.png")
        if mp4.exists():
            print(f"  {label}: {mp4} ({mp4.stat().st_size / 1024:.0f} KB)")
        if png.exists():
            print(f"  {label} first frame: {png} ({png.stat().st_size / 1024:.0f} KB)")
        if not mp4.exists() and not png.exists():
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
