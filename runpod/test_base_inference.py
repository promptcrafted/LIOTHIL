"""Quick base model test — no LoRA, 5 steps, just verify model generates coherently.

Also tests with LoRA at proper resolution to compare.

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
    """Print diagnostic info about pipeline output."""
    print(f"\n  [{label}] Output type: {type(frames)}")
    if isinstance(frames, np.ndarray):
        print(f"  [{label}] Shape: {frames.shape}, dtype: {frames.dtype}")
        print(f"  [{label}] Value range: min={frames.min()}, max={frames.max()}, mean={frames.mean():.1f}")
        if frames.ndim == 4:
            mid = frames[frames.shape[0] // 2]
            print(f"  [{label}] Mid frame: mean={mid.mean():.1f}, std={mid.std():.1f}")
    elif isinstance(frames, (list, tuple)):
        print(f"  [{label}] List length: {len(frames)}")
        if frames and isinstance(frames[0], (list, tuple)):
            print(f"  [{label}] Inner length: {len(frames[0])}")
            elem = frames[0][0]
            print(f"  [{label}] Element type: {type(elem)}")
            if isinstance(elem, np.ndarray):
                print(f"  [{label}] Element shape: {elem.shape}")


def main():
    PROMPT = "A woman with dark hair walks down a city street, morning light"
    NEG = ""  # empty negative for simple test
    LORA_PATH = "/workspace/outputs/test2-unified-only/unified/test_lora_unified_epoch003.safetensors"

    # ── Load T5 and encode prompts ────────────────────────────────
    print("Loading T5 encoder...")
    from transformers import AutoTokenizer, UMT5EncoderModel, UMT5Config

    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
    config = UMT5Config.from_pretrained("google/umt5-xxl")
    text_encoder = UMT5EncoderModel(config)
    t5_sd = torch.load(
        "/workspace/models/models_t5_umt5-xxl-enc-bf16.pth",
        map_location="cpu", weights_only=True
    )
    text_encoder.load_state_dict(t5_sd, strict=False)
    del t5_sd
    text_encoder = text_encoder.to("cuda", dtype=torch.bfloat16)
    text_encoder.eval()

    prompt_embeds = encode_prompt(tokenizer, text_encoder, PROMPT)
    neg_embeds = encode_prompt(tokenizer, text_encoder, NEG)
    print(f"  Prompt embeds: {prompt_embeds.shape}")

    # Free T5
    del text_encoder, tokenizer
    clean()
    print(f"  T5 freed. VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # ── Load base model ───────────────────────────────────────────
    print("\nLoading base model (low noise expert)...")
    from diffusers.models import WanTransformer3DModel
    model = WanTransformer3DModel.from_single_file(
        "/workspace/models/wan2.2_t2v_low_noise_14B_fp16.safetensors",
        torch_dtype=torch.bfloat16
    )
    model = model.to("cuda")
    model.eval()
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # ── Build pipeline ────────────────────────────────────────────
    print("\nBuilding pipeline...")
    from diffusers import WanPipeline, FlowMatchEulerDiscreteScheduler
    from diffusers.models import AutoencoderKLWan

    vae = AutoencoderKLWan.from_single_file(
        "/workspace/models/wan_2.1_vae.safetensors",
        torch_dtype=torch.bfloat16
    )
    vae = vae.to("cuda")
    scheduler = FlowMatchEulerDiscreteScheduler()

    pipeline = WanPipeline(
        transformer=model,
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        scheduler=scheduler,
    )

    # ── Test 1: Base model only, 5 steps, small ──────────────────
    print("\n" + "=" * 60)
    print("  TEST A: Base model, no LoRA, 5 steps, 480x832, 17 frames")
    print("=" * 60)

    generator = torch.Generator(device="cuda").manual_seed(42)
    with torch.no_grad():
        output = pipeline(
            prompt_embeds=prompt_embeds.to("cuda", dtype=torch.bfloat16),
            negative_prompt_embeds=neg_embeds.to("cuda", dtype=torch.bfloat16),
            num_inference_steps=5,
            guidance_scale=5.0,
            height=480,
            width=832,
            num_frames=17,
            generator=generator,
        )

    frames = output.frames if hasattr(output, "frames") else output
    inspect_output(frames, "BASE")

    from dimljus.training.sampler import _save_frames_to_video
    out_a = Path("/workspace/test_A_base_only.mp4")
    _save_frames_to_video(frames, out_a, fps=16)

    # ── Test 2: With LoRA, 5 steps ──────────────────────────────
    print("\n" + "=" * 60)
    print("  TEST B: With LoRA, 5 steps, 480x832, 17 frames")
    print("=" * 60)

    # Remove model from pipeline temporarily
    pipeline.transformer = None
    del pipeline
    clean()

    # Apply LoRA
    print("  Applying LoRA...")
    from dimljus.training.wan.modules import create_lora_on_model, inject_lora_state_dict
    from dimljus.training.wan.constants import T2V_LORA_TARGETS
    from safetensors.torch import load_file

    lora_sd = load_file(LORA_PATH)
    print(f"  LoRA: {len(lora_sd)} keys, {sum(v.numel() for v in lora_sd.values()) / 1e6:.1f}M params")

    model = create_lora_on_model(
        model=model,
        target_modules=T2V_LORA_TARGETS,
        rank=16,
        alpha=16,
        dropout=0.0,
    )
    inject_lora_state_dict(model, lora_sd)
    model.eval()

    # Rebuild pipeline
    pipeline = WanPipeline(
        transformer=model,
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        scheduler=FlowMatchEulerDiscreteScheduler(),
    )

    generator = torch.Generator(device="cuda").manual_seed(42)
    with torch.no_grad():
        output = pipeline(
            prompt_embeds=prompt_embeds.to("cuda", dtype=torch.bfloat16),
            negative_prompt_embeds=neg_embeds.to("cuda", dtype=torch.bfloat16),
            num_inference_steps=5,
            guidance_scale=5.0,
            height=480,
            width=832,
            num_frames=17,
            generator=generator,
        )

    frames = output.frames if hasattr(output, "frames") else output
    inspect_output(frames, "LORA")

    out_b = Path("/workspace/test_B_with_lora.mp4")
    _save_frames_to_video(frames, out_b, fps=16)

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for label, path in [("Base", out_a), ("LoRA", out_b)]:
        if path.exists():
            print(f"  {label}: {path} ({path.stat().st_size / 1024:.0f} KB)")
        else:
            png_dir = path.with_suffix("")
            if png_dir.exists():
                pngs = list(png_dir.glob("*.png"))
                print(f"  {label}: {len(pngs)} PNGs in {png_dir}")
            else:
                print(f"  {label}: FAILED — no output")

    # Cleanup
    pipeline.transformer = None
    del pipeline, vae, model
    clean()
    print("\nDone!")


if __name__ == "__main__":
    main()
