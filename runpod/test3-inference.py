"""Post-training inference test for Test 3 (experts-only).

Loads the final merged LoRA from test3-experts-only training and generates
a sample video using the standard diffusers pipeline + load_lora_weights().

This validates the full pipeline: train → save → load → generate.

Usage:
    python /workspace/dimljus/runpod/test3-inference.py
"""
import torch
import gc
import numpy as np
from pathlib import Path


def clean():
    gc.collect()
    torch.cuda.empty_cache()


def main():
    OUTPUT_DIR = Path("/workspace/outputs/test3-experts-only")
    MERGED_LORA = OUTPUT_DIR / "final" / "test_lora_merged.safetensors"

    # Also check per-expert checkpoints
    HIGH_LORA_DIR = OUTPUT_DIR / "high_noise"
    LOW_LORA_DIR = OUTPUT_DIR / "low_noise"

    MODEL_HIGH = "/workspace/models/wan2.2_t2v_high_noise_14B_fp16.safetensors"
    MODEL_LOW = "/workspace/models/wan2.2_t2v_low_noise_14B_fp16.safetensors"
    VAE_PATH = "/workspace/models/wan_2.1_vae.safetensors"
    T5_PATH = "/workspace/models/models_t5_umt5-xxl-enc-bf16.pth"

    PROMPTS = [
        "Medium shot, Holly Golightly walks up an indoor staircase, looking back over her shoulder, morning light.",
        "Close up, Holly Golightly sits at a vanity mirror, adjusting a pearl necklace, soft warm lighting.",
    ]
    NEG = "blurry, low quality, distorted"
    STEPS = 30
    GUIDANCE = 4.0
    GUIDANCE_2 = 3.0  # Low-noise expert uses slightly lower CFG
    SHIFT = 5.0       # FlowMatchEulerDiscreteScheduler shift (ComfyUI quality path)
    BOUNDARY = 0.5    # 50/50 split between experts

    # ── Check what checkpoints exist ──────────────────────────────
    print("=" * 60)
    print("  TEST 3 — Post-Training Inference")
    print("=" * 60)

    print(f"\n  Output dir: {OUTPUT_DIR}")
    if MERGED_LORA.exists():
        size_mb = MERGED_LORA.stat().st_size / 1024 / 1024
        print(f"  Merged LoRA: {MERGED_LORA} ({size_mb:.1f} MB)")
    else:
        print(f"  Merged LoRA: NOT FOUND at {MERGED_LORA}")

    for label, lora_dir in [("High-noise", HIGH_LORA_DIR), ("Low-noise", LOW_LORA_DIR)]:
        if lora_dir.exists():
            files = sorted(lora_dir.glob("*.safetensors"))
            print(f"  {label} checkpoints: {len(files)}")
            for f in files:
                print(f"    {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            print(f"  {label} dir: NOT FOUND")

    # ── Load T5 and encode prompts ────────────────────────────────
    print("\nLoading T5 encoder...")
    from transformers import AutoTokenizer, UMT5EncoderModel, UMT5Config
    from dimljus.training.wan.inference import WanInferencePipeline

    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
    config = UMT5Config.from_pretrained("google/umt5-xxl")
    text_encoder = UMT5EncoderModel(config)
    t5_sd = torch.load(T5_PATH, map_location="cpu", weights_only=True)
    text_encoder.load_state_dict(t5_sd, strict=False)
    del t5_sd
    WanInferencePipeline._fix_t5_embed_tokens(text_encoder)
    text_encoder = text_encoder.to("cuda", dtype=torch.bfloat16)
    text_encoder.eval()

    # Encode all prompts + negative
    all_embeds = {}
    for text in PROMPTS + [NEG]:
        tokens = tokenizer(
            text, max_length=512, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            output = text_encoder(
                input_ids=tokens.input_ids.to("cuda"),
                attention_mask=tokens.attention_mask.to("cuda"),
            )
        all_embeds[text] = output.last_hidden_state

    del text_encoder, tokenizer
    clean()
    print(f"  Encoded {len(PROMPTS)} prompts + neg. VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # ── Load both expert models ───────────────────────────────────
    print("\nLoading both expert models...")
    from diffusers.models import WanTransformer3DModel
    from diffusers import WanPipeline, FlowMatchEulerDiscreteScheduler
    from diffusers.models import AutoencoderKLWan

    # Explicit config= prevents a known diffusers bug where from_single_file
    # can infer Wan 2.1 config instead of Wan 2.2 (diffusers#12329).
    model_high = WanTransformer3DModel.from_single_file(
        MODEL_HIGH, torch_dtype=torch.bfloat16,
        config="Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="transformer",
    ).to("cuda").eval()

    model_low = WanTransformer3DModel.from_single_file(
        MODEL_LOW, torch_dtype=torch.bfloat16,
        config="Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="transformer_2",
    ).to("cuda").eval()

    vae = AutoencoderKLWan.from_single_file(
        VAE_PATH, torch_dtype=torch.float32
    ).to("cuda")

    print(f"  Both experts + VAE loaded. VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # ── Build pipeline ────────────────────────────────────────────
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

    # ── Test A: Base model (no LoRA) ──────────────────────────────
    print("\n" + "=" * 60)
    print(f"  TEST A: Base model, no LoRA, {STEPS} steps")
    print("=" * 60)

    out_dir = OUTPUT_DIR / "inference_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    from dimljus.training.sampler import _save_frames_to_video

    neg_embeds = all_embeds[NEG].to("cuda", dtype=torch.bfloat16)

    for i, prompt in enumerate(PROMPTS):
        generator = torch.Generator(device="cuda").manual_seed(42 + i)
        prompt_embeds = all_embeds[prompt].to("cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            output = pipeline(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=neg_embeds,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
                guidance_scale_2=GUIDANCE_2,
                height=480, width=832, num_frames=17,
                generator=generator,
            )
        frames = output.frames if hasattr(output, "frames") else output
        out_path = out_dir / f"base_prompt{i}.mp4"
        _save_frames_to_video(frames, out_path, fps=16)

    # ── Test B: With merged LoRA ──────────────────────────────────
    if not MERGED_LORA.exists():
        print(f"\n  SKIPPED Test B: Merged LoRA not found at {MERGED_LORA}")
        print("  Run test3-experts-only training first.")
    else:
        print("\n" + "=" * 60)
        print(f"  TEST B: Merged LoRA via load_lora_weights(), {STEPS} steps")
        print("=" * 60)

        from safetensors.torch import load_file
        from dimljus.training.wan.checkpoint_io import has_diffusers_prefix, add_diffusers_prefix

        lora_sd = load_file(str(MERGED_LORA))
        n_keys = len(lora_sd)
        n_params = sum(v.numel() for v in lora_sd.values())
        print(f"  LoRA: {n_keys} keys, {n_params / 1e6:.1f}M params")

        # Check key format
        sample_keys = list(lora_sd.keys())[:3]
        print(f"  Sample keys: {sample_keys}")
        has_prefix = has_diffusers_prefix(lora_sd)
        print(f"  Has diffusers prefix: {has_prefix}")

        # Count transformer vs transformer_2 keys
        t1_keys = sum(1 for k in lora_sd if k.startswith("transformer."))
        t2_keys = sum(1 for k in lora_sd if k.startswith("transformer_2."))
        print(f"  transformer keys: {t1_keys}, transformer_2 keys: {t2_keys}")

        if not has_prefix:
            print("  Adding transformer. prefix...")
            lora_sd = add_diffusers_prefix(lora_sd, prefix="transformer")

        pipeline.load_lora_weights(lora_sd, adapter_name="dimljus")
        print(f"  LoRA applied. VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

        for i, prompt in enumerate(PROMPTS):
            generator = torch.Generator(device="cuda").manual_seed(42 + i)
            prompt_embeds = all_embeds[prompt].to("cuda", dtype=torch.bfloat16)

            with torch.no_grad():
                output = pipeline(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=neg_embeds,
                    num_inference_steps=STEPS,
                    guidance_scale=GUIDANCE,
                    guidance_scale_2=GUIDANCE_2,
                    height=480, width=832, num_frames=17,
                    generator=generator,
                )
            frames = output.frames if hasattr(output, "frames") else output
            out_path = out_dir / f"lora_prompt{i}.mp4"
            _save_frames_to_video(frames, out_path, fps=16)

        pipeline.unload_lora_weights()

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    if out_dir.exists():
        for f in sorted(out_dir.glob("*.mp4")):
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name} ({size_kb:.0f} KB)")
        for f in sorted(out_dir.glob("*.png")):
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name} ({size_kb:.0f} KB)")

    clean()
    print("\nDone!")


if __name__ == "__main__":
    main()
