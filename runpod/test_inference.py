"""Isolated inference test — generate video from a trained LoRA checkpoint.

Loads the Wan model, applies a trained LoRA, runs inference, and saves
the output video. Tests the full sampling chain independently of training.

Usage:
    python /workspace/dimljus/runpod/test_inference.py \
        --lora /workspace/outputs/test2-unified-only/unified/test_lora_unified_epoch003.safetensors \
        --prompt "A woman with dark hair styled in an updo walks down a city street, wearing a black dress and pearls, morning light." \
        --output /workspace/test_sample.mp4
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated inference test")
    parser.add_argument("--lora", required=True, help="Path to trained LoRA safetensors")
    parser.add_argument("--prompt", default="A woman with dark hair styled in an updo walks down a city street, wearing a black dress and pearls, morning light.")
    parser.add_argument("--neg", default="blurry, low quality, distorted")
    parser.add_argument("--output", default="/workspace/test_sample.mp4")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    # Model paths (defaults match setup.sh)
    parser.add_argument("--dit", default="/workspace/models/wan2.2_t2v_low_noise_14B_fp16.safetensors")
    parser.add_argument("--vae", default="/workspace/models/wan_2.1_vae.safetensors")
    parser.add_argument("--t5", default="/workspace/models/models_t5_umt5-xxl-enc-bf16.pth")
    args = parser.parse_args()

    import torch
    import numpy as np

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB")
    print(f"LoRA: {args.lora}")
    print(f"Prompt: {args.prompt[:80]}...")
    print()

    # ── Step 1: Load base model ───────────────────────────────────────
    print("Step 1: Loading base model...")
    t0 = time.time()
    from diffusers.models import WanTransformer3DModel
    # config= prevents diffusers from guessing wrong model config (diffusers#12329).
    # Without it, from_single_file auto-detects Wan 2.1 config instead of Wan 2.2
    # because the transformer weight shapes are identical — silent garbage output.
    model = WanTransformer3DModel.from_single_file(
        args.dit, torch_dtype=torch.bfloat16,
        config="Wan-AI/Wan2.2-T2V-A14B-Diffusers", subfolder="transformer",
    )
    model = model.to("cuda")
    model.eval()
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # ── Step 2: Apply LoRA ────────────────────────────────────────────
    print("\nStep 2: Applying LoRA...")
    from safetensors.torch import load_file
    from dimljus.training.wan.modules import (
        create_lora_on_model, inject_lora_state_dict,
    )
    from dimljus.training.wan.constants import T2V_LORA_TARGETS

    lora_sd = load_file(args.lora)
    print(f"  LoRA keys: {len(lora_sd)}")
    print(f"  LoRA size: {sum(v.numel() for v in lora_sd.values()) / 1e6:.1f}M params")

    # Get rank from first lora_A weight
    rank = 16
    for key, val in lora_sd.items():
        if "lora_A" in key:
            rank = val.shape[0]
            break
    print(f"  Rank: {rank}")
    print(f"  Target modules: {T2V_LORA_TARGETS}")

    # Create PEFT LoRA on the model using our standard function
    model = create_lora_on_model(
        model=model,
        target_modules=T2V_LORA_TARGETS,
        rank=rank,
        alpha=rank,
        dropout=0.0,
    )
    # Inject the trained weights
    inject_lora_state_dict(model, lora_sd)
    model.eval()
    print(f"  LoRA applied. VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # ── Step 3: Encode prompt with T5 ────────────────────────────────
    print("\nStep 3: Encoding prompt with T5...")
    t0 = time.time()
    from transformers import AutoTokenizer, UMT5EncoderModel, UMT5Config

    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
    config = UMT5Config.from_pretrained("google/umt5-xxl")
    text_encoder = UMT5EncoderModel(config)
    t5_sd = torch.load(args.t5, map_location="cpu", weights_only=True)
    text_encoder.load_state_dict(t5_sd, strict=False)
    del t5_sd
    text_encoder = text_encoder.to("cuda", dtype=torch.bfloat16)
    text_encoder.eval()

    def encode_prompt(text: str) -> torch.Tensor:
        tokens = tokenizer(text, max_length=512, padding="max_length",
                          truncation=True, return_tensors="pt")
        with torch.no_grad():
            output = text_encoder(
                input_ids=tokens.input_ids.to("cuda"),
                attention_mask=tokens.attention_mask.to("cuda"),
            )
        return output.last_hidden_state.cpu()

    prompt_embeds = encode_prompt(args.prompt)
    neg_embeds = encode_prompt(args.neg) if args.neg else None
    print(f"  Prompt encoded: {prompt_embeds.shape}")
    print(f"  T5 encoding took {time.time() - t0:.1f}s")

    # Free T5
    del text_encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  T5 freed. VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # ── Step 4: Build pipeline and generate ──────────────────────────
    print("\nStep 4: Building pipeline and generating...")
    from diffusers import WanPipeline, FlowMatchEulerDiscreteScheduler
    from diffusers.models import AutoencoderKLWan

    vae = AutoencoderKLWan.from_single_file(args.vae, torch_dtype=torch.bfloat16)
    vae = vae.to("cuda")
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)

    pipeline = WanPipeline(
        transformer=model,
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        scheduler=scheduler,
    )

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    print(f"  VRAM before generation: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
    t0 = time.time()

    with torch.no_grad():
        output = pipeline(
            prompt_embeds=prompt_embeds.to("cuda", dtype=torch.bfloat16),
            negative_prompt_embeds=neg_embeds.to("cuda", dtype=torch.bfloat16) if neg_embeds is not None else None,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
        )

    gen_time = time.time() - t0
    print(f"  Generation took {gen_time:.1f}s ({gen_time / args.steps:.1f}s/step)")

    # ── Step 5: Save output ──────────────────────────────────────────
    print("\nStep 5: Saving output...")
    frames = output.frames if hasattr(output, "frames") else output
    print(f"  Raw output type: {type(frames)}")
    if isinstance(frames, (list, tuple)):
        print(f"  Outer length: {len(frames)}")
        if len(frames) > 0:
            inner = frames[0]
            print(f"  Inner type: {type(inner)}")
            if isinstance(inner, (list, tuple)):
                print(f"  Inner length: {len(inner)}")
                if len(inner) > 0:
                    elem = inner[0]
                    print(f"  Element type: {type(elem)}")
                    if isinstance(elem, np.ndarray):
                        print(f"  Element shape: {elem.shape}, dtype: {elem.dtype}")
            elif isinstance(inner, np.ndarray):
                print(f"  Inner shape: {inner.shape}, dtype: {inner.dtype}")

    # Use our sampler's save function
    from dimljus.training.sampler import _save_frames_to_video
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_frames_to_video(frames, output_path, fps=16)

    # Check result
    if output_path.exists() and output_path.stat().st_size > 0:
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"\n  SUCCESS: {output_path} ({size_mb:.1f} MB)")
    else:
        # Check fallback
        png_dir = output_path.with_suffix("")
        if png_dir.exists():
            pngs = list(png_dir.glob("*.png"))
            if pngs:
                total = sum(p.stat().st_size for p in pngs)
                print(f"\n  FALLBACK: {len(pngs)} PNGs in {png_dir} ({total / 1024 / 1024:.1f} MB)")
            else:
                print(f"\n  FAILED: PNG dir exists but empty: {png_dir}")
        else:
            print(f"\n  FAILED: No output at {output_path}")

    # Cleanup
    pipeline.transformer = None
    del pipeline, vae
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Final VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
    print("\nDone!")


if __name__ == "__main__":
    main()
