"""Isolate which component causes noise by swapping one thing at a time.

The from_pretrained test (test C) produces coherent video. Our manual test
(test A) produces noise. This script swaps individual components to find
which one causes the noise.

Test D: from_pretrained pipeline, but replace T5 with our .pth loading
  → If noise: T5 loading is the bug
  → If works: T5 is fine, issue is elsewhere

Test E: from_pretrained pipeline, but replace transformers with from_single_file
  → If noise: from_single_file transformer loading is the bug
  → If works: transformers are fine

Test F: from_pretrained pipeline, but replace scheduler with FlowMatchEuler
  → If noise: scheduler/shift is the bug
  → If works: scheduler is fine

We test one variable at a time against the known-good from_pretrained baseline.
"""
import torch
import gc
import os
import numpy as np
from pathlib import Path
from PIL import Image

# Ensure HF token is available for gated repos.
# Set HF_TOKEN in your environment (RunPod pod settings or export HF_TOKEN=...).
if not os.environ.get("HF_TOKEN"):
    print("WARNING: HF_TOKEN not set. Downloads from gated repos may fail.")
    print("  Fix: export HF_TOKEN=hf_your_token_here")

PROMPT = "A woman with dark hair walks down a city street, morning light"
NEG = "blurry, low quality, distorted"
REPO = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
T5_PATH = "/workspace/models/models_t5_umt5-xxl-enc-bf16.pth"
MODEL_HIGH = "/workspace/models/wan2.2_t2v_high_noise_14B_fp16.safetensors"
MODEL_LOW = "/workspace/models/wan2.2_t2v_low_noise_14B_fp16.safetensors"

HEIGHT, WIDTH, NUM_FRAMES = 480, 832, 17
STEPS = 30
GUIDANCE_SCALE = 4.0


def clean():
    gc.collect()
    torch.cuda.empty_cache()


def save_grid(frames, path):
    """Save keyframe grid from numpy output."""
    arr = frames
    while arr.ndim > 4:
        arr = arr[0]
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    pil_frames = [Image.fromarray(arr[i]) for i in range(arr.shape[0])]

    n = len(pil_frames)
    indices = [int(i * (n - 1) / min(4, n - 1)) for i in range(min(5, n))]
    selected = [pil_frames[i] for i in indices]

    w, h = selected[0].size
    grid = Image.new("RGB", (w * len(selected), h))
    for i, img in enumerate(selected):
        grid.paste(img, (i * w, 0))

    grid_path = path.with_suffix(".grid.png")
    grid.save(grid_path)
    print(f"  Grid: {grid_path} ({grid_path.stat().st_size / 1024:.0f} KB)")
    print(f"  Values: min={frames.min():.3f}, max={frames.max():.3f}, mean={frames.mean():.3f}")

    # Save first frame
    png_path = path.with_suffix(".png")
    selected[0].save(png_path)
    print(f"  First frame: {png_path}")


def run_test(label, pipeline, use_prompt_embeds=False, prompt_embeds=None, neg_embeds=None):
    """Run inference and save output."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    generator = torch.Generator(device="cuda").manual_seed(42)
    with torch.no_grad():
        if use_prompt_embeds:
            output = pipeline(
                prompt_embeds=prompt_embeds.to("cuda", dtype=torch.bfloat16),
                negative_prompt_embeds=neg_embeds.to("cuda", dtype=torch.bfloat16),
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE_SCALE,
                height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
                generator=generator,
            )
        else:
            output = pipeline(
                prompt=PROMPT,
                negative_prompt=NEG,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE_SCALE,
                height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
                generator=generator,
            )

    frames = output.frames if hasattr(output, "frames") else output
    if isinstance(frames, np.ndarray):
        print(f"  Output shape: {frames.shape}")
        save_grid(frames, Path(f"/workspace/test_{label.split(':')[0].strip()}.mp4"))
    else:
        print(f"  Output type: {type(frames)} (unexpected)")


def main():
    import sys

    # Parse which test to run from command line
    test = sys.argv[1] if len(sys.argv) > 1 else "D"
    test = test.upper()

    if test == "D":
        # ══════════════════════════════════════════════════════════
        # TEST D: from_pretrained pipeline + our .pth T5 loading
        # If noise → T5 .pth loading is the bug
        # ══════════════════════════════════════════════════════════
        print("\n" + "=" * 60)
        print("  TEST D: from_pretrained pipeline + .pth T5")
        print("  Isolates: Is the T5 .pth loading causing noise?")
        print("=" * 60)

        # Load T5 from .pth (our way)
        print("\nLoading T5 from .pth file (our way)...")
        from transformers import AutoTokenizer, UMT5EncoderModel, UMT5Config

        tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
        config = UMT5Config.from_pretrained("google/umt5-xxl")
        text_encoder = UMT5EncoderModel(config)
        t5_sd = torch.load(T5_PATH, map_location="cpu", weights_only=True)
        text_encoder.load_state_dict(t5_sd, strict=False)
        del t5_sd

        # Fix embed_tokens weight tying (same as our code does)
        if hasattr(text_encoder, "encoder") and hasattr(text_encoder.encoder, "embed_tokens"):
            if hasattr(text_encoder, "shared"):
                text_encoder.encoder.embed_tokens.weight = text_encoder.shared.weight

        text_encoder = text_encoder.to("cuda", dtype=torch.bfloat16).eval()

        # Encode prompts
        tokens = tokenizer(PROMPT, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            prompt_embeds = text_encoder(input_ids=tokens.input_ids.to("cuda"), attention_mask=tokens.attention_mask.to("cuda")).last_hidden_state
        tokens_neg = tokenizer(NEG, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            neg_embeds = text_encoder(input_ids=tokens_neg.input_ids.to("cuda"), attention_mask=tokens_neg.attention_mask.to("cuda")).last_hidden_state

        print(f"  Prompt embeds: {prompt_embeds.shape}, std={prompt_embeds.std():.3f}")
        del text_encoder, tokenizer
        clean()

        # Load pipeline from HF but without text encoder (we provide our own embeds)
        print("\nLoading from_pretrained pipeline (without T5, we provide embeds)...")
        from diffusers import WanPipeline
        pipeline = WanPipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16, text_encoder=None, tokenizer=None)
        pipeline.vae = pipeline.vae.to(dtype=torch.float32)
        pipeline = pipeline.to("cuda")
        print(f"  Pipeline loaded. VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        print(f"  Scheduler: {type(pipeline.scheduler).__name__}")
        print(f"  boundary_ratio: {pipeline.config.boundary_ratio}")

        run_test("D: from_pretrained + pth T5", pipeline, use_prompt_embeds=True, prompt_embeds=prompt_embeds, neg_embeds=neg_embeds)

    elif test == "E":
        # ══════════════════════════════════════════════════════════
        # TEST E: from_pretrained but swap transformers with from_single_file
        # If noise → from_single_file loading is the bug
        # ══════════════════════════════════════════════════════════
        print("\n" + "=" * 60)
        print("  TEST E: from_pretrained + from_single_file transformers")
        print("  Isolates: Is from_single_file transformer loading causing noise?")
        print("=" * 60)

        from diffusers import WanPipeline
        from diffusers.models import WanTransformer3DModel

        # Load full pipeline first to get everything except transformers
        print("\nLoading base pipeline from HF...")
        pipeline = WanPipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16)

        # Now replace the transformers with from_single_file versions
        print("\nReplacing transformers with from_single_file versions...")
        del pipeline.transformer, pipeline.transformer_2
        clean()

        model_high = WanTransformer3DModel.from_single_file(
            MODEL_HIGH, torch_dtype=torch.bfloat16,
            config=REPO, subfolder="transformer",
        )
        model_low = WanTransformer3DModel.from_single_file(
            MODEL_LOW, torch_dtype=torch.bfloat16,
            config=REPO, subfolder="transformer_2",
        )
        pipeline.transformer = model_high
        pipeline.transformer_2 = model_low

        pipeline.vae = pipeline.vae.to(dtype=torch.float32)
        pipeline = pipeline.to("cuda")
        print(f"  Pipeline loaded. VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

        run_test("E: from_pretrained + single_file transformers", pipeline)

    elif test == "F":
        # ══════════════════════════════════════════════════════════
        # TEST F: from_pretrained but swap scheduler to FlowMatchEuler
        # If noise → scheduler is the bug
        # ══════════════════════════════════════════════════════════
        print("\n" + "=" * 60)
        print("  TEST F: from_pretrained + FlowMatchEuler scheduler")
        print("  Isolates: Is the scheduler causing noise?")
        print("=" * 60)

        from diffusers import WanPipeline, FlowMatchEulerDiscreteScheduler

        pipeline = WanPipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16)
        pipeline.scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
        pipeline.vae = pipeline.vae.to(dtype=torch.float32)
        pipeline = pipeline.to("cuda")
        print(f"  Pipeline loaded. VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        print(f"  Scheduler: {type(pipeline.scheduler).__name__}")
        print(f"  boundary_ratio: {pipeline.config.boundary_ratio}")

        run_test("F: from_pretrained + FlowMatchEuler", pipeline)

    else:
        print(f"Unknown test: {test}. Use D, E, or F.")

    clean()
    print("\nDone!")


if __name__ == "__main__":
    main()
