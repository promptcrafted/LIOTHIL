"""Test LoRA inference with isolation checkpoints (test4/test5).

Last verified inference script — uses validated settings:
  - FlowMatchEulerDiscreteScheduler shift=5.0
  - boundary_ratio=0.6 (inference, NOT training's 0.875)
  - guidance_scale=4.0, 30 steps, 480x832, 17 frames

Generates 4 comparison videos:
  A. Base model (no LoRA) — reference baseline
  B. Low-noise LoRA only (test4, epoch 5) — fine detail expert
  C. High-noise LoRA only (test5, epoch 5) — composition expert
  D. Both LoRAs combined

Requires: diffusers==0.35.0, transformers>=4.46,<5, imageio, imageio-ffmpeg

Usage:
    HF_HOME=/workspace/.cache/huggingface HF_TOKEN=hf_xxx \
    PYTHONPATH=/workspace/dimljus python /workspace/dimljus/runpod/test_lora_inference.py
"""
import torch
import gc
import os
import numpy as np
from pathlib import Path
from PIL import Image

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

MODEL_HIGH = "/workspace/models/wan2.2_t2v_high_noise_14B_fp16.safetensors"
MODEL_LOW = "/workspace/models/wan2.2_t2v_low_noise_14B_fp16.safetensors"
VAE_PATH = "/workspace/models/wan_2.1_vae.safetensors"
HF_REPO = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
LORA_LOW = "/workspace/outputs/test4-low-only/low_noise/test_lora_low_epoch005.safetensors"
LORA_HIGH = "/workspace/outputs/test5-high-only/high_noise/test_lora_high_epoch005.safetensors"
OUT_DIR = Path("/workspace/outputs/lora_inference_test")


def clean():
    gc.collect()
    torch.cuda.empty_cache()


def save_outputs(frames_array, base_path, label):
    """Save keyframe grid + mp4 video from pipeline output."""
    from diffusers.utils import export_to_video

    arr = frames_array
    while arr.ndim > 4:
        arr = arr[0]
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)

    pil_frames = [Image.fromarray(arr[i]) for i in range(arr.shape[0])]

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

    # Video
    mp4_path = base_path.with_suffix(".mp4")
    export_to_video(pil_frames, str(mp4_path), fps=16)
    print(f"  [{label}] Grid: {grid_path} ({grid_path.stat().st_size / 1024:.0f} KB)")
    print(f"  [{label}] Video: {mp4_path} ({mp4_path.stat().st_size / 1024:.0f} KB)")


def generate(pipeline, prompt, seed=42):
    """Run inference with validated settings."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        output = pipeline(
            prompt=prompt,
            negative_prompt="blurry, low quality, distorted",
            num_inference_steps=30,
            guidance_scale=4.0,
            height=480,
            width=832,
            num_frames=17,
            generator=generator,
        )
    return output.frames if hasattr(output, "frames") else output


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import diffusers
    import transformers

    print(f"diffusers: {diffusers.__version__}, transformers: {transformers.__version__}")

    # Load pipeline from HF (gets correct T5 + text encoding pipeline)
    print("\n=== Loading pipeline ===")
    from diffusers import WanPipeline, FlowMatchEulerDiscreteScheduler
    from diffusers.models import WanTransformer3DModel, AutoencoderKLWan

    pipeline = WanPipeline.from_pretrained(HF_REPO, torch_dtype=torch.bfloat16)
    pipeline.vae = pipeline.vae.to(dtype=torch.float32)

    # Override scheduler: FlowMatchEuler shift=5.0 (validated, ComfyUI-aligned)
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)

    # Override boundary_ratio: 0.6 for inference (HF default 0.875 is training value)
    pipeline._internal_dict["boundary_ratio"] = 0.6

    # Fix T5 embed_tokens weight tying
    t5 = pipeline.text_encoder
    if hasattr(t5, "shared") and hasattr(t5, "encoder"):
        if not torch.equal(t5.shared.weight, t5.encoder.embed_tokens.weight):
            t5.encoder.embed_tokens.weight = t5.shared.weight
            print("  Fixed T5 embed_tokens")

    print(f"  Scheduler: {type(pipeline.scheduler).__name__} (shift=5.0)")
    print(f"  boundary_ratio: 0.6 (inference)")

    # Swap transformers to local from_single_file copies
    del pipeline.transformer, pipeline.transformer_2
    clean()

    pipeline.transformer = WanTransformer3DModel.from_single_file(
        MODEL_HIGH,
        torch_dtype=torch.bfloat16,
        config=HF_REPO,
        subfolder="transformer",
    )
    pipeline.transformer_2 = WanTransformer3DModel.from_single_file(
        MODEL_LOW,
        torch_dtype=torch.bfloat16,
        config=HF_REPO,
        subfolder="transformer_2",
    )
    pipeline.vae = AutoencoderKLWan.from_single_file(VAE_PATH, torch_dtype=torch.float32)
    pipeline = pipeline.to("cuda")
    print(f"  Ready. VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    prompt = "Medium shot, Holly Golightly walks down a city street, morning light, elegant black dress"

    # === A: Base model ===
    print("\n=== A: Base model, no LoRA ===")
    frames = generate(pipeline, prompt)
    save_outputs(frames, OUT_DIR / "A_base", "BASE")

    # === B: Low-noise LoRA only ===
    print("\n=== B: Low-noise LoRA (test4 ep5) ===")
    from safetensors.torch import load_file

    lora_low_sd = load_file(LORA_LOW)
    pipeline.load_lora_weights(lora_low_sd, adapter_name="low_noise")
    print(f"  Loaded {len(lora_low_sd)} LoRA keys (transformer_2.*)")
    frames = generate(pipeline, prompt)
    save_outputs(frames, OUT_DIR / "B_low_lora", "LOW")
    pipeline.unload_lora_weights()
    clean()

    # === C: High-noise LoRA only ===
    print("\n=== C: High-noise LoRA (test5 ep5) ===")
    lora_high_sd = load_file(LORA_HIGH)
    pipeline.load_lora_weights(lora_high_sd, adapter_name="high_noise")
    print(f"  Loaded {len(lora_high_sd)} LoRA keys (transformer.*)")
    frames = generate(pipeline, prompt)
    save_outputs(frames, OUT_DIR / "C_high_lora", "HIGH")
    pipeline.unload_lora_weights()
    clean()

    # === D: Both LoRAs ===
    print("\n=== D: Both LoRAs combined ===")
    combined = {}
    combined.update(lora_low_sd)
    combined.update(lora_high_sd)
    pipeline.load_lora_weights(combined, adapter_name="combined")
    print(f"  Loaded {len(combined)} combined LoRA keys")
    frames = generate(pipeline, prompt)
    save_outputs(frames, OUT_DIR / "D_both_loras", "BOTH")
    pipeline.unload_lora_weights()
    clean()

    print("\n=== DONE ===")
    for f in sorted(OUT_DIR.glob("*")):
        print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
