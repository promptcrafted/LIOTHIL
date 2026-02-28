"""Manual forward pass test — bypass WanPipeline entirely.

Runs ONE denoising step directly through the model and checks if the
predicted velocity is reasonable. If direct forward pass works but
WanPipeline produces noise, the bug is in the pipeline's __call__().
"""
import torch
import gc
from pathlib import Path

MODEL_HIGH = "/workspace/models/wan2.2_t2v_high_noise_14B_fp16.safetensors"
VAE_PATH = "/workspace/models/wan_2.1_vae.safetensors"
T5_PATH = "/workspace/models/models_t5_umt5-xxl-enc-bf16.pth"
HF_REPO = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
OUT_DIR = Path("/workspace/outputs/manual_forward_test")

def clean():
    gc.collect()
    torch.cuda.empty_cache()

def main():
    print("=== Manual forward pass test ===")
    import diffusers
    print(f"diffusers: {diffusers.__version__}, torch: {torch.__version__}")

    # 1. Encode prompt
    print("\n--- Encoding prompt ---")
    from transformers import AutoTokenizer, UMT5EncoderModel, UMT5Config
    from dimljus.training.wan.inference import WanInferencePipeline

    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
    config = UMT5Config.from_pretrained("google/umt5-xxl")
    text_encoder = UMT5EncoderModel(config)
    t5_sd = torch.load(T5_PATH, map_location="cpu", weights_only=True)
    text_encoder.load_state_dict(t5_sd, strict=False)
    del t5_sd
    WanInferencePipeline._fix_t5_embed_tokens(text_encoder)
    text_encoder = text_encoder.to("cuda", dtype=torch.bfloat16).eval()

    prompt = "Medium shot, Holly Golightly walks up an indoor staircase, looking back over her shoulder, morning light."
    neg = "blurry, low quality, distorted"

    with torch.no_grad():
        tok = tokenizer(prompt, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
        pe = text_encoder(input_ids=tok.input_ids.to("cuda"), attention_mask=tok.attention_mask.to("cuda")).last_hidden_state
        tok_n = tokenizer(neg, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
        ne = text_encoder(input_ids=tok_n.input_ids.to("cuda"), attention_mask=tok_n.attention_mask.to("cuda")).last_hidden_state

    del text_encoder, tokenizer
    clean()
    print(f"prompt_embeds: {pe.shape}, mean={pe.float().mean():.4f}")
    print(f"neg_embeds:    {ne.shape}, mean={ne.float().mean():.4f}")

    # 2. Load model (single expert only for simplicity)
    print("\n--- Loading high-noise expert ---")
    from diffusers.models import WanTransformer3DModel
    model = WanTransformer3DModel.from_single_file(
        MODEL_HIGH, torch_dtype=torch.bfloat16,
        config=HF_REPO, subfolder="transformer",
    ).to("cuda").eval()
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # 3. Create noise latents (same as WanPipeline.prepare_latents)
    print("\n--- Creating noise latents ---")
    gen = torch.Generator(device="cuda").manual_seed(42)
    # Shape: (batch, channels, temporal, height, width)
    # For 480x832 at 17 frames: temporal=5, height=60, width=104
    latents = torch.randn(1, 16, 5, 60, 104, device="cuda", dtype=torch.float32, generator=gen)
    print(f"Noise latents: {latents.shape}, mean={latents.mean():.4f}, std={latents.std():.4f}")

    # 4. Setup scheduler
    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    scheduler.set_timesteps(30, device="cuda")
    timesteps = scheduler.timesteps
    print(f"Timesteps: {timesteps[:5]}... (total {len(timesteps)})")

    # 5. Manual denoising loop — 30 steps with CFG
    print("\n--- Manual denoising (30 steps with CFG) ---")
    for i, t in enumerate(timesteps):
        latent_input = latents.to(torch.bfloat16)
        ts = t.expand(1)  # batch_size=1

        with torch.no_grad():
            # Conditional prediction
            noise_pred_cond = model(
                hidden_states=latent_input,
                timestep=ts,
                encoder_hidden_states=pe,
                return_dict=False,
            )[0]

            # Unconditional prediction
            noise_pred_uncond = model(
                hidden_states=latent_input,
                timestep=ts,
                encoder_hidden_states=ne,
                return_dict=False,
            )[0]

        # CFG
        noise_pred = noise_pred_uncond + 4.0 * (noise_pred_cond - noise_pred_uncond)

        # Scheduler step
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        if i in [0, 5, 14, 29]:
            print(f"  Step {i:2d} t={t.item():.1f}: latent mean={latents.float().mean():.4f}, "
                  f"std={latents.float().std():.4f}, pred_mean={noise_pred.float().mean():.4f}")

    print(f"\nFinal latents: mean={latents.float().mean():.4f}, std={latents.float().std():.4f}")

    # 6. Decode through VAE
    print("\n--- VAE decode ---")
    from diffusers.models import AutoencoderKLWan
    vae = AutoencoderKLWan.from_single_file(VAE_PATH, torch_dtype=torch.float32).to("cuda")

    latents_for_vae = latents.to(torch.float32)
    # Apply VAE normalization (same as WanPipeline)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to("cuda", torch.float32)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to("cuda", torch.float32)
    latents_for_vae = latents_for_vae / latents_std + latents_mean
    print(f"Normalized latents: mean={latents_for_vae.mean():.4f}, std={latents_for_vae.std():.4f}")

    with torch.no_grad():
        video = vae.decode(latents_for_vae, return_dict=False)[0]
    print(f"Decoded video: {video.shape}, mean={video.float().mean():.4f}")

    # Convert to frames and save grid
    video = video.clamp(0, 1).cpu().float().numpy()
    if video.ndim == 5:
        video = video[0]  # remove batch dim -> (C, F, H, W) or (F, C, H, W) or (F, H, W, C)
    print(f"Video array: shape={video.shape}")

    # Save grid
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import numpy as np
    from PIL import Image

    # Handle different possible shapes
    if video.shape[0] == 3:  # (C, F, H, W)
        video = np.transpose(video, (1, 2, 3, 0))  # -> (F, H, W, C)
    elif video.shape[1] == 3:  # (F, C, H, W)
        video = np.transpose(video, (0, 2, 3, 1))  # -> (F, H, W, C)

    n_frames = video.shape[0]
    indices = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]
    selected = [video[min(i, n_frames - 1)] for i in indices]

    frames_uint8 = [(f * 255).clip(0, 255).astype(np.uint8) for f in selected]
    pil_frames = [Image.fromarray(f) for f in frames_uint8]
    w, h = pil_frames[0].size
    grid = Image.new("RGB", (w * len(pil_frames), h))
    for i, img in enumerate(pil_frames):
        grid.paste(img, (i * w, 0))
    grid_path = OUT_DIR / "manual_forward_grid.png"
    grid.save(grid_path)
    arr = np.array(grid).astype(np.float32) / 255.0
    print(f"Grid: {grid_path} | mean={arr.mean():.4f}, std={arr.std():.4f}")
    print("\n=== DONE ===")

if __name__ == "__main__":
    main()
