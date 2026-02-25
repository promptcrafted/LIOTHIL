"""
Dimljus — Wan 2.2 LoRA Training for RunPod
============================================
Unified training script for T2V and I2V, both noise-level experts.
Uses musubi-tuner as the training backend while dimljus's native
pipeline matures.

Usage:
  cd /workspace/musubi-tuner
  tmux new -s train

  # Train high-noise expert (T2V)
  python /workspace/dimljus/runpod/train.py --variant t2v --noise_level high

  # Train low-noise expert (T2V)
  python /workspace/dimljus/runpod/train.py --variant t2v --noise_level low

  # Train both experts sequentially (T2V)
  python /workspace/dimljus/runpod/train.py --variant t2v --noise_level both

  # I2V training
  python /workspace/dimljus/runpod/train.py --variant i2v --noise_level high

  # Custom config
  python /workspace/dimljus/runpod/train.py --variant t2v --noise_level high \\
      --lr 5e-5 --rank 32 --epochs 30

  # Resume from checkpoint
  python /workspace/dimljus/runpod/train.py --variant t2v --noise_level high \\
      --resume_from /workspace/outputs/my-lora-e25.safetensors

  # Merge a speed LoRA before training
  python /workspace/dimljus/runpod/train.py --variant t2v --noise_level high \\
      --merge lightning --merge_strength 0.8
"""

import os
import sys
import subprocess
import datetime
import argparse
import glob
from pathlib import Path


# =============================================================================
# VARIANT CONFIGURATION
# =============================================================================
# Per-variant and per-expert defaults following dimljus conventions.
# All values are strings because they're passed as CLI args to musubi-tuner.

VARIANT_CONFIG = {
    "t2v": {
        "task": "t2v-A14B",
        "boundary": 875,
        "flow_shift": "3.0",
        "i2v_flag": False,
        "experts": {
            "high": {
                "learning_rate": "1e-4",
                "rank": "16",
                "alpha": "16",
                "max_epochs": "30",
                "save_every": "5",
            },
            "low": {
                "learning_rate": "8e-5",
                "rank": "16",
                "alpha": "16",
                "max_epochs": "50",
                "save_every": "5",
            },
        },
    },
    "i2v": {
        "task": "i2v-A14B",
        "boundary": 900,
        "flow_shift": "5.0",
        "i2v_flag": True,
        "experts": {
            "high": {
                "learning_rate": "1e-4",
                "rank": "16",
                "alpha": "16",
                "max_epochs": "30",
                "save_every": "5",
            },
            "low": {
                "learning_rate": "8e-5",
                "rank": "16",
                "alpha": "16",
                "max_epochs": "50",
                "save_every": "5",
            },
        },
    },
}

# --- Shared Defaults ---
OUTPUT_NAME         = "dimljus-wan22"
LR_SCHEDULER        = "cosine_with_min_lr"
MIN_LR_RATIO        = "0.01"
OPTIMIZER           = "adamw8bit"
SEED                = "42"

# --- Paths on RunPod ---
WORKSPACE       = "/workspace"
MODELS_DIR      = f"{WORKSPACE}/models"
DATASETS_DIR    = f"{WORKSPACE}/datasets"
OUTPUTS_DIR     = f"{WORKSPACE}/outputs"
DATASET_CONFIG  = f"{WORKSPACE}/dimljus/runpod/dataset-config.toml"
MUSUBI_DIR      = f"{WORKSPACE}/musubi-tuner"
RESUME_DIR      = f"{WORKSPACE}/resume_checkpoints"

# --- Model files (pre-downloaded by setup.sh) ---
MODEL_FILES = {
    "t2v": {
        "dit_high": f"{MODELS_DIR}/wan2.2_t2v_high_noise_14B_fp16.safetensors",
        "dit_low":  f"{MODELS_DIR}/wan2.2_t2v_low_noise_14B_fp16.safetensors",
    },
    "i2v": {
        "dit_high": f"{MODELS_DIR}/wan2.2_i2v_high_noise_14B_fp16.safetensors",
        "dit_low":  f"{MODELS_DIR}/wan2.2_i2v_low_noise_14B_fp16.safetensors",
    },
    # Shared across all variants
    "vae": f"{MODELS_DIR}/wan_2.1_vae.safetensors",
    "t5":  f"{MODELS_DIR}/models_t5_umt5-xxl-enc-bf16.pth",
}

# --- Merge Presets ---
# Speed LoRAs that can be baked into the DiT before training.
# The resulting character LoRA then works WITH the speed LoRA at inference.
MERGE_PRESETS = {
    "lightning": {
        "repo_id": "lightx2v/Wan2.2-Lightning",
        "t2v": {
            "high": "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors",
            "low":  "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors",
        },
        "i2v": {
            "high": "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors",
            "low":  "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors",
        },
    },
    "lightning-v1.1": {
        "repo_id": "lightx2v/Wan2.2-Lightning",
        "t2v": {
            "high": "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors",
            "low":  "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors",
        },
    },
}
MERGED_DITS_DIR = f"{WORKSPACE}/merged_dits"
MERGE_LORA_CACHE = f"{WORKSPACE}/merge_lora_cache"


# =============================================================================
# Merge Helper Functions
# =============================================================================

def resolve_merge_lora(merge_value, variant, noise_level):
    """Resolve --merge value to a local .safetensors path.

    Accepts either a preset name ('lightning') or a path to a .safetensors file.
    """
    # Direct file path
    if merge_value.endswith(".safetensors"):
        if not os.path.exists(merge_value):
            print(f"  ERROR: Merge LoRA file not found: {merge_value}")
            return None, None
        return merge_value, os.path.basename(merge_value).replace(".safetensors", "")

    # Named preset
    preset = MERGE_PRESETS.get(merge_value)
    if preset is None:
        print(f"  ERROR: Unknown merge preset '{merge_value}'")
        print(f"  Available presets: {', '.join(MERGE_PRESETS.keys())}")
        return None, None

    task_files = preset.get(variant)
    if task_files is None:
        print(f"  ERROR: Preset '{merge_value}' has no {variant} files")
        return None, None

    filename = task_files.get(noise_level)
    if filename is None:
        print(f"  ERROR: Preset '{merge_value}' has no {variant}/{noise_level} file")
        return None, None

    try:
        import huggingface_hub
        os.makedirs(MERGE_LORA_CACHE, exist_ok=True)

        lora_basename = os.path.basename(filename)
        local_lora = os.path.join(MERGE_LORA_CACHE, lora_basename)

        if os.path.exists(local_lora):
            print(f"  [CACHED] Merge LoRA: {local_lora}")
            return local_lora, merge_value

        print(f"  [DOWNLOADING] {preset['repo_id']}/{filename}...")
        downloaded = huggingface_hub.hf_hub_download(
            repo_id=preset["repo_id"],
            filename=filename,
        )
        import shutil
        shutil.copy2(downloaded, local_lora)
        print(f"  Downloaded and cached: {local_lora}")
        return local_lora, merge_value

    except Exception as e:
        print(f"  ERROR downloading merge LoRA: {e}")
        return None, None


def merge_lora_into_dit(dit_path, lora_path, output_path, strength=1.0):
    """Merge a LoRA's learned deltas into the base DiT weights.

    This creates a new .safetensors file with the LoRA baked in.
    Used for training character LoRAs on top of speed LoRAs.
    """
    import torch
    from safetensors.torch import load_file, save_file

    print(f"Loading base DiT: {dit_path}")
    print(f"  (this is ~28GB for 14B fp16 — loading on CPU)")
    base_sd = load_file(dit_path, device="cpu")
    print(f"  Loaded {len(base_sd)} base model keys")

    print(f"Loading merge LoRA: {lora_path}")
    lora_sd = load_file(lora_path, device="cpu")
    print(f"  Loaded {len(lora_sd)} LoRA keys")

    base_keys = set(base_sd.keys())

    # Find LoRA A/B pairs and map them to base model keys
    lora_pairs = {}
    unmapped_keys = []

    for key in lora_sd:
        if ".lora_down.weight" not in key:
            continue

        down_key = key
        up_key = key.replace(".lora_down.weight", ".lora_up.weight")
        alpha_key = key.replace(".lora_down.weight", ".alpha")

        if up_key not in lora_sd:
            print(f"  WARNING: Found down without up: {key}")
            continue

        # Try to map LoRA key to base model key
        module_path = key.replace(".lora_down.weight", "")
        candidates = [f"{module_path}.weight", module_path]
        for prefix in ["diffusion_model.", "lora_unet_", "lora_te_"]:
            if module_path.startswith(prefix):
                stripped = module_path[len(prefix):]
                candidates.extend([f"{stripped}.weight", stripped])
        if not module_path.startswith("diffusion_model."):
            candidates.extend([
                f"diffusion_model.{module_path}.weight",
                f"diffusion_model.{module_path}",
            ])

        base_key = None
        for candidate in candidates:
            if candidate in base_keys:
                base_key = candidate
                break

        if base_key is None:
            unmapped_keys.append(key)
            if len(unmapped_keys) <= 3:
                print(f"  WARNING: Cannot map: {key}")
            continue

        alpha = lora_sd[alpha_key].item() if alpha_key in lora_sd else None
        lora_pairs[base_key] = {
            "down": lora_sd[down_key],
            "up": lora_sd[up_key],
            "alpha": alpha,
        }

    print(f"\n  Found {len(lora_pairs)} LoRA layer pairs to merge")
    if unmapped_keys:
        print(f"  WARNING: {len(unmapped_keys)} keys could not be mapped")
    if len(lora_pairs) == 0:
        print("  ERROR: No LoRA pairs mapped — aborting merge")
        return False

    # Apply LoRA deltas: W' = W + scale * (up @ down)
    merged_count = 0
    for base_key, pair in lora_pairs.items():
        down = pair["down"].float()
        up = pair["up"].float()
        alpha = pair["alpha"]
        rank = down.shape[0]
        scale = strength * (alpha / rank) if alpha is not None else strength
        base_weight = base_sd[base_key].float()

        if down.dim() == 2 and up.dim() == 2:
            delta = (up @ down) * scale
        elif down.dim() == 5 and up.dim() == 5:
            if up.shape[2:] == (1, 1, 1):
                down_2d = down.reshape(rank, -1)
                up_2d = up.reshape(up.shape[0], rank)
                delta = (up_2d @ down_2d).reshape(base_weight.shape) * scale
            else:
                print(f"  WARNING: Complex Conv3d for {base_key}, skipping")
                continue
        elif down.dim() == 3 and up.dim() == 3:
            down_2d = down.reshape(rank, -1)
            up_2d = up.reshape(up.shape[0], rank)
            delta = (up_2d @ down_2d).reshape(base_weight.shape) * scale
        else:
            print(f"  WARNING: Unexpected dims for {base_key}, skipping")
            continue

        base_sd[base_key] = (base_weight + delta).to(base_sd[base_key].dtype)
        merged_count += 1

    print(f"  Merged {merged_count}/{len(lora_pairs)} layers (strength={strength})")

    print(f"  Saving merged DiT to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_file(base_sd, output_path)
    merged_size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"  Saved ({merged_size_gb:.1f} GB)")

    del base_sd, lora_sd
    return True


# =============================================================================
# Training
# =============================================================================

def train_expert(args, variant, noise_level):
    """Train one noise-level expert.

    Runs the musubi-tuner 3-step pipeline:
      1. Cache latents (VAE encode videos/images)
      2. Cache text encoder outputs (T5 encode captions)
      3. Train LoRA (accelerate launch)
    """
    is_high = noise_level == "high"
    label = "HIGH-NOISE" if is_high else "LOW-NOISE"
    vcfg = VARIANT_CONFIG[variant]
    expert = vcfg["experts"][noise_level]

    # --- Resolve hyperparameters (CLI overrides > per-expert defaults) ---
    lr          = args.lr or expert["learning_rate"]
    scheduler   = args.scheduler or LR_SCHEDULER
    min_lr      = args.min_lr or MIN_LR_RATIO
    optimizer   = args.optimizer or OPTIMIZER
    rank        = args.rank or expert["rank"]
    alpha       = args.alpha or expert["alpha"]
    epochs      = args.epochs or expert["max_epochs"]
    save_every  = args.save_every or expert["save_every"]
    seed        = args.seed or SEED
    flow_shift  = args.flow_shift or vcfg["flow_shift"]
    output_name = args.output_name or f"{OUTPUT_NAME}-{variant}"
    dataset_cfg = args.dataset_config or DATASET_CONFIG

    # --- Validate prerequisites ---
    dit_key = "dit_high" if is_high else "dit_low"
    dit_path = MODEL_FILES[variant][dit_key]
    vae_path = MODEL_FILES["vae"]
    t5_path  = MODEL_FILES["t5"]

    missing = []
    for name, path in [("DiT", dit_path), ("VAE", vae_path), ("T5", t5_path)]:
        if not os.path.exists(path):
            missing.append(f"  {name}: {path}")
    if missing:
        print("ERROR: Model files not found. Run setup.sh first.")
        for m in missing:
            print(m)
        sys.exit(1)

    if not os.path.exists(dataset_cfg):
        print(f"ERROR: Dataset config not found: {dataset_cfg}")
        print()
        print("  Edit the template at: /workspace/dimljus/runpod/dataset-config.toml")
        print("  Or specify a different config: --dataset_config /path/to/config.toml")
        sys.exit(1)

    # --- Detect musubi-tuner repo structure ---
    if os.path.exists(f"{MUSUBI_DIR}/src/musubi_tuner/wan_train_network.py"):
        script_prefix = "src/musubi_tuner/"
    else:
        script_prefix = ""

    # =================================================================
    # LoRA Merge (optional)
    # =================================================================
    merge_strength = float(args.merge_strength) if args.merge_strength else 1.0
    merge_preset_name = None

    if args.merge:
        print(f"\n{'='*60}")
        print(f"  LORA MERGE — {variant.upper()} {label} expert")
        print(f"  Merge: {args.merge}")
        print(f"  Strength: {merge_strength}")
        print(f"{'='*60}")

        lora_path, merge_preset_name = resolve_merge_lora(
            args.merge, variant, noise_level
        )
        if lora_path is None:
            print("  ERROR: Could not resolve merge LoRA. Aborting.")
            sys.exit(1)

        os.makedirs(MERGED_DITS_DIR, exist_ok=True)
        merge_tag = f"wan22_{variant}_{noise_level}_{merge_preset_name}_s{merge_strength}"
        merged_dit_path = os.path.join(MERGED_DITS_DIR, f"{merge_tag}.safetensors")

        if os.path.exists(merged_dit_path):
            print(f"  [CACHED] Merged DiT exists: {merged_dit_path}")
            dit_path = merged_dit_path
        else:
            success = merge_lora_into_dit(
                dit_path, lora_path, merged_dit_path, merge_strength
            )
            if success:
                print(f"  Merge complete. Using merged DiT for training.")
                dit_path = merged_dit_path
            else:
                print(f"  ERROR: Merge failed. Aborting.")
                sys.exit(1)

    # --- Timestamped output ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    name = f"{output_name}-{noise_level}-{timestamp}"
    run_output_dir = f"{OUTPUTS_DIR}/{output_name}"
    os.makedirs(run_output_dir, exist_ok=True)

    # =================================================================
    # Resume from checkpoint (optional)
    # =================================================================
    resume_weights = None
    os.makedirs(RESUME_DIR, exist_ok=True)

    if args.resume_from:
        if os.path.exists(args.resume_from):
            resume_weights = args.resume_from
            print(f"\n  RESUME: Using specified checkpoint:")
            print(f"    {resume_weights}")
        else:
            print(f"\n  WARNING: --resume_from path not found: {args.resume_from}")
            print(f"  Training will start from scratch.")
    else:
        # Auto-detect resume checkpoint
        candidates = sorted(
            glob.glob(f"{RESUME_DIR}/*.safetensors"),
            key=os.path.getmtime,
            reverse=True,
        )
        if candidates:
            resume_weights = candidates[0]
            print(f"\n  RESUME: Auto-detected checkpoint in {RESUME_DIR}/:")
            print(f"    {resume_weights}")
            if len(candidates) > 1:
                print(f"    ({len(candidates)} files — using newest)")
        else:
            print(f"\n  No resume checkpoint found. Starting from scratch.")
            print(f"  (To resume: drop a .safetensors in {RESUME_DIR}/ or use --resume_from)")

    if resume_weights:
        size_mb = os.path.getsize(resume_weights) / (1024 * 1024)
        print(f"    File size: {size_mb:.1f} MB")
        print(f"    NOTE: Loading weights only — LR schedule starts fresh.")

    # --- Print summary ---
    print("=" * 60)
    print(f"  Wan 2.2 {variant.upper()} — {label} Expert (RunPod)")
    if args.merge:
        print(f"  Merge: {args.merge} (strength={merge_strength})")
    else:
        print(f"  Training against stock DiT")
    print(f"  Output: {name}")
    print(f"  Dir:    {run_output_dir}")
    print(f"  LR: {lr} | Scheduler: {scheduler}")
    print(f"  Rank: {rank} | Alpha: {alpha}")
    print(f"  Epochs: {epochs} | Flow Shift: {flow_shift}")
    if resume_weights:
        print(f"  RESUMING from: {os.path.basename(resume_weights)}")
    print("=" * 60)

    # =================================================================
    # Step 1: Cache latents
    # =================================================================
    print(f"\n{'='*60}")
    print(f"  Step 1/3: Caching latents...")
    print(f"{'='*60}")
    cache_latents_cmd = [
        "python",
        f"{script_prefix}wan_cache_latents.py",
        "--dataset_config", dataset_cfg,
        "--vae", vae_path,
        "--vae_cache_cpu",
    ]
    if vcfg["i2v_flag"]:
        cache_latents_cmd.append("--i2v")
    subprocess.run(cache_latents_cmd, check=True)

    # =================================================================
    # Step 2: Cache text encoder outputs
    # =================================================================
    print(f"\n{'='*60}")
    print(f"  Step 2/3: Caching text encoder outputs...")
    print(f"{'='*60}")
    subprocess.run([
        "python",
        f"{script_prefix}wan_cache_text_encoder_outputs.py",
        "--dataset_config", dataset_cfg,
        "--t5", t5_path,
        "--batch_size", "16",
        "--fp8_t5",
    ], check=True)

    # =================================================================
    # Step 3: Train LoRA
    # =================================================================
    print(f"\n{'='*60}")
    print(f"  Step 3/3: Training {label} expert...")
    print(f"  DiT: {dit_path}")
    print(f"{'='*60}")

    # Expert timestep boundaries
    boundary = str(vcfg["boundary"])
    if is_high:
        min_ts, max_ts = boundary, "1000"
    else:
        min_ts, max_ts = "0", boundary

    train_cmd = [
        "accelerate", "launch",
        "--num_cpu_threads_per_process", "1",
        "--mixed_precision", "fp16",
        f"{script_prefix}wan_train_network.py",
        "--task", vcfg["task"],
        "--dit", dit_path,
        "--vae", vae_path,
        "--t5", t5_path,
        "--dataset_config", dataset_cfg,
        "--sdpa",
        "--mixed_precision", "fp16",
        "--fp8_base",
        "--fp8_scaled",
        "--vae_cache_cpu",
        # --- Expert timestep range ---
        "--min_timestep", min_ts,
        "--max_timestep", max_ts,
        "--preserve_distribution_shape",
        # --- Optimizer ---
        "--optimizer_type", optimizer,
        "--optimizer_args", "weight_decay=0.01",
        # --- Learning Rate ---
        "--learning_rate", lr,
        # --- LR Scheduler ---
        "--lr_scheduler", scheduler,
        "--lr_scheduler_min_lr_ratio", min_lr,
        # --- Memory ---
        "--gradient_checkpointing",
        "--max_data_loader_n_workers", "2",
        "--persistent_data_loader_workers",
        # --- LoRA Config ---
        "--network_module", "networks.lora_wan",
        "--network_dim", rank,
        "--network_alpha", alpha,
        # --- Timestep / Flow ---
        "--timestep_sampling", "shift",
        "--discrete_flow_shift", flow_shift,
        # --- Training Duration ---
        "--max_train_epochs", epochs,
        "--save_every_n_epochs", save_every,
        # --- Output ---
        "--seed", seed,
        "--output_dir", run_output_dir,
        "--output_name", name,
        # --- Logging ---
        "--log_with", "tensorboard",
        "--logging_dir", f"{run_output_dir}/logs",
    ]

    if resume_weights:
        train_cmd += ["--network_weights", resume_weights]

    print(f"\nTraining command:\n{' '.join(train_cmd)}\n")
    result = subprocess.run(train_cmd)

    if result.returncode != 0:
        print(f"ERROR: Training failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  {label} expert training complete!")
    print(f"{'='*60}")
    print(f"\nContents of {run_output_dir}:")
    for root, dirs, files in os.walk(run_output_dir):
        level = root.replace(run_output_dir, "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
        for f in sorted(files):
            filepath = os.path.join(root, f)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"{subindent}{f} ({size_mb:.1f} MB)")

    return run_output_dir


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Dimljus — Wan 2.2 LoRA Training for RunPod",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train high-noise expert (T2V)
  python train.py --variant t2v --noise_level high

  # Train both experts sequentially (T2V)
  python train.py --variant t2v --noise_level both

  # I2V with custom learning rate
  python train.py --variant i2v --noise_level low --lr 5e-5

  # Resume from checkpoint
  python train.py --variant t2v --noise_level high --resume_from /workspace/outputs/my-lora.safetensors

  # Merge Lightning speed LoRA before training
  python train.py --variant t2v --noise_level high --merge lightning
        """,
    )

    # --- Required ---
    parser.add_argument(
        "--variant", required=True, choices=["t2v", "i2v"],
        help="Model variant: 't2v' (text-to-video) or 'i2v' (image-to-video)",
    )
    parser.add_argument(
        "--noise_level", required=True, choices=["high", "low", "both"],
        help="Which expert to train: 'high', 'low', or 'both' (sequential)",
    )

    # --- Hyperparameters (override per-expert defaults) ---
    parser.add_argument("--lr", default=None, help="Learning rate")
    parser.add_argument("--rank", default=None, help="LoRA rank (default: 16 high, 24 low)")
    parser.add_argument("--alpha", default=None, help="LoRA alpha (default: matches rank)")
    parser.add_argument("--epochs", default=None, help="Max epochs")
    parser.add_argument("--save_every", default=None, help="Save checkpoint every N epochs")
    parser.add_argument("--scheduler", default=None, help=f"LR scheduler (default: {LR_SCHEDULER})")
    parser.add_argument("--min_lr", default=None, help=f"Min LR ratio (default: {MIN_LR_RATIO})")
    parser.add_argument("--optimizer", default=None, help=f"Optimizer (default: {OPTIMIZER})")
    parser.add_argument("--seed", default=None, help=f"Random seed (default: {SEED})")
    parser.add_argument("--flow_shift", default=None, help="Discrete flow shift")

    # --- Paths ---
    parser.add_argument("--output_name", default=None, help=f"Output name prefix")
    parser.add_argument("--dataset_config", default=None, help=f"Dataset TOML config path")
    parser.add_argument("--resume_from", default=None,
                        help="Path to a .safetensors checkpoint to resume from")

    # --- Merge ---
    parser.add_argument("--merge", default=None,
                        help="Merge a speed LoRA into the DiT before training. "
                             "Use a preset name ('lightning') or path to .safetensors")
    parser.add_argument("--merge_strength", default=None,
                        help="Merge strength (default: 1.0)")

    return parser.parse_args()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    if not os.path.exists(MUSUBI_DIR):
        print(f"ERROR: musubi-tuner not found at {MUSUBI_DIR}")
        print(f"Run setup.sh first: bash /workspace/dimljus/runpod/setup.sh")
        sys.exit(1)

    os.chdir(MUSUBI_DIR)
    print(f"Working directory: {os.getcwd()}")

    args = parse_args()

    if args.noise_level == "both":
        # Train both experts sequentially (high first, then low)
        print("\n" + "=" * 60)
        print("  Training BOTH experts sequentially")
        print("  Order: high-noise first, then low-noise")
        print("=" * 60)
        train_expert(args, args.variant, "high")
        print("\n\n")
        train_expert(args, args.variant, "low")
        print("\n" + "=" * 60)
        print("  Both experts complete!")
        print("=" * 60)
    else:
        train_expert(args, args.variant, args.noise_level)

    print(f"\nTo download results:")
    print(f"  Use Jupyter Lab to browse /workspace/outputs/")
    print(f"  Or: scp -P PORT root@HOST:/workspace/outputs/*.safetensors .")
