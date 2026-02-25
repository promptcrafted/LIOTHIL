"""
Exact Comparison: Wan 2.2 T2V Low-Noise Expert vs Wan 2.1 T2V Weights

This script answers a precise question: are the low-noise expert weights in
Wan 2.2 T2V EXACTLY identical to the Wan 2.1 T2V weights, byte for byte?

Background:
    Our task vector analysis (task_vector_analysis.py) found that the low-noise
    expert moved only ~0.04% from the Wan 2.1 base — suggesting Alibaba may have
    simply copied Wan 2.1 as-is into the low-noise expert slot, then only retrained
    the high-noise expert. But ~0.04% could be rounding noise, dtype conversion
    artifacts, or genuine tiny changes. This script settles the question definitively.

Model layout (Wan 2.2 T2V diffusers format):
    transformer/    → HIGH-noise expert (early denoising, global composition)
    transformer_2/  → LOW-noise expert  (late denoising, fine detail)

    The pipeline's boundary_ratio=0.875 controls when to switch between experts.
    "transformer" handles timesteps with high noise (t > boundary),
    "transformer_2" handles timesteps with low noise (t < boundary).

Usage:
    python scripts/compare_experts_to_21.py

Requirements:
    - safetensors (pip install safetensors)
    - torch (pip install torch)
    - ~CPU only, no GPU needed
    - Takes several minutes to scan ~55 GB of weights
"""

import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open


# ──────────────────────────────────────────────────────────────────────
# Configuration — edit these paths to match your local setup
# ──────────────────────────────────────────────────────────────────────

# Wan 2.1 T2V (single transformer, the "base" model)
WAN21_DIR = Path(r"C:\Users\minta\Projects\dimljus\models\Wan2.1-T2V-14B-Diffusers\transformer")

# Wan 2.2 T2V experts (two separate transformers)
WAN22_HIGH_NOISE_DIR = Path(r"C:\Users\minta\Projects\dimljus\models\Wan2.2-T2V-A14B-Diffusers\transformer")
WAN22_LOW_NOISE_DIR = Path(r"C:\Users\minta\Projects\dimljus\models\Wan2.2-T2V-A14B-Diffusers\transformer_2")


# ──────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────

def print_header(title: str) -> None:
    """Print a nicely formatted section header."""
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)
    print()


def load_index(model_dir: Path) -> dict[str, str]:
    """Load the safetensors shard index for a model directory.

    The index file maps each tensor name to the shard filename that contains it.
    For example: {"blocks.0.attn1.to_q.weight": "diffusion_pytorch_model-00001-of-00012.safetensors"}

    Returns:
        Dict mapping tensor_name -> shard_filename
    """
    index_path = model_dir / "diffusion_pytorch_model.safetensors.index.json"
    if not index_path.exists():
        print(f"ERROR: Index file not found: {index_path}")
        sys.exit(1)
    with open(index_path) as f:
        data = json.load(f)
    return data["weight_map"]


def list_directory_structure(model_dir: Path, label: str) -> None:
    """Print the directory structure of a model path for inspection."""
    print(f"{label}:")
    print(f"  Path: {model_dir}")

    if not model_dir.exists():
        print(f"  ERROR: Directory does not exist!")
        return

    safetensors_files = sorted(model_dir.glob("*.safetensors"))
    json_files = sorted(model_dir.glob("*.json"))

    for jf in json_files:
        size_kb = jf.stat().st_size / 1024
        print(f"    {jf.name:60s}  {size_kb:8.1f} KB")

    total_size_gb = 0
    for sf in safetensors_files:
        size_gb = sf.stat().st_size / (1024 ** 3)
        total_size_gb += size_gb
        print(f"    {sf.name:60s}  {size_gb:6.2f} GB")

    print(f"  Total safetensors: {len(safetensors_files)} files, {total_size_gb:.2f} GB")


def compare_tensors_exact(
    dir_a: Path,
    index_a: dict[str, str],
    dir_b: Path,
    index_b: dict[str, str],
    label_a: str,
    label_b: str,
) -> dict:
    """Compare every tensor between two models, byte by byte.

    Uses safetensors lazy loading so we only hold one tensor pair in memory at a time.
    This keeps memory usage under ~1 GB even though the models are ~55 GB each.

    Args:
        dir_a: Path to first model's transformer directory
        index_a: Tensor-to-shard mapping for model A
        dir_b: Path to second model's transformer directory
        index_b: Tensor-to-shard mapping for model B
        label_a: Human-readable name for model A (e.g., "Wan 2.2 Low-Noise")
        label_b: Human-readable name for model B (e.g., "Wan 2.1 T2V")

    Returns:
        Summary dict with comparison results
    """
    # Figure out which tensors exist in each model
    keys_a = set(index_a.keys())
    keys_b = set(index_b.keys())
    common_keys = sorted(keys_a & keys_b)
    only_in_a = sorted(keys_a - keys_b)
    only_in_b = sorted(keys_b - keys_a)

    # Report any tensors that exist in one model but not the other
    if only_in_a:
        print(f"  WARNING: {len(only_in_a)} tensors only in {label_a}:")
        for k in only_in_a:
            print(f"    - {k}")
        print()

    if only_in_b:
        print(f"  WARNING: {len(only_in_b)} tensors only in {label_b}:")
        for k in only_in_b:
            print(f"    - {k}")
        print()

    print(f"  Tensors in common: {len(common_keys)}")
    print(f"  Tensors only in {label_a}: {len(only_in_a)}")
    print(f"  Tensors only in {label_b}: {len(only_in_b)}")
    print()

    # Track results
    identical_count = 0
    different_count = 0
    shape_mismatch_count = 0
    dtype_mismatch_count = 0
    differences = []  # List of (tensor_name, max_abs_diff, relative_diff, dtype_a, dtype_b)

    # Cache open file handles per shard (avoid reopening the same shard many times)
    # Safetensors files are memory-mapped, so this is efficient
    open_files_a: dict[str, object] = {}
    open_files_b: dict[str, object] = {}

    total = len(common_keys)
    start_time = time.time()
    last_progress_time = start_time

    # We group tensors by their shard to minimize file opens.
    # Build shard -> [tensor_names] mapping for both models
    shard_groups_a: dict[str, list[str]] = {}
    for key in common_keys:
        shard = index_a[key]
        shard_groups_a.setdefault(shard, []).append(key)

    # Process tensors grouped by shard_a to minimize file opening
    # But we still need random access to shard_b, so we track which
    # shard_b files are open
    processed = 0

    for shard_a_name in sorted(shard_groups_a.keys()):
        # Open shard from model A
        shard_a_path = str(dir_a / shard_a_name)
        f_a = safe_open(shard_a_path, framework="pt", device="cpu")

        # Track which model B shards we need for this group
        needed_b_shards: dict[str, object] = {}

        for tensor_name in shard_groups_a[shard_a_name]:
            # Load tensor from model A
            tensor_a = f_a.get_tensor(tensor_name)

            # Load tensor from model B (open shard if not cached)
            shard_b_name = index_b[tensor_name]
            if shard_b_name not in needed_b_shards:
                shard_b_path = str(dir_b / shard_b_name)
                needed_b_shards[shard_b_name] = safe_open(shard_b_path, framework="pt", device="cpu")
            f_b = needed_b_shards[shard_b_name]
            tensor_b = f_b.get_tensor(tensor_name)

            processed += 1

            # Check shapes match
            if tensor_a.shape != tensor_b.shape:
                shape_mismatch_count += 1
                differences.append({
                    "name": tensor_name,
                    "issue": "SHAPE_MISMATCH",
                    "shape_a": list(tensor_a.shape),
                    "shape_b": list(tensor_b.shape),
                    "dtype_a": str(tensor_a.dtype),
                    "dtype_b": str(tensor_b.dtype),
                })
                continue

            # Record dtype information
            dtypes_match = tensor_a.dtype == tensor_b.dtype

            # Exact byte-level comparison (same dtype, same bits)
            # torch.equal checks exact equality including dtype
            is_exact = torch.equal(tensor_a, tensor_b)

            if is_exact:
                identical_count += 1
            else:
                different_count += 1

                # Compute how different they are
                # Cast to float32 for numerical comparison
                a_f32 = tensor_a.to(torch.float32)
                b_f32 = tensor_b.to(torch.float32)
                diff = (a_f32 - b_f32).abs()
                max_abs_diff = diff.max().item()
                mean_abs_diff = diff.mean().item()

                # Relative magnitude: how big is the difference compared to the weights?
                weight_norm = a_f32.abs().mean().item()
                if weight_norm > 0:
                    relative_pct = (mean_abs_diff / weight_norm) * 100
                else:
                    relative_pct = float("inf") if mean_abs_diff > 0 else 0.0

                # Count how many individual values differ
                num_different = (tensor_a != tensor_b).sum().item()
                total_elements = tensor_a.numel()
                pct_different = (num_different / total_elements) * 100

                differences.append({
                    "name": tensor_name,
                    "issue": "VALUE_MISMATCH",
                    "dtype_a": str(tensor_a.dtype),
                    "dtype_b": str(tensor_b.dtype),
                    "dtypes_match": dtypes_match,
                    "shape": list(tensor_a.shape),
                    "max_abs_diff": max_abs_diff,
                    "mean_abs_diff": mean_abs_diff,
                    "relative_pct": relative_pct,
                    "num_different": num_different,
                    "total_elements": total_elements,
                    "pct_different": pct_different,
                })

                if not dtypes_match:
                    dtype_mismatch_count += 1

            # Print progress every 5 seconds (these files are big, user needs feedback)
            now = time.time()
            if now - last_progress_time >= 5.0 or processed == total:
                elapsed = now - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total - processed) / rate if rate > 0 else 0
                print(
                    f"  [{processed:4d}/{total}] "
                    f"identical={identical_count}, different={different_count}, "
                    f"shape_mismatch={shape_mismatch_count}  "
                    f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)",
                    flush=True,
                )
                last_progress_time = now

        # Close file handles for this shard group (Python GC handles safetensors,
        # but let's be explicit about not accumulating references)
        del f_a
        for fh in needed_b_shards.values():
            del fh
        needed_b_shards.clear()

    return {
        "total_compared": len(common_keys),
        "identical": identical_count,
        "different": different_count,
        "shape_mismatch": shape_mismatch_count,
        "dtype_mismatch": dtype_mismatch_count,
        "only_in_a": only_in_a,
        "only_in_b": only_in_b,
        "differences": differences,
    }


def print_summary(results: dict, label_a: str, label_b: str) -> None:
    """Print a clear, readable summary of the comparison results."""
    print_header("RESULTS SUMMARY")

    total = results["total_compared"]
    identical = results["identical"]
    different = results["different"]
    shape_mm = results["shape_mismatch"]

    print(f"  Comparison: {label_a} vs {label_b}")
    print()
    print(f"  Total tensors compared:     {total}")
    print(f"  Exactly identical:          {identical}  ({identical/total*100:.1f}%)")
    print(f"  Different values:           {different}  ({different/total*100:.1f}%)")
    print(f"  Shape mismatches:           {shape_mm}")
    print()

    # Overall verdict
    if different == 0 and shape_mm == 0 and not results["only_in_a"] and not results["only_in_b"]:
        print("  *** VERDICT: IDENTICAL ***")
        print()
        print("  The Wan 2.2 low-noise expert is an EXACT copy of Wan 2.1 T2V weights.")
        print("  Every tensor matches byte-for-byte. Alibaba did not modify the low-noise")
        print("  expert at all — they copied Wan 2.1 wholesale and only retrained the")
        print("  high-noise expert to create Wan 2.2's MoE architecture.")
        print()
        print("  Implication for LoRA training:")
        print("    - A Wan 2.1 T2V LoRA IS a low-noise expert LoRA")
        print("    - 'Unified warmup' = just train on Wan 2.1 first")
        print("    - The high-noise expert is where ALL the 2.2 specialization lives")
    elif different == 0 and shape_mm == 0:
        print("  *** VERDICT: IDENTICAL (with caveats) ***")
        print()
        print("  All shared tensors match exactly, but the models have different tensor sets.")
        print("  See warnings above for details.")
    else:
        print("  *** VERDICT: DIFFERS ***")
        print()
        if different > 0:
            print(f"  {different} tensor(s) have different values:")
            print()

            # Sort differences by magnitude for easy scanning
            value_diffs = [d for d in results["differences"] if d["issue"] == "VALUE_MISMATCH"]
            value_diffs.sort(key=lambda x: x["max_abs_diff"], reverse=True)

            # Print header row
            print(f"  {'Tensor Name':<55s} {'Max Abs Diff':>12s} {'Mean Abs Diff':>13s} {'Rel %':>7s} {'% Changed':>10s} {'dtypes':>12s}")
            print(f"  {'-'*55} {'-'*12} {'-'*13} {'-'*7} {'-'*10} {'-'*12}")

            for d in value_diffs:
                dtype_note = "" if d["dtypes_match"] else f" {d['dtype_a']}/{d['dtype_b']}"
                print(
                    f"  {d['name']:<55s} "
                    f"{d['max_abs_diff']:12.2e} "
                    f"{d['mean_abs_diff']:13.2e} "
                    f"{d['relative_pct']:6.3f}% "
                    f"{d['pct_different']:9.2f}% "
                    f"{dtype_note}"
                )

            # Summary statistics across all differences
            if value_diffs:
                print()
                all_max = max(d["max_abs_diff"] for d in value_diffs)
                all_mean = sum(d["mean_abs_diff"] for d in value_diffs) / len(value_diffs)
                all_rel = sum(d["relative_pct"] for d in value_diffs) / len(value_diffs)
                print(f"  Across all differing tensors:")
                print(f"    Largest max absolute difference:   {all_max:.2e}")
                print(f"    Average mean absolute difference:  {all_mean:.2e}")
                print(f"    Average relative difference:       {all_rel:.4f}%")

        if shape_mm > 0:
            print()
            shape_diffs = [d for d in results["differences"] if d["issue"] == "SHAPE_MISMATCH"]
            print(f"  {shape_mm} tensor(s) have shape mismatches:")
            for d in shape_diffs:
                print(f"    {d['name']}: {d['shape_a']} vs {d['shape_b']}")

    print()


# ──────────────────────────────────────────────────────────────────────
# Also compare the HIGH-NOISE expert for completeness
# ──────────────────────────────────────────────────────────────────────

def quick_check_high_noise(
    dir_high: Path,
    index_high: dict[str, str],
    dir_21: Path,
    index_21: dict[str, str],
) -> None:
    """Do a quick spot-check of the high-noise expert vs Wan 2.1.

    We already know from the task vector analysis that the high-noise expert
    moved ~46% from Wan 2.1. This just confirms it's NOT identical, as a
    sanity check that our comparison method works.

    Only checks the FIRST 5 tensors to save time.
    """
    print_header("SANITY CHECK: High-Noise Expert vs Wan 2.1 (first 5 tensors)")

    common = sorted(set(index_high.keys()) & set(index_21.keys()))[:5]

    for tensor_name in common:
        # Load from high-noise expert
        shard_h = index_high[tensor_name]
        with safe_open(str(dir_high / shard_h), framework="pt", device="cpu") as f:
            t_high = f.get_tensor(tensor_name)

        # Load from Wan 2.1
        shard_21 = index_21[tensor_name]
        with safe_open(str(dir_21 / shard_21), framework="pt", device="cpu") as f:
            t_21 = f.get_tensor(tensor_name)

        is_exact = torch.equal(t_high, t_21)

        if is_exact:
            print(f"  {tensor_name}: IDENTICAL")
        else:
            diff = (t_high.to(torch.float32) - t_21.to(torch.float32)).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            print(f"  {tensor_name}: DIFFERS (max={max_diff:.2e}, mean={mean_diff:.2e})")

    print()
    print("  (If the high-noise expert differs from 2.1, our comparison method is working correctly.)")
    print()


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print_header("Wan 2.2 Low-Noise Expert vs Wan 2.1 T2V — Exact Comparison")

    # ── Step 1: Show the directory structure ──────────────────────────
    print_header("DIRECTORY STRUCTURE")

    list_directory_structure(WAN21_DIR, "Wan 2.1 T2V (base model)")
    print()
    list_directory_structure(WAN22_LOW_NOISE_DIR, "Wan 2.2 T2V Low-Noise Expert (transformer_2)")
    print()
    list_directory_structure(WAN22_HIGH_NOISE_DIR, "Wan 2.2 T2V High-Noise Expert (transformer)")
    print()

    # ── Step 2: Load shard indices ────────────────────────────────────
    print_header("LOADING SHARD INDICES")

    index_21 = load_index(WAN21_DIR)
    print(f"  Wan 2.1 T2V:          {len(index_21)} tensors across {len(set(index_21.values()))} shards")

    index_low = load_index(WAN22_LOW_NOISE_DIR)
    print(f"  Wan 2.2 Low-Noise:    {len(index_low)} tensors across {len(set(index_low.values()))} shards")

    index_high = load_index(WAN22_HIGH_NOISE_DIR)
    print(f"  Wan 2.2 High-Noise:   {len(index_high)} tensors across {len(set(index_high.values()))} shards")
    print()

    # Quick sanity check: do all three models have the same tensor names?
    keys_21 = set(index_21.keys())
    keys_low = set(index_low.keys())
    keys_high = set(index_high.keys())

    if keys_21 == keys_low == keys_high:
        print("  All three models have identical tensor name sets.")
    else:
        if keys_21 != keys_low:
            diff = keys_21.symmetric_difference(keys_low)
            print(f"  WARNING: Wan 2.1 and Low-Noise differ in {len(diff)} tensor names")
        if keys_21 != keys_high:
            diff = keys_21.symmetric_difference(keys_high)
            print(f"  WARNING: Wan 2.1 and High-Noise differ in {len(diff)} tensor names")
        if keys_low != keys_high:
            diff = keys_low.symmetric_difference(keys_high)
            print(f"  WARNING: Low-Noise and High-Noise differ in {len(diff)} tensor names")

    # ── Step 3: Sanity check — high-noise should DIFFER ───────────────
    quick_check_high_noise(WAN22_HIGH_NOISE_DIR, index_high, WAN21_DIR, index_21)

    # ── Step 4: Full comparison — low-noise vs 2.1 ───────────────────
    print_header("FULL COMPARISON: Wan 2.2 Low-Noise vs Wan 2.1 T2V")
    print("  Comparing every tensor byte-by-byte. This will take a few minutes...")
    print("  (Loading ~55 GB per model via lazy safetensors access)")
    print()

    start = time.time()
    results = compare_tensors_exact(
        dir_a=WAN22_LOW_NOISE_DIR,
        index_a=index_low,
        dir_b=WAN21_DIR,
        index_b=index_21,
        label_a="Wan 2.2 Low-Noise",
        label_b="Wan 2.1 T2V",
    )
    elapsed = time.time() - start

    print()
    print(f"  Comparison completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")

    # ── Step 5: Print summary ─────────────────────────────────────────
    print_summary(results, "Wan 2.2 Low-Noise Expert", "Wan 2.1 T2V")


if __name__ == "__main__":
    main()
