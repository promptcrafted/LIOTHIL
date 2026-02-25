"""
Task Vector Analysis: Wan 2.1 -> Wan 2.2 Expert Divergence

Computes task vectors (expert_weights - base_weights) for both Wan 2.2 experts
relative to Wan 2.1 as baseline, then analyzes:
  - How far each expert moved from the base (magnitude)
  - Whether the two experts moved in the same or different directions (cosine similarity)
  - Per-block and per-layer-type aggregation

This tells us:
  - Which blocks specialized the most during MoE training
  - Which blocks moved together (shared improvement) vs apart (clean specialization)
  - Directly informs LoRA training strategy: unified warmup targets vs divergent targets

Memory-efficient: loads one layer at a time via safetensors lazy loading.
Requires only CPU — no GPU needed.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


# ──────────────────────────────────────────────────────────────────────
# Configuration — edit these paths to match your local setup
# ──────────────────────────────────────────────────────────────────────

BASE_MODEL = Path(r"C:\Users\minta\Projects\dimljus\models\Wan2.1-T2V-14B-Diffusers\transformer")
HIGH_NOISE = Path(r"C:\Users\minta\Projects\dimljus\models\Wan2.2-T2V-A14B-Diffusers\transformer")
LOW_NOISE = Path(r"C:\Users\minta\Projects\dimljus\models\Wan2.2-T2V-A14B-Diffusers\transformer_2")


def load_index(model_dir: Path) -> dict[str, str]:
    """Load the safetensors index file that maps layer names to shard files.

    Returns a dict of {layer_name: shard_filename}.
    """
    index_path = model_dir / "diffusion_pytorch_model.safetensors.index.json"
    with open(index_path) as f:
        data = json.load(f)
    return data["weight_map"]


def load_tensor(model_dir: Path, shard_file: str, layer_name: str) -> torch.Tensor:
    """Lazy-load a single tensor from a safetensors shard.

    This only reads the specific tensor from disk, not the entire shard.
    Converts to float32 for consistent computation (models may be bf16).
    """
    shard_path = model_dir / shard_file
    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
        tensor = f.get_tensor(layer_name)
    return tensor.to(torch.float32)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two flattened tensors."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    dot = torch.dot(a_flat, b_flat)
    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return (dot / (norm_a * norm_b)).item()


def l2_norm(t: torch.Tensor) -> float:
    """L2 norm of a flattened tensor."""
    return torch.norm(t.flatten()).item()


def relative_magnitude(task_vec: torch.Tensor, base: torch.Tensor) -> float:
    """How large the task vector is relative to the base weights.

    Expressed as a percentage: (||task_vec|| / ||base||) * 100.
    This tells us what fraction of the base weight magnitude was 'moved'
    during specialization.
    """
    base_norm = l2_norm(base)
    if base_norm == 0:
        return 0.0
    return (l2_norm(task_vec) / base_norm) * 100


def parse_layer_name(name: str) -> dict:
    """Parse a diffusers layer name into block index, component, and sublayer.

    Examples:
        'blocks.5.attn1.to_q.weight' -> {block: 5, component: 'self_attn', sublayer: 'to_q', param: 'weight'}
        'blocks.12.ffn.net.0.proj.weight' -> {block: 12, component: 'ffn', sublayer: 'up_proj', param: 'weight'}
        'blocks.0.norm2.weight' -> {block: 0, component: 'norm', sublayer: 'norm2', param: 'weight'}
        'blocks.0.scale_shift_table' -> {block: 0, component: 'modulation', sublayer: 'scale_shift_table', param: 'table'}
        'norm_out.weight' -> {block: -1, component: 'global', sublayer: 'norm_out', param: 'weight'}
    """
    # Block-level layers
    block_match = re.match(r"blocks\.(\d+)\.(.*)", name)
    if block_match:
        block_idx = int(block_match.group(1))
        rest = block_match.group(2)

        # Self-attention (attn1)
        if rest.startswith("attn1."):
            sublayer = rest.replace("attn1.", "").replace(".0.", "_")
            return {"block": block_idx, "component": "self_attn", "sublayer": sublayer, "param": name.split(".")[-1]}

        # Cross-attention (attn2)
        if rest.startswith("attn2."):
            sublayer = rest.replace("attn2.", "").replace(".0.", "_")
            return {"block": block_idx, "component": "cross_attn", "sublayer": sublayer, "param": name.split(".")[-1]}

        # FFN
        if rest.startswith("ffn."):
            if "net.0.proj" in rest:
                sublayer = "up_proj"
            elif "net.2" in rest:
                sublayer = "down_proj"
            else:
                sublayer = rest.replace("ffn.", "")
            return {"block": block_idx, "component": "ffn", "sublayer": sublayer, "param": name.split(".")[-1]}

        # Norms
        if rest.startswith("norm"):
            return {"block": block_idx, "component": "norm", "sublayer": rest.split(".")[0], "param": name.split(".")[-1]}

        # Modulation (scale_shift_table)
        if "scale_shift_table" in rest:
            return {"block": block_idx, "component": "modulation", "sublayer": "scale_shift_table", "param": "table"}

        return {"block": block_idx, "component": "other", "sublayer": rest, "param": name.split(".")[-1]}

    # Global layers (not in a block)
    return {"block": -1, "component": "global", "sublayer": name, "param": name.split(".")[-1]}


def is_lora_target(parsed: dict) -> bool:
    """Would this layer get a LoRA adapter in standard Wan training?

    LoRA targets: attention Q/K/V/O projections + FFN up/down projections.
    LoRA excludes: norms, modulation, global layers (embeddings, head).
    """
    if parsed["component"] in ("self_attn", "cross_attn"):
        return parsed["sublayer"].startswith("to_") and parsed["param"] == "weight"
    if parsed["component"] == "ffn":
        return parsed["param"] == "weight"
    return False


def main():
    print("=" * 80)
    print("TASK VECTOR ANALYSIS: Wan 2.1 -> Wan 2.2 Expert Specialization")
    print("=" * 80)
    print()

    # ── Load index files ──────────────────────────────────────────────
    print("Loading index files...")
    try:
        base_index = load_index(BASE_MODEL)
        high_index = load_index(HIGH_NOISE)
        low_index = load_index(LOW_NOISE)
    except FileNotFoundError as e:
        print(f"\nERROR: Could not find model files: {e}")
        print("Make sure all three models are downloaded.")
        sys.exit(1)

    # Find layers present in all three models
    common_layers = sorted(set(base_index.keys()) & set(high_index.keys()) & set(low_index.keys()))
    print(f"  Base model layers:       {len(base_index)}")
    print(f"  High-noise expert layers: {len(high_index)}")
    print(f"  Low-noise expert layers:  {len(low_index)}")
    print(f"  Common layers:           {len(common_layers)}")

    # Layers unique to one model (e.g., I2V has extra cross-attn projections)
    base_only = set(base_index.keys()) - set(high_index.keys())
    high_only = set(high_index.keys()) - set(base_index.keys())
    low_only = set(low_index.keys()) - set(base_index.keys())
    if base_only:
        print(f"  Base-only layers:        {len(base_only)} (e.g., {list(base_only)[:3]})")
    if high_only:
        print(f"  High-noise-only layers:  {len(high_only)} (e.g., {list(high_only)[:3]})")
    if low_only:
        print(f"  Low-noise-only layers:   {len(low_only)} (e.g., {list(low_only)[:3]})")
    print()

    # ── Analyze each layer ────────────────────────────────────────────
    print("Analyzing layers (this processes ~53GB of weights, may take a few minutes)...")
    print()

    # Storage for per-layer results
    results = []

    for i, layer_name in enumerate(common_layers):
        parsed = parse_layer_name(layer_name)

        # Load tensors from all three models
        base_tensor = load_tensor(BASE_MODEL, base_index[layer_name], layer_name)
        high_tensor = load_tensor(HIGH_NOISE, high_index[layer_name], layer_name)
        low_tensor = load_tensor(LOW_NOISE, low_index[layer_name], layer_name)

        # Compute task vectors: how far each expert moved from base
        tv_high = high_tensor - base_tensor  # high-noise task vector
        tv_low = low_tensor - base_tensor    # low-noise task vector

        # Metrics
        result = {
            "layer": layer_name,
            "block": parsed["block"],
            "component": parsed["component"],
            "sublayer": parsed["sublayer"],
            "is_lora_target": is_lora_target(parsed),
            "numel": base_tensor.numel(),
            # How far each expert moved from base (as % of base magnitude)
            "high_rel_magnitude": relative_magnitude(tv_high, base_tensor),
            "low_rel_magnitude": relative_magnitude(tv_low, base_tensor),
            # Absolute magnitudes of task vectors
            "high_tv_norm": l2_norm(tv_high),
            "low_tv_norm": l2_norm(tv_low),
            # Did the experts move in the same direction?
            # +1 = same direction, 0 = orthogonal, -1 = opposite
            "tv_cosine": cosine_similarity(tv_high, tv_low),
            # How similar are the experts to each other?
            "expert_cosine": cosine_similarity(high_tensor, low_tensor),
            # How similar is each expert to the base?
            "high_base_cosine": cosine_similarity(high_tensor, base_tensor),
            "low_base_cosine": cosine_similarity(low_tensor, base_tensor),
        }
        results.append(result)

        # Free memory immediately
        del base_tensor, high_tensor, low_tensor, tv_high, tv_low

        # Progress
        if (i + 1) % 50 == 0 or i == len(common_layers) - 1:
            print(f"  Processed {i + 1}/{len(common_layers)} layers...")

    print()

    # ── Per-Layer Report (LoRA targets only) ──────────────────────────
    lora_results = [r for r in results if r["is_lora_target"]]
    non_lora_results = [r for r in results if not r["is_lora_target"] and r["block"] >= 0]

    print("=" * 80)
    print("PER-LAYER RESULTS: LoRA Target Layers (weights only)")
    print("=" * 80)
    print()
    print(f"{'Layer':<50} {'High%':>6} {'Low%':>6} {'TV cos':>7} {'Exp cos':>8}")
    print("-" * 80)

    for r in sorted(lora_results, key=lambda x: x["tv_cosine"]):
        short_name = r["layer"].replace("diffusion_pytorch_model.", "")
        print(f"{short_name:<50} {r['high_rel_magnitude']:>5.2f}% {r['low_rel_magnitude']:>5.2f}% {r['tv_cosine']:>7.4f} {r['expert_cosine']:>8.5f}")

    # ── Per-Block Aggregation ─────────────────────────────────────────
    print()
    print("=" * 80)
    print("PER-BLOCK AGGREGATION (LoRA targets only)")
    print("=" * 80)
    print()
    print("  High%  = how far the high-noise expert moved from Wan 2.1 (avg relative magnitude)")
    print("  Low%   = how far the low-noise expert moved from Wan 2.1")
    print("  TV cos = did both experts move in the same direction? (+1=same, 0=orthogonal, -1=opposite)")
    print("  E cos  = how similar are the two experts to each other?")
    print()

    block_agg = defaultdict(lambda: {
        "high_mag": [], "low_mag": [], "tv_cos": [], "exp_cos": [],
        "high_base_cos": [], "low_base_cos": [],
    })

    for r in lora_results:
        b = r["block"]
        block_agg[b]["high_mag"].append(r["high_rel_magnitude"])
        block_agg[b]["low_mag"].append(r["low_rel_magnitude"])
        block_agg[b]["tv_cos"].append(r["tv_cosine"])
        block_agg[b]["exp_cos"].append(r["expert_cosine"])
        block_agg[b]["high_base_cos"].append(r["high_base_cosine"])
        block_agg[b]["low_base_cos"].append(r["low_base_cosine"])

    print(f"{'Block':>5}  {'High%':>7}  {'Low%':>7}  {'TV cos':>7}  {'E cos':>7}  {'H>base':>7}  {'L>base':>7}  {'Interpretation'}")
    print("-" * 100)

    for block_idx in sorted(block_agg.keys()):
        agg = block_agg[block_idx]
        avg = lambda lst: sum(lst) / len(lst) if lst else 0
        h_mag = avg(agg["high_mag"])
        l_mag = avg(agg["low_mag"])
        tv_cos = avg(agg["tv_cos"])
        e_cos = avg(agg["exp_cos"])
        h_base = avg(agg["high_base_cos"])
        l_base = avg(agg["low_base_cos"])

        # Interpret the pattern
        if tv_cos > 0.7:
            interp = "SHARED improvement (both moved together)"
        elif tv_cos > 0.3:
            interp = "MIXED (partially shared, partially divergent)"
        elif tv_cos > -0.3:
            interp = "SPECIALIZED (moved in different directions)"
        else:
            interp = "OPPOSED (moved in opposite directions)"

        print(f"  {block_idx:>3}   {h_mag:>6.2f}%  {l_mag:>6.2f}%  {tv_cos:>7.4f}  {e_cos:>7.5f}  {h_base:>7.5f}  {l_base:>7.5f}  {interp}")

    # ── Per-Component Aggregation ─────────────────────────────────────
    print()
    print("=" * 80)
    print("PER-COMPONENT AGGREGATION (across all blocks)")
    print("=" * 80)
    print()

    comp_agg = defaultdict(lambda: {"high_mag": [], "low_mag": [], "tv_cos": [], "exp_cos": []})

    for r in lora_results:
        comp = r["component"]
        sub = r["sublayer"]
        key = f"{comp}.{sub}"
        comp_agg[key]["high_mag"].append(r["high_rel_magnitude"])
        comp_agg[key]["low_mag"].append(r["low_rel_magnitude"])
        comp_agg[key]["tv_cos"].append(r["tv_cosine"])
        comp_agg[key]["exp_cos"].append(r["expert_cosine"])

    print(f"{'Component':<30} {'High%':>7}  {'Low%':>7}  {'TV cos':>7}  {'E cos':>7}")
    print("-" * 70)

    for key in sorted(comp_agg.keys()):
        agg = comp_agg[key]
        avg = lambda lst: sum(lst) / len(lst) if lst else 0
        print(f"{key:<30} {avg(agg['high_mag']):>6.2f}%  {avg(agg['low_mag']):>6.2f}%  {avg(agg['tv_cos']):>7.4f}  {avg(agg['exp_cos']):>7.5f}")

    # ── Non-LoRA layers (norms, modulation) ───────────────────────────
    print()
    print("=" * 80)
    print("NON-LORA LAYERS (norms, modulation — for comparison)")
    print("=" * 80)
    print()

    nonlora_comp = defaultdict(lambda: {"high_mag": [], "low_mag": [], "tv_cos": [], "exp_cos": []})
    for r in non_lora_results:
        key = f"{r['component']}.{r['sublayer']}"
        nonlora_comp[key]["high_mag"].append(r["high_rel_magnitude"])
        nonlora_comp[key]["low_mag"].append(r["low_rel_magnitude"])
        nonlora_comp[key]["tv_cos"].append(r["tv_cosine"])
        nonlora_comp[key]["exp_cos"].append(r["expert_cosine"])

    print(f"{'Component':<30} {'High%':>7}  {'Low%':>7}  {'TV cos':>7}  {'E cos':>7}")
    print("-" * 70)

    for key in sorted(nonlora_comp.keys()):
        agg = nonlora_comp[key]
        avg = lambda lst: sum(lst) / len(lst) if lst else 0
        print(f"{key:<30} {avg(agg['high_mag']):>6.2f}%  {avg(agg['low_mag']):>6.2f}%  {avg(agg['tv_cos']):>7.4f}  {avg(agg['exp_cos']):>7.5f}")

    # ── Global layers ─────────────────────────────────────────────────
    global_results = [r for r in results if r["block"] == -1]
    if global_results:
        print()
        print("=" * 80)
        print("GLOBAL LAYERS (embeddings, head — not in blocks)")
        print("=" * 80)
        print()
        print(f"{'Layer':<50} {'High%':>6} {'Low%':>6} {'TV cos':>7} {'Exp cos':>8}")
        print("-" * 80)
        for r in global_results:
            print(f"{r['layer']:<50} {r['high_rel_magnitude']:>5.2f}% {r['low_rel_magnitude']:>5.2f}% {r['tv_cosine']:>7.4f} {r['expert_cosine']:>8.5f}")

    # ── Summary statistics ────────────────────────────────────────────
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    avg = lambda lst: sum(lst) / len(lst) if lst else 0

    all_tv_cos = [r["tv_cosine"] for r in lora_results]
    all_high_mag = [r["high_rel_magnitude"] for r in lora_results]
    all_low_mag = [r["low_rel_magnitude"] for r in lora_results]

    print(f"  LoRA target layers analyzed: {len(lora_results)}")
    print(f"  Avg task vector cosine (direction agreement): {avg(all_tv_cos):.4f}")
    print(f"  Avg high-noise relative magnitude:           {avg(all_high_mag):.2f}%")
    print(f"  Avg low-noise relative magnitude:            {avg(all_low_mag):.2f}%")
    print()

    # Blocks sorted by specialization (lowest TV cosine = most specialized)
    block_specialization = []
    for block_idx in sorted(block_agg.keys()):
        agg = block_agg[block_idx]
        block_specialization.append((block_idx, avg(agg["tv_cos"]), avg(agg["high_mag"]), avg(agg["low_mag"])))

    block_specialization.sort(key=lambda x: x[1])

    print("  MOST SPECIALIZED blocks (experts moved in different directions):")
    for block_idx, tv_cos, h_mag, l_mag in block_specialization[:5]:
        print(f"    Block {block_idx:>2}: TV cosine = {tv_cos:.4f}, high moved {h_mag:.2f}%, low moved {l_mag:.2f}%")

    print()
    print("  MOST SHARED blocks (experts moved in the same direction):")
    for block_idx, tv_cos, h_mag, l_mag in block_specialization[-5:]:
        print(f"    Block {block_idx:>2}: TV cosine = {tv_cos:.4f}, high moved {h_mag:.2f}%, low moved {l_mag:.2f}%")

    print()
    print("  LEAST MOVED blocks (stayed closest to Wan 2.1):")
    block_by_movement = sorted(block_specialization, key=lambda x: x[2] + x[3])
    for block_idx, tv_cos, h_mag, l_mag in block_by_movement[:5]:
        print(f"    Block {block_idx:>2}: high moved {h_mag:.2f}%, low moved {l_mag:.2f}%, TV cosine = {tv_cos:.4f}")

    print()
    print("  MOST MOVED blocks (changed the most from Wan 2.1):")
    for block_idx, tv_cos, h_mag, l_mag in block_by_movement[-5:]:
        print(f"    Block {block_idx:>2}: high moved {h_mag:.2f}%, low moved {l_mag:.2f}%, TV cosine = {tv_cos:.4f}")

    # ── Training strategy implications ────────────────────────────────
    print()
    print("=" * 80)
    print("TRAINING STRATEGY IMPLICATIONS")
    print("=" * 80)
    print()
    print("  UNIFIED WARMUP should target blocks where TV cosine is HIGH (>0.5):")
    print("  Both experts moved together here — a single LoRA benefits both.")
    unified_blocks = [b for b, tc, _, _ in block_specialization if tc > 0.5]
    print(f"    Candidate blocks: {unified_blocks}")
    print()
    print("  DIVERGENT SPECIALIZATION should target blocks where TV cosine is LOW (<0.3):")
    print("  The experts went different directions — each needs its own LoRA adjustments.")
    divergent_blocks = [b for b, tc, _, _ in block_specialization if tc < 0.3]
    print(f"    Candidate blocks: {divergent_blocks}")
    print()
    print("  SKIP candidates — blocks that barely moved from Wan 2.1 at all:")
    skip_blocks = [b for b, _, hm, lm in block_by_movement if hm + lm < avg(all_high_mag) + avg(all_low_mag) * 0.5]
    print(f"    Candidate blocks: {skip_blocks[:10]}")

    # ── Save raw results to JSON for further analysis ─────────────────
    output_path = Path(__file__).parent / "task_vector_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print()
    print(f"  Raw results saved to: {output_path}")
    print()


if __name__ == "__main__":
    main()
