"""
Generate publication-quality charts from task vector analysis results.

Reads T2V and I2V task vector JSON files and produces 8 charts as PNG files
in docs/charts/. These visualize how Wan 2.2 MoE experts diverged from
the Wan 2.1 baseline -- the core finding that drives differential MoE training.

Usage:
    python scripts/generate_report_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
T2V_PATH = SCRIPT_DIR / "task_vector_results.json"
I2V_PATH = SCRIPT_DIR / "task_vector_results_i2v.json"
OUT_DIR = SCRIPT_DIR.parent / "docs" / "charts"
DPI = 200

plt.style.use("seaborn-v0_8-darkgrid")
COLOR_T2V_HIGH = "#FF6B6B"
COLOR_T2V_LOW = "#4ECDC4"
COLOR_I2V_HIGH = "#FFB347"
COLOR_I2V_LOW = "#7EC8E3"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12,
    "legend.fontsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 10,
    "figure.titlesize": 16,
})


def load_lora_targets(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [d for d in data if d.get("is_lora_target")]


def agg_block(data, field):
    buckets = {}
    for d in data:
        buckets.setdefault(d["block"], []).append(d[field])
    return {b: mean(v) for b, v in sorted(buckets.items())}


def agg_comp(data, field):
    buckets = {}
    for d in data:
        buckets.setdefault(d["component"], []).append(d[field])
    return {c: mean(v) for c, v in sorted(buckets.items())}


def chart_01(t2v):
    high = agg_block(t2v, "high_rel_magnitude")
    low = agg_block(t2v, "low_rel_magnitude")
    blocks = sorted(high.keys())
    x = np.arange(len(blocks))
    w = 0.38
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - w / 2, [high[b] for b in blocks], w,
           label="High-Noise Expert", color=COLOR_T2V_HIGH, edgecolor="none")
    ax.bar(x + w / 2, [low[b] for b in blocks], w,
           label="Low-Noise Expert", color=COLOR_T2V_LOW, edgecolor="none")
    ax.set_xlabel("Transformer Block")
    ax.set_ylabel("Relative Movement (%)")
    ax.set_title("T2V Expert Movement from Wan 2.1 Baseline (per block)")
    ax.set_xticks(x)
    ax.set_xticklabels(blocks, fontsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_t2v_block_movement.png", dpi=DPI)
    plt.close(fig)
    print("  01_t2v_block_movement.png")


def chart_02(i2v):
    high = agg_block(i2v, "high_rel_magnitude")
    low = agg_block(i2v, "low_rel_magnitude")
    blocks = sorted(high.keys())
    x = np.arange(len(blocks))
    w = 0.38
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - w / 2, [high[b] for b in blocks], w,
           label="High-Noise Expert", color=COLOR_I2V_HIGH, edgecolor="none")
    ax.bar(x + w / 2, [low[b] for b in blocks], w,
           label="Low-Noise Expert", color=COLOR_I2V_LOW, edgecolor="none")
    ax.set_xlabel("Transformer Block")
    ax.set_ylabel("Relative Movement (%)")
    ax.set_title("I2V Expert Movement from Wan 2.1 I2V Baseline (per block)")
    ax.set_xticks(x)
    ax.set_xticklabels(blocks, fontsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_i2v_block_movement.png", dpi=DPI)
    plt.close(fig)
    print("  02_i2v_block_movement.png")


def chart_03(t2v, i2v):
    t2v_h = agg_block(t2v, "high_rel_magnitude")
    i2v_h = agg_block(i2v, "high_rel_magnitude")
    blocks = sorted(t2v_h.keys())
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(blocks, [t2v_h[b] for b in blocks],
            marker="o", markersize=4, linewidth=2,
            color=COLOR_T2V_HIGH, label="T2V High-Noise")
    ax.plot(blocks, [i2v_h[b] for b in blocks],
            marker="s", markersize=4, linewidth=2,
            color=COLOR_I2V_HIGH, label="I2V High-Noise")
    ax.set_xlabel("Transformer Block")
    ax.set_ylabel("Relative Movement (%)")
    ax.set_title("High-Noise Expert: T2V vs I2V Movement Pattern")
    ax.legend()
    ax.set_xticks(blocks)
    ax.set_xticklabels(blocks, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_comparison_high_noise.png", dpi=DPI)
    plt.close(fig)
    print("  03_comparison_high_noise.png")


def chart_04(t2v, i2v):
    t2v_l = agg_block(t2v, "low_rel_magnitude")
    i2v_l = agg_block(i2v, "low_rel_magnitude")
    blocks = sorted(t2v_l.keys())
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(blocks, [t2v_l[b] for b in blocks],
            marker="o", markersize=4, linewidth=2,
            color=COLOR_T2V_LOW, label="T2V Low-Noise")
    ax.plot(blocks, [i2v_l[b] for b in blocks],
            marker="s", markersize=4, linewidth=2,
            color=COLOR_I2V_LOW, label="I2V Low-Noise")
    ax.set_yscale("log")
    ax.set_xlabel("Transformer Block")
    ax.set_ylabel("Relative Movement (%, log scale)")
    ax.set_title("Low-Noise Expert: T2V vs I2V Movement Pattern")
    ax.legend()
    ax.set_xticks(blocks)
    ax.set_xticklabels(blocks, fontsize=7)
    mid = 20
    tv = t2v_l[mid]
    iv = i2v_l[mid]
    ratio = iv / tv if tv > 0 else float("inf")
    gap_label = "{:.0f}x gap".format(ratio)
    ax.annotate(
        gap_label,
        xy=(mid, (tv * iv) ** 0.5),
        fontsize=11, fontweight="bold", color="white", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#555555", alpha=0.8),
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_comparison_low_noise.png", dpi=DPI)
    plt.close(fig)
    print("  04_comparison_low_noise.png")


def chart_05(t2v, i2v):
    t2v_c = agg_block(t2v, "tv_cosine")
    i2v_c = agg_block(i2v, "tv_cosine")
    blocks = sorted(t2v_c.keys())
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(blocks, [t2v_c[b] for b in blocks],
            marker="o", markersize=4, linewidth=2,
            color=COLOR_T2V_HIGH, label="T2V")
    ax.plot(blocks, [i2v_c[b] for b in blocks],
            marker="s", markersize=4, linewidth=2,
            color=COLOR_I2V_HIGH, label="I2V")
    ax.axhline(y=0, color="white", linewidth=1, linestyle="--", alpha=0.6)
    ax.annotate(
        "Orthogonal (independent directions)",
        xy=(0.5, 0), xycoords=("axes fraction", "data"),
        fontsize=9, color="white", alpha=0.7, ha="center",
        xytext=(0, -15), textcoords="offset points",
    )
    ax.set_xlabel("Transformer Block")
    ax.set_ylabel("Cosine Similarity (-1 to +1)")
    ax.set_title("Task Vector Direction Agreement (per block)")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="upper right")
    ax.set_xticks(blocks)
    ax.set_xticklabels(blocks, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_tv_cosine_by_block.png", dpi=DPI)
    plt.close(fig)
    print("  05_tv_cosine_by_block.png")


def chart_06(t2v, i2v):
    t2v_h = agg_comp(t2v, "high_rel_magnitude")
    t2v_l = agg_comp(t2v, "low_rel_magnitude")
    i2v_h = agg_comp(i2v, "high_rel_magnitude")
    i2v_l = agg_comp(i2v, "low_rel_magnitude")
    label_map = {
        "self_attn": "Self-Attention",
        "cross_attn": "Cross-Attention",
        "ffn": "FFN",
    }
    comps = ["self_attn", "cross_attn", "ffn"]
    labels = [label_map[c] for c in comps]
    x = np.arange(len(comps))
    w = 0.18
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5 * w, [t2v_h[c] for c in comps], w,
           label="T2V High-Noise", color=COLOR_T2V_HIGH)
    ax.bar(x - 0.5 * w, [t2v_l[c] for c in comps], w,
           label="T2V Low-Noise", color=COLOR_T2V_LOW)
    ax.bar(x + 0.5 * w, [i2v_h[c] for c in comps], w,
           label="I2V High-Noise", color=COLOR_I2V_HIGH)
    ax.bar(x + 1.5 * w, [i2v_l[c] for c in comps], w,
           label="I2V Low-Noise", color=COLOR_I2V_LOW)
    ax.set_xlabel("Component Type")
    ax.set_ylabel("Mean Relative Movement (%)")
    ax.set_title("Expert Movement by Component Type: T2V vs I2V")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_component_comparison.png", dpi=DPI)
    plt.close(fig)
    print("  06_component_comparison.png")


def chart_07(t2v, i2v):
    t2v_h = agg_block(t2v, "high_rel_magnitude")
    t2v_l = agg_block(t2v, "low_rel_magnitude")
    i2v_h = agg_block(i2v, "high_rel_magnitude")
    i2v_l = agg_block(i2v, "low_rel_magnitude")
    blocks = sorted(t2v_h.keys())
    t2v_r, i2v_r = [], []
    for b in blocks:
        t2v_r.append(
            t2v_h[b] / t2v_l[b] if t2v_l[b] > 1e-10 else t2v_h[b] / 1e-10
        )
        i2v_r.append(
            i2v_h[b] / i2v_l[b] if i2v_l[b] > 1e-10 else i2v_h[b] / 1e-10
        )
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(blocks, t2v_r, marker="o", markersize=5, linewidth=2,
            color=COLOR_T2V_HIGH, label="T2V (one-sided specialization)")
    ax.plot(blocks, i2v_r, marker="s", markersize=5, linewidth=2,
            color=COLOR_I2V_HIGH, label="I2V (both experts trained)")
    ax.set_yscale("log")
    ax.set_xlabel("Transformer Block")
    ax.set_ylabel("High / Low Movement Ratio (log scale)")
    ax.set_title(
        "Expert Asymmetry: How Much More Did High-Noise Move? (per block)"
    )
    ax.axhline(y=1, color="white", linewidth=1, linestyle="--", alpha=0.5)
    ax.annotate(
        "Equal movement", xy=(39, 1),
        fontsize=9, color="white", alpha=0.6,
        xytext=(-10, 10), textcoords="offset points", ha="right",
    )
    ax.legend()
    ax.set_xticks(blocks)
    ax.set_xticklabels(blocks, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "07_asymmetry_ratio.png", dpi=DPI)
    plt.close(fig)
    print("  07_asymmetry_ratio.png")


def chart_08(t2v, i2v):
    t2v_c = agg_block(t2v, "expert_cosine")
    i2v_c = agg_block(i2v, "expert_cosine")
    blocks = sorted(t2v_c.keys())
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(blocks, [t2v_c[b] for b in blocks],
            marker="o", markersize=4, linewidth=2,
            color=COLOR_T2V_HIGH, label="T2V Expert Cosine")
    ax.plot(blocks, [i2v_c[b] for b in blocks],
            marker="s", markersize=4, linewidth=2,
            color=COLOR_I2V_HIGH, label="I2V Expert Cosine")
    t2v_m = mean(t2v_c[b] for b in blocks)
    i2v_m = mean(i2v_c[b] for b in blocks)
    ax.axhline(y=t2v_m, color=COLOR_T2V_HIGH, linewidth=1,
               linestyle=":", alpha=0.5)
    ax.axhline(y=i2v_m, color=COLOR_I2V_HIGH, linewidth=1,
               linestyle=":", alpha=0.5)
    t2v_label = "T2V mean: {:.4f}".format(t2v_m)
    i2v_label = "I2V mean: {:.4f}".format(i2v_m)
    ax.annotate(t2v_label, xy=(0.02, 0.95),
                xycoords="axes fraction", fontsize=9, color=COLOR_T2V_HIGH)
    ax.annotate(i2v_label, xy=(0.02, 0.88),
                xycoords="axes fraction", fontsize=9, color=COLOR_I2V_HIGH)
    ax.set_xlabel("Transformer Block")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Expert-to-Expert Similarity (per block)")
    ax.legend(loc="lower right")
    ax.set_xticks(blocks)
    ax.set_xticklabels(blocks, fontsize=7)
    all_v = [t2v_c[b] for b in blocks] + [i2v_c[b] for b in blocks]
    ax.set_ylim(min(all_v) - 0.005, max(all_v) + 0.005)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "08_expert_cosine_by_block.png", dpi=DPI)
    plt.close(fig)
    print("  08_expert_cosine_by_block.png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t2v_msg = "Loading T2V data from {}".format(T2V_PATH)
    print(t2v_msg)
    t2v = load_lora_targets(T2V_PATH)
    print("  {} LoRA-target layers".format(len(t2v)))
    i2v_msg = "Loading I2V data from {}".format(I2V_PATH)
    print(i2v_msg)
    i2v = load_lora_targets(I2V_PATH)
    print("  {} LoRA-target layers".format(len(i2v)))
    print("\nGenerating charts in {}/".format(OUT_DIR))
    chart_01(t2v)
    chart_02(i2v)
    chart_03(t2v, i2v)
    chart_04(t2v, i2v)
    chart_05(t2v, i2v)
    chart_06(t2v, i2v)
    chart_07(t2v, i2v)
    chart_08(t2v, i2v)
    print("\nDone! 8 charts saved to {}/".format(OUT_DIR))


if __name__ == "__main__":
    main()
