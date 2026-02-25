import json, statistics
from pathlib import Path
from collections import defaultdict

SDIR = Path("C:/Users/minta/Projects/dimljus-kit/scripts")
with open(SDIR / "task_vector_results.json") as f:
    t2v_list = json.load(f)
with open(SDIR / "task_vector_results_i2v.json") as f:
    i2v_list = json.load(f)

def pct(v): return f"{v*100:.2f}%"
def fcos(v): return f"{v:.4f}"

def agg_block(data):
    blocks = defaultdict(list)
    for r in data:
        if r["is_lora_target"] and r["block"] >= 0:
            blocks[r["block"]].append(r)
    result = {}
    for bn, ly in sorted(blocks.items()):
        result[bn] = {
            "high": statistics.mean(r["high_rel_magnitude"] for r in ly),
            "low": statistics.mean(r["low_rel_magnitude"] for r in ly),
            "tv_cos": statistics.mean(r["tv_cosine"] for r in ly),
            "expert_cos": statistics.mean(r["expert_cosine"] for r in ly),
            "count": len(ly),
        }
    return result

def pearson(xs, ys):
    n = len(xs)
    if n < 2: return 0.0
    mx, my = sum(xs)/n, sum(ys)/n
    cov = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    sx = sum((x-mx)**2 for x in xs)**0.5
    sy = sum((y-my)**2 for y in ys)**0.5
    return cov/(sx*sy) if sx and sy else 0.0