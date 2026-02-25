"""Compare task vector analysis: T2V vs I2V expert specialization."""

import json, statistics
from pathlib import Path
from collections import defaultdict

SDIR = Path(__file__).parent
with open(SDIR / "task_vector_results.json") as f:
    t2v_list = json.load(f)
with open(SDIR / "task_vector_results_i2v.json") as f:
    i2v_list = json.load(f)

def pct(v):
    return "{:.2f}%".format(v * 100)

def fcos(v):
    return "{:.4f}".format(v)

def section(t):
    print()
    print("=" * 80)
    print("  " + t)
    print("=" * 80)
    print()

def subsection(t):
    print()
    print("--- " + t + " ---")
    print()

def agg_block(data):
    blocks = defaultdict(list)
    for r in data:
        if r["is_lora_target"] and r["block"] >= 0:
            blocks[r["block"]].append(r)
    out = {}
    for bn, ly in sorted(blocks.items()):
        out[bn] = dict(
            high=statistics.mean(r["high_rel_magnitude"] for r in ly),
            low=statistics.mean(r["low_rel_magnitude"] for r in ly),
            tv_cos=statistics.mean(r["tv_cosine"] for r in ly),
            expert_cos=statistics.mean(r["expert_cosine"] for r in ly),
            count=len(ly))
    return out

def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (sx * sy) if sx and sy else 0.0

def show_block_row(b, t, iv):
    dh = (iv["high"] - t["high"]) * 100
    dl = (iv["low"] - t["low"]) * 100
    dtv = iv["tv_cos"] - t["tv_cos"]
    th, tl, ttv = t["high"] * 100, t["low"] * 100, t["tv_cos"]
    ih, il, itv = iv["high"] * 100, iv["low"] * 100, iv["tv_cos"]
    print("{:>3} | {:>7.2f}% {:>7.2f}% {:>8.4f} | {:>7.2f}% {:>7.2f}% {:>8.4f} | {:>+6.2f}% {:>+6.2f}% {:>+7.4f}".format(
        b, th, tl, ttv, ih, il, itv, dh, dl, dtv))

def main():
    print("TASK VECTOR COMPARISON: T2V vs I2V Expert Specialization")
    print("T2V: {} layers | I2V: {} layers".format(len(t2v_list), len(i2v_list)))

    section("1. OVERALL SUMMARY")
    for label, data in [("T2V (Wan 2.1->2.2 T2V)", t2v_list), ("I2V (Wan 2.1->2.2 I2V)", i2v_list)]:
        lora = [r for r in data if r["is_lora_target"]]
        print("  " + label)
        print("    Total: {} | LoRA targets: {}".format(len(data), len(lora)))
        for sl, sub in [("ALL", data), ("LORA", lora)]:
            atv = statistics.mean(r["tv_cosine"] for r in sub)
            ah = statistics.mean(r["high_rel_magnitude"] for r in sub)
            al = statistics.mean(r["low_rel_magnitude"] for r in sub)
            ae = statistics.mean(r["expert_cosine"] for r in sub)
            ratio = ah / al if al > 0 else float("inf")
            print("    {:>4}: tv_cos={} exp_cos={} high={} low={} H/L={:.1f}x".format(
                sl, fcos(atv), fcos(ae), pct(ah), pct(al), ratio))
        print()

    section("2. PER-BLOCK COMPARISON (LoRA targets only)")
    t2v_b = agg_block(t2v_list)
    i2v_b = agg_block(i2v_list)
    allb = sorted(set(t2v_b) | set(i2v_b))
    hdr = "Blk |  T2V Hi%  T2V Lo%   T2V TV |  I2V Hi%  I2V Lo%   I2V TV |   dHi%   dLo%     dTV"
    print(hdr)
    print("-" * len(hdr))
    for b in allb:
        t, iv = t2v_b.get(b), i2v_b.get(b)
        if t and iv:
            show_block_row(b, t, iv)
    print("-" * len(hdr))
    thi = statistics.mean(t2v_b[b]["high"] for b in allb) * 100
    tlo = statistics.mean(t2v_b[b]["low"] for b in allb) * 100
    ttv = statistics.mean(t2v_b[b]["tv_cos"] for b in allb)
    ihi = statistics.mean(i2v_b[b]["high"] for b in allb) * 100
    ilo = statistics.mean(i2v_b[b]["low"] for b in allb) * 100
    itv = statistics.mean(i2v_b[b]["tv_cos"] for b in allb)
    print("AVG | {:>7.2f}% {:>7.2f}% {:>8.4f} | {:>7.2f}% {:>7.2f}% {:>8.4f} | {:>+6.2f}% {:>+6.2f}% {:>+7.4f}".format(
        thi, tlo, ttv, ihi, ilo, itv, ihi - thi, ilo - tlo, itv - ttv))

    section("3. PER-COMPONENT COMPARISON (LoRA targets only)")
    tgt = ["self_attn", "cross_attn", "ffn"]
    def agg_comp(data):
        comps = defaultdict(list)
        for r in data:
            if r["is_lora_target"] and r["component"] in tgt:
                comps[r["component"]].append(r)
        out = {}
        for c, ly in comps.items():
            out[c] = dict(
                high=statistics.mean(r["high_rel_magnitude"] for r in ly),
                low=statistics.mean(r["low_rel_magnitude"] for r in ly),
                tv_cos=statistics.mean(r["tv_cosine"] for r in ly),
                expert_cos=statistics.mean(r["expert_cosine"] for r in ly),
                count=len(ly))
        return out
    tc, ic = agg_comp(t2v_list), agg_comp(i2v_list)
    ch = "{:>12} | {:>8} {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} {:>8} | {:>7} {:>7}".format(
        "Component", "T2V Hi%", "T2V Lo%", "T2V TV", "T2V Ex", "I2V Hi%", "I2V Lo%", "I2V TV", "I2V Ex", "dHi%", "dLo%")
    print(ch)
    print("-" * len(ch))
    for comp in tgt:
        t, iv = tc.get(comp), ic.get(comp)
        if t and iv:
            dh = (iv["high"] - t["high"]) * 100
            dl = (iv["low"] - t["low"]) * 100
            print("{:>12} | {:>7.2f}% {:>7.2f}% {:>8.4f} {:>8.4f} | {:>7.2f}% {:>7.2f}% {:>8.4f} {:>8.4f} | {:>+6.2f}% {:>+6.2f}%".format(
                comp, t["high"]*100, t["low"]*100, t["tv_cos"], t["expert_cos"],
                iv["high"]*100, iv["low"]*100, iv["tv_cos"], iv["expert_cos"], dh, dl))

    subsection("Per-sublayer breakdown")
    def agg_sl(data):
        sls = defaultdict(list)
        for r in data:
            if r["is_lora_target"]:
                sls[r["sublayer"]].append(r)
        out = {}
        for sl, ly in sls.items():
            out[sl] = dict(
                high=statistics.mean(r["high_rel_magnitude"] for r in ly),
                low=statistics.mean(r["low_rel_magnitude"] for r in ly),
                tv_cos=statistics.mean(r["tv_cosine"] for r in ly),
                count=len(ly))
        return out
    tsl, isl = agg_sl(t2v_list), agg_sl(i2v_list)
    slh = "{:>16} {:>4} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} | {:>7} {:>7}".format(
        "Sublayer", "N", "T2V Hi%", "T2V Lo%", "T2V TV", "I2V Hi%", "I2V Lo%", "I2V TV", "dHi%", "dLo%")
    print(slh)
    print("-" * len(slh))
    for sl in sorted(set(tsl) | set(isl)):
        t, iv = tsl.get(sl), isl.get(sl)
        if t and iv:
            dh = (iv["high"] - t["high"]) * 100
            dl = (iv["low"] - t["low"]) * 100
            print("{:>16} {:>4} | {:>7.2f}% {:>7.2f}% {:>8.4f} | {:>7.2f}% {:>7.2f}% {:>8.4f} | {:>+6.2f}% {:>+6.2f}%".format(
                sl, t["count"], t["high"]*100, t["low"]*100, t["tv_cos"],
                iv["high"]*100, iv["low"]*100, iv["tv_cos"], dh, dl))

    return t2v_b, i2v_b

def key_questions(t2v_b, i2v_b):
    section("4. KEY QUESTIONS")
    t2v_lora = [r for r in t2v_list if r["is_lora_target"]]
    i2v_lora = [r for r in i2v_list if r["is_lora_target"]]

    subsection("Q1: Did the I2V experts specialize MORE or LESS than T2V?")
    t2v_ah = statistics.mean(r["high_rel_magnitude"] for r in t2v_lora)
    t2v_al = statistics.mean(r["low_rel_magnitude"] for r in t2v_lora)
    i2v_ah = statistics.mean(r["high_rel_magnitude"] for r in i2v_lora)
    i2v_al = statistics.mean(r["low_rel_magnitude"] for r in i2v_lora)
    t2v_aec = statistics.mean(r["expert_cosine"] for r in t2v_lora)
    i2v_aec = statistics.mean(r["expert_cosine"] for r in i2v_lora)
    print("  High-noise movement:  T2V={}, I2V={}".format(pct(t2v_ah), pct(i2v_ah)))
    print("  Low-noise movement:   T2V={}, I2V={}".format(pct(t2v_al), pct(i2v_al)))
    print("  Expert cosine:        T2V={}, I2V={}".format(fcos(t2v_aec), fcos(i2v_aec)))
    print()
    tot_t, tot_i = t2v_ah + t2v_al, i2v_ah + i2v_al
    if tot_i > tot_t:
        print("  ANSWER: I2V specialized MORE ({:.2f}x). I2V {} vs T2V {}".format(tot_i/tot_t, pct(tot_i), pct(tot_t)))
    else:
        print("  ANSWER: I2V specialized LESS ({:.2f}x). T2V {} vs I2V {}".format(tot_t/tot_i, pct(tot_t), pct(tot_i)))
    t2v_rat = t2v_ah / t2v_al if t2v_al > 0 else float("inf")
    i2v_rat = i2v_ah / i2v_al if i2v_al > 0 else float("inf")
    print("  High/Low asymmetry: T2V={:.1f}x, I2V={:.1f}x".format(t2v_rat, i2v_rat))

    subsection("Q2: Are the same blocks most/least specialized?")
    t2v_ranked = sorted(t2v_b.items(), key=lambda x: x[1]["high"], reverse=True)
    i2v_ranked = sorted(i2v_b.items(), key=lambda x: x[1]["high"], reverse=True)
    print("  Top 5 most specialized (high-noise):")
    print("    T2V: {}".format([b for b, _ in t2v_ranked[:5]]))
    print("    I2V: {}".format([b for b, _ in i2v_ranked[:5]]))
    print("  Bottom 5 least specialized:")
    print("    T2V: {}".format([b for b, _ in t2v_ranked[-5:]]))
    print("    I2V: {}".format([b for b, _ in i2v_ranked[-5:]]))
    t2v_rk = {b: rk for rk, (b, _) in enumerate(t2v_ranked)}
    i2v_rk = {b: rk for rk, (b, _) in enumerate(i2v_ranked)}
    shared = sorted(set(t2v_rk) & set(i2v_rk))
    n = len(shared)
    dsq = sum((t2v_rk[b] - i2v_rk[b]) ** 2 for b in shared)
    rho = 1 - (6 * dsq) / (n * (n ** 2 - 1))
    print("  Spearman rho: {:.4f}".format(rho))
    if rho > 0.8:
        print("  ANSWER: YES - very similar block specialization patterns.")
    elif rho > 0.5:
        print("  ANSWER: PARTIALLY - moderately similar patterns.")
    else:
        print("  ANSWER: NO - different block specialization patterns.")

    subsection("Q3: Is the low-noise-barely-moved pattern also true for I2V?")
    print("  T2V low-noise avg: {}".format(pct(t2v_al)))
    print("  I2V low-noise avg: {}".format(pct(i2v_al)))
    t2v_u1 = sum(1 for r in t2v_lora if r["low_rel_magnitude"] < 0.01)
    i2v_u1 = sum(1 for r in i2v_lora if r["low_rel_magnitude"] < 0.01)
    t2v_u5 = sum(1 for r in t2v_lora if r["low_rel_magnitude"] < 0.05)
    i2v_u5 = sum(1 for r in i2v_lora if r["low_rel_magnitude"] < 0.05)
    print("  Low<1%: T2V={}/{}, I2V={}/{}".format(t2v_u1, len(t2v_lora), i2v_u1, len(i2v_lora)))
    print("  Low<5%: T2V={}/{}, I2V={}/{}".format(t2v_u5, len(t2v_lora), i2v_u5, len(i2v_lora)))
    if i2v_al < 0.01:
        print("  ANSWER: YES - I2V low-noise also barely moved (< 1%).")
    elif i2v_al < 0.05:
        print("  ANSWER: PARTIALLY - I2V low-noise moved more but still small (< 5%).")
    else:
        print("  ANSWER: NO - I2V low-noise moved significantly ({}). BOTH experts changed.".format(pct(i2v_al)))

    subsection("Q4: Blocks with BIGGEST T2V-vs-I2V difference?")
    bdiffs = []
    for b in shared:
        t, iv = t2v_b[b], i2v_b[b]
        dh = iv["high"] - t["high"]
        dl = iv["low"] - t["low"]
        dtv = iv["tv_cos"] - t["tv_cos"]
        bdiffs.append((b, abs(dh) + abs(dl) + abs(dtv), dh, dl, dtv))
    bdiffs.sort(key=lambda x: x[1], reverse=True)
    print("  {:>5}  {:>10}  {:>10}  {:>10}  {:>10}".format("Block", "TotalDiff", "dHigh", "dLow", "dTVcos"))
    print("  Top 10 most different:")
    for b, tot, dh, dl, dtv in bdiffs[:10]:
        print("  {:>5}  {:>10.4f}  {:>+10.4f}  {:>+10.4f}  {:>+10.4f}".format(b, tot, dh, dl, dtv))
    print("  Bottom 5 most similar:")
    for b, tot, dh, dl, dtv in bdiffs[-5:]:
        print("  {:>5}  {:>10.4f}  {:>+10.4f}  {:>+10.4f}  {:>+10.4f}".format(b, tot, dh, dl, dtv))

    subsection("Q5: Any I2V-only layers?")
    t2v_set = {r["layer"] for r in t2v_list}
    i2v_set = {r["layer"] for r in i2v_list}
    i2v_only = i2v_set - t2v_set
    if i2v_only:
        print("  {} I2V-only layers:".format(len(i2v_only)))
    else:
        print("  No I2V-only layers. Both models have identical layer names.")
        print("  (I2V extra input channels are in input projection, not transformer blocks)")

    return t2v_lora, i2v_lora, t2v_ah, t2v_al, i2v_ah, i2v_al

def cross_model(t2v_b, i2v_b, t2v_lora, i2v_lora, t2v_ah, t2v_al, i2v_ah, i2v_al):
    section("5. CROSS-MODEL ANALYSIS")
    t2v_by = {r["layer"]: r for r in t2v_lora}
    i2v_by = {r["layer"]: r for r in i2v_lora}
    shared_l = sorted(set(t2v_by) & set(i2v_by))

    subsection("Layer-level Pearson correlations (LoRA targets)")
    rh = pearson([t2v_by[l]["high_rel_magnitude"] for l in shared_l], [i2v_by[l]["high_rel_magnitude"] for l in shared_l])
    rl = pearson([t2v_by[l]["low_rel_magnitude"] for l in shared_l], [i2v_by[l]["low_rel_magnitude"] for l in shared_l])
    rt = pearson([t2v_by[l]["tv_cosine"] for l in shared_l], [i2v_by[l]["tv_cosine"] for l in shared_l])
    re = pearson([t2v_by[l]["expert_cosine"] for l in shared_l], [i2v_by[l]["expert_cosine"] for l in shared_l])
    print("  High-noise rel magnitude: r = {:.4f}".format(rh))
    print("  Low-noise rel magnitude:  r = {:.4f}".format(rl))
    print("  Task vector cosine:       r = {:.4f}".format(rt))
    print("  Expert cosine:            r = {:.4f}".format(re))
    if rh > 0.8:
        print("  -> HIGHLY correlated: same blocks moved most in both.")
    elif rh > 0.5:
        print("  -> MODERATELY correlated.")
    else:
        print("  -> WEAKLY correlated: different specialization patterns.")

    subsection("Training strategy: quartile overlap")
    def quartile(bd, m="high"):
        items = sorted(bd.items(), key=lambda x: x[1][m])
        nn = len(items)
        return set(b for b, _ in items[:nn // 4]), set(b for b, _ in items[3 * nn // 4:])
    t2v_loq, t2v_hiq = quartile(t2v_b)
    i2v_loq, i2v_hiq = quartile(i2v_b)
    print("  High-movement blocks (top quartile):")
    print("    T2V: {}".format(sorted(t2v_hiq)))
    print("    I2V: {}".format(sorted(i2v_hiq)))
    olap_hi = sorted(t2v_hiq & i2v_hiq)
    print("    Overlap: {} ({}/{})".format(olap_hi, len(olap_hi), len(t2v_hiq)))
    print("  Low-movement blocks (bottom quartile):")
    print("    T2V: {}".format(sorted(t2v_loq)))
    print("    I2V: {}".format(sorted(i2v_loq)))
    olap_lo = sorted(t2v_loq & i2v_loq)
    print("    Overlap: {} ({}/{})".format(olap_lo, len(olap_lo), len(t2v_loq)))

    subsection("Expert divergence (lowest expert_cosine = most diverged)")
    t2v_ec = sorted(t2v_b.items(), key=lambda x: x[1]["expert_cos"])
    i2v_ec = sorted(i2v_b.items(), key=lambda x: x[1]["expert_cos"])
    print("  Most diverged: T2V={}".format([(b, round(v["expert_cos"], 4)) for b, v in t2v_ec[:5]]))
    print("                 I2V={}".format([(b, round(v["expert_cos"], 4)) for b, v in i2v_ec[:5]]))
    print("  Least diverged: T2V={}".format([(b, round(v["expert_cos"], 4)) for b, v in t2v_ec[-5:]]))
    print("                  I2V={}".format([(b, round(v["expert_cos"], 4)) for b, v in i2v_ec[-5:]]))

    subsection("Movement gradient: early vs late blocks")
    for label, blocks in [("T2V", t2v_b), ("I2V", i2v_b)]:
        eh = [v["high"] for b, v in blocks.items() if 0 <= b <= 9]
        mh = [v["high"] for b, v in blocks.items() if 10 <= b <= 29]
        lh = [v["high"] for b, v in blocks.items() if 30 <= b <= 39]
        el = [v["low"] for b, v in blocks.items() if 0 <= b <= 9]
        ml = [v["low"] for b, v in blocks.items() if 10 <= b <= 29]
        ll = [v["low"] for b, v in blocks.items() if 30 <= b <= 39]
        print("  {}: Early(0-9) hi={} lo={} | Mid(10-29) hi={} lo={} | Late(30-39) hi={} lo={}".format(
            label, pct(statistics.mean(eh)), pct(statistics.mean(el)),
            pct(statistics.mean(mh)), pct(statistics.mean(ml)),
            pct(statistics.mean(lh)), pct(statistics.mean(ll))))

    subsection("SUMMARY: Unified vs Divergent LoRA Strategy")
    print("  T2V: high={} low={} -> ONE-SIDED (high retrained, low = base)".format(pct(t2v_ah), pct(t2v_al)))
    print("  I2V: high={} low={}".format(pct(i2v_ah), pct(i2v_al)), end="")
    if i2v_al < 0.01:
        print(" -> ALSO ONE-SIDED")
        print("  IMPLICATION: Same differential strategy works for both.")
    elif i2v_al < 0.05:
        print(" -> Low moved MORE but still modest")
        print("  IMPLICATION: I2V may need slightly more aggressive low-noise training.")
    else:
        print(" -> TWO-SIDED (both experts changed)")
        print("  IMPLICATION: I2V strategy should DIFFER from T2V!")
        print("  Both experts need meaningful training. Image conditioning forces low-noise adaptation.")

    t2v_tva = statistics.mean(r["tv_cosine"] for r in t2v_lora)
    i2v_tva = statistics.mean(r["tv_cosine"] for r in i2v_lora)
    print()
    print("  Task vector cosine (direction agreement): T2V={}, I2V={}".format(fcos(t2v_tva), fcos(i2v_tva)))
    if abs(t2v_tva) < 0.1 and abs(i2v_tva) < 0.1:
        print("  Both near zero -> ORTHOGONAL. Independent per-expert LoRA justified for both.")
    elif i2v_tva > 0.3:
        print("  I2V experts moved SIMILARLY -> Unified LoRA may work for I2V.")
    elif i2v_tva < -0.3:
        print("  I2V experts moved OPPOSITELY -> Separate per-expert LoRA needed.")
    else:
        if abs(i2v_tva) > abs(t2v_tva):
            print("  I2V experts more aligned than T2V -- some shared signal between experts.")
        else:
            print("  I2V experts also moved in largely independent directions.")


if __name__ == "__main__":
    t2v_b, i2v_b = main()
    t2v_lora, i2v_lora, t2v_ah, t2v_al, i2v_ah, i2v_al = key_questions(t2v_b, i2v_b)
    cross_model(t2v_b, i2v_b, t2v_lora, i2v_lora, t2v_ah, t2v_al, i2v_ah, i2v_al)
