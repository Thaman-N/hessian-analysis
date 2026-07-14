"""
gqa_ratio_spectrum.py

Tests whether QK suppression / VO amplification scales with Q:KV ratio.
All data from cached JSON files — no model loading.

Key design decisions vs v1:
  - Try mqafalcon (without 'c') first — that's what the paper used.
    mqacfalcon appears to be a different aggregation that doesn't reproduce paper numbers.
  - Also extract attn_out_drift from Falcon as ρ_O equivalent.
  - Report all gens for Falcon (not just gen4) to check consistency with paper.
  - Architecture confound is made explicit: SmolLM→Llama is ratio-only (both sequential GQA);
    Llama→Falcon adds parallel residual stream on top of ratio change, so monotonicity
    across all three is not expected — the table is two separate comparisons.

Run: python scripts/gqa_ratio_spectrum.py
Out: results/circuit_analysis/gqa_ratio_spectrum.json
     results/circuit_analysis/gqa_ratio_spectrum.png
"""

import json
import os
import sys
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

ROOT    = r"D:\Thaman\Work\hessian-spectral-analysis"
OUT_DIR = os.path.join(ROOT, "results", "circuit_analysis")
CACHE   = os.path.join(OUT_DIR, "gqa_ratio_spectrum.json")
PLOT    = os.path.join(OUT_DIR, "gqa_ratio_spectrum.png")

GATE_SPLIT_CACHE = os.path.join(OUT_DIR, "llama_mlp_gate_split.json")
FALCON_FIM_FILE  = os.path.join(ROOT, "results", "falcon-7b_treatment_gen_0", "perblock_fim.json")

# Try mqafalcon (paper version) first, then mqacfalcon as fallback, across all gens
FALCON_DRIFT_CANDIDATES = []
for gen in [4, 3, 2, 1]:
    FALCON_DRIFT_CANDIDATES.append(
        (f"mqafalcon-7b_drift_split_gen_{gen}",
         os.path.join(ROOT, "results", f"mqafalcon-7b_drift_split_gen_{gen}.json"), gen))
    FALCON_DRIFT_CANDIDATES.append(
        (f"mqacfalcon-7b_drift_split_gen_{gen}",
         os.path.join(ROOT, "results", f"mqacfalcon-7b_drift_split_gen_{gen}.json"), gen))

os.makedirs(OUT_DIR, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def spearman(x, y):
    pairs = [(a, b) for a, b in zip(x, y)
             if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)]
    if len(pairs) < 4:
        return None, None, len(pairs)
    xs, ys = zip(*pairs)
    rho, p = stats.spearmanr(xs, ys)
    return float(rho), float(p), len(pairs)


def sig(p):
    if p is None: return ""
    return "**" if p < 0.01 else ("*" if p < 0.05 else "")


def fmt(rho, p):
    if rho is None: return "  N/A  "
    return f"{rho:+.3f}{sig(p)}"


def load_fim_attn(fim_file):
    with open(fim_file) as f:
        d = json.load(f)
    return [b["attention"]["top"] for b in d["blocks"]]


# ── load SmolLM and Llama ─────────────────────────────────────────────────────

def load_smollm_llama():
    if not os.path.exists(GATE_SPLIT_CACHE):
        print(f"ERROR: {GATE_SPLIT_CACHE} not found — run llama_mlp_gate_split.py first.")
        sys.exit(1)

    with open(GATE_SPLIT_CACHE) as f:
        data = json.load(f)

    results = []
    for entry in data:
        name     = entry["model"]
        fim_attn = entry["fim_attn"]
        blocks   = entry["block_drifts"]
        log_fim  = [np.log10(v) if v > 0 else np.nan for v in fim_attn]

        q = [b["attn_q_proj"] for b in blocks]
        k = [b["attn_k_proj"] for b in blocks]
        v = [b["attn_v_proj"] for b in blocks]
        o = [b["attn_o_proj"] for b in blocks]

        rho_q, p_q, n = spearman(log_fim, q)
        rho_k, p_k, _ = spearman(log_fim, k)
        rho_v, p_v, _ = spearman(log_fim, v)
        rho_o, p_o, _ = spearman(log_fim, o)

        qk_mean = float(np.mean([r for r in [rho_q, rho_k] if r is not None]))
        vo_mean = float(np.mean([r for r in [rho_v, rho_o] if r is not None]))

        q_heads, kv_heads = (9, 3) if "SmolLM" in name else (32, 8)

        results.append({
            "model": name, "arch": "SEQ",
            "q_heads": q_heads, "kv_heads": kv_heads,
            "ratio": q_heads / kv_heads,
            "n_blocks": entry["n_layers"],
            "log_fim": log_fim,
            "q_drifts": q, "k_drifts": k, "v_drifts": v, "o_drifts": o,
            "correlations": {
                "rho_q": rho_q, "p_q": p_q,
                "rho_k": rho_k, "p_k": p_k,
                "rho_v": rho_v, "p_v": p_v,
                "rho_o": rho_o, "p_o": p_o,
                "rho_qk_mean": qk_mean, "rho_vo_mean": vo_mean, "n": n,
            }
        })

        print(f"  {name} ({q_heads}Q/{kv_heads}KV, {q_heads/kv_heads:.0f}:1, SEQ), n={entry['n_layers']}")
        print(f"    ρ_Q={fmt(rho_q,p_q)}  ρ_K={fmt(rho_k,p_k)}  "
              f"ρ_V={fmt(rho_v,p_v)}  ρ_O={fmt(rho_o,p_o)}")
        print(f"    QK_mean={qk_mean:+.3f}  VO_mean={vo_mean:+.3f}")

    return results


# ── load Falcon across all gens ───────────────────────────────────────────────

def load_falcon_all_gens():
    """
    Load Falcon MQA split for every available gen, prefer mqafalcon over mqacfalcon.
    Returns list of per-gen dicts so we can check consistency with paper table.
    """
    fim_attn = load_fim_attn(FALCON_FIM_FILE)
    log_fim  = [np.log10(v) if v > 0 else np.nan for v in fim_attn]
    n_blocks = len(fim_attn)

    gen_results = {}

    # Group candidates by gen, prefer mqafalcon over mqacfalcon
    from collections import defaultdict
    by_gen = defaultdict(dict)
    for label, path, gen in FALCON_DRIFT_CANDIDATES:
        variant = "paper" if "mqacfalcon" not in label else "corrected"
        if os.path.exists(path) and variant not in by_gen[gen]:
            by_gen[gen][variant] = (label, path)

    if not by_gen:
        print("  No Falcon drift split files found.")
        return None, None

    for gen in sorted(by_gen.keys(), reverse=True):
        # prefer mqafalcon (paper), fall back to mqacfalcon
        variant, (label, path) = next(iter(by_gen[gen].items()))
        # actually pick paper first
        if "paper" in by_gen[gen]:
            label, path = by_gen[gen]["paper"]
        elif "corrected" in by_gen[gen]:
            label, path = by_gen[gen]["corrected"]

        with open(path) as f:
            raw = json.load(f)

        # Extract per-block Q, KV, O drifts
        q_drifts, kv_drifts, o_drifts, block_indices = [], [], [], []

        blocks_src = raw.get("blocks", raw if isinstance(raw, dict) else {})
        if isinstance(blocks_src, dict):
            for idx_str in sorted(blocks_src.keys(), key=int):
                b = blocks_src[idx_str]
                if not isinstance(b, dict):
                    continue
                q  = b.get("attn_q_drift")  or b.get("q_relative_drift")  or b.get("q_drift")
                kv = b.get("attn_kv_drift") or b.get("kv_relative_drift") or b.get("kv_drift")
                out= b.get("attn_out_drift") or b.get("out_relative_drift") or b.get("o_drift")
                if q is not None:
                    q_drifts.append(float(q))
                    kv_drifts.append(float(kv) if kv is not None else None)
                    o_drifts.append(float(out) if out is not None else None)
                    block_indices.append(int(idx_str))

        if not q_drifts:
            print(f"  Gen{gen} ({label}): could not parse — keys: "
                  f"{list(list(blocks_src.values())[0].keys()) if blocks_src else 'unknown'}")
            continue

        # Align FIM to block_indices
        log_fim_aligned = [log_fim[i] if i < n_blocks and np.isfinite(log_fim[i])
                           else np.nan for i in block_indices]

        rho_q,  p_q,  n = spearman(log_fim_aligned, q_drifts)
        rho_kv, p_kv, _ = spearman(log_fim_aligned, kv_drifts)
        rho_o,  p_o,  _ = spearman(log_fim_aligned, o_drifts)

        gen_results[gen] = {
            "gen": gen, "label": label,
            "n_blocks": len(q_drifts),
            "q_drifts": q_drifts, "kv_drifts": kv_drifts, "o_drifts": o_drifts,
            "log_fim": log_fim_aligned,
            "correlations": {
                "rho_q":  rho_q,  "p_q":  p_q,
                "rho_kv": rho_kv, "p_kv": p_kv,
                "rho_o":  rho_o,  "p_o":  p_o,
                # unified table fields
                "rho_qk_mean": float(rho_q)  if rho_q  is not None else None,
                "rho_vo_mean": float(rho_kv) if rho_kv is not None else None,
                "n": n,
            }
        }

        print(f"  Falcon Gen{gen} ({label}), n={len(q_drifts)} blocks:")
        print(f"    ρ_Q={fmt(rho_q,p_q)}  ρ_KV={fmt(rho_kv,p_kv)}  ρ_O={fmt(rho_o,p_o)}")

    if not gen_results:
        return None, None

    # Use last gen for the spectrum table; return all gens for trajectory check
    last_gen = max(gen_results.keys())
    best     = gen_results[last_gen]

    falcon_entry = {
        "model": "Falcon-7B", "arch": "PAR (MQA)",
        "q_heads": 71, "kv_heads": 1, "ratio": 71.0,
        "n_blocks": best["n_blocks"],
        "log_fim": best["log_fim"],
        "q_drifts": best["q_drifts"],
        "kv_drifts": best["kv_drifts"],
        "o_drifts": best["o_drifts"],
        "gen_used": last_gen,
        "correlations": best["correlations"],
        "all_gens": gen_results,
    }
    return falcon_entry, gen_results


# ── analysis and printing ─────────────────────────────────────────────────────

def print_falcon_trajectory(gen_results):
    print(f"\n  Falcon correlation trajectory (check vs paper: Q neg*, KV pos**):")
    print(f"  {'Gen':>4}  {'file':>35}  {'ρ_Q':>10}  {'ρ_KV':>10}  {'ρ_O':>10}")
    print(f"  {'─'*75}")
    for gen in sorted(gen_results.keys()):
        c = gen_results[gen]["correlations"]
        label = gen_results[gen]["label"]
        print(f"  {gen:>4}  {label:>35}  "
              f"{fmt(c['rho_q'],c['p_q']):>10}  "
              f"{fmt(c['rho_kv'],c['p_kv']):>10}  "
              f"{fmt(c['rho_o'],c['p_o']):>10}")
    print(f"  Paper (from published table): "
          f"G1: Q=-0.37* KV=+0.60**  G2: Q=-0.39* KV=+0.14  "
          f"G3: Q=-0.41* KV=+0.46**  G4: Q=-0.36* KV=+0.57**")


def print_summary(all_models, falcon_gens):
    print("\n" + "="*90)
    print("GQA RATIO SPECTRUM — QK Suppression and VO/KV Amplification")
    print("="*90)
    print(f"  NOTE: SmolLM and Llama are SEQUENTIAL (SEQ); Falcon is PARALLEL (PAR).")
    print(f"  SEQ→SEQ comparison (3:1 vs 4:1) isolates GQA ratio effect.")
    print(f"  SEQ→PAR comparison (Llama vs Falcon) conflates ratio with architecture.")
    print()
    print(f"  {'Model':<16} {'Arch':<10} {'Ratio':>8}  {'n':>4}  "
          f"{'ρ_Q':>10} {'ρ_K/KV':>10} {'ρ_V':>8} {'ρ_O':>10}  "
          f"{'QK_mean':>9} {'VO_mean':>9}")
    print("  " + "─"*95)

    for m in all_models:
        c   = m["correlations"]
        rho_k_or_kv = c.get("rho_k") or c.get("rho_kv")
        p_k_or_kv   = c.get("p_k")   or c.get("p_kv")
        print(f"  {m['model']:<16} {m['arch']:<10} "
              f"{m['q_heads']}Q/{m['kv_heads']}KV ({m['ratio']:.0f}:1)  "
              f"{m['n_blocks']:>4}  "
              f"{fmt(c['rho_q'],c['p_q']):>10} "
              f"{fmt(rho_k_or_kv,p_k_or_kv):>10} "
              f"{fmt(c.get('rho_v'),c.get('p_v')):>8} "
              f"{fmt(c.get('rho_o'),c.get('p_o')):>10}  "
              f"{c['rho_qk_mean']:>+9.3f} {c['rho_vo_mean']:>+9.3f}"
              if c.get('rho_qk_mean') is not None else
              f"  (N/A)")

    print()
    print("  GQA RATIO EFFECT (sequential only, 3:1 → 4:1):")
    smollm = next(m for m in all_models if "SmolLM" in m["model"])
    llama  = next(m for m in all_models if "Llama"  in m["model"])
    print(f"    QK_mean: {smollm['correlations']['rho_qk_mean']:+.3f} → "
          f"{llama['correlations']['rho_qk_mean']:+.3f}  "
          f"({'more negative ✓' if llama['correlations']['rho_qk_mean'] < smollm['correlations']['rho_qk_mean'] else 'less negative ✗'})")
    print(f"    VO_mean: {smollm['correlations']['rho_vo_mean']:+.3f} → "
          f"{llama['correlations']['rho_vo_mean']:+.3f}  "
          f"({'more positive ✓' if llama['correlations']['rho_vo_mean'] > smollm['correlations']['rho_vo_mean'] else 'less positive ✗'})")

    if falcon_gens:
        print()
        print("  FALCON (PAR + MQA 71:1) — architecture confound prevents spectrum comparison:")
        print_falcon_trajectory(falcon_gens)

    print()
    print("  KEY INTERPRETATIONS:")
    print("  1. SEQ GQA effect (3:1→4:1): if QK more negative and VO more positive,")
    print("     GQA ratio modulates dissociation even within sequential architectures.")
    print("  2. Falcon near-zero combined ρ is not a ratio saturation effect —")
    print("     it is the parallel architecture cancelling both QK suppression and KV amplification.")
    print("  3. The spectrum claim requires a parallel GQA model to complete the design.")
    print("     Pythia uses standard MHA (no GQA). A parallel GQA model (e.g. GPT-J-6B)")
    print("     would let you test ratio effect within parallel architecture.")


# ── plotting ──────────────────────────────────────────────────────────────────

def plot(all_models, falcon_gens):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ratios  = [m["ratio"]                        for m in all_models]
    qk_vals = [m["correlations"]["rho_qk_mean"]  for m in all_models]
    vo_vals = [m["correlations"]["rho_vo_mean"]  for m in all_models]
    names   = [m["model"]                        for m in all_models]
    archs   = [m["arch"]                         for m in all_models]
    colors  = ["#2196F3" if "SEQ" in a else "#F44336" for a in archs]
    markers = ["o" if "SEQ" in a else "^" for a in archs]

    # ── Left: QK_mean and VO_mean vs ratio ───────────────────────────────────
    ax = axes[0]
    ax.axhline(0, color="k", lw=0.8, ls="--")

    # Draw connecting line for SEQ models only (valid spectrum comparison)
    seq_models = [(m["ratio"], m["correlations"]["rho_qk_mean"],
                   m["correlations"]["rho_vo_mean"])
                  for m in all_models if "SEQ" in m["arch"]]
    if len(seq_models) >= 2:
        seq_models.sort()
        rs, qks, vos = zip(*seq_models)
        ax.plot(rs, qks, color="#2196F3", lw=1.5, ls="-",  alpha=0.6, label="SEQ QK trend")
        ax.plot(rs, vos, color="#2196F3", lw=1.5, ls="--", alpha=0.6, label="SEQ VO trend")

    for r, qk, vo, name, col, mk in zip(ratios, qk_vals, vo_vals, names, colors, markers):
        label_name = name.split("-")[0] + ("\n(SEQ)" if "SEQ" in archs[names.index(name)]
                                            else "\n(PAR)")
        ax.scatter(r, qk, color=col, marker=mk, s=130, zorder=5)
        ax.scatter(r, vo, color=col, marker=mk, s=130, zorder=5, facecolors="none",
                   linewidths=2)
        ax.annotate(f"{name.split('-')[0]}\nQK", (r, qk),
                    textcoords="offset points", xytext=(6, -14), fontsize=7.5, color=col)
        ax.annotate(f"{name.split('-')[0]}\nVO", (r, vo),
                    textcoords="offset points", xytext=(6, 4),  fontsize=7.5, color=col)

    ax.set_xscale("log")
    ax.set_xlabel("Q:KV ratio (log scale)", fontsize=10)
    ax.set_ylabel("Spearman ρ(log10 FIM_attn, drift)", fontsize=9)
    ax.set_title("QK (filled) vs VO (open) mean ρ\nBlue=SEQ, Red=PAR")
    ax.set_xticks([3, 4, 71])
    ax.set_xticklabels(["3:1\nSmolLM\n(SEQ)", "4:1\nLlama\n(SEQ)", "71:1\nFalcon\n(PAR)"])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ── Middle: Falcon gen trajectory vs paper ────────────────────────────────
    ax = axes[1]
    if falcon_gens:
        gens    = sorted(falcon_gens.keys())
        rho_q_  = [falcon_gens[g]["correlations"]["rho_q"]  for g in gens]
        rho_kv_ = [falcon_gens[g]["correlations"]["rho_kv"] for g in gens]
        rho_o_  = [falcon_gens[g]["correlations"]["rho_o"]  for g in gens]

        ax.plot(gens, [r if r else np.nan for r in rho_q_],  "o-", color="#e74c3c",
                label="ρ_Q (our data)")
        ax.plot(gens, [r if r else np.nan for r in rho_kv_], "s-", color="#3498db",
                label="ρ_KV (our data)")
        ax.plot(gens, [r if r else np.nan for r in rho_o_],  "^-", color="#2ecc71",
                label="ρ_O (our data)")

        # Paper reference values
        paper_q  = {1: -0.37, 2: -0.39, 3: -0.41, 4: -0.36}
        paper_kv = {1: +0.60, 2: +0.14, 3: +0.46, 4: +0.57}
        pg = sorted(set(gens) & set(paper_q.keys()))
        ax.plot(pg, [paper_q[g]  for g in pg], "o--", color="#e74c3c", alpha=0.5,
                label="ρ_Q (paper)", lw=1)
        ax.plot(pg, [paper_kv[g] for g in pg], "s--", color="#3498db", alpha=0.5,
                label="ρ_KV (paper)", lw=1)

        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("Generation"); ax.set_ylabel("Spearman ρ")
        ax.set_title("Falcon MQA split trajectory\n(solid=our data, dashed=paper)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        ax.set_xticks(gens)
    else:
        ax.text(0.5, 0.5, "No Falcon data", transform=ax.transAxes, ha="center")

    # ── Right: per-component ρ by model ──────────────────────────────────────
    ax = axes[2]
    comp_labels = ["Q", "K/KV", "V", "O"]
    x = np.arange(len(comp_labels))
    width = 0.25
    offsets = np.linspace(-width, width, len(all_models))

    for i, (m, col) in enumerate(zip(all_models, colors)):
        c = m["correlations"]
        vals = [
            c.get("rho_q"),
            c.get("rho_k") or c.get("rho_kv"),
            c.get("rho_v"),
            c.get("rho_o"),
        ]
        vals = [v if v is not None else 0.0 for v in vals]
        alpha = [1.0 if c.get(f"p_{['q','k','v','o'][j]}") is not None and
                 c.get(f"p_{['q','k','v','o'][j]}", 1) < 0.05 else 0.4
                 for j in range(4)]
        bars = ax.bar(x + offsets[i], vals, width * 0.9,
                      color=col, alpha=0.85,
                      label=f"{m['model'].split('-')[0]} ({m['arch'][:3]})")
        # dim non-significant bars
        for bar, a in zip(bars, alpha):
            bar.set_alpha(a)

    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(comp_labels)
    ax.set_ylabel("Spearman ρ(log10 FIM_attn, drift)")
    ax.set_title("Per-component ρ by model\n(dim = not significant)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "GQA Ratio Spectrum: QK Suppression and VO Amplification\n"
        "SEQ→SEQ (blue, solid line) is a valid ratio comparison. "
        "SEQ→PAR (Falcon, red) conflates architecture + ratio.",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOT, dpi=150, bbox_inches="tight")
    print(f"\nPlot: {PLOT}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    if os.path.exists(CACHE):
        print(f"Cache hit: {CACHE}  — delete to rerun.")
        with open(CACHE) as f:
            saved = json.load(f)
        all_models  = saved["models"]
        falcon_gens = saved.get("falcon_all_gens")
        plot(all_models, falcon_gens)
        print_summary(all_models, falcon_gens)
        return

    print("Loading SmolLM and Llama from gate-split cache...")
    all_models = load_smollm_llama()

    print("\nLoading Falcon (all gens, prefer mqafalcon over mqacfalcon)...")
    falcon_entry, falcon_gens = load_falcon_all_gens()
    if falcon_entry is not None:
        all_models.append(falcon_entry)

    all_models.sort(key=lambda m: m["ratio"])

    with open(CACHE, "w") as f:
        json.dump({"models": all_models, "falcon_all_gens": falcon_gens}, f, indent=2)
    print(f"\nCached: {CACHE}")

    plot(all_models, falcon_gens)
    print_summary(all_models, falcon_gens)


if __name__ == "__main__":
    main()