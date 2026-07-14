"""
phi_circuit_completion.py

Closes README Part 5 open question: "Run OV drift and attn/MLP split for Phi-1.5."

Phi-1.5 (parallel): attn_vs_mlp_split gave rho_attn=-0.172 (ns), rho_mlp=-0.228 (ns),
rho_combined=+0.208 (ns). Weaker than Pythia's rho~+0.8.

This script runs component-level correlations across all 5 generations to answer:
  (a) Do component-level signals emerge from Gen3 onward? (matches FIM paper finding)
  (b) Are signs consistently positive despite not reaching significance?
  (c) Does OV drift (per head W_V @ W_O) show the FIM_attn positive correlation?

Phi-1.5 architecture (PhiDecoderLayer):
  attention: self_attn — q_proj, k_proj, v_proj, dense (= o_proj)
  mlp:       mlp       — fc1 (up), fc2 (down)
  Residual stream: PARALLEL (attn output + mlp output added to same x)

Drift files (results root takes priority):
  results/phi-1_5_drift_gen_1.json through gen_5
  Format: {"blocks": {"0": {"attn_relative_drift": ..., "mlp_relative_drift": ...}, ...}}

FIM file: results/phi-1_5_treatment_gen_0/perblock_fim.json

Run: python scripts/phi_circuit_completion.py
Out: results/circuit_analysis/phi_circuit_completion.json
     results/circuit_analysis/phi_circuit_completion.png
"""

import json
import os
import numpy as np
import torch
from scipy import stats
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT    = r"D:\Thaman\Work\hessian-spectral-analysis"
OUT_DIR = os.path.join(ROOT, "results", "circuit_analysis")
CACHE   = os.path.join(OUT_DIR, "phi_circuit_completion.json")
PLOT    = os.path.join(OUT_DIR, "phi_circuit_completion.png")

PHI_GEN0 = "microsoft/phi-1_5"
PHI_GENS = {
    n: os.path.join(ROOT, "models", f"phi-1_5_treatment_gen_{n}")
    for n in range(1, 6)
}

FIM_FILE = os.path.join(ROOT, "results", "phi-1_5_treatment_gen_0", "perblock_fim.json")

# Drift files: check results root first, then results/parameterdrift/
def phi_drift_path(gen_n):
    primary   = os.path.join(ROOT, "results", f"phi-1_5_drift_gen_{gen_n}.json")
    secondary = os.path.join(ROOT, "results", "parameterdrift", f"phi-1_5_drift_gen_{gen_n}.json")
    if os.path.exists(primary):
        return primary
    if os.path.exists(secondary):
        return secondary
    return None

os.makedirs(OUT_DIR, exist_ok=True)


# ── loaders ───────────────────────────────────────────────────────────────────

def load_fim():
    with open(FIM_FILE) as f:
        d = json.load(f)
    fim_attn = [b["attention"]["top"] for b in d["blocks"]]
    fim_mlp  = [b["mlp"]["top"]       for b in d["blocks"]]
    return fim_attn, fim_mlp


def load_drift_json(path):
    """Returns {block_idx: {attn_relative_drift, mlp_relative_drift}}"""
    with open(path) as f:
        d = json.load(f)
    return {int(k): v for k, v in d["blocks"].items()}


def rel_drift(w0, wn):
    return (wn.float() - w0.float()).norm().item() / (w0.float().norm().item() + 1e-8)


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


# ── component drift from model weights ───────────────────────────────────────

def compute_component_drifts(m0, mn, n_layers, n_heads, head_dim):
    """
    Phi-1.5 PhiDecoderLayer:
      self_attn: PhiAttention — q_proj, k_proj, v_proj, dense
      mlp:       PhiMLP       — fc1, fc2
    """
    rows = []
    for i in range(n_layers):
        a0 = m0.model.layers[i].self_attn
        an = mn.model.layers[i].self_attn
        l0 = m0.model.layers[i].mlp
        ln = mn.model.layers[i].mlp

        row = {"layer": i}

        # Attention projections
        for comp in ["q_proj", "k_proj", "v_proj", "dense"]:
            row[f"attn_{comp}"] = rel_drift(
                getattr(a0, comp).weight, getattr(an, comp).weight)

        # Attention aggregate
        attn_w0 = torch.cat([getattr(a0, c).weight.float().flatten()
                              for c in ["q_proj", "k_proj", "v_proj", "dense"]])
        attn_wn = torch.cat([getattr(an, c).weight.float().flatten()
                              for c in ["q_proj", "k_proj", "v_proj", "dense"]])
        row["attn_agg"] = rel_drift(attn_w0, attn_wn)

        # MLP projections
        for comp in ["fc1", "fc2"]:
            row[f"mlp_{comp}"] = rel_drift(
                getattr(l0, comp).weight, getattr(ln, comp).weight)

        mlp_w0 = torch.cat([getattr(l0, c).weight.float().flatten() for c in ["fc1", "fc2"]])
        mlp_wn = torch.cat([getattr(ln, c).weight.float().flatten() for c in ["fc1", "fc2"]])
        row["mlp_agg"] = rel_drift(mlp_w0, mlp_wn)

        # OV drift: W_V @ W_O per head
        # v_proj: (n_heads*head_dim, hidden); dense: (hidden, n_heads*head_dim)
        try:
            v0 = a0.v_proj.weight.float()   # (n_heads*hd, hidden)
            vn = an.v_proj.weight.float()
            o0 = a0.dense.weight.float()    # (hidden, n_heads*hd)
            on = an.dense.weight.float()
            ov_drifts = []
            for h in range(n_heads):
                wv0 = v0[h*head_dim:(h+1)*head_dim, :]
                wvn = vn[h*head_dim:(h+1)*head_dim, :]
                wo0 = o0[:, h*head_dim:(h+1)*head_dim]
                won = on[:, h*head_dim:(h+1)*head_dim]
                ov0 = wv0 @ wo0  # (head_dim, hidden) @ ... approximation
                ovn = wvn @ won
                ov_drifts.append(rel_drift(ov0, ovn))
            row["ov_mean"] = float(np.mean(ov_drifts))
        except Exception as e:
            print(f"    OV drift failed layer {i}: {e}")
            row["ov_mean"] = None

        rows.append(row)
    return rows


# ── analysis across generations ───────────────────────────────────────────────

def run_all_gens():
    fim_attn, fim_mlp = load_fim()
    n_blocks = len(fim_attn)
    log_fim_attn = [np.log10(v) if v > 0 else np.nan for v in fim_attn]
    log_fim_mlp  = [np.log10(v) if v > 0 else np.nan for v in fim_mlp]
    print(f"FIM loaded: {n_blocks} blocks from {FIM_FILE}")

    from transformers import AutoModelForCausalLM
    print("Loading Phi-1.5 Gen0...")
    m0 = AutoModelForCausalLM.from_pretrained(PHI_GEN0, torch_dtype=torch.float32,
                                               trust_remote_code=True)
    n_layers = m0.config.num_hidden_layers
    n_heads  = m0.config.num_attention_heads
    hidden   = m0.config.hidden_size
    head_dim = hidden // n_heads
    print(f"  n_layers={n_layers}, n_heads={n_heads}, head_dim={head_dim}")

    gen_results = {}

    for gen_n, gen_path in PHI_GENS.items():
        if not os.path.exists(gen_path):
            print(f"  Gen{gen_n} not found at {gen_path}, skipping.")
            continue

        print(f"\n── Gen{gen_n} ──")
        mn = AutoModelForCausalLM.from_pretrained(gen_path, torch_dtype=torch.float32,
                                                   trust_remote_code=True)

        block_drifts = compute_component_drifts(m0, mn, n_layers, n_heads, head_dim)
        del mn

        # Component drift arrays
        comp_drifts = {
            "attn_q":    [b["attn_q_proj"] for b in block_drifts],
            "attn_k":    [b["attn_k_proj"] for b in block_drifts],
            "attn_v":    [b["attn_v_proj"] for b in block_drifts],
            "attn_o":    [b["attn_dense"]  for b in block_drifts],
            "attn_agg":  [b["attn_agg"]    for b in block_drifts],
            "mlp_fc1":   [b["mlp_fc1"]     for b in block_drifts],
            "mlp_fc2":   [b["mlp_fc2"]     for b in block_drifts],
            "mlp_agg":   [b["mlp_agg"]     for b in block_drifts],
            "ov_mean":   [b["ov_mean"]     for b in block_drifts],
        }

        # Also load existing aggregate drift for cross-check
        dp = phi_drift_path(gen_n)
        agg_from_file = None
        if dp:
            existing = load_drift_json(dp)
            agg_from_file = {
                "attn": [existing.get(b, {}).get("attn_relative_drift") for b in range(n_layers)],
                "mlp":  [existing.get(b, {}).get("mlp_relative_drift")  for b in range(n_layers)],
            }

        corr = {}
        print(f"  {'Component':<12} {'ρ(FIM_attn)':>13} {'ρ(FIM_mlp)':>12}")
        print(f"  {'─'*40}")
        for comp, dvals in comp_drifts.items():
            rho_a, p_a, n_a = spearman(log_fim_attn, dvals)
            rho_m, p_m, n_m = spearman(log_fim_mlp,  dvals)
            corr[comp] = {
                "rho_attn_fim": rho_a, "p_attn_fim": p_a,
                "rho_mlp_fim":  rho_m, "p_mlp_fim":  p_m,
                "n": n_a,
            }
            def fv(r, p): return f"{r:+.3f}{sig(p)}" if r is not None else " N/A"
            print(f"  {comp:<12} {fv(rho_a,p_a):>13} {fv(rho_m,p_m):>12}")

        gen_results[str(gen_n)] = {
            "gen":          gen_n,
            "block_drifts": block_drifts,
            "correlations": corr,
            "agg_from_file": agg_from_file,
        }

    del m0

    return {
        "model":    "Phi-1.5",
        "arch":     "PAR",
        "n_layers": n_layers,
        "fim_attn": fim_attn,
        "fim_mlp":  fim_mlp,
        "generations": gen_results,
    }


# ── plotting ──────────────────────────────────────────────────────────────────

def plot(results):
    gens = sorted(results["generations"].keys(), key=int)
    if not gens:
        print("Nothing to plot.")
        return

    gen_nums = [int(g) for g in gens]
    fim_attn = results["fim_attn"]
    fim_mlp  = results["fim_mlp"]
    log_fa   = [np.log10(v) if v > 0 else np.nan for v in fim_attn]
    log_fm   = [np.log10(v) if v > 0 else np.nan for v in fim_mlp]
    n_layers = results["n_layers"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    def rho_traj(comp, fim_key):
        vals = []
        for g in gens:
            c = results["generations"][g]["correlations"].get(comp, {})
            vals.append(c.get(fim_key))
        return [v if v is not None else np.nan for v in vals]

    # Top-left: attn components vs FIM_attn across gens
    ax = axes[0][0]
    for comp in ["attn_q", "attn_k", "attn_v", "attn_o", "attn_agg"]:
        ax.plot(gen_nums, rho_traj(comp, "rho_attn_fim"), "o-", label=comp)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_title("Attn components vs log10(FIM_attn)")
    ax.set_xlabel("Generation"); ax.set_ylabel("Spearman ρ")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xticks(gen_nums)

    # Top-middle: MLP components vs FIM_mlp across gens
    ax = axes[0][1]
    for comp in ["mlp_fc1", "mlp_fc2", "mlp_agg"]:
        ax.plot(gen_nums, rho_traj(comp, "rho_mlp_fim"), "o-", label=comp)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_title("MLP components vs log10(FIM_mlp)")
    ax.set_xlabel("Generation"); ax.set_ylabel("Spearman ρ")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xticks(gen_nums)

    # Top-right: OV drift at last gen vs FIM_attn
    ax = axes[0][2]
    last_g = gens[-1]
    ov_vals = [b["ov_mean"] for b in results["generations"][last_g]["block_drifts"]]
    layers  = list(range(n_layers))
    fa_arr  = np.array(log_fa)
    valid   = fa_arr[np.isfinite(fa_arr)]
    if len(valid) and any(v is not None for v in ov_vals):
        sc = ax.scatter(layers, ov_vals, c=fa_arr, cmap="RdYlGn_r", s=55, zorder=3,
                        vmin=valid.min(), vmax=valid.max())
        plt.colorbar(sc, ax=ax, label="log10(FIM_attn)")
    else:
        ax.plot(layers, ov_vals, "o-")
    rho_ov, p_ov, _ = spearman(log_fa, ov_vals)
    title = f"OV drift at Gen{last_g}"
    if rho_ov is not None:
        title += f"\nρ(FIM_attn)={rho_ov:+.3f}{sig(p_ov)}"
    ax.set_title(title); ax.set_xlabel("Layer"); ax.set_ylabel("Mean OV drift")
    ax.grid(True, alpha=0.3)

    # Bottom-left: MLP drift heatmap (layers × gens)
    ax = axes[1][0]
    mat = np.full((n_layers, len(gens)), np.nan)
    for gi, g in enumerate(gens):
        for b in results["generations"][g]["block_drifts"]:
            mat[b["layer"], gi] = b["mlp_agg"]
    im = ax.imshow(mat, aspect="auto", cmap="hot", origin="lower")
    plt.colorbar(im, ax=ax, label="MLP agg drift")
    ax.set_xticks(range(len(gens))); ax.set_xticklabels([f"G{g}" for g in gens])
    ax.set_xlabel("Generation"); ax.set_ylabel("Layer")
    ax.set_title("MLP agg drift: layers × gens")

    # Bottom-middle: Attn drift heatmap
    ax = axes[1][1]
    mat2 = np.full((n_layers, len(gens)), np.nan)
    for gi, g in enumerate(gens):
        for b in results["generations"][g]["block_drifts"]:
            mat2[b["layer"], gi] = b["attn_agg"]
    im2 = ax.imshow(mat2, aspect="auto", cmap="hot", origin="lower")
    plt.colorbar(im2, ax=ax, label="Attn agg drift")
    ax.set_xticks(range(len(gens))); ax.set_xticklabels([f"G{g}" for g in gens])
    ax.set_xlabel("Generation"); ax.set_ylabel("Layer")
    ax.set_title("Attn agg drift: layers × gens")

    # Bottom-right: Summary ρ trajectory (attn_agg, mlp_agg, ov_mean)
    ax = axes[1][2]
    ax.plot(gen_nums, rho_traj("attn_agg", "rho_attn_fim"), "o-", label="attn_agg vs FIM_attn")
    ax.plot(gen_nums, rho_traj("mlp_agg",  "rho_mlp_fim"),  "s-", label="mlp_agg vs FIM_mlp")
    ax.plot(gen_nums, rho_traj("ov_mean",  "rho_attn_fim"), "^-", label="ov_mean vs FIM_attn")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_title("Summary ρ trajectory (PAR)")
    ax.set_xlabel("Generation"); ax.set_ylabel("Spearman ρ")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xticks(gen_nums)

    plt.suptitle(
        "Phi-1.5 Circuit Completion (Parallel architecture)\n"
        "Prediction: positive ρ for all components; signal should emerge Gen3+",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOT, dpi=150, bbox_inches="tight")
    print(f"\nPlot: {PLOT}")


def print_summary(results):
    gens = sorted(results["generations"].keys(), key=int)
    print("\n" + "="*80)
    print("SUMMARY — Phi-1.5 Component Correlations (PAR, prediction: all positive)")
    print("="*80)
    print(f"  {'Gen':<5} {'attn_agg':>10} {'mlp_agg':>10} {'ov_mean':>10} "
          f"{'mlp_fc1':>10} {'mlp_fc2':>10}")
    print("  " + "-"*55)
    for g in gens:
        c = results["generations"][g]["correlations"]
        def f(comp, key):
            v = c.get(comp, {}).get(key)
            p = c.get(comp, {}).get(key.replace("rho", "p"))
            return f"{v:+.3f}{sig(p)}" if v is not None else "N/A"
        print(f"  G{g:<4} "
              f"{f('attn_agg','rho_attn_fim'):>10} "
              f"{f('mlp_agg','rho_mlp_fim'):>10} "
              f"{f('ov_mean','rho_attn_fim'):>10} "
              f"{f('mlp_fc1','rho_mlp_fim'):>10} "
              f"{f('mlp_fc2','rho_mlp_fim'):>10}")
    print()
    print("DIAGNOSIS:")
    print("  All near-zero all gens → cascade degeneration scrambles signal")
    print("  Positive signs emerge Gen3+ → matches FIM paper, parallel mechanism present but slow")
    print("  fc1 vs fc2 diverge → fc1 (up-proj) more sensitive than fc2 (down)")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    if os.path.exists(CACHE):
        print(f"Cache hit: {CACHE}\nDelete to rerun.")
        with open(CACHE) as f:
            results = json.load(f)
        plot(results)
        print_summary(results)
        return

    results = run_all_gens()

    with open(CACHE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCached: {CACHE}")

    plot(results)
    print_summary(results)


if __name__ == "__main__":
    main()