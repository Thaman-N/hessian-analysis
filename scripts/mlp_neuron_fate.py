"""
mlp_neuron_fate_v2.py

Fixes from v1:
  - Dead/sat thresholds were calibrated for catastrophic collapse. At actual collapse
    levels (GPT-2 +143% PPL, Pythia +1426%), no neurons cross 10x or <10% thresholds
    because MLP activation ratios cluster in 1.0-1.5x range.
  - mean_ratio has a DEPTH confound: high-FIM blocks happen to be early layers, and
    early layers always show lower activation expansion than late layers regardless of
    architecture. Spearman(FIM, ratio) picks up depth not FIM protection.
  - Fix: compute partial Spearman controlling for layer index (rank residualise both
    FIM and ratio against layer position before correlating).
  - Also: replace dead/sat with percentile-based thresholds (top/bottom 10% of the
    actual ratio distribution) so the metric is always well-defined.

Run: python scripts/mlp_neuron_fate_v2.py
Out: results/circuit_analysis/mlp_neuron_fate_v2.json
     results/circuit_analysis/mlp_neuron_fate_v2.png
"""

import json
import os
import numpy as np
import torch
from scipy import stats
import matplotlib.pyplot as plt
from datasets import load_dataset

ROOT    = r"D:\Thaman\Work\hessian-spectral-analysis"
OUT_DIR = os.path.join(ROOT, "results", "circuit_analysis")
CACHE   = os.path.join(OUT_DIR, "mlp_neuron_fate_v2.json")
PLOT    = os.path.join(OUT_DIR, "mlp_neuron_fate_v2.png")

N_PROMPTS = 200

MODELS = [
    {
        "name":      "GPT-2",
        "arch":      "gpt2",
        "gen0_path": "gpt2",
        "gen5_path": os.path.join(ROOT, "models", "gpt2_treatment_gen_5"),
        "fim_file":  os.path.join(ROOT, "results", "fimgpt2_gen_0", "perblock_fim.json"),
    },
    {
        "name":      "Pythia-1.4b",
        "arch":      "pythia",
        "gen0_path": "EleutherAI/pythia-1.4b",
        "gen5_path": os.path.join(ROOT, "models", "pythia-1.4b_treatment_gen_5"),
        "fim_file":  os.path.join(ROOT, "results", "pythia-1.4b_treatment_gen_0", "perblock_fim.json"),
    },
]

os.makedirs(OUT_DIR, exist_ok=True)


def load_fim_mlp(fim_file):
    with open(fim_file) as f:
        d = json.load(f)
    fim_attn = [b["attention"]["top"] for b in d["blocks"]]
    fim_mlp  = [b["mlp"]["top"]       for b in d["blocks"]]
    return fim_attn, fim_mlp


def collect_activations_gpt2(model, tokenizer, prompts, device):
    n = model.config.n_layer
    acc = {i: [] for i in range(n)}
    def make_hook(i):
        def h(m, inp, out):
            acc[i].append(out.detach().cpu().float().abs().mean(dim=(0, 1)))
        return h
    handles = [model.transformer.h[i].mlp.c_fc.register_forward_hook(make_hook(i))
               for i in range(n)]
    model.eval()
    with torch.no_grad():
        for p in prompts:
            enc = tokenizer(p, return_tensors="pt", truncation=True, max_length=128).to(device)
            model(**enc)
    for h in handles: h.remove()
    return {i: torch.stack(acc[i]).mean(0).numpy() for i in range(n)}


def collect_activations_pythia(model, tokenizer, prompts, device):
    n = model.config.num_hidden_layers
    acc = {i: [] for i in range(n)}
    def make_hook(i):
        def h(m, inp, out):
            acc[i].append(out.detach().cpu().float().abs().mean(dim=(0, 1)))
        return h
    handles = [model.gpt_neox.layers[i].mlp.dense_h_to_4h.register_forward_hook(make_hook(i))
               for i in range(n)]
    model.eval()
    with torch.no_grad():
        for p in prompts:
            enc = tokenizer(p, return_tensors="pt", truncation=True, max_length=128).to(device)
            model(**enc)
    for h in handles: h.remove()
    return {i: torch.stack(acc[i]).mean(0).numpy() for i in range(n)}


def partial_spearman(x, y, covariate):
    """
    Partial Spearman correlation between x and y controlling for covariate.
    Residualise ranks of x and y on ranks of covariate, then correlate residuals.
    This removes the depth confound (layer index is the covariate).
    """
    def rank(v):
        return np.array(stats.rankdata(v), dtype=float)

    rx = rank(x)
    ry = rank(y)
    rc = rank(covariate)

    def residualise(v, c):
        slope, intercept, _, _, _ = stats.linregress(c, v)
        return v - (slope * c + intercept)

    rx_resid = residualise(rx, rc)
    ry_resid = residualise(ry, rc)

    rho, p = stats.pearsonr(rx_resid, ry_resid)   # Pearson on rank residuals = partial Spearman
    return float(rho), float(p)


def sig(p):
    if p is None: return ""
    return "**" if p < 0.01 else ("*" if p < 0.05 else "")


def analyze(cfg, prompts, device):
    name = cfg["name"]
    print(f"\n{'='*60}\n  {name}\n{'='*60}")

    fim_attn, fim_mlp = load_fim_mlp(cfg["fim_file"])
    log_fim_mlp = np.array([np.log10(v) if v > 0 else np.nan for v in fim_mlp])
    n_blocks = len(fim_mlp)

    if cfg["arch"] == "gpt2":
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        m0 = GPT2LMHeadModel.from_pretrained(cfg["gen0_path"]).to(device)
        tok = GPT2Tokenizer.from_pretrained(cfg["gen0_path"])
        tok.pad_token = tok.eos_token
        m5 = GPT2LMHeadModel.from_pretrained(cfg["gen5_path"]).to(device)
        fn = collect_activations_gpt2
    else:
        from transformers import GPTNeoXForCausalLM, AutoTokenizer
        m0 = GPTNeoXForCausalLM.from_pretrained(cfg["gen0_path"]).to(device)
        tok = AutoTokenizer.from_pretrained(cfg["gen0_path"])
        m5 = GPTNeoXForCausalLM.from_pretrained(cfg["gen5_path"]).to(device)
        fn = collect_activations_pythia

    print(f"  Collecting activations ({N_PROMPTS} prompts)...")
    acts0 = fn(m0, tok, prompts, device)
    acts5 = fn(m5, tok, prompts, device)
    del m0, m5
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Per-block mean |activation| ratio Gen5/Gen0
    eps = 1e-8
    ratios = np.array([float((acts5[b] / (acts0[b] + eps)).mean()) for b in range(n_blocks)])
    layers = np.arange(n_blocks, dtype=float)

    # Percentile-based thresholds from the actual distribution
    # "suppressed" = bottom 20% of ratio; "expanded" = top 20%
    p20 = np.percentile(ratios, 20)
    p80 = np.percentile(ratios, 80)
    suppressed_frac = np.array([(ratios[b] < p20) for b in range(n_blocks)], dtype=float)
    expanded_frac   = np.array([(ratios[b] > p80) for b in range(n_blocks)], dtype=float)

    print(f"\n  ratio range: [{ratios.min():.3f}, {ratios.max():.3f}]  "
          f"p20={p20:.3f}  p80={p80:.3f}")
    print(f"\n  {'Blk':>3}  {'ratio':>7}  {'log10FIM':>9}  {'layer_idx':>10}")
    print(f"  {'─'*38}")
    for b in range(n_blocks):
        print(f"  {b:>3}  {ratios[b]:>7.3f}  {log_fim_mlp[b]:>9.3f}  {b:>10}")

    # ── Spearman (raw) ────────────────────────────────────────────────────────
    valid = [(log_fim_mlp[b], ratios[b]) for b in range(n_blocks)
             if np.isfinite(log_fim_mlp[b])]
    xs, ys = zip(*valid)
    rho_raw, p_raw = stats.spearmanr(xs, ys)

    # ── Partial Spearman controlling for depth (layer index) ─────────────────
    fim_valid = np.array([log_fim_mlp[b] for b in range(n_blocks) if np.isfinite(log_fim_mlp[b])])
    rat_valid = np.array([ratios[b]      for b in range(n_blocks) if np.isfinite(log_fim_mlp[b])])
    dep_valid = np.array([float(b)       for b in range(n_blocks) if np.isfinite(log_fim_mlp[b])])

    rho_partial, p_partial = partial_spearman(fim_valid, rat_valid, dep_valid)

    # ── Depth correlation (to confirm confound) ───────────────────────────────
    rho_depth_ratio, p_depth_ratio = stats.spearmanr(dep_valid, rat_valid)
    rho_depth_fim,   p_depth_fim   = stats.spearmanr(dep_valid, fim_valid)

    n = len(valid)
    print(f"\n  Correlations (n={n}):")
    print(f"    ρ(FIM_mlp, ratio) raw      = {rho_raw:+.3f}{sig(p_raw)}   p={p_raw:.4f}")
    print(f"    ρ(FIM_mlp, ratio) partial  = {rho_partial:+.3f}{sig(p_partial)}   p={p_partial:.4f}  "
          f"← controlling for layer depth")
    print(f"    ρ(depth, ratio)            = {rho_depth_ratio:+.3f}{sig(p_depth_ratio)}   "
          f"p={p_depth_ratio:.4f}  (confound magnitude)")
    print(f"    ρ(depth, FIM_mlp)          = {rho_depth_fim:+.3f}{sig(p_depth_fim)}   "
          f"p={p_depth_fim:.4f}  (FIM-depth colinearity)")

    blocks_out = [
        {
            "block": b,
            "ratio": float(ratios[b]),
            "fim_mlp": fim_mlp[b],
            "log10_fim_mlp": float(log_fim_mlp[b]) if np.isfinite(log_fim_mlp[b]) else None,
        }
        for b in range(n_blocks)
    ]

    return {
        "model": name,
        "arch":  cfg["arch"],
        "blocks": blocks_out,
        "p20": float(p20), "p80": float(p80),
        "correlations": {
            "rho_raw":      float(rho_raw),     "p_raw":      float(p_raw),
            "rho_partial":  float(rho_partial),  "p_partial":  float(p_partial),
            "rho_depth_ratio": float(rho_depth_ratio),
            "rho_depth_fim":   float(rho_depth_fim),
            "n": n,
        }
    }


def plot(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    for row, res in enumerate(all_results):
        blocks    = res["blocks"]
        layers    = [b["block"]         for b in blocks]
        ratios    = [b["ratio"]         for b in blocks]
        log_fim   = np.array([b["log10_fim_mlp"] if b["log10_fim_mlp"] is not None else np.nan
                               for b in blocks])
        arch = "SEQ" if res["arch"] == "gpt2" else "PAR"
        c = res["correlations"]

        # Left: ratio by layer, coloured by FIM
        ax = axes[row][0]
        valid_fim = log_fim[np.isfinite(log_fim)]
        sc = ax.scatter(layers, ratios, c=log_fim, cmap="RdYlGn_r", s=60, zorder=3,
                        vmin=valid_fim.min(), vmax=valid_fim.max())
        plt.colorbar(sc, ax=ax, label="log10(FIM_mlp)")
        ax.axhline(1.0, color="k", lw=0.8, ls="--", label="ratio=1 (no change)")
        ax.set_xlabel("Layer"); ax.set_ylabel("Mean |act| ratio (Gen5/Gen0)")
        ax.set_title(f"{res['model']} ({arch}) — activation ratio by layer\n"
                     f"ρ_raw={c['rho_raw']:+.3f}{sig(c['p_raw'])}  "
                     f"ρ_partial={c['rho_partial']:+.3f}{sig(c['p_partial'])}")
        ax.grid(True, alpha=0.3)

        # Right: FIM vs ratio scatter with depth as size
        ax = axes[row][1]
        fim_vals = [b["log10_fim_mlp"] for b in blocks if b["log10_fim_mlp"] is not None]
        rat_vals = [b["ratio"]         for b in blocks if b["log10_fim_mlp"] is not None]
        lay_vals = [b["block"]         for b in blocks if b["log10_fim_mlp"] is not None]
        # size encodes layer depth
        max_l = max(lay_vals) if lay_vals else 1
        sizes = [30 + 120 * (l / max_l) for l in lay_vals]
        sc2 = ax.scatter(fim_vals, rat_vals, s=sizes, c=lay_vals,
                         cmap="viridis", alpha=0.8, zorder=3)
        plt.colorbar(sc2, ax=ax, label="Layer index")
        ax.set_xlabel("log10(FIM_mlp)"); ax.set_ylabel("Mean |act| ratio")
        ax.set_title(f"{res['model']} ({arch}) — FIM vs ratio\n"
                     f"Point size & color = layer depth\n"
                     f"ρ_partial(depth removed)={c['rho_partial']:+.3f}{sig(c['p_partial'])}")
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "MLP Neuron Activation Ratio Gen0→Gen5 (v2)\n"
        "Raw ρ is confounded by depth (high FIM = early layer = naturally lower ratio).\n"
        "Partial ρ removes depth to isolate the true FIM effect.",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOT, dpi=150, bbox_inches="tight")
    print(f"\nPlot: {PLOT}")


def main():
    if os.path.exists(CACHE):
        print(f"Cache hit: {CACHE}")
        with open(CACHE) as f:
            results = json.load(f)
        plot(results)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    prompts = [item["text"][:256] for item in ds if len(prompts := []) == 0 or True][:N_PROMPTS]
    # simpler loading:
    prompts = []
    ds2 = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    for item in ds2:
        prompts.append(item["text"][:256])
        if len(prompts) >= N_PROMPTS:
            break
    print(f"  Loaded {len(prompts)} prompts.")

    results = [analyze(cfg, prompts, device) for cfg in MODELS]

    with open(CACHE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCached: {CACHE}")
    plot(results)

    print("\n" + "="*70)
    print("SUMMARY — Partial Spearman (depth-controlled)")
    print("="*70)
    print(f"{'Model':<15} {'Arch':<6} {'ρ_raw':>8} {'ρ_partial':>11} "
          f"{'ρ(depth,ratio)':>16} {'ρ(depth,FIM)':>14}")
    print("-"*70)
    for r in results:
        c = r["correlations"]
        arch = "SEQ" if r["arch"] == "gpt2" else "PAR"
        print(f"{r['model']:<15} {arch:<6} "
              f"{c['rho_raw']:>+8.3f}{sig(c['p_raw']):<2} "
              f"{c['rho_partial']:>+11.3f}{sig(c['p_partial']):<2} "
              f"{c['rho_depth_ratio']:>+16.3f}  "
              f"{c['rho_depth_fim']:>+14.3f}")
    print()
    print("KEY: If ρ_partial flips sign vs ρ_raw, depth was the dominant driver.")
    print("     If ρ_partial stays negative for BOTH: both models show same response,")
    print("     meaning activation expansion doesn't differentiate seq vs par at this scale.")
    print("     True FIM effect would need ρ_partial(seq) negative, ρ_partial(par) positive.")


if __name__ == "__main__":
    main()