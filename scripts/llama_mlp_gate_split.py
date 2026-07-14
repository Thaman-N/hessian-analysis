"""
llama_mlp_gate_split.py

Llama-3.2-1B shows rho_attn=-0.374, rho_mlp=+0.397, rho_combined=+0.076 —
a sequential model whose MLP is NOT protected.

Hypothesis: SwiGLU gating (output = gate_proj * silu(up_proj), then down_proj)
creates two MLP weight paths with different gradient structures. vt suppression
may apply differently to gate_proj vs up_proj, and their opposing effects cancel
when aggregated at the block level — producing near-zero combined rho.

SmolLM2-135M uses identical SwiGLU/LlamaAttention but shows rho_mlp=-0.612**.
Comparing the gate/up/down split between Llama and SmolLM reveals whether:
  (a) Gate vs up drift cancels in Llama but not SmolLM (structural explanation)
  (b) Both positive in Llama (scale/depth drives vt saturation at 1B)
  (c) Both negative in both (attn/mlp split script was correct; anomaly elsewhere)

Run: python scripts/llama_mlp_gate_split.py
Out: results/circuit_analysis/llama_mlp_gate_split.json
     results/circuit_analysis/llama_mlp_gate_split.png
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
CACHE   = os.path.join(OUT_DIR, "llama_mlp_gate_split.json")
PLOT    = os.path.join(OUT_DIR, "llama_mlp_gate_split.png")

MODELS = [
    {
        "name":      "Llama-3.2-1B",
        "arch_tag":  "SEQ",
        "gen0_path": "meta-llama/Llama-3.2-1B",
        "gen5_path": os.path.join(ROOT, "models", "llama_treatment_gen_5"),
        "fim_file":  os.path.join(ROOT, "results", "fimllama_treatment_gen_0", "perblock_fim.json"),
    },
    {
        "name":      "SmolLM2-135M",
        "arch_tag":  "SEQ",
        "gen0_path": "HuggingFaceTB/SmolLM2-135M",
        "gen5_path": os.path.join(ROOT, "models", "treatment_gen_5"),
        "fim_file":  os.path.join(ROOT, "results", "fimsmollm0", "perblock_fim.json"),
    },
]

os.makedirs(OUT_DIR, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_fim(fim_file):
    with open(fim_file) as f:
        d = json.load(f)
    fim_attn = [b["attention"]["top"] for b in d["blocks"]]
    fim_mlp  = [b["mlp"]["top"]       for b in d["blocks"]]
    return fim_attn, fim_mlp


def rel_drift(w0: torch.Tensor, w5: torch.Tensor) -> float:
    return (w5.float() - w0.float()).norm().item() / (w0.float().norm().item() + 1e-8)


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


def compute_block_drifts(m0, m5, n_layers):
    """
    Both Llama-3.2-1B and SmolLM2-135M use LlamaDecoder structure:
      self_attn: q_proj, k_proj, v_proj, o_proj
      mlp:       gate_proj, up_proj, down_proj  (SwiGLU)
    """
    rows = []
    for i in range(n_layers):
        b0 = m0.model.layers[i]
        b5 = m5.model.layers[i]
        row = {"layer": i}

        # MLP components
        for comp in ["gate_proj", "up_proj", "down_proj"]:
            w0 = getattr(b0.mlp, comp).weight
            w5 = getattr(b5.mlp, comp).weight
            row[f"mlp_{comp}"] = rel_drift(w0, w5)

        # MLP aggregate (matches attn_vs_mlp_split definition)
        all_w0 = torch.cat([getattr(b0.mlp, c).weight.float().flatten()
                             for c in ["gate_proj", "up_proj", "down_proj"]])
        all_w5 = torch.cat([getattr(b5.mlp, c).weight.float().flatten()
                             for c in ["gate_proj", "up_proj", "down_proj"]])
        row["mlp_agg"] = rel_drift(all_w0, all_w5)

        # Attention components
        for comp in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            w0 = getattr(b0.self_attn, comp).weight
            w5 = getattr(b5.self_attn, comp).weight
            row[f"attn_{comp}"] = rel_drift(w0, w5)

        attn_w0 = torch.cat([getattr(b0.self_attn, c).weight.float().flatten()
                              for c in ["q_proj", "k_proj", "v_proj", "o_proj"]])
        attn_w5 = torch.cat([getattr(b5.self_attn, c).weight.float().flatten()
                              for c in ["q_proj", "k_proj", "v_proj", "o_proj"]])
        row["attn_agg"] = rel_drift(attn_w0, attn_w5)

        rows.append(row)
    return rows


def analyze(cfg):
    name = cfg["name"]
    print(f"\n{'='*60}\n  {name}\n{'='*60}")

    from transformers import AutoModelForCausalLM
    print("  Loading Gen0...")
    m0 = AutoModelForCausalLM.from_pretrained(cfg["gen0_path"], torch_dtype=torch.float32)
    print("  Loading Gen5...")
    m5 = AutoModelForCausalLM.from_pretrained(cfg["gen5_path"], torch_dtype=torch.float32)

    n_layers = m0.config.num_hidden_layers
    print(f"  n_layers = {n_layers}")

    fim_attn, fim_mlp = load_fim(cfg["fim_file"])
    log_fim_attn = [np.log10(v) if v > 0 else np.nan for v in fim_attn]
    log_fim_mlp  = [np.log10(v) if v > 0 else np.nan for v in fim_mlp]

    print("  Computing component drifts...")
    block_drifts = compute_block_drifts(m0, m5, n_layers)
    del m0, m5

    # Print table
    comps = ["gate_proj", "up_proj", "down_proj", "agg"]
    hdr = f"  {'L':>3}  " + "  ".join(f"{'mlp_'+c:>12}" for c in comps) + \
          "  " + "  ".join(f"{'attn_'+c:>12}" for c in ["q_proj","k_proj","v_proj","o_proj","agg"]) + \
          f"  {'log10FIM_m':>10}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for b in block_drifts:
        i = b["layer"]
        row_str = f"  {i:>3}  "
        row_str += "  ".join(f"{b[f'mlp_{c}']:>12.5f}" for c in comps)
        row_str += "  "
        row_str += "  ".join(f"{b[f'attn_{c}']:>12.5f}"
                              for c in ["q_proj","k_proj","v_proj","o_proj","agg"])
        row_str += f"  {log_fim_mlp[i]:>10.3f}"
        print(row_str)

    # Correlations: test each MLP component vs log10(FIM_mlp)
    # and each attn component vs log10(FIM_attn)
    mlp_comps  = {"gate_proj": "mlp_gate_proj", "up_proj": "mlp_up_proj",
                  "down_proj": "mlp_down_proj", "agg": "mlp_agg"}
    attn_comps = {"q_proj": "attn_q_proj", "k_proj": "attn_k_proj",
                  "v_proj": "attn_v_proj", "o_proj": "attn_o_proj", "agg": "attn_agg"}

    corr = {}

    print(f"\n  ── MLP components vs log10(FIM_mlp) ──")
    print(f"  {'Component':<15} {'rho':>8}   {'p':>8}   n")
    print(f"  {'─'*40}")
    for label, key in mlp_comps.items():
        vals = [b[key] for b in block_drifts]
        rho, p, n = spearman(log_fim_mlp, vals)
        corr[f"mlp_{label}"] = {"rho": rho, "p": p, "n": n}
        rho_str = f"{rho:+.3f}{sig(p)}" if rho is not None else "N/A"
        p_str   = f"{p:.4f}"            if p   is not None else "N/A"
        print(f"  {label:<15} {rho_str:>10}   {p_str:>8}   {n}")

    print(f"\n  ── Attention components vs log10(FIM_attn) ──")
    print(f"  {'Component':<15} {'rho':>8}   {'p':>8}   n")
    print(f"  {'─'*40}")
    for label, key in attn_comps.items():
        vals = [b[key] for b in block_drifts]
        rho, p, n = spearman(log_fim_attn, vals)
        corr[f"attn_{label}"] = {"rho": rho, "p": p, "n": n}
        rho_str = f"{rho:+.3f}{sig(p)}" if rho is not None else "N/A"
        p_str   = f"{p:.4f}"            if p   is not None else "N/A"
        print(f"  {label:<15} {rho_str:>10}   {p_str:>8}   {n}")

    return {
        "model":        name,
        "arch":         cfg["arch_tag"],
        "n_layers":     n_layers,
        "block_drifts": block_drifts,
        "fim_attn":     fim_attn,
        "fim_mlp":      fim_mlp,
        "correlations": corr,
    }


# ── plotting ──────────────────────────────────────────────────────────────────

def plot(all_results):
    n_models = len(all_results)
    fig, axes = plt.subplots(n_models, 4, figsize=(20, 5 * n_models))
    if n_models == 1:
        axes = [axes]

    mlp_plot_comps = ["gate_proj", "up_proj", "down_proj", "agg"]
    titles = ["gate_proj drift", "up_proj drift", "down_proj drift", "MLP aggregate drift"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    for row, res in enumerate(all_results):
        log_fim = np.array([np.log10(v) if v > 0 else np.nan for v in res["fim_mlp"]])
        layers  = [b["layer"] for b in res["block_drifts"]]

        for col, (comp, title, color) in enumerate(zip(mlp_plot_comps, titles, colors)):
            ax = axes[row][col]
            key    = f"mlp_{comp}"
            drifts = [b[key] for b in res["block_drifts"]]

            valid = log_fim[np.isfinite(log_fim)]
            if len(valid):
                sc = ax.scatter(layers, drifts, c=log_fim, cmap="coolwarm",
                                s=55, zorder=3, vmin=valid.min(), vmax=valid.max())
                plt.colorbar(sc, ax=ax, label="log10(FIM_mlp)")
            else:
                ax.plot(layers, drifts, "o-", color=color)

            c = res["correlations"].get(f"mlp_{comp}", {})
            rho, p = c.get("rho"), c.get("p")
            if rho is not None:
                ax.text(0.03, 0.97, f"ρ={rho:+.3f}{sig(p)}",
                        transform=ax.transAxes, va="top", fontsize=9,
                        bbox=dict(boxstyle="round", fc="white", alpha=0.8))

            ax.set_xlabel("Layer")
            ax.set_ylabel("Relative drift")
            ax.set_title(f"{res['model']} ({res['arch']})\n{title}")
            ax.grid(True, alpha=0.3)

    plt.suptitle(
        "SwiGLU Gate/Up/Down Split: Does gate_proj cancel vt protection in Llama?\n"
        "If gate_rho > 0 while up_rho < 0: gate structure explains Llama anomaly\n"
        "Compare with SmolLM (same arch, 135M) which shows full protection",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOT, dpi=150, bbox_inches="tight")
    print(f"\nPlot: {PLOT}")


def print_summary(all_results):
    print("\n" + "="*75)
    print("SUMMARY — SwiGLU Component Split")
    print("="*75)
    print(f"{'Model':<16} {'gate_rho':<12} {'up_rho':<12} {'down_rho':<12} {'mlp_agg_rho':<13}")
    print("-"*65)
    for res in all_results:
        c = res["correlations"]
        def f(key):
            v, p = c.get(key, {}).get("rho"), c.get(key, {}).get("p")
            return f"{v:+.3f}{sig(p)}" if v is not None else "N/A"
        print(f"{res['model']:<16} "
              f"{f('mlp_gate_proj'):<12} "
              f"{f('mlp_up_proj'):<12} "
              f"{f('mlp_down_proj'):<12} "
              f"{f('mlp_agg'):<13}")
    print()
    print("INTERPRETATION:")
    print("  gate >> up (gate+, up-): SwiGLU gate cancels protection → Llama anomaly explained")
    print("  both + in Llama, both - in SmolLM: scale (1B vs 135M) is the differentiator")
    print("  both - in Llama: attn_vs_mlp_split rho_mlp=+0.397 was a false positive (n=16)")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    if os.path.exists(CACHE):
        print(f"Cache hit: {CACHE}\nDelete to rerun.")
        with open(CACHE) as f:
            results = json.load(f)
        plot(results)
        print_summary(results)
        return

    results = []
    for cfg in MODELS:
        results.append(analyze(cfg))

    with open(CACHE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCached: {CACHE}")

    plot(results)
    print_summary(results)


if __name__ == "__main__":
    main()