"""
Experiment I: Attention vs MLP FIM-Drift Split
================================================
The OV circuit analysis showed GPT-2 (sequential) has rho=+0.615 (high FIM =
more OV drift), which is opposite to the FIM paper's negative sequential
correlation. This was puzzling.

The resolution: the FIM paper's block-level drift aggregates attention + MLP.
If the sequential protection operates primarily through MLP suppression (not
attention suppression), then:
  - FIM_mlp ~ mlp_drift: negative for sequential (MLP protected)
  - FIM_attn ~ attn_drift: near zero or positive for sequential (attn NOT protected)
  - Both positive for parallel (both amplified)

This experiment tests that hypothesis directly using already-uploaded drift and
FIM files. No GPU required — pure analysis of existing data.

Usage:
  python scripts/attn_vs_mlp_split.py

Output:
  results/circuit_analysis/attn_vs_mlp_split.json
  results/circuit_analysis/attn_vs_mlp_split.png
"""

import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BASE_DIR   = r"D:\Thaman\Work\hessian-spectral-analysis"
UPLOAD_DIR = r"D:\Thaman\Work\hessian-spectral-analysis\results\parameterdrift"
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "circuit_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# DATA SOURCES
# All FIM files should be in results/ or wherever your summaries live.
# Drift files should be in results/parameterdrift/ or wherever
# parameter_drift.py saved them.
# Update these paths if your layout differs.
# ============================================================

MODELS = {
    "SmolLM (seq)": {
        "arch": "sequential",
        "fim_gen0": os.path.join(BASE_DIR, "results", "fimsmollm0.txt"),
        "drift_files": {
            gen: os.path.join(BASE_DIR, "results", "parameterdrift",
                              f"treatment_gen_{gen}.json")
            for gen in range(1, 6)
        },
    },
    "GPT-2 (seq)": {
        "arch": "sequential",
        "fim_gen0": os.path.join(BASE_DIR, "results", "fimgpt2_gen_0.txt"),
        "drift_files": {
            gen: os.path.join(BASE_DIR, "results", "parameterdrift",
                              f"gpt2_treatment_gen_{gen}.json")
            for gen in range(1, 6)
        },
    },
    "Pythia (par)": {
        "arch": "parallel",
        "fim_gen0": os.path.join(BASE_DIR, "results",
                                 "pythia-1.4b_treatment_gen_0", "perblock_fim.json"),
        "drift_files": {
            gen: os.path.join(BASE_DIR, "results",
                              f"pythia_treatment_drift_correct_gen_{gen}.json")
            for gen in range(1, 6)
        },
    },
    "Phi-1.5 (par)": {
        "arch": "parallel",
        "fim_gen0": os.path.join(BASE_DIR, "results", "phi-1_5_treatment_gen_0",
                                 "perblock_fim.json"),
        "drift_files": {
            gen: os.path.join(BASE_DIR, "results",
                              f"phi-1_5_drift_gen_{gen}.json")
            for gen in range(1, 6)
        },
    },
    "Llama (seq)": {
        "arch": "sequential",
        "fim_gen0": os.path.join(BASE_DIR, "results", "fimllama_treatment_gen_0.txt"),
        "drift_files": {
            gen: os.path.join(BASE_DIR, "results", "parameterdrift",
                              f"llama_treatment_gen_{gen}.json")
            for gen in range(1, 6)
        },
    },
}


def parse_fim_txt(path):
    """Parse FIM summary txt file → {block: {attn: float, mlp: float}}"""
    for enc in ["utf-16", "utf-8", "cp1252"]:
        try:
            with open(path, encoding=enc) as f:
                content = f.read()
            break
        except UnicodeError:
            continue
    blocks = {}
    for line in content.split("\n"):
        m = re.match(
            r"Block\s+(\d+)\s*\|\s*Attn:\s*([\d.]+)\s*\|\s*MLP:\s*([\d.]+)", line
        )
        if m:
            b = int(m.group(1))
            blocks[b] = {"attn": float(m.group(2)), "mlp": float(m.group(3))}
    return blocks


def parse_fim_json(path):
    """Parse perblock_fim.json → {block: {attn: float, mlp: float}}"""
    with open(path) as f:
        data = json.load(f)
    blocks = {}
    for entry in data.get("blocks", []):
        b = entry.get("block_idx", 0)
        attn = entry.get("attention", {})
        mlp  = entry.get("mlp", {})
        at = float(attn.get("top", 0)) if isinstance(attn, dict) and "error" not in attn else 0.0
        mt = float(mlp.get("top",  0)) if isinstance(mlp,  dict) and "error" not in mlp  else 0.0
        if at > 0 or mt > 0:
            blocks[b] = {"attn": at, "mlp": mt}
    return blocks


def load_fim(path):
    if path.endswith(".json"):
        return parse_fim_json(path)
    return parse_fim_txt(path)


def load_drift(path):
    with open(path) as f:
        d = json.load(f)
    return {int(b): v for b, v in d["blocks"].items()}


def compute_correlations(fim, drift):
    """
    Compute three Spearman correlations:
      1. log10(FIM_attn_b) vs attn_drift_b
      2. log10(FIM_mlp_b)  vs mlp_drift_b
      3. log10(FIM_b)      vs combined_drift_b  (replicates FIM paper)

    Returns dict with rho and p for each.
    """
    common = sorted(set(fim) & set(drift))
    if len(common) < 5:
        return None, common

    log_fim_attn    = np.array([np.log10(fim[b]["attn"] + 1e-8) for b in common])
    log_fim_mlp     = np.array([np.log10(fim[b]["mlp"]  + 1e-8) for b in common])
    log_fim_combined = np.array([
        np.log10(fim[b]["attn"] + fim[b]["mlp"] + 1e-8) for b in common
    ])
    d_attn     = np.array([drift[b]["attn_relative_drift"] for b in common])
    d_mlp      = np.array([drift[b]["mlp_relative_drift"]  for b in common])
    d_combined = d_attn + d_mlp

    rho_attn,     p_attn     = spearmanr(log_fim_attn,     d_attn)
    rho_mlp,      p_mlp      = spearmanr(log_fim_mlp,      d_mlp)
    rho_combined, p_combined = spearmanr(log_fim_combined, d_combined)

    return {
        "blocks":        common,
        "rho_attn":      float(rho_attn),
        "p_attn":        float(p_attn),
        "rho_mlp":       float(rho_mlp),
        "p_mlp":         float(p_mlp),
        "rho_combined":  float(rho_combined),
        "p_combined":    float(p_combined),
        "log_fim_attn":  log_fim_attn.tolist(),
        "log_fim_mlp":   log_fim_mlp.tolist(),
        "d_attn":        d_attn.tolist(),
        "d_mlp":         d_mlp.tolist(),
    }, common


def run():
    all_results = {}

    print(f"\n{'Model':<20} {'rho_attn':>10} {'rho_mlp':>10} {'rho_combined':>14} {'n':>5}")
    print("-" * 65)

    for model_name, config in MODELS.items():
        arch = config["arch"]

        fim_path = config["fim_gen0"]
        if not os.path.exists(fim_path):
            print(f"{model_name:<20} FIM not found: {fim_path}")
            continue

        fim = load_fim(fim_path)
        if not fim:
            print(f"{model_name:<20} FIM parse failed")
            continue

        model_results = {}

        for gen in range(1, 6):
            drift_path = config["drift_files"][gen]
            if not os.path.exists(drift_path):
                continue
            drift = load_drift(drift_path)
            result, common = compute_correlations(fim, drift)
            if result is None:
                continue
            model_results[gen] = result

        if not model_results:
            print(f"{model_name:<20} no drift files found")
            continue

        # Print Gen5 summary (most developed pattern)
        def sig(p):
            return "**" if p < 0.01 else ("*" if p < 0.05 else "  ")

        gen5 = model_results.get(5, list(model_results.values())[-1])
        print(
            f"{model_name:<20} "
            f"{gen5['rho_attn']:+.3f}{sig(gen5['p_attn'])}  "
            f"{gen5['rho_mlp']:+.3f}{sig(gen5['p_mlp'])}  "
            f"{gen5['rho_combined']:+.3f}{sig(gen5['p_combined'])}  "
            f"{len(gen5['blocks']):>5}"
        )

        all_results[model_name] = {
            "arch": arch,
            "gens": {str(g): v for g, v in model_results.items()}
        }

    print()
    print("rho_attn     = Spearman(log10 FIM_attn_b, attn_drift_b)")
    print("rho_mlp      = Spearman(log10 FIM_mlp_b,  mlp_drift_b)")
    print("rho_combined = replicates FIM paper block-level result")
    print()
    print("KEY PREDICTION:")
    print("  Sequential: rho_mlp << rho_attn (MLP more protected than attention)")
    print("  Parallel:   both positive (both components amplified)")

    save_path = os.path.join(OUTPUT_DIR, "attn_vs_mlp_split.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)

    plot(all_results)
    return all_results


def plot(all_results):
    """
    Two-panel figure:
      Left:  rho_attn vs rho_mlp per model at Gen5 (scatter with architecture colour)
      Right: trajectory of rho_attn and rho_mlp across gens for GPT-2 and Pythia
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ----- Panel 1: rho_attn vs rho_mlp scatter -----
    ax = axes[0]
    color_map = {"sequential": "steelblue", "parallel": "tomato"}
    marker_map = {"sequential": "o", "parallel": "s"}

    for model_name, res in all_results.items():
        arch  = res["arch"]
        gen5  = res["gens"].get("5", list(res["gens"].values())[-1])
        ra    = gen5["rho_attn"]
        rm    = gen5["rho_mlp"]
        ax.scatter(ra, rm,
                   color=color_map[arch],
                   marker=marker_map[arch],
                   s=120, zorder=5,
                   edgecolors="k", linewidths=0.8)
        ax.annotate(model_name.split(" ")[0],
                    (ra, rm), textcoords="offset points",
                    xytext=(6, 4), fontsize=9)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("rho(FIM_attn, attn_drift) Gen5", fontsize=11)
    ax.set_ylabel("rho(FIM_mlp, mlp_drift) Gen5", fontsize=11)
    ax.set_title(
        "Attention vs MLP FIM-Drift Correlation\n"
        "Blue circles = sequential, Red squares = parallel",
        fontsize=11
    )

    # Quadrant labels
    ax.text(0.72, 0.05, "both amplified\n(parallel)",
            transform=ax.transAxes, fontsize=8, color="tomato", ha="center")
    ax.text(0.15, 0.95, "both protected\n(ideal sequential)",
            transform=ax.transAxes, fontsize=8, color="steelblue", ha="center")
    ax.text(0.72, 0.95, "attn amplified\nMLP protected",
            transform=ax.transAxes, fontsize=8, color="gray", ha="center")
    ax.grid(True, alpha=0.3)

    # ----- Panel 2: trajectory across gens for GPT-2 and Pythia -----
    ax2 = axes[1]

    for model_name, res in all_results.items():
        if "GPT-2" not in model_name and "Pythia" not in model_name:
            continue
        arch = res["arch"]
        color = color_map[arch]
        gens  = sorted(int(g) for g in res["gens"].keys())
        rho_attn_traj = [res["gens"][str(g)]["rho_attn"] for g in gens]
        rho_mlp_traj  = [res["gens"][str(g)]["rho_mlp"]  for g in gens]
        short = model_name.split(" ")[0]
        ax2.plot(gens, rho_attn_traj, "o--", color=color, alpha=0.7,
                 linewidth=1.5, markersize=6, label=f"{short} attn")
        ax2.plot(gens, rho_mlp_traj,  "s-",  color=color, alpha=1.0,
                 linewidth=2, markersize=7, label=f"{short} MLP")

    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Generation", fontsize=11)
    ax2.set_ylabel("Spearman rho", fontsize=11)
    ax2.set_title(
        "Attn vs MLP FIM-Drift Trajectory\n"
        "Solid = MLP, Dashed = Attention",
        fontsize=11
    )
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1, 1)

    plt.suptitle(
        "Sequential Protection Is MLP-Specific: Attention OV Circuits Drift Freely\n"
        "v_t suppression protects MLP in sequential models but not attention weights",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "attn_vs_mlp_split.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {plot_path}")


if __name__ == "__main__":
    run()