"""
Experiment C: OV Circuit Drift Analysis
=========================================
Based on: "Circuits Updates - July 2025" (Anthropic) and
          "Towards Understanding Fine-Tuning Mechanisms via Circuit Analysis" (2025)

Key finding from literature: circuit NODES stay stable across fine-tuning but
EDGES (the OV and QK weight matrices that determine what each head reads and writes)
undergo significant changes.

Your setting is different: recursive self-distillation is NOT standard fine-tuning.
The question is whether the same node-stable/edge-changing pattern holds here,
or whether recursive collapse on synthetic data breaks node stability too.

What this experiment measures:
  The OV circuit of an attention head is W_V @ W_O (value followed by output projection).
  This matrix determines: given a token the head attends to, what does it write to
  the residual stream? If the OV matrix is stable, the head's "writing behaviour"
  is preserved. If it changes, the head is doing fundamentally different things.

  The QK circuit is W_Q @ W_K.T (query-key interaction).
  This matrix determines: which tokens does the head attend to?
  If QK changes, the head is looking at different things in context.

Prediction from FIM paper:
  In SEQUENTIAL models: high-FIM blocks have suppressed drift → OV circuits should
  be more stable in high-FIM blocks than low-FIM blocks.

  In PARALLEL models: high-FIM blocks drift more → OV circuits in high-FIM blocks
  should show MORE change than low-FIM blocks.

  This would be a circuit-level confirmation of the FIM paper's block-level finding.

Architecture note:
  GPT-2: W_V has shape [d_model, d_head], W_O has shape [n_heads * d_head, d_model]
  Pythia: uses fused QKV projection
  We compute the effective OV matrix per head and measure its drift via
  normalized Frobenius norm (same metric as the FIM paper for consistency).

Usage:
  python scripts/ov_circuit_drift.py
"""

import os
import gc
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM

BASE_DIR = r"D:\Thaman\Work\hessian-spectral-analysis"

MODELS = {
    "GPT2": {
        "arch": "sequential",
        "checkpoints": {
            0: "gpt2",
            1: os.path.join(BASE_DIR, "models", "gpt2_treatment_gen_1"),
            2: os.path.join(BASE_DIR, "models", "gpt2_treatment_gen_2"),
            3: os.path.join(BASE_DIR, "models", "gpt2_treatment_gen_3"),
            4: os.path.join(BASE_DIR, "models", "gpt2_treatment_gen_4"),
            5: os.path.join(BASE_DIR, "models", "gpt2_treatment_gen_5"),
        },
        "n_layers": 12,
        "n_heads": 12,
        "d_model": 768,
        "d_head": 64,
    },
    "Pythia": {
        "arch": "parallel",
        "checkpoints": {
            0: "EleutherAI/pythia-1.4b",
            1: os.path.join(BASE_DIR, "models", "pythia-1.4b_treatment_gen_1"),
            2: os.path.join(BASE_DIR, "models", "pythia-1.4b_treatment_gen_2"),
            3: os.path.join(BASE_DIR, "models", "pythia-1.4b_treatment_gen_3"),
            4: os.path.join(BASE_DIR, "models", "pythia-1.4b_treatment_gen_4"),
            5: os.path.join(BASE_DIR, "models", "pythia-1.4b_treatment_gen_5"),
        },
        "n_layers": 24,
        "n_heads": 16,
        "d_model": 2048,
        "d_head": 128,
    },
}

# Gen0 FIM per block (from original paper)
GEN0_FIM = {
    "GPT2": {},   # loaded from file
    "Pythia": {
        0: 545.3385, 1: 1340.6135, 2: 1008.3357, 3: 1000.3466,
        4: 918.8504, 5: 818.8515, 6: 568.5288, 7: 665.3354,
        8: 669.4224, 9: 656.7953, 10: 508.1771, 11: 392.9493,
        12: 248.4828, 13: 203.8360, 14: 160.7173, 15: 694.9300,
        16: 93.7418, 17: 119.3484, 18: 111.0569, 19: 133.6137,
        20: 146.9509, 21: 125.1650, 22: 162.9082, 23: 128.3309,
    },
}

OUTPUT_DIR = os.path.join(BASE_DIR, "results", "circuit_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_fim_gpt2():
    """Load GPT-2 Gen0 FIM from file."""
    paths = [
        os.path.join(BASE_DIR, "results", "gpt2_treatment_gen_0", "perblock_fim.json"),
        os.path.join(BASE_DIR, "results", "fimgpt2_gen_0.txt"),
    ]
    import re
    for path in paths:
        if not os.path.exists(path):
            continue
        if path.endswith(".json"):
            with open(path) as f:
                data = json.load(f)
            result = {}
            for entry in data.get("blocks", []):
                b = entry.get("block_idx", 0)
                attn = entry.get("attention", {})
                mlp = entry.get("mlp", {})
                at = float(attn.get("top", 0)) if isinstance(attn, dict) and "error" not in attn else 0.0
                mt = float(mlp.get("top", 0)) if isinstance(mlp, dict) and "error" not in mlp else 0.0
                if at > 0 or mt > 0:
                    result[b] = at + mt
            if result:
                return result
        elif path.endswith(".txt"):
            for enc in ["utf-16", "utf-8"]:
                try:
                    with open(path, encoding=enc) as f:
                        content = f.read()
                    result = {}
                    for line in content.split("\n"):
                        m = re.match(r"Block\s+(\d+)\s*\|\s*Attn:\s*([\d.]+)\s*\|\s*MLP:\s*([\d.]+)", line)
                        if m:
                            result[int(m.group(1))] = float(m.group(2)) + float(m.group(3))
                    if result:
                        return result
                except UnicodeError:
                    continue
    return {}


def extract_ov_matrices(model, model_name):
    """
    Extract the OV matrix per head per layer.
    OV = W_V @ W_O (effective writing matrix of each head)

    Returns dict: {(layer, head): OV_matrix as numpy array}
    """
    ov_matrices = {}
    state_dict = dict(model.named_parameters())

    if model_name == "GPT2":
        # GPT-2 uses Conv1D: weight shape is [in_features, out_features]
        # i.e. TRANSPOSED compared to standard nn.Linear [out, in]
        # c_attn.weight: [d_model, 3*d_model] → Q=[0:d], K=[d:2d], V=[2d:3d] along dim=1
        # c_proj.weight: [d_model, d_model]
        n_layers = MODELS["GPT2"]["n_layers"]
        n_heads  = MODELS["GPT2"]["n_heads"]
        d_model  = MODELS["GPT2"]["d_model"]
        d_head   = MODELS["GPT2"]["d_head"]

        for layer in range(n_layers):
            c_attn_key = f"transformer.h.{layer}.attn.c_attn.weight"
            c_proj_key = f"transformer.h.{layer}.attn.c_proj.weight"

            if c_attn_key not in state_dict or c_proj_key not in state_dict:
                continue

            # Conv1D weights: [in, out] — need to transpose for standard W @ x
            c_attn = state_dict[c_attn_key].float()  # [d_model, 3*d_model]
            c_proj = state_dict[c_proj_key].float()  # [d_model, d_model]

            # W_V: columns [2*d_model : 3*d_model], shape [d_model, d_model]
            # For head h: W_V_h = c_attn[:, 2*d_model + h*d_head : 2*d_model + (h+1)*d_head]
            # Shape: [d_model, d_head] — this maps input to value
            W_V_all = c_attn[:, 2*d_model:]  # [d_model, n_heads*d_head]

            # W_O (c_proj): [d_model, d_model] in Conv1D = [in=n_heads*d_head, out=d_model]
            # For head h: W_O_h = c_proj[h*d_head:(h+1)*d_head, :] shape [d_head, d_model]
            W_O = c_proj  # [n_heads*d_head, d_model]

            for head in range(n_heads):
                W_V_h = W_V_all[:, head*d_head:(head+1)*d_head]   # [d_model, d_head]
                W_O_h = W_O[head*d_head:(head+1)*d_head, :]        # [d_head, d_model]
                # OV circuit: what does the head write to residual stream?
                # OV = W_V_h @ W_O_h : [d_model, d_model]
                # This is correct: input (d_model) -> V projection (d_head) -> O projection (d_model)
                OV = (W_V_h @ W_O_h).detach().numpy()
                ov_matrices[(layer, head)] = OV

    elif model_name == "Pythia":
        # Pythia (GPT-NeoX): query_key_value has shape [d_model, 3*d_model]
        #                    dense (output proj) has shape [d_model, d_model]
        n_layers = MODELS["Pythia"]["n_layers"]
        n_heads  = MODELS["Pythia"]["n_heads"]
        d_model  = MODELS["Pythia"]["d_model"]
        d_head   = MODELS["Pythia"]["d_head"]

        for layer in range(n_layers):
            qkv_key   = f"gpt_neox.layers.{layer}.attention.query_key_value.weight"
            dense_key = f"gpt_neox.layers.{layer}.attention.dense.weight"

            if qkv_key not in state_dict or dense_key not in state_dict:
                continue

            qkv   = state_dict[qkv_key].float()   # [3*d_model, d_model]
            dense = state_dict[dense_key].float()  # [d_model, d_model]

            # V part: rows [2*d_model : 3*d_model]
            W_V_all = qkv[2*d_model:, :]   # [d_model, d_model]
            W_O     = dense                  # [d_model, d_model]

            for head in range(n_heads):
                W_V_h = W_V_all[head*d_head:(head+1)*d_head, :]  # [d_head, d_model]
                W_O_h = W_O[:, head*d_head:(head+1)*d_head]       # [d_model, d_head]
                # OV = W_V_h.T @ W_O_h.T: [d_model, d_model]
                OV = (W_V_h.T @ W_O_h.T).detach().numpy()
                ov_matrices[(layer, head)] = OV

    return ov_matrices


def compute_ov_drift(ov_gen0, ov_genN):
    """
    Compute normalized Frobenius drift for each (layer, head) OV matrix.
    Same normalization as the FIM paper: ||OV_N - OV_0||_F / ||OV_0||_F
    """
    drifts = {}
    for key in ov_gen0:
        if key not in ov_genN:
            continue
        m0 = ov_gen0[key]
        mN = ov_genN[key]
        norm_base = np.linalg.norm(m0, 'fro')
        if norm_base < 1e-8:
            continue
        drift = np.linalg.norm(mN - m0, 'fro') / norm_base
        drifts[key] = float(drift)
    return drifts


def run_analysis():
    # Load GPT-2 FIM
    GEN0_FIM["GPT2"] = load_fim_gpt2()
    if not GEN0_FIM["GPT2"]:
        print("WARNING: GPT-2 Gen0 FIM not found. Correlation for GPT-2 will be skipped.")

    results = {}
    output_path = os.path.join(OUTPUT_DIR, "ov_circuit_drift.json")

    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        print("Loaded existing OV drift results.")

    for model_name, config in MODELS.items():
        if model_name in results:
            print(f"{model_name}: cached.")
            continue

        print(f"\n{'='*60}")
        print(f"OV Circuit Drift: {model_name} ({config['arch']})")
        print(f"{'='*60}")

        # Load Gen0
        print("  Loading Gen0...")
        try:
            m0 = AutoModelForCausalLM.from_pretrained(
                config["checkpoints"][0],
                torch_dtype=torch.float32,
                device_map="cpu",
            )
            ov_gen0 = extract_ov_matrices(m0, model_name)
            del m0; gc.collect()
        except Exception as e:
            print(f"  Gen0 load failed: {e}")
            continue

        model_results = {}
        for gen in range(1, 6):
            ckpt = config["checkpoints"].get(gen)
            if not ckpt or not os.path.exists(str(ckpt)):
                print(f"  Gen{gen}: not found, skipping.")
                continue

            print(f"  Loading Gen{gen}...")
            try:
                mN = AutoModelForCausalLM.from_pretrained(
                    ckpt,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                )
                ov_genN = extract_ov_matrices(mN, model_name)
                del mN; gc.collect()
            except Exception as e:
                print(f"  Gen{gen} failed: {e}")
                continue

            drifts = compute_ov_drift(ov_gen0, ov_genN)
            # Convert tuple keys to strings for JSON
            model_results[str(gen)] = {
                f"{l}_{h}": v for (l, h), v in drifts.items()
            }
            print(f"  Gen{gen}: computed {len(drifts)} head OV drifts")

        results[model_name] = model_results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    return results


def analyze_and_plot(results):
    from scipy.stats import spearmanr

    print("\n" + "="*60)
    print("OV CIRCUIT DRIFT: PER-BLOCK ANALYSIS")
    print("="*60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, (model_name, model_results) in enumerate(results.items()):
        config = MODELS[model_name]
        arch = config["arch"]
        n_layers = config["n_layers"]
        n_heads = config["n_heads"]
        fim_data = GEN0_FIM.get(model_name, {})

        if model_name == "GPT2":
            fim_data = load_fim_gpt2()

        # Per-block mean OV drift at Gen5
        gen5_data = model_results.get("5", {})
        if not gen5_data:
            print(f"  {model_name}: no Gen5 data")
            continue

        # Aggregate per block (mean across heads)
        block_ov_drifts = {}
        for key, val in gen5_data.items():
            layer = int(key.split("_")[0])
            if layer not in block_ov_drifts:
                block_ov_drifts[layer] = []
            block_ov_drifts[layer].append(val)

        blocks = sorted(block_ov_drifts.keys())
        mean_drifts = [np.mean(block_ov_drifts[b]) for b in blocks]
        fim_vals    = [fim_data.get(b, np.nan) for b in blocks]

        # Trajectory plot (top row)
        ax_traj = axes[0, col]
        for gen_key in ["1", "2", "3", "4", "5"]:
            gen_data = model_results.get(gen_key, {})
            if not gen_data:
                continue
            gen_block_drifts = {}
            for key, val in gen_data.items():
                layer = int(key.split("_")[0])
                if layer not in gen_block_drifts:
                    gen_block_drifts[layer] = []
                gen_block_drifts[layer].append(val)
            gen_blocks = sorted(gen_block_drifts.keys())
            gen_means = [np.mean(gen_block_drifts[b]) for b in gen_blocks]
            ax_traj.plot(gen_blocks, gen_means, 'o-', label=f"Gen{gen_key}",
                        linewidth=1.5, markersize=3, alpha=0.8)

        ax_traj.set_xlabel("Block", fontsize=11)
        ax_traj.set_ylabel("Mean OV Drift (Fro norm)", fontsize=11)
        ax_traj.set_title(f"{model_name} ({arch})\nOV Circuit Drift Per Block", fontsize=11)
        ax_traj.legend(fontsize=8)
        ax_traj.grid(True, alpha=0.3)

        # FIM correlation scatter (bottom row)
        ax_corr = axes[1, col]
        valid = [(f, d) for f, d in zip(fim_vals, mean_drifts) if not np.isnan(f)]
        if valid:
            f_vals, d_vals = zip(*valid)
            log_f = np.log10(np.array(f_vals) + 1e-8)
            ax_corr.scatter(log_f, d_vals, alpha=0.7, s=60,
                           color='steelblue' if arch == 'sequential' else 'tomato',
                           edgecolors='k')
            rho, p = spearmanr(f_vals, d_vals)
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            ax_corr.set_title(
                f"{model_name}: FIM vs OV Drift at Gen5\n"
                f"rho={rho:+.3f}{sig} p={p:.4f}",
                fontsize=11
            )
            # Annotate blocks
            for b, f, d in zip(blocks, fim_vals, mean_drifts):
                if not np.isnan(f):
                    ax_corr.annotate(str(b), (np.log10(f + 1e-8), d),
                                    fontsize=7, ha='center', alpha=0.6)

            print(f"\n{model_name} ({arch}):")
            print(f"  OV Drift ~ FIM: rho={rho:+.3f}{sig} p={p:.4f}")
            print(f"  Expected: {'rho < 0 (high FIM protected)' if arch == 'sequential' else 'rho > 0 (high FIM drifts more)'}")
            print(f"  {'CONFIRMED' if (arch == 'sequential' and rho < -0.3) or (arch == 'parallel' and rho > 0.3) else 'NOT CONFIRMED or WEAK'}")

        ax_corr.set_xlabel("log10(FIM_b) Gen0", fontsize=11)
        ax_corr.set_ylabel("Mean OV Drift Gen5", fontsize=11)
        ax_corr.grid(True, alpha=0.3)

    plt.suptitle(
        "OV Circuit Drift Across Recursive Self-Distillation Generations\n"
        "Circuit-level test of FIM paper: does high FIM predict OV stability?",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "ov_circuit_drift.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {plot_path}")


if __name__ == "__main__":
    print("OV Circuit Drift Analysis")
    print("Measures how each attention head's writing behaviour changes across generations")
    print()
    results = run_analysis()
    GEN0_FIM["GPT2"] = load_fim_gpt2()
    analyze_and_plot(results)