"""
Experiment D: Circuit Node vs Edge Stability
=============================================
Based on: "Towards Understanding Fine-Tuning Mechanisms via Circuit Analysis" (2025)
Key finding: circuit NODES stay stable but EDGES change significantly after fine-tuning.

Your question: Does the same hold for recursive collapse, or does collapse also
destroy node stability? This is the clearest way to distinguish recursive collapse
from standard fine-tuning at the circuit level.

Definitions:
  NODE = an attention head or MLP layer (its existence and basic function)
  EDGE = the information flow between nodes (captured by weight matrices)

  Node stability: measured by how much the embedding of each head changes.
    We use the singular value spectrum of each head's weight matrix.
    If the top singular values stay the same, the head is doing the same thing.

  Edge stability: measured by how much the composition between heads changes.
    Composition = how much one head's output affects another head's key/query.
    K-composition = head B's key is computed from head A's output.
    If K-composition changes, the "routing" between heads has changed.

Why this matters:
  If nodes are stable but edges change → the collapse is a routing problem,
  not a representation problem. The individual heads still know what to do
  but they're no longer talking to each other correctly.

  If nodes also change → collapse is more fundamental.

This directly extends the fine-tuning circuit analysis literature to the
recursive self-distillation setting, which is your unique contribution.

Usage:
  python scripts/node_edge_stability.py
"""

import os
import gc
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM

BASE_DIR = r"D:\Thaman\Work\hessian-spectral-analysis"

MODELS = {
    "GPT2": {
        "arch": "sequential",
        "checkpoints": {
            0: "gpt2",
            1: os.path.join(BASE_DIR, "models", "gpt2_treatment_gen_1"),
            3: os.path.join(BASE_DIR, "models", "gpt2_treatment_gen_3"),
            5: os.path.join(BASE_DIR, "models", "gpt2_treatment_gen_5"),
        },
        "n_layers": 12, "n_heads": 12, "d_model": 768, "d_head": 64,
    },
    "Pythia": {
        "arch": "parallel",
        "checkpoints": {
            0: "EleutherAI/pythia-1.4b",
            1: os.path.join(BASE_DIR, "models", "pythia-1.4b_treatment_gen_1"),
            3: os.path.join(BASE_DIR, "models", "pythia-1.4b_treatment_gen_3"),
            5: os.path.join(BASE_DIR, "models", "pythia-1.4b_treatment_gen_5"),
        },
        "n_layers": 24, "n_heads": 16, "d_model": 2048, "d_head": 128,
    },
}

OUTPUT_DIR = os.path.join(BASE_DIR, "results", "circuit_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_head_weight_matrices(model, model_name):
    """
    Extract Q, K, V, O weight matrices per head per layer.
    Returns dict: {(layer, head, 'Q'/'K'/'V'/'O'): weight_matrix}
    """
    sd = dict(model.named_parameters())
    matrices = {}

    if model_name == "GPT2":
        n_layers = MODELS["GPT2"]["n_layers"]
        n_heads  = MODELS["GPT2"]["n_heads"]
        d_model  = MODELS["GPT2"]["d_model"]
        d_head   = MODELS["GPT2"]["d_head"]

        for layer in range(n_layers):
            c_attn = sd.get(f"transformer.h.{layer}.attn.c_attn.weight")
            c_proj = sd.get(f"transformer.h.{layer}.attn.c_proj.weight")
            if c_attn is None:
                continue

            # GPT-2 Conv1D weight shape: [in_features, out_features] = [d_model, 3*d_model]
            # NOT transposed like standard Linear [out, in]
            # Q cols [0:d_model], K cols [d_model:2*d_model], V cols [2*d_model:3*d_model]
            c_attn = c_attn.float()   # [d_model, 3*d_model]
            W_Q_all = c_attn[:, :d_model]           # [d_model, d_model]
            W_K_all = c_attn[:, d_model:2*d_model]  # [d_model, d_model]
            W_V_all = c_attn[:, 2*d_model:]         # [d_model, d_model]
            # c_proj: [d_model, d_model] = [n_heads*d_head, d_model] in Conv1D
            W_O_all = c_proj.float() if c_proj is not None else None

            for head in range(n_heads):
                # W_Q, W_K, W_V: slice columns for this head → [d_model, d_head]
                matrices[(layer, head, 'Q')] = W_Q_all[:, head*d_head:(head+1)*d_head].detach().numpy()
                matrices[(layer, head, 'K')] = W_K_all[:, head*d_head:(head+1)*d_head].detach().numpy()
                matrices[(layer, head, 'V')] = W_V_all[:, head*d_head:(head+1)*d_head].detach().numpy()
                # W_O: slice rows for this head → [d_head, d_model]
                if W_O_all is not None:
                    matrices[(layer, head, 'O')] = W_O_all[head*d_head:(head+1)*d_head, :].detach().numpy()

    elif model_name == "Pythia":
        n_layers = MODELS["Pythia"]["n_layers"]
        n_heads  = MODELS["Pythia"]["n_heads"]
        d_model  = MODELS["Pythia"]["d_model"]
        d_head   = MODELS["Pythia"]["d_head"]

        for layer in range(n_layers):
            qkv   = sd.get(f"gpt_neox.layers.{layer}.attention.query_key_value.weight")
            dense = sd.get(f"gpt_neox.layers.{layer}.attention.dense.weight")
            if qkv is None:
                continue

            qkv = qkv.float()   # [3*d_model, d_model]
            W_Q_all = qkv[:d_model, :]
            W_K_all = qkv[d_model:2*d_model, :]
            W_V_all = qkv[2*d_model:, :]
            W_O_all = dense.float() if dense is not None else None

            for head in range(n_heads):
                matrices[(layer, head, 'Q')] = W_Q_all[head*d_head:(head+1)*d_head, :].detach().numpy()
                matrices[(layer, head, 'K')] = W_K_all[head*d_head:(head+1)*d_head, :].detach().numpy()
                matrices[(layer, head, 'V')] = W_V_all[head*d_head:(head+1)*d_head, :].detach().numpy()
                if W_O_all is not None:
                    matrices[(layer, head, 'O')] = W_O_all[:, head*d_head:(head+1)*d_head].detach().numpy()

    return matrices


def node_stability(mats_gen0, mats_genN):
    """
    Node stability: compare singular value spectra of each head's weight matrices.
    If SVD spectrum is stable → node is stable (doing the same computation).

    Metric: cosine similarity between top-k singular value vectors.
    High cosine sim = stable node. Low = node has changed function.
    """
    node_similarities = {}

    for key in mats_gen0:
        if key not in mats_genN:
            continue
        layer, head, mat_type = key
        M0 = mats_gen0[key]
        MN = mats_genN[key]

        # SVD of each matrix
        try:
            _, s0, _ = np.linalg.svd(M0, full_matrices=False)
            _, sN, _ = np.linalg.svd(MN, full_matrices=False)
        except np.linalg.LinAlgError:
            continue

        # Keep top-k singular values
        k = min(10, len(s0), len(sN))
        s0_top = s0[:k]
        sN_top = sN[:k]

        # Cosine similarity of singular value vectors
        norm0 = np.linalg.norm(s0_top)
        normN = np.linalg.norm(sN_top)
        if norm0 < 1e-8 or normN < 1e-8:
            continue
        cos_sim = np.dot(s0_top, sN_top) / (norm0 * normN)
        node_similarities[(layer, head, mat_type)] = float(cos_sim)

    return node_similarities


def edge_stability(mats_gen0, mats_genN, model_name):
    """
    Edge stability: K-composition between heads.
    K-composition of head B from head A = ||W_OV_A @ W_K_B||_F / ||W_K_B||_F
    where W_OV_A = W_V_A @ W_O_A (writing circuit of head A)

    If this value changes between Gen0 and GenN, the routing from A to B has changed.
    This is the EDGE drift metric.
    """
    config = MODELS[model_name]
    n_layers = config["n_layers"]
    n_heads  = config["n_heads"]

    edge_drifts = {}

    # Only compute within-layer (adjacent layer) composition for efficiency
    for layer in range(1, n_layers):
        for head_A in range(n_heads):
            # OV circuit of head A in previous layer
            V0_A = mats_gen0.get((layer-1, head_A, 'V'))
            O0_A = mats_gen0.get((layer-1, head_A, 'O'))
            VN_A = mats_genN.get((layer-1, head_A, 'V'))
            ON_A = mats_genN.get((layer-1, head_A, 'O'))

            if any(x is None for x in [V0_A, O0_A, VN_A, ON_A]):
                continue

            # OV matrices
            OV_0 = V0_A.T @ O0_A.T   # [d_model, d_model]
            OV_N = VN_A.T @ ON_A.T

            for head_B in range(n_heads):
                # K matrix of head B in this layer
                K0_B = mats_gen0.get((layer, head_B, 'K'))
                KN_B = mats_genN.get((layer, head_B, 'K'))

                if K0_B is None or KN_B is None:
                    continue

                # K-composition score at Gen0 and GenN
                # Score = ||OV_A @ K_B||_F / ||K_B||_F
                try:
                    kcomp_0 = np.linalg.norm(OV_0 @ K0_B.T, 'fro') / (np.linalg.norm(K0_B) + 1e-8)
                    kcomp_N = np.linalg.norm(OV_N @ KN_B.T, 'fro') / (np.linalg.norm(KN_B) + 1e-8)
                except Exception:
                    continue

                # Edge drift = |score_N - score_0| / score_0
                if kcomp_0 > 1e-6:
                    edge_drift = abs(kcomp_N - kcomp_0) / kcomp_0
                    edge_drifts[(layer-1, head_A, layer, head_B)] = float(edge_drift)

    return edge_drifts


def run_analysis():
    results = {}
    output_path = os.path.join(OUTPUT_DIR, "node_edge_stability.json")

    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        print("Loaded existing node/edge stability results.")

    for model_name, config in MODELS.items():
        if model_name in results:
            print(f"{model_name}: cached.")
            continue

        print(f"\n{'='*60}")
        print(f"Node/Edge Stability: {model_name} ({config['arch']})")
        print(f"{'='*60}")

        print("  Loading Gen0...")
        try:
            m0 = AutoModelForCausalLM.from_pretrained(
                config["checkpoints"][0],
                torch_dtype=torch.float32, device_map="cpu"
            )
            mats_gen0 = get_head_weight_matrices(m0, model_name)
            del m0; gc.collect()
        except Exception as e:
            print(f"  Gen0 failed: {e}")
            continue

        model_results = {}
        for gen, ckpt in config["checkpoints"].items():
            if gen == 0:
                continue
            if not os.path.exists(str(ckpt)):
                continue

            print(f"  Gen{gen}...")
            try:
                mN = AutoModelForCausalLM.from_pretrained(
                    ckpt, torch_dtype=torch.float32, device_map="cpu"
                )
                mats_genN = get_head_weight_matrices(mN, model_name)
                del mN; gc.collect()
            except Exception as e:
                print(f"  Gen{gen} failed: {e}")
                continue

            node_sims  = node_stability(mats_gen0, mats_genN)
            edge_drifts = edge_stability(mats_gen0, mats_genN, model_name)

            # Summary stats
            node_mean = float(np.mean(list(node_sims.values()))) if node_sims else float('nan')
            edge_mean = float(np.mean(list(edge_drifts.values()))) if edge_drifts else float('nan')

            model_results[str(gen)] = {
                "node_mean_cosine_sim": node_mean,
                "edge_mean_drift": edge_mean,
                "n_nodes": len(node_sims),
                "n_edges": len(edge_drifts),
            }

            print(f"  Gen{gen}: node_stability={node_mean:.4f}, edge_drift={edge_mean:.4f}")

        results[model_name] = model_results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    return results


def plot_and_interpret(results):
    print("\n" + "="*60)
    print("NODE vs EDGE STABILITY ACROSS GENERATIONS")
    print("="*60)
    print("Literature benchmark (standard fine-tuning):")
    print("  Node stability (cosine sim): stays HIGH (~0.99)")
    print("  Edge drift: increases significantly")
    print()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors_node = {'GPT2': 'steelblue', 'Pythia': 'tomato'}
    colors_edge = {'GPT2': 'navy', 'Pythia': 'darkred'}

    gens = [1, 3, 5]

    for ax_idx, metric in enumerate(['node_mean_cosine_sim', 'edge_mean_drift']):
        ax = axes[ax_idx]
        for model_name, model_results in results.items():
            arch = MODELS[model_name]["arch"]
            y_vals = [model_results.get(str(g), {}).get(metric, np.nan) for g in gens]
            color = colors_node[model_name] if ax_idx == 0 else colors_edge[model_name]
            ax.plot(gens, y_vals, 'o-', color=color,
                   label=f"{model_name} ({arch})", linewidth=2, markersize=8)

            print(f"{model_name} ({arch}) - {metric}:")
            for g, v in zip(gens, y_vals):
                print(f"  Gen{g}: {v:.4f}")

        ax.set_xlabel("Generation", fontsize=12)
        if ax_idx == 0:
            ax.set_ylabel("Mean Cosine Sim (higher=more stable)", fontsize=11)
            ax.set_title("NODE Stability\n(Do heads preserve their function?)", fontsize=12)
            ax.set_ylim(0, 1.05)
            ax.axhline(y=0.99, color='green', linestyle='--', alpha=0.5,
                      label='Fine-tuning benchmark (~0.99)')
        else:
            ax.set_ylabel("Mean Edge Drift (higher=more change)", fontsize=11)
            ax.set_title("EDGE Stability\n(Does routing between heads change?)", fontsize=12)

        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Node vs Edge Stability in Recursive Self-Distillation\n"
        "Does recursive collapse behave like fine-tuning (node-stable, edge-changing)?",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "node_edge_stability.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {plot_path}")

    print("\nINTERPRETATION:")
    for model_name, model_results in results.items():
        arch = MODELS[model_name]["arch"]
        n1 = model_results.get("1", {}).get("node_mean_cosine_sim", np.nan)
        n5 = model_results.get("5", {}).get("node_mean_cosine_sim", np.nan)
        e1 = model_results.get("1", {}).get("edge_mean_drift", np.nan)
        e5 = model_results.get("5", {}).get("edge_mean_drift", np.nan)

        if not np.isnan(n5) and not np.isnan(e5):
            node_stable = n5 > 0.95
            edge_changed = e5 > e1 * 1.5 if not np.isnan(e1) else False
            print(f"\n{model_name} ({arch}):")
            print(f"  Node stability Gen5: {n5:.4f} ({'STABLE like fine-tuning' if node_stable else 'UNSTABLE — collapse destroys nodes'})")
            print(f"  Edge drift Gen5: {e5:.4f} ({'INCREASED like fine-tuning' if edge_changed else 'stable'})")
            if node_stable and edge_changed:
                print(f"  → SAME PATTERN as fine-tuning: nodes stable, edges change")
            elif not node_stable:
                print(f"  → DIFFERENT from fine-tuning: recursive collapse also destroys node function")


if __name__ == "__main__":
    print("Node vs Edge Stability Analysis")
    print("Compares recursive collapse circuit dynamics to standard fine-tuning")
    print()
    results = run_analysis()
    plot_and_interpret(results)