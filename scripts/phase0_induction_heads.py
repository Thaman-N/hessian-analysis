"""
Phase 0: Induction Head Feasibility Check
==========================================
Run this FIRST before committing to the full circuit analysis.

What it does:
- Loads your existing SmolLM2 and GPT-2 checkpoints (Gen0-Gen5)
- Computes induction head score for every attention head at every generation
- Plots the trajectory of the top induction heads across generations
- Outputs a JSON with all scores

Decision gate:
  - If TOP induction head scores drop by >20% from Gen0 to Gen5 → interesting,
    proceed to full analysis
  - If scores stay within 10% of Gen0 → circuits are preserved, stop here

Usage:
  cd D:/Thaman/Work/hessian-spectral-analysis
  python scripts/phase0_induction_heads.py

Requirements:
  pip install transformer_lens matplotlib

Critical note on architecture:
  SmolLM2-135M uses LlamaAttention with RoPE. Classical induction heads were
  characterized in GPT-2 which uses learned positional embeddings. RoPE changes
  how attention patterns work so classical induction head detection may behave
  differently. GPT-2 results are the primary comparison; SmolLM is secondary.
"""

import os
import gc
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


try:
    from transformer_lens import HookedTransformer
    TL_AVAILABLE = True
    print("TransformerLens available.")
except ImportError:
    TL_AVAILABLE = False
    print("TransformerLens not found. Using HuggingFace fallback.")
    # from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# CONFIG — update paths to match your setup
# ============================================================
BASE_DIR = r"D:\Thaman\Work\hessian-spectral-analysis"

MODELS = {
    "SmolLM": {
        "hf_id": "HuggingFaceTB/SmolLM2-135M",
        "checkpoints": {
            0: "HuggingFaceTB/SmolLM2-135M",
            1: os.path.join(BASE_DIR, "models", "treatment_gen_1"),
            2: os.path.join(BASE_DIR, "models", "treatment_gen_2"),
            3: os.path.join(BASE_DIR, "models", "treatment_gen_3"),
            4: os.path.join(BASE_DIR, "models", "treatment_gen_4"),
            5: os.path.join(BASE_DIR, "models", "treatment_gen_5"),
        },
        "arch": "sequential",
        "n_layers": 30,
        "n_heads": 9,
        "tl_name": "HuggingFaceTB/SmolLM2-135M",  # may need adjustment
    },
    "GPT2": {
        "hf_id": "gpt2",
        "checkpoints": {
            0: "gpt2",
            1: os.path.join(BASE_DIR, "models", "gpt2_treatment_gen_1"),
            2: os.path.join(BASE_DIR, "models", "gpt2_treatment_gen_2"),
            3: os.path.join(BASE_DIR, "models", "gpt2_treatment_gen_3"),
            4: os.path.join(BASE_DIR, "models", "gpt2_treatment_gen_4"),
            5: os.path.join(BASE_DIR, "models", "gpt2_treatment_gen_5"),
        },
        "arch": "sequential",
        "n_layers": 12,
        "n_heads": 12,
        "tl_name": "gpt2",  # GPT-2 is natively supported in TransformerLens
    },
    "Pythia": {
        "hf_id": "EleutherAI/pythia-1.4b",
        "checkpoints": {
            0: "EleutherAI/pythia-1.4b",
            1: os.path.join(BASE_DIR, "models", "pythia-1.4b_treatment_gen_1"),
            2: os.path.join(BASE_DIR, "models", "pythia-1.4b_treatment_gen_2"),
            3: os.path.join(BASE_DIR, "models", "pythia-1.4b_treatment_gen_3"),
            4: os.path.join(BASE_DIR, "models", "pythia-1.4b_treatment_gen_4"),
            5: os.path.join(BASE_DIR, "models", "pythia-1.4b_treatment_gen_5"),
        },
        "arch": "parallel",
        "n_layers": 24,
        "n_heads": 16,
        "tl_name": "EleutherAI/pythia-1.4b",
    },
}

OUTPUT_DIR = os.path.join(BASE_DIR, "results", "circuit_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 50  # length of the repeated sequence for induction head detection


# ============================================================
# INDUCTION HEAD DETECTION — TransformerLens path
# ============================================================

def compute_induction_scores_tl(model_path, n_layers, n_heads, tl_name):
    """
    Compute induction head score for every head using TransformerLens.

    Method: Create a random sequence and repeat it.
    [A B C D E ... A B C D E ...]
    An induction head at position i in the second half attends to position i-SEQ_LEN.
    Score = mean attention weight at the "copy" offset.

    Returns: dict {(layer, head): score}
    """
    print(f"  Loading via TransformerLens: {model_path}")

    # Try loading — first try with the checkpoint path, then fall back to base
    try:
        if os.path.exists(str(model_path)):
            # Local checkpoint: load base then override with checkpoint weights
            model = HookedTransformer.from_pretrained(
                tl_name,
                hf_model=AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32
                )
            )
        else:
            model = HookedTransformer.from_pretrained(model_path)
    except Exception as e:
        print(f"  TransformerLens load failed: {e}")
        return None

    model = model.to(DEVICE)
    model.eval()

    # Generate random sequence of length SEQ_LEN, then repeat it
    vocab_size = model.cfg.d_vocab
    rand_seq = torch.randint(50, vocab_size - 50, (1, SEQ_LEN)).to(DEVICE)
    repeated_seq = torch.cat([rand_seq, rand_seq], dim=1)  # shape [1, 2*SEQ_LEN]

    scores = {}
    with torch.no_grad():
        _, cache = model.run_with_cache(repeated_seq)

        for layer in range(n_layers):
            attn_key = f"blocks.{layer}.attn.hook_pattern"
            if attn_key not in cache:
                continue

            # attn_pattern shape: [batch, n_heads, seq_len, seq_len]
            attn = cache[attn_key][0]  # [n_heads, seq_len, seq_len]

            for head in range(n_heads):
                # For positions SEQ_LEN to 2*SEQ_LEN-1,
                # check attention to position i - SEQ_LEN
                # This is the "copy" offset for induction heads
                copy_attentions = []
                for pos in range(SEQ_LEN, 2 * SEQ_LEN):
                    copy_pos = pos - SEQ_LEN
                    copy_attentions.append(attn[head, pos, copy_pos].item())

                scores[(layer, head)] = float(np.mean(copy_attentions))

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return scores


# ============================================================
# INDUCTION HEAD DETECTION — HuggingFace fallback path
# (for when TransformerLens doesn't support the architecture)
# ============================================================

def compute_induction_scores_hf(model_path, n_layers, n_heads):
    """
    HuggingFace fallback for induction head detection.
    Uses output_attentions=True to get attention patterns directly.

    Note: This works for most HuggingFace models but attention patterns
    may be returned in different formats. Check output shape carefully.
    """
    print(f"  Loading via HuggingFace: {model_path}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            attn_implementation="eager",  # needed for output_attentions
            output_attentions=True,
        ).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path if os.path.exists(str(model_path)) else model_path
        )
    except Exception as e:
        print(f"  HuggingFace load failed: {e}")
        return None

    model.eval()

    # Use token IDs directly to avoid tokenizer issues
    vocab_size = model.config.vocab_size
    rand_seq = torch.randint(100, min(vocab_size - 100, 5000), (1, SEQ_LEN)).to(DEVICE)
    repeated_seq = torch.cat([rand_seq, rand_seq], dim=1)

    scores = {}
    with torch.no_grad():
        outputs = model(repeated_seq, output_attentions=True)

    # outputs.attentions is a tuple of length n_layers
    # each element shape: [batch, n_heads, seq_len, seq_len]
    if outputs.attentions is None:
        print("  WARNING: model did not return attentions. Try eager implementation.")
        return None

    for layer_idx, attn_layer in enumerate(outputs.attentions):
        if layer_idx >= n_layers:
            break
        attn = attn_layer[0]  # [n_heads, seq_len, seq_len]

        actual_n_heads = attn.shape[0]
        for head in range(actual_n_heads):
            copy_attentions = []
            for pos in range(SEQ_LEN, 2 * SEQ_LEN):
                copy_pos = pos - SEQ_LEN
                if copy_pos < attn.shape[2] and pos < attn.shape[1]:
                    copy_attentions.append(attn[head, pos, copy_pos].item())

            if copy_attentions:
                scores[(layer_idx, head)] = float(np.mean(copy_attentions))

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return scores


# ============================================================
# MAIN ANALYSIS
# ============================================================

def compute_induction_scores(model_path, n_layers, n_heads, tl_name):
    """Try TransformerLens first, fall back to HuggingFace."""
    if TL_AVAILABLE:
        scores = compute_induction_scores_tl(model_path, n_layers, n_heads, tl_name)
        if scores is not None:
            return scores
        print("  TransformerLens failed, trying HuggingFace fallback...")

    return compute_induction_scores_hf(model_path, n_layers, n_heads)


def run_analysis():
    all_results = {}

    for model_name, config in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({config['arch']})")
        print(f"{'='*60}")

        results_path = os.path.join(OUTPUT_DIR, f"{model_name}_induction_scores.json")

        if os.path.exists(results_path):
            print(f"  Existing results found, loading...")
            with open(results_path) as f:
                model_results = json.load(f)
        else:
            model_results = {}

        for gen, ckpt_path in config["checkpoints"].items():
            gen_key = str(gen)
            if gen_key in model_results:
                print(f"  Gen {gen}: already computed, skipping.")
                continue

            if not (os.path.exists(str(ckpt_path)) or "EleutherAI" in str(ckpt_path)
                    or "HuggingFace" in str(ckpt_path) or ckpt_path == "gpt2"):
                print(f"  Gen {gen}: checkpoint not found at {ckpt_path}, skipping.")
                continue

            print(f"\n  Gen {gen}: {ckpt_path}")
            scores = compute_induction_scores(
                ckpt_path,
                config["n_layers"],
                config["n_heads"],
                config["tl_name"]
            )

            if scores is not None:
                # Convert tuple keys to strings for JSON
                model_results[gen_key] = {
                    f"{l}_{h}": v for (l, h), v in scores.items()
                }
                print(f"  Gen {gen}: computed {len(scores)} head scores")
                print(f"  Top 3 induction heads: "
                      + ", ".join(f"L{l}H{h}={v:.4f}"
                                  for (l, h), v in sorted(
                                      scores.items(), key=lambda x: x[1], reverse=True
                                  )[:3]))
            else:
                print(f"  Gen {gen}: FAILED to compute scores")
                model_results[gen_key] = {}

            # Save after each generation (resume support)
            with open(results_path, 'w') as f:
                json.dump(model_results, f, indent=2)

        all_results[model_name] = model_results

    return all_results


def plot_results(all_results):
    """
    Plot induction head score trajectory across generations.
    For each model, find the top-3 induction heads at Gen0 and track them.
    """
    fig, axes = plt.subplots(1, len(all_results), figsize=(6 * len(all_results), 5))
    if len(all_results) == 1:
        axes = [axes]

    for ax, (model_name, model_results) in zip(axes, all_results.items()):
        config = MODELS[model_name]
        arch = config["arch"]

        # Find top induction heads at Gen0
        gen0 = model_results.get("0", {})
        if not gen0:
            ax.set_title(f"{model_name} — no Gen0 data")
            continue

        # Sort by score at Gen0, take top 5
        sorted_heads = sorted(gen0.items(), key=lambda x: x[1], reverse=True)[:5]
        top_heads = [h for h, _ in sorted_heads]

        # Track these heads across generations
        gens = sorted([int(g) for g in model_results.keys()])
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_heads)))

        for head_key, color in zip(top_heads, colors):
            scores_across_gens = []
            for gen in gens:
                gen_data = model_results.get(str(gen), {})
                score = gen_data.get(head_key, np.nan)
                scores_across_gens.append(score)

            layer, head = head_key.split("_")
            ax.plot(gens, scores_across_gens, 'o-',
                    color=color, label=f"L{layer}H{head}",
                    linewidth=2, markersize=6)

        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Induction Head Score", fontsize=12)
        ax.set_title(f"{model_name} ({arch})\nTop Induction Heads", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # Add text showing Gen0→Gen5 change for best head
        if len(gens) >= 2 and sorted_heads:
            best_head = sorted_heads[0][0]
            gen0_score = model_results.get("0", {}).get(best_head, np.nan)
            gen5_score = model_results.get("5", {}).get(best_head, np.nan)
            if not np.isnan(gen0_score) and not np.isnan(gen5_score):
                change = (gen5_score - gen0_score) / (gen0_score + 1e-8) * 100
                ax.text(0.05, 0.05, f"Best head: {change:+.1f}% Gen0→Gen5",
                        transform=ax.transAxes, fontsize=10,
                        color='red' if change < -20 else 'green')

    plt.suptitle("Induction Head Score Across Recursive Self-Distillation Generations",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "induction_head_trajectory.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    plt.show()


def interpret_results(all_results):
    """
    Print a clear decision output: proceed or stop.
    """
    print("\n" + "="*60)
    print("DECISION GATE: Should we proceed with full circuit analysis?")
    print("="*60)

    decisions = {}
    for model_name, model_results in all_results.items():
        gen0 = model_results.get("0", {})
        gen5 = model_results.get("5", {})

        if not gen0 or not gen5:
            print(f"\n{model_name}: INSUFFICIENT DATA — missing Gen0 or Gen5")
            decisions[model_name] = "insufficient"
            continue

        # Top induction head at Gen0
        best_head = max(gen0.items(), key=lambda x: x[1])
        best_head_key, gen0_score = best_head
        gen5_score = gen5.get(best_head_key, np.nan)

        if np.isnan(gen5_score):
            print(f"\n{model_name}: MISSING Gen5 score for best head {best_head_key}")
            decisions[model_name] = "insufficient"
            continue

        change_pct = (gen5_score - gen0_score) / (gen0_score + 1e-8) * 100

        # Mean over all heads
        common_heads = set(gen0.keys()) & set(gen5.keys())
        mean_change = np.mean([
            (gen5[h] - gen0[h]) / (gen0[h] + 1e-8) * 100
            for h in common_heads
        ])

        arch = MODELS[model_name]["arch"]
        print(f"\n{model_name} ({arch}):")
        print(f"  Best induction head ({best_head_key}): "
              f"Gen0={gen0_score:.4f} → Gen5={gen5_score:.4f} "
              f"({change_pct:+.1f}%)")
        print(f"  Mean change across all heads: {mean_change:+.1f}%")

        if abs(change_pct) > 20:
            print(f"  → SIGNIFICANT CHANGE: proceed with full circuit analysis")
            decisions[model_name] = "proceed"
        elif abs(change_pct) > 10:
            print(f"  → MODERATE CHANGE: worth investigating further")
            decisions[model_name] = "borderline"
        else:
            print(f"  → STABLE: induction heads preserved, circuit-level "
                  f"analysis may not add much")
            decisions[model_name] = "stable"

    print("\n" + "-"*60)
    any_interesting = any(d in ("proceed", "borderline") for d in decisions.values())
    if any_interesting:
        print("VERDICT: Proceed with full circuit analysis.")
        print("At least one model shows significant induction head degradation.")
        print("Run residual_stream_analysis.py next.")
    else:
        print("VERDICT: Circuit-level structure is stable.")
        print("The collapse is not degrading induction circuits significantly.")
        print("The FIM-drift paper is the right level of analysis.")
        print("Consider: is the collapse manifesting as weight direction changes")
        print("without destroying circuit function? That could still be interesting.")

    return decisions


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("Phase 0: Induction Head Feasibility Check")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"TransformerLens available: {TL_AVAILABLE}")
    print(f"Output dir: {OUTPUT_DIR}")
    print()

    all_results = run_analysis()
    plot_results(all_results)
    decisions = interpret_results(all_results)

    # Save final summary
    summary_path = os.path.join(OUTPUT_DIR, "phase0_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            "decisions": decisions,
            "results": {
                model: {
                    gen: {k: v for k, v in heads.items()}
                    for gen, heads in gens.items()
                }
                for model, gens in all_results.items()
            }
        }, f, indent=2)
    print(f"\nFull results saved to: {summary_path}")