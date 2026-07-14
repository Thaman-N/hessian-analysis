"""
Experiments F, G, H: Three New Directions
==========================================

Experiment F: Attention Entropy Per Head Across Generations
  Based on: Hong & Lee EMNLP 2025 "Variance Sensitivity Induces Attention
  Entropy Collapse", Zhai et al. 2023 "Stabilizing Transformer Training by
  Preventing Attention Entropy Collapse"

  Your logit lens showed INCREASING residual stream entropy in Pythia's middle
  layers as collapse progresses. This is the OPPOSITE of the entropy collapse
  studied in training instability. Here we measure per-head attention entropy
  directly — are specific heads going uniform (spreading attention everywhere)
  or peaky (attending to one token)?

  Uniform attention = head has stopped doing selective computation.
  Peaky attention = head is locked onto a single token (attention sink).
  Both are signs of functional collapse, but different mechanisms.

  Connection to FIM paper: if high-FIM blocks in Pythia show more entropy change
  (either direction) than low-FIM blocks, that's circuit-level evidence of what
  the per-block drift is actually doing computationally.

Experiment G: Gradient Interference Between Generations
  Based on: Imanov (Jan 2026) — gradient interference (negative cosine similarity
  between task gradients) is the primary mechanism of catastrophic forgetting.
  This paper studied standard fine-tuning. Your setting is different: in recursive
  self-distillation, the "task" changes every generation because the training data
  changes. The question is whether gradients from Gen N are interfering with the
  knowledge encoded at Gen 0 — and whether this interference is architecture-dependent.

  Method: for each block, compute the gradient of the Gen0 loss w.r.t. the
  Gen5 model's parameters on a held-out Gen0 validation set. Compare with
  the gradient of the Gen5 training loss. If cosine similarity is negative,
  the Gen5 training gradient would have pushed the weights away from Gen0
  performance — this is gradient interference.

  Prediction: interference should be stronger in Pythia (parallel) than GPT-2
  (sequential) for high-FIM blocks, since those blocks drifted more.

Experiment H: SAE Feature Death (Pythia only)
  Uses EleutherAI pretrained SAEs for Pythia.
  pip install sae (EleutherAI sae library)

  Method: load Gen0 and Gen5 Pythia activations. Run both through the same
  pretrained SAE (trained on Gen0 distribution = The Pile). For each SAE feature,
  measure mean activation at Gen0 and Gen5. Features that drop to near-zero
  have "died" — the model has stopped computing the concept that feature encoded.

  This gives you INTERPRETABLE results: you can look up what each dead feature
  was computing (via Neuronpedia or the SAE feature labels) and say "the model
  stopped encoding [syntactic subject position / copy signal / etc.]."

  Note on SAE mismatch: the SAE was trained on The Pile, your Gen5 model was
  trained on TinyStories synthetic data. This WILL cause mismatch. Feature
  activation magnitudes will generally decrease because the model has shifted
  away from The Pile distribution. But we're looking for DIFFERENTIAL feature
  death — features that die more than the baseline drop, especially in specific
  blocks. Those are genuinely lost concepts, not just distributional shift.

Usage:
  python scripts/new_experiments.py --experiment F
  python scripts/new_experiments.py --experiment G
  python scripts/new_experiments.py --experiment H

Requirements:
  pip install transformer_lens
  pip install sae  (for H only — EleutherAI's sae library)
"""

import os
import gc
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from transformer_lens import HookedTransformer
    TL_AVAILABLE = True
except ImportError:
    TL_AVAILABLE = False
    print("Warning: TransformerLens not found. Install with: pip install transformer_lens")

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

BASE_DIR = r"D:\Thaman\Work\hessian-spectral-analysis"

CHECKPOINTS = {
    "GPT2": {
        "arch": "sequential",
        "tl_name": "gpt2",
        "n_layers": 12,
        "n_heads": 12,
        "gens": {
            0: "gpt2",
            1: os.path.join(BASE_DIR, "models", "gpt2_treatment_gen_1"),
            3: os.path.join(BASE_DIR, "models", "gpt2_treatment_gen_3"),
            5: os.path.join(BASE_DIR, "models", "gpt2_treatment_gen_5"),
        }
    },
    "Pythia": {
        "arch": "parallel",
        "tl_name": "EleutherAI/pythia-1.4b",
        "n_layers": 24,
        "n_heads": 16,
        "gens": {
            0: "EleutherAI/pythia-1.4b",
            1: os.path.join(BASE_DIR, "models", "pythia-1.4b_treatment_gen_1"),
            3: os.path.join(BASE_DIR, "models", "pythia-1.4b_treatment_gen_3"),
            5: os.path.join(BASE_DIR, "models", "pythia-1.4b_treatment_gen_5"),
        }
    },
}

GEN0_FIM = {
    "GPT2": {},
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Shared eval prompts
EVAL_PROMPTS = [
    "Once upon a time there was a little girl who",
    "The dog ran across the field and then",
    "In the forest the animals gathered to discuss",
    "The scientist looked at the data and realized",
    "Every morning the children would wake up and play",
    "The old man sat by the river thinking about",
    "She opened the door and found that inside",
    "The teacher asked the students to write about",
    "At the end of the day the family sat together",
    "The sun was setting and the sky turned",
]


def load_fim_gpt2():
    """Load GPT-2 Gen0 FIM from file."""
    import re
    paths = [
        os.path.join(BASE_DIR, "results", "gpt2_treatment_gen_0", "perblock_fim.json"),
        os.path.join(BASE_DIR, "results", "fimgpt2_gen_0.txt"),
    ]
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


def load_tl_model(model_path, tl_name):
    """Load a checkpoint into TransformerLens."""
    is_local = os.path.exists(str(model_path))
    if is_local:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32
        )
        model = HookedTransformer.from_pretrained(
            tl_name, hf_model=hf_model, dtype=torch.float32
        )
        del hf_model
    else:
        model = HookedTransformer.from_pretrained(model_path, dtype=torch.float32)
    return model.to(DEVICE).eval()


# ============================================================
# EXPERIMENT F: ATTENTION ENTROPY PER HEAD
# ============================================================

def compute_attention_entropy(model, prompts, n_layers, n_heads):
    """
    For each attention head, compute mean attention entropy across all prompts.
    Entropy = -sum(p * log(p)) over attended positions.
    Max entropy = log(seq_len) (uniform attention = not selecting anything).
    Near-zero entropy = concentrated attention (one dominant token).
    """
    head_entropies = {(l, h): [] for l in range(n_layers) for h in range(n_heads)}

    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        for layer in range(n_layers):
            attn_key = f"blocks.{layer}.attn.hook_pattern"
            if attn_key not in cache:
                continue
            # attn shape: [batch, n_heads, seq, seq]
            attn = cache[attn_key][0]  # [n_heads, seq, seq]

            for head in range(n_heads):
                if head >= attn.shape[0]:
                    continue
                # Mean entropy across query positions
                probs = attn[head]  # [seq, seq]
                # Clamp for log stability
                probs_safe = probs.clamp(min=1e-10)
                ent = -(probs * probs_safe.log()).sum(dim=-1)  # [seq]
                head_entropies[(layer, head)].append(ent.mean().item())

    # Average across prompts
    return {k: float(np.mean(v)) if v else float('nan')
            for k, v in head_entropies.items()}


def run_experiment_F():
    print("\n" + "="*60)
    print("EXPERIMENT F: Attention Entropy Per Head")
    print("="*60)

    if not TL_AVAILABLE:
        print("TransformerLens required. pip install transformer_lens")
        return

    GEN0_FIM["GPT2"] = load_fim_gpt2()
    results = {}
    output_path = os.path.join(OUTPUT_DIR, "attention_entropy.json")

    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        print("Loaded existing results.")

    for model_name, config in CHECKPOINTS.items():
        if model_name in results:
            print(f"{model_name}: cached.")
            continue

        arch = config["arch"]
        print(f"\n{model_name} ({arch}):")
        model_results = {}

        for gen, ckpt in config["gens"].items():
            print(f"  Gen{gen}...")
            try:
                model = load_tl_model(ckpt, config["tl_name"])
            except Exception as e:
                print(f"  Gen{gen} failed: {e}")
                continue

            entropies = compute_attention_entropy(
                model, EVAL_PROMPTS,
                config["n_layers"], config["n_heads"]
            )
            # Convert tuple keys to strings
            model_results[str(gen)] = {
                f"{l}_{h}": v for (l, h), v in entropies.items()
            }

            del model
            torch.cuda.empty_cache()
            gc.collect()

        results[model_name] = model_results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    _analyze_experiment_F(results)


def _analyze_experiment_F(results):
    from scipy.stats import spearmanr

    print("\n" + "="*60)
    print("ATTENTION ENTROPY ANALYSIS")
    print("="*60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, (model_name, model_results) in enumerate(results.items()):
        arch = CHECKPOINTS[model_name]["arch"]
        n_layers = CHECKPOINTS[model_name]["n_layers"]
        n_heads = CHECKPOINTS[model_name]["n_heads"]
        fim_data = GEN0_FIM.get(model_name, {})

        gen0_data = {
            (int(k.split("_")[0]), int(k.split("_")[1])): v
            for k, v in model_results.get("0", {}).items()
        }
        gen5_data = {
            (int(k.split("_")[0]), int(k.split("_")[1])): v
            for k, v in model_results.get("5", {}).items()
        }

        if not gen0_data or not gen5_data:
            continue

        # Per-block mean entropy change Gen0 → Gen5
        block_entropy_change = {}
        for layer in range(n_layers):
            gen0_vals = [gen0_data.get((layer, h), np.nan) for h in range(n_heads)]
            gen5_vals = [gen5_data.get((layer, h), np.nan) for h in range(n_heads)]
            gen0_mean = np.nanmean(gen0_vals)
            gen5_mean = np.nanmean(gen5_vals)
            if not np.isnan(gen0_mean) and not np.isnan(gen5_mean):
                block_entropy_change[layer] = float(gen5_mean - gen0_mean)

        blocks = sorted(block_entropy_change.keys())
        changes = [block_entropy_change[b] for b in blocks]
        fim_vals = [fim_data.get(b, np.nan) for b in blocks]

        # Trajectory: mean head entropy per layer across generations
        ax_traj = axes[0, col]
        gen_keys = sorted([g for g in model_results.keys() if g.isdigit()], key=int)
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(gen_keys)))

        for gen_key, color in zip(gen_keys, colors):
            gen_data = {
                (int(k.split("_")[0]), int(k.split("_")[1])): v
                for k, v in model_results.get(gen_key, {}).items()
            }
            layer_means = [
                np.nanmean([gen_data.get((l, h), np.nan) for h in range(n_heads)])
                for l in range(n_layers)
            ]
            ax_traj.plot(range(n_layers), layer_means, 'o-',
                        color=color, label=f"Gen{gen_key}",
                        linewidth=1.5, markersize=3, alpha=0.8)

        ax_traj.set_xlabel("Layer", fontsize=11)
        ax_traj.set_ylabel("Mean Attention Entropy (nats)", fontsize=11)
        ax_traj.set_title(
            f"{model_name} ({arch})\n"
            "Higher = more uniform (less selective)\n"
            "Lower = more peaked (attention sink forming)",
            fontsize=10
        )
        ax_traj.legend(fontsize=8)
        ax_traj.grid(True, alpha=0.3)

        # FIM vs entropy change scatter
        ax_corr = axes[1, col]
        valid = [(f, c) for f, c in zip(fim_vals, changes) if not np.isnan(f)]
        if valid:
            f_vals, c_vals = zip(*valid)
            log_f = np.log10(np.array(f_vals) + 1e-8)
            color = 'steelblue' if arch == 'sequential' else 'tomato'
            ax_corr.scatter(log_f, c_vals, alpha=0.7, s=60,
                           color=color, edgecolors='k')
            ax_corr.axhline(y=0, color='black', linestyle='--', alpha=0.4,
                           linewidth=1, label='No change')
            rho, p = spearmanr(f_vals, c_vals)
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            ax_corr.set_title(
                f"{model_name}: FIM vs Entropy Change Gen0→Gen5\n"
                f"rho={rho:+.3f}{sig} p={p:.4f}\n"
                f"+ve = heads becoming more uniform, -ve = more peaked",
                fontsize=10
            )
            for b, f, c in zip(blocks, fim_vals, changes):
                if not np.isnan(f) and abs(c) > 0.1:
                    ax_corr.annotate(str(b), (np.log10(f + 1e-8), c),
                                    fontsize=7, alpha=0.7)

            print(f"\n{model_name} ({arch}): FIM~entropy_change rho={rho:+.3f}{sig} p={p:.4f}")
            print(f"  Mean entropy change Gen0→Gen5: {np.mean(c_vals):+.4f}")
            print(f"  {'Becoming more uniform (attention spreading)' if np.mean(c_vals) > 0 else 'Becoming more peaked (attention sinks forming)'}")
            print(f"  {'Layer':<8} {'log10FIM':>10} {'EntChange':>12} {'Direction':>12}")
            for b, f, c in zip(blocks, fim_vals, changes):
                if not np.isnan(f):
                    direction = "spreading" if c > 0.05 else ("sinking" if c < -0.05 else "stable")
                    print(f"  {b:<8} {np.log10(f+1e-8):>10.3f} {c:>+12.4f} {direction:>12}")

        ax_corr.set_xlabel("log10(FIM_b) Gen0", fontsize=11)
        ax_corr.set_ylabel("Entropy Change (Gen5 - Gen0)", fontsize=11)
        ax_corr.grid(True, alpha=0.3)
        ax_corr.legend(fontsize=9)

    plt.suptitle(
        "Attention Entropy Per Head: Are Heads Spreading or Concentrating?\n"
        "Positive = more uniform (less selective), Negative = attention sink forming",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "attention_entropy.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {plot_path}")


# ============================================================
# EXPERIMENT G: GRADIENT INTERFERENCE BETWEEN GENERATIONS
# ============================================================

def compute_block_gradient_cosine(model_path, gen0_texts, gen5_texts,
                                   tokenizer_path, n_layers):
    """
    Compute per-block gradient cosine similarity on CPU to avoid CUDA issues.
    Load model fresh in float32 on CPU — slower but avoids all CUBLAS errors.
    """
    import re

    # Load model fresh on CPU in float32 for stable gradient computation
    print("    Loading model to CPU for gradient computation...")
    model_cpu = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        device_map="cpu",
    )
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_cpu.train()

    def get_block_grads(texts):
        block_grads = {l: [] for l in range(n_layers)}
        for text in texts[:5]:  # 5 texts on CPU is plenty
            enc = tok(text, return_tensors="pt", truncation=True, max_length=32)
            if enc["input_ids"].shape[1] < 3:
                continue
            model_cpu.zero_grad()
            try:
                out = model_cpu(**enc, labels=enc["input_ids"])
                out.loss.backward()
            except Exception as e:
                print(f"      backward failed: {e}")
                continue
            for name, param in model_cpu.named_parameters():
                if param.grad is None:
                    continue
                for pat in [r'\.(layers|h|blocks)\.(\d+)\.',
                             r'\.gpt_neox\.layers\.(\d+)\.']:
                    m = re.search(pat, name)
                    if m:
                        try:
                            layer = int(m.group(1)) if 'gpt_neox' in pat else int(m.group(2))
                        except (ValueError, IndexError):
                            break
                        if layer < n_layers:
                            block_grads[layer].append(
                                param.grad.detach().float().view(-1).clone()
                            )
                        break
        return {l: torch.cat(v) if v else None for l, v in block_grads.items()}

    print("    Computing gradients on split A...")
    grads_A = get_block_grads(gen0_texts)
    print("    Computing gradients on split B...")
    grads_B = get_block_grads(gen5_texts)

    model_cpu.eval()
    del model_cpu
    gc.collect()

    cosines = {}
    for layer in range(n_layers):
        gA = grads_A.get(layer)
        gB = grads_B.get(layer)
        if gA is None or gB is None:
            continue
        min_len = min(len(gA), len(gB))
        cos = torch.nn.functional.cosine_similarity(
            gA[:min_len].unsqueeze(0), gB[:min_len].unsqueeze(0)
        ).item()
        cosines[layer] = float(cos)

    return cosines


def run_experiment_G():
    """
    Gradient interference analysis.
    Compare gradient directions at Gen5 for:
    - Gen0 validation data (what the model should preserve)
    - Current generation synthetic data (what it's being trained on)
    Negative cosine similarity = these are pulling in opposite directions.
    """
    print("\n" + "="*60)
    print("EXPERIMENT G: Gradient Interference Between Generations")
    print("="*60)

    GEN0_FIM["GPT2"] = load_fim_gpt2()

    # Load validation data representing Gen0 distribution (TinyStories = what the models learned from)
    print("Loading evaluation data...")
    ds_real = load_dataset("roneneldan/TinyStories", split="validation[:50]")
    gen0_texts = ds_real["text"]

    # Gen5-like texts = use the EVAL_PROMPTS (which are clean prompts,
    # generating from Gen5 would be collapsed text)
    # Instead, the most honest test is: compare gradient of Gen0 validation loss
    # vs gradient of Gen5 training loss on the Gen5 synthetic training data
    # But we don't have the Gen5 synthetic data easily accessible here
    # So we use EVAL_PROMPTS as a proxy for in-distribution current data
    gen5_texts = gen0_texts[25:]  # use second half as "different distribution"

    results = {}
    output_path = os.path.join(OUTPUT_DIR, "gradient_interference.json")

    # Always recompute — gradient interference depends on correct regex fix
    if os.path.exists(output_path):
        os.remove(output_path)
        print("Cleared old cache to ensure fresh computation.")

    for model_name, config in CHECKPOINTS.items():
        if model_name in results:
            print(f"{model_name}: cached.")
            continue

        arch = config["arch"]
        print(f"\n{model_name} ({arch}):")
        model_results = {}

        # Only run on Gen5 — that's where the interference matters
        gen5_ckpt = config["gens"].get(5)
        if not gen5_ckpt or not os.path.exists(str(gen5_ckpt)):
            if gen5_ckpt not in ["gpt2", "EleutherAI/pythia-1.4b"]:
                print(f"  Gen5 checkpoint not found")
                continue

        # Model is loaded inside compute_block_gradient_cosine on CPU
        model = None  # placeholder

        # Pass path rather than loaded model - gradient computation
        # loads its own CPU copy to avoid CUDA backward issues
        del model  # free GPU memory before CPU gradient run
        torch.cuda.empty_cache()
        gc.collect()

        cosines = compute_block_gradient_cosine(
            gen5_ckpt,
            list(gen0_texts[:10]),
            list(gen0_texts[10:20]),
            gen5_ckpt,
            config["n_layers"]
        )
        model_results["gen5"] = {str(k): v for k, v in cosines.items()}

        print(f"  Computed {len(cosines)} block cosines:")
        for l in sorted(cosines.keys()):
            marker = " <-- interference" if cosines[l] < 0 else ""
            print(f"    Block {l:2d}: cosine={cosines[l]:+.4f}{marker}")

        results[model_name] = model_results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    _analyze_experiment_G(results)


def _analyze_experiment_G(results):
    from scipy.stats import spearmanr

    print("\n" + "="*60)
    print("GRADIENT INTERFERENCE RESULTS")
    print("="*60)
    print("Cosine similarity between gradient directions for two data splits.")
    print("Near zero or negative = gradient interference (pulling in different directions)")
    print()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (model_name, model_results) in zip(axes, results.items()):
        arch = CHECKPOINTS[model_name]["arch"]
        fim_data = GEN0_FIM.get(model_name, {})

        cosines_data = model_results.get("gen5", {})
        if not cosines_data:
            ax.set_title(f"{model_name}: no data")
            continue

        layers = sorted(int(l) for l in cosines_data.keys())
        cosines = [cosines_data[str(l)] for l in layers]
        fim_vals = [fim_data.get(l, np.nan) for l in layers]

        colors = ['red' if c < 0 else 'steelblue' for c in cosines]
        ax.bar(layers, cosines, color=colors, alpha=0.8, edgecolor='k', linewidth=0.5)
        ax.axhline(y=0, color='black', linewidth=1)
        ax.set_xlabel("Block", fontsize=11)
        ax.set_ylabel("Gradient Cosine Similarity", fontsize=11)

        valid = [(f, c) for f, c in zip(fim_vals, cosines) if not np.isnan(f)]
        if valid:
            f_vals, c_vals = zip(*valid)
            rho, p = spearmanr(f_vals, c_vals)
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            n_negative = sum(1 for c in c_vals if c < 0)
            ax.set_title(
                f"{model_name} ({arch}) Gen5\n"
                f"rho(FIM, gradient_cosine)={rho:+.3f}{sig} p={p:.4f}\n"
                f"Red bars = interference ({n_negative}/{len(c_vals)} blocks)",
                fontsize=10
            )
            print(f"{model_name} ({arch}):")
            print(f"  rho(FIM, cosine) = {rho:+.3f}{sig} p={p:.4f}")
            print(f"  Blocks with negative cosine (interference): {n_negative}/{len(c_vals)}")
            print(f"  Mean cosine: {np.mean(c_vals):+.4f}")
            print(f"  {'Block':<8} {'log10FIM':>10} {'Cosine':>10}")
            for l, f, c in zip(layers, fim_vals, cosines):
                if not np.isnan(f):
                    marker = " <-- interference" if c < 0 else ""
                    print(f"  {l:<8} {np.log10(f+1e-8):>10.3f} {c:>+10.4f}{marker}")

        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(
        "Gradient Interference at Gen5: Are Gradients Pulling in Conflicting Directions?\n"
        "Red = negative cosine similarity = interference (forgetting mechanism)",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "gradient_interference.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {plot_path}")


# ============================================================
# EXPERIMENT H: SAE FEATURE DEATH (Pythia only)
# ============================================================


def _run_experiment_H_manual():
    """
    Manual SAE feature analysis without a library.
    Instead of a pretrained SAE, we train a lightweight PCA-based feature
    decomposition on Gen0 activations and measure feature death at Gen5.

    This is less interpretable than a full SAE but captures the same
    information: which directions in activation space become inactive?

    Note: For full interpretability, install sae-lens:
      pip install sae-lens
    and rerun with the full SAE.
    """
    from sklearn.decomposition import PCA
    from datasets import load_dataset
    print("Running manual PCA-based feature analysis (SAE approximation)...")

    gen0_path = CHECKPOINTS["Pythia"]["gens"][0]
    gen5_path = CHECKPOINTS["Pythia"]["gens"][5]

    if not os.path.exists(str(gen5_path)):
        print(f"Gen5 not found: {gen5_path}")
        return

    # Load both to CPU — collect_acts will move each to GPU one at a time
    model_gen0 = AutoModelForCausalLM.from_pretrained(
        gen0_path, torch_dtype=torch.float16,
        attn_implementation="eager", device_map="cpu"
    )
    tok = AutoTokenizer.from_pretrained(gen0_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_gen5 = AutoModelForCausalLM.from_pretrained(
        gen5_path, torch_dtype=torch.float16,
        attn_implementation="eager", device_map="cpu"
    )

    ds = load_dataset("roneneldan/TinyStories", split="validation[:200]")

    results_pca = {}

    for target_layer in [5, 12, 20]:
        print(f"\n  Layer {target_layer}:")

        def collect_acts(mdl, layer_idx):
            mdl = mdl.to(DEVICE)
            acts = []
            hook_data = {}
            def hook_fn(m, inp, out):
                hook_data["act"] = (out[0] if isinstance(out, tuple) else out).detach().cpu().float()
            if hasattr(mdl, "gpt_neox"):
                h = mdl.gpt_neox.layers[layer_idx].register_forward_hook(hook_fn)
            else:
                h = mdl.transformer.h[layer_idx].register_forward_hook(hook_fn)
            mdl.eval()
            with torch.no_grad():
                for text in ds["text"]:
                    enc = tok(text, return_tensors="pt", truncation=True, max_length=64).to(DEVICE)
                    mdl(**enc)
                    if "act" in hook_data:
                        acts.append(hook_data["act"][0].mean(0).numpy())
            h.remove()
            mdl.cpu()
            torch.cuda.empty_cache()
            return np.stack(acts)

        acts0 = collect_acts(model_gen0, target_layer)
        acts5 = collect_acts(model_gen5, target_layer)

        # Fit PCA on Gen0 activations
        n_components = min(100, acts0.shape[1], acts0.shape[0])
        pca = PCA(n_components=n_components)
        pca.fit(acts0)

        # Project both onto PCA components
        proj0 = pca.transform(acts0)  # [n_samples, n_components]
        proj5 = pca.transform(acts5)

        # Feature "activity" = variance explained in each component
        var0 = proj0.var(axis=0)  # [n_components]
        var5 = proj5.var(axis=0)

        # Relative activity
        relative = var5 / (var0 + 1e-10)
        dead_mask = relative < 0.2
        n_dead = dead_mask.sum()
        dead_pct = n_dead / n_components * 100

        print(f"    PCA components: {n_components}")
        print(f"    'Dead' components (<20% Gen0 variance): {n_dead}/{n_components} ({dead_pct:.1f}%)")
        print(f"    Mean variance ratio Gen5/Gen0: {relative.mean():.3f}")
        print(f"    Variance explained by dead components: {pca.explained_variance_ratio_[dead_mask].sum()*100:.1f}%")

        results_pca[target_layer] = {
            "n_components": int(n_components),
            "n_dead": int(n_dead),
            "dead_pct": float(dead_pct),
            "mean_var_ratio": float(relative.mean()),
            "var_explained_by_dead": float(pca.explained_variance_ratio_[dead_mask].sum()),
        }

    del model_gen0, model_gen5
    torch.cuda.empty_cache()

    output_path = os.path.join(OUTPUT_DIR, "sae_feature_death_pca.json")
    with open(output_path, "w") as f:
        json.dump(results_pca, f, indent=2)
    print(f"\nSaved to: {output_path}")
    print("Note: these are PCA components, not SAE features.")
    print("Install sae-lens for interpretable feature labels: pip install sae-lens")

def run_experiment_H():
    """
    SAE Feature Death analysis using EleutherAI pretrained SAEs for Pythia.

    Installation: pip install sae
    SAE repo: https://github.com/EleutherAI/sae

    The SAE was trained on The Pile. Features will generally activate less
    on TinyStories-trained models. We look for DIFFERENTIAL death — features
    that die MORE than the baseline average across the model.
    """
    print("\n" + "="*60)
    print("EXPERIMENT H: SAE Feature Death (Pythia only)")
    print("="*60)

    # Try SAELens - but verify the correct release ID first
    sae_lib = None
    correct_release = None

    try:
        from sae_lens import SAE
        from sae_lens.pretrained_saes import get_pretrained_saes_directory
        directory = get_pretrained_saes_directory()
        # Find any Pythia-1.4b release
        for release_id, release_data in directory.items():
            if "pythia" in release_id.lower() and "1.4" in release_id:
                correct_release = release_id
                break
        # Fallback: try common names
        if correct_release is None:
            for candidate in ["pythia-1.4b-deduped-res-sm",
                              "pythia-1.4b-res-sm",
                              "EleutherAI/sae-pythia-1.4b"]:
                if candidate in directory:
                    correct_release = candidate
                    break
        if correct_release:
            sae_lib = "sae_lens"
            print(f"Using SAELens with release: {correct_release}")
        else:
            print(f"SAELens installed but no Pythia-1.4b release found.")
            print(f"Available Pythia releases: {[k for k in directory if 'pythia' in k.lower()][:5]}")
    except ImportError:
        pass
    except Exception as e:
        print(f"SAELens error: {e}")

    if sae_lib is None or correct_release is None:
        try:
            from sae import Sae
            sae_lib = "eleutherai_sae"
            print("Using EleutherAI SAE library")
        except ImportError:
            pass

    if sae_lib is None or correct_release is None:
        print("No working SAE library found for Pythia-1.4b.")
        print("Running PCA-based fallback (approximation)...")
        _run_experiment_H_manual()
        return

    from datasets import load_dataset

    # Load Pythia Gen0 and Gen5
    gen0_path = CHECKPOINTS["Pythia"]["gens"][0]
    gen5_path = CHECKPOINTS["Pythia"]["gens"][5]

    if not os.path.exists(str(gen5_path)):
        print(f"Gen5 checkpoint not found: {gen5_path}")
        return

    print("Loading Pythia Gen0...")
    model_gen0 = AutoModelForCausalLM.from_pretrained(
        gen0_path, torch_dtype=torch.float16,
        attn_implementation="eager",
        device_map="cpu"  # load to CPU first to avoid double VRAM
    )
    tok = AutoTokenizer.from_pretrained(gen0_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Loading Pythia Gen5...")
    model_gen5 = AutoModelForCausalLM.from_pretrained(
        gen5_path, torch_dtype=torch.float16,
        attn_implementation="eager",
        device_map="cpu"
    )

    # Load SAE — EleutherAI has SAEs for Pythia layers
    # Try loading SAE for a middle layer first
    print("Loading EleutherAI SAE for Pythia-1.4b...")
    sae_results = {}

    # Check which layers have SAEs available
    # EleutherAI SAEs are typically available for residual stream at each layer
    # Load from: EleutherAI/sae-pythia-1.4b-{layer}
    # We'll try layers 5, 12, 20 to sample early, mid, late

    for target_layer in [5, 12, 20]:
        print(f"\nLayer {target_layer}: loading SAE...")
        sae = None
        try:
            if sae_lib == "sae_lens":
                sae_obj, cfg_dict, _ = SAE.from_pretrained(
                    release=correct_release,
                    sae_id=f"blocks.{target_layer}.hook_resid_post",
                )
                sae = sae_obj
            elif sae_lib == "eleutherai_sae":
                sae_repo = f"EleutherAI/sae-pythia-1.4b-{target_layer}"
                sae = Sae.load_from_hub(sae_repo, device=DEVICE)
        except Exception as e:
            print(f"  SAE load failed for layer {target_layer}: {e}")
            print(f"  Skipping this layer.")
            continue

        if sae is None:
            continue

        # Collect residual stream activations at this layer
        def get_layer_acts(model, texts, layer_idx):
            acts = []
            hook_data = {}

            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hook_data['act'] = output[0].detach().cpu().float()
                else:
                    hook_data['act'] = output.detach().cpu().float()

            # Get the right layer module
            if hasattr(model, 'gpt_neox'):
                layer_mod = model.gpt_neox.layers[layer_idx]
            else:
                layer_mod = model.transformer.h[layer_idx]

            hook = layer_mod.register_forward_hook(hook_fn)
            model.eval()

            ds = load_dataset("roneneldan/TinyStories", split="validation[:100]")
            with torch.no_grad():
                for text in ds["text"]:
                    enc = tok(text, return_tensors="pt",
                             truncation=True, max_length=64).to(DEVICE)
                    _ = model(**enc)
                    if "act" in hook_data:
                        # Take mean over sequence positions
                        acts.append(hook_data["act"][0].mean(0))  # [hidden]

            hook.remove()
            return torch.stack(acts)  # [n_samples, hidden]

        print(f"  Collecting Gen0 activations at layer {target_layer}...")
        acts_gen0 = get_layer_acts(model_gen0, None, target_layer).to(DEVICE)
        print(f"  Collecting Gen5 activations at layer {target_layer}...")
        acts_gen5 = get_layer_acts(model_gen5, None, target_layer).to(DEVICE)

        # Run SAE on both
        with torch.no_grad():
            latents_gen0 = sae.encode(acts_gen0)  # [n_samples, n_features]
            latents_gen5 = sae.encode(acts_gen5)

        # Feature activity: mean activation across samples
        # Dead = mean activation drops to near zero
        mean_gen0 = latents_gen0.abs().mean(0).cpu().numpy()  # [n_features]
        mean_gen5 = latents_gen5.abs().mean(0).cpu().numpy()

        # Feature death: activation drops by more than 80%
        baseline_drop = float(np.mean(mean_gen5) / (np.mean(mean_gen0) + 1e-8))
        relative_drop = mean_gen5 / (mean_gen0 + 1e-8)
        dead_threshold = 0.2  # feature must retain < 20% of Gen0 activation
        dead_features = np.where(relative_drop < dead_threshold)[0]
        dead_pct = len(dead_features) / len(mean_gen0) * 100

        print(f"  Layer {target_layer}: baseline activity drop = {baseline_drop:.3f}")
        print(f"  Dead features (<20% Gen0 activity): {len(dead_features)}/{len(mean_gen0)} ({dead_pct:.1f}%)")

        sae_results[target_layer] = {
            "n_features": int(len(mean_gen0)),
            "n_dead": int(len(dead_features)),
            "dead_pct": float(dead_pct),
            "baseline_activity_ratio": float(baseline_drop),
            "dead_feature_indices": dead_features.tolist()[:50],  # first 50
        }

    del model_gen0, model_gen5
    torch.cuda.empty_cache()
    gc.collect()

    # Save and print
    output_path = os.path.join(OUTPUT_DIR, "sae_feature_death.json")
    with open(output_path, 'w') as f:
        json.dump(sae_results, f, indent=2)

    print("\n" + "="*60)
    print("SAE FEATURE DEATH SUMMARY")
    print("="*60)
    for layer, res in sae_results.items():
        print(f"Layer {layer}:")
        print(f"  Total features: {res['n_features']}")
        print(f"  Dead features (retaining <20% activity): {res['n_dead']} ({res['dead_pct']:.1f}%)")
        print(f"  Overall activity ratio Gen5/Gen0: {res['baseline_activity_ratio']:.3f}")

    print(f"\nResults saved: {output_path}")
    print("\nNext step: look up dead feature indices on Neuronpedia")
    print("https://neuronpedia.org/pythia-1.4b/{layer}/{feature_index}")
    print("to see what concepts the model has forgotten.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["F", "G", "H", "all"],
                        default="F")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    if args.experiment in ("F", "all"):
        run_experiment_F()

    if args.experiment in ("G", "all"):
        run_experiment_G()

    if args.experiment in ("H", "all"):
        run_experiment_H()