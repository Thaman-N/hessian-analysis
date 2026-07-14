"""
Circuit-Level Collapse Analysis: Two Experiments
==================================================

Experiment A: Logit Lens Trajectory
  At each layer and generation, decode the residual stream directly to vocabulary.
  Tracks WHERE in the network the model commits to its prediction across generations.
  Prediction: Pythia (parallel) commits earlier as collapse progresses (late blocks
  become unreliable). SmolLM (sequential) commitment layer stays stable.

Experiment B: Causal Restoration Patching
  For prompts where Gen5 Pythia fails, patch Gen0 residual stream block-by-block.
  Measure how much performance is recovered when each block is restored.
  Correlate recovery amount with block FIM from the original paper.
  Prediction: High-FIM blocks, when restored, recover the most performance.
  This would be direct causal evidence for the FIM paper's mechanistic claim.

Requirements:
  pip install transformer_lens
  pip install transformer_lens>=1.0.0

Run order:
  1. Run Experiment A first (cheap, ~30 mins)
  2. If logit lens shows interesting dynamics, run Experiment B

Usage:
  cd D:/Thaman/Work/hessian-spectral-analysis
  python scripts/circuit_analysis_experiments.py --experiment A
  python scripts/circuit_analysis_experiments.py --experiment B
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
    import transformer_lens.utils as utils
    TL_AVAILABLE = True
except ImportError:
    raise ImportError(
        "TransformerLens required. Install with: pip install transformer_lens"
    )

from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = r"D:\Thaman\Work\hessian-spectral-analysis"

CHECKPOINTS = {
    "GPT2": {
        "arch": "sequential",
        "tl_name": "gpt2",
        "n_layers": 12,
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
        "gens": {
            0: "EleutherAI/pythia-1.4b",
            1: os.path.join(BASE_DIR, "models", "pythia-1.4b_treatment_gen_1"),
            3: os.path.join(BASE_DIR, "models", "pythia-1.4b_treatment_gen_3"),
            5: os.path.join(BASE_DIR, "models", "pythia-1.4b_treatment_gen_5"),
        }
    },
}

import re as _re

def load_fim_for_model(model_name):
    """
    Load Gen0 FIM values for a given model from the results directory.
    Tries JSON first, then txt summary files.
    Returns dict {block_idx: fim_value}
    """
    # Map model names to their Gen0 FIM file paths
    fim_paths = {
        "GPT2": [
            os.path.join(BASE_DIR, "results", "gpt2_treatment_gen_0", "perblock_fim.json"),
            os.path.join(BASE_DIR, "results", "fimgpt2_gen_0.txt"),
        ],
        "Pythia": [
            os.path.join(BASE_DIR, "results", "pythia-1.4b_treatment_gen_0", "perblock_fim.json"),
            os.path.join(BASE_DIR, "results", "fimpythia_gen_0.txt"),
        ],
        "SmolLM": [
            os.path.join(BASE_DIR, "results", "treatment_gen_0", "perblock_fim.json"),
            os.path.join(BASE_DIR, "results", "fimtreatment0.txt"),
        ],
    }

    for path in fim_paths.get(model_name, []):
        if not os.path.exists(path):
            continue
        if path.endswith(".json"):
            try:
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
                    print(f"  Loaded FIM for {model_name} from {path}")
                    return result
            except Exception as e:
                print(f"  Failed to load FIM JSON {path}: {e}")
        elif path.endswith(".txt"):
            try:
                for enc in ["utf-16", "utf-8", "cp1252"]:
                    try:
                        with open(path, encoding=enc) as f:
                            content = f.read()
                        break
                    except UnicodeError:
                        continue
                result = {}
                for line in content.split("\n"):
                    m = _re.match(r"Block\s+(\d+)\s*\|\s*Attn:\s*([\d.]+)\s*\|\s*MLP:\s*([\d.]+)", line)
                    if m:
                        b = int(m.group(1))
                        result[b] = float(m.group(2)) + float(m.group(3))
                if result:
                    print(f"  Loaded FIM for {model_name} from {path}")
                    return result
            except Exception as e:
                print(f"  Failed to load FIM txt {path}: {e}")

    print(f"  WARNING: No FIM file found for {model_name}. Correlation will be skipped.")
    return {}

OUTPUT_DIR = os.path.join(BASE_DIR, "results", "circuit_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fixed evaluation prompts — same across all experiments for comparability
EVAL_PROMPTS = [
    "Once upon a time there was a little girl who",
    "The dog ran across the field and then",
    "In the forest, the animals gathered to",
    "The scientist looked at the results and",
    "Every morning the children would wake up and",
    "The old man sat by the river and thought about",
    "When the rain started falling the town",
    "She opened the box and found inside",
    "The teacher asked the class to",
    "At the end of the day the family",
]


# ============================================================
# HELPER: Load model via TransformerLens
# ============================================================

def load_tl_model(model_path, tl_name):
    """
    Load a HuggingFace checkpoint into TransformerLens.
    TransformerLens needs the architecture name even for local checkpoints.
    """
    is_local = os.path.exists(str(model_path))

    if is_local:
        # Load HF model first, then convert to TransformerLens
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
        )
        model = HookedTransformer.from_pretrained(
            tl_name,
            hf_model=hf_model,
            dtype=torch.float32,
        )
        del hf_model
    else:
        model = HookedTransformer.from_pretrained(
            model_path,
            dtype=torch.float32,
        )

    model = model.to(DEVICE)
    model.eval()
    return model


# ============================================================
# EXPERIMENT A: Logit Lens
# ============================================================

def logit_lens_layer_entropy(model, prompts, layer_idx):
    """
    At a given layer, decode the residual stream to vocabulary logits.
    Return the entropy of the resulting distribution (lower = more committed).
    Average over all prompts and token positions.
    """
    entropies = []

    for prompt in prompts:
        tokens = model.to_tokens(prompt)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        # Residual stream after this layer
        resid = cache[f"blocks.{layer_idx}.hook_resid_post"]  # [1, seq, d_model]

        # Apply layer norm and unembed to get logits
        # TransformerLens: model.ln_final and model.unembed
        normed = model.ln_final(resid)  # [1, seq, d_model]
        logits = model.unembed(normed)  # [1, seq, vocab]
        probs = torch.softmax(logits[0], dim=-1)  # [seq, vocab]

        # Entropy at each position (lower = model is more certain)
        ent = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # [seq]
        entropies.extend(ent.detach().cpu().numpy().tolist())

        del resid, normed, logits, probs, cache

    return float(np.mean(entropies))


def logit_lens_commitment_layer(model, prompts, n_layers):
    """
    Find the layer where entropy drops most sharply (= model commits to prediction).
    Returns: (commitment_layer, layer_entropies)
    """
    layer_entropies = []
    for layer in range(n_layers):
        ent = logit_lens_layer_entropy(model, prompts, layer)
        layer_entropies.append(ent)

    # Commitment layer = where entropy drops most from previous layer
    drops = [layer_entropies[i-1] - layer_entropies[i]
             for i in range(1, len(layer_entropies))]
    commitment_layer = int(np.argmax(drops)) + 1

    return commitment_layer, layer_entropies


def run_experiment_A():
    """
    Track commitment layer across generations for SmolLM and Pythia.
    """
    print("\n" + "="*60)
    print("EXPERIMENT A: Logit Lens Commitment Layer Trajectory")
    print("="*60)

    results = {}
    output_path = os.path.join(OUTPUT_DIR, "logit_lens_results.json")

    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        print("Loaded existing results.")

    for model_name, config in CHECKPOINTS.items():
        if model_name not in results:
            results[model_name] = {}

        arch = config["arch"]
        n_layers = config["n_layers"]
        print(f"\n{model_name} ({arch}):")

        for gen, ckpt_path in config["gens"].items():
            gen_key = str(gen)
            if gen_key in results[model_name]:
                print(f"  Gen {gen}: cached ({results[model_name][gen_key]['commitment_layer']})")
                continue

            print(f"  Gen {gen}: loading...")
            try:
                model = load_tl_model(ckpt_path, config["tl_name"])
            except Exception as e:
                print(f"  Gen {gen}: FAILED — {e}")
                continue

            commitment_layer, layer_entropies = logit_lens_commitment_layer(
                model, EVAL_PROMPTS, n_layers
            )

            results[model_name][gen_key] = {
                "commitment_layer": commitment_layer,
                "layer_entropies": layer_entropies,
            }

            print(f"  Gen {gen}: commitment layer = {commitment_layer} "
                  f"(out of {n_layers})")

            del model
            torch.cuda.empty_cache()
            gc.collect()

            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['blue', 'orange', 'green', 'red']
    gen_keys = ['0', '1', '3', '5']

    for ax, (model_name, model_results) in zip(axes, results.items()):
        arch = CHECKPOINTS[model_name]["arch"]
        n_layers = CHECKPOINTS[model_name]["n_layers"]

        for gen_key, color in zip(gen_keys, colors):
            if gen_key not in model_results:
                continue
            layer_entropies = model_results[gen_key]["layer_entropies"]
            commitment = model_results[gen_key]["commitment_layer"]
            ax.plot(range(len(layer_entropies)), layer_entropies,
                    color=color, label=f"Gen{gen_key} (commit@L{commitment})",
                    linewidth=1.5)
            ax.axvline(x=commitment, color=color, linestyle='--', alpha=0.4)

        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Residual Stream Entropy (nats)", fontsize=12)
        ax.set_title(f"{model_name} ({arch})\nLogit Lens: Lower = More Committed",
                     fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Logit Lens: Where Does the Model Commit to Its Prediction?",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "logit_lens_trajectory.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {plot_path}")

    # Interpretation
    print("\n" + "-"*60)
    print("INTERPRETATION:")
    for model_name, model_results in results.items():
        arch = CHECKPOINTS[model_name]["arch"]
        gen0_commit = model_results.get("0", {}).get("commitment_layer", None)
        gen5_commit = model_results.get("5", {}).get("commitment_layer", None)
        if gen0_commit and gen5_commit:
            shift = gen5_commit - gen0_commit
            direction = "earlier" if shift < 0 else "later"
            print(f"{model_name} ({arch}): commitment layer "
                  f"Gen0={gen0_commit} → Gen5={gen5_commit} "
                  f"(shifted {abs(shift)} layers {direction})")
            if arch == "parallel" and shift < -2:
                print(f"  → CONFIRMS prediction: parallel model commits earlier "
                      f"as late blocks become unreliable")
            elif arch == "sequential" and abs(shift) <= 2:
                print(f"  → CONFIRMS prediction: sequential model commitment "
                      f"layer is stable")

    return results


# ============================================================
# EXPERIMENT B: Causal Restoration Patching
# ============================================================

def perplexity_from_logits(model, tokens):
    """Compute perplexity for a given token sequence."""
    with torch.no_grad():
        logits = model(tokens)
    log_probs = torch.nn.functional.log_softmax(logits[0, :-1], dim=-1)
    target_log_probs = log_probs[
        torch.arange(tokens.shape[1]-1), tokens[0, 1:]
    ]
    return float(torch.exp(-target_log_probs.mean()).item())


def find_failure_prompts(model_gen0, model_gen5, corpus_prompts, n_failures=500):
    """
    Find prompts where Gen5 predicts the WRONG next token but Gen0 predicts
    the CORRECT one. These are the cases where collapse has causally broken
    something — not just reduced confidence but actually flipped the prediction.

    Returns list of (prompt_prefix, correct_token_id) pairs.
    Avoids the positional bias of perplexity patching by using a binary
    correct/incorrect measure instead.
    """
    print(f"  Finding failure prompts (need {n_failures})...")

    # Expand corpus with TinyStories validation set
    try:
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split="validation[:1000]")
        corpus_prompts = list(corpus_prompts) + ds["text"]
        print(f"  Corpus expanded to {len(corpus_prompts)} prompts")
    except Exception as e:
        print(f"  Could not load TinyStories: {e}. Using provided corpus only.")

    failures = []

    for prompt in corpus_prompts:
        if len(failures) >= n_failures:
            break

        tokens = model_gen0.to_tokens(prompt)
        if tokens.shape[1] < 6:
            continue

        # Test multiple positions within each prompt
        for pos in range(3, min(tokens.shape[1] - 1, 20)):
            prefix = tokens[:, :pos]
            correct_next = tokens[0, pos].item()

            with torch.no_grad():
                logits_gen0 = model_gen0(prefix)[0, -1, :]
                logits_gen5 = model_gen5(prefix)[0, -1, :]

            pred_gen0 = logits_gen0.argmax().item()
            pred_gen5 = logits_gen5.argmax().item()

            # Gen0 correct, Gen5 wrong — this is a collapse-induced failure
            if pred_gen0 == correct_next and pred_gen5 != correct_next:
                failures.append({
                    "prefix_tokens": prefix[0].cpu().tolist(),
                    "correct_token": correct_next,
                    "gen5_prediction": pred_gen5,
                })
                if len(failures) >= n_failures:
                    break

    print(f"  Found {len(failures)} failure cases")
    return failures


def causal_restoration_sweep(model_gen0, model_gen5, prompts, n_layers):
    """
    Causal restoration using corrupted/clean pair design.

    For each layer, patch Gen0 residual stream into Gen5 at that layer
    and measure how often Gen5 now predicts the correct token on failure cases.

    Recovery metric: fraction of failure cases where patching restores
    correct prediction. This removes positional bias because we only
    measure on cases where Gen5 specifically fails and Gen0 succeeds.

    High recovery at layer L = Gen5's failure at those cases is caused
    by damage at layer L specifically.
    """
    print("  Step 1: Finding failure prompts...")

    # Build corpus from eval prompts — expand each into many positions
    corpus = []
    for p in prompts:
        tokens = model_gen0.to_tokens(p)
        for start in range(0, max(1, tokens.shape[1] - 20), 5):
            end = min(start + 40, tokens.shape[1])
            sub = model_gen0.tokenizer.decode(tokens[0, start:end].tolist())
            corpus.append(sub)
    corpus = corpus * 5  # repeat to get enough failures

    failures = find_failure_prompts(model_gen0, model_gen5, corpus, n_failures=200)

    if len(failures) < 10:
        print("  WARNING: Too few failure cases found. Falling back to PPL method.")
        # Fall back to original PPL method if not enough failures
        return _ppl_sweep_fallback(model_gen0, model_gen5, prompts, n_layers)

    # Baseline: Gen5 accuracy on failure cases (should be ~0 by construction)
    baseline_gen5_correct = 0  # always 0 — these are failure cases by definition
    print(f"  Step 2: Patching each layer and measuring recovery...")
    print(f"  (Using {len(failures)} failure cases)")

    layer_results = {}

    for patch_layer in range(n_layers):
        restored_count = 0

        for case in failures:
            prefix_tokens = torch.tensor(
                [case["prefix_tokens"]], dtype=torch.long
            ).to(DEVICE)
            correct_token = case["correct_token"]

            # Get Gen0 residual stream at this layer
            with torch.no_grad():
                _, cache_gen0 = model_gen0.run_with_cache(prefix_tokens)
            gen0_resid = cache_gen0[
                f"blocks.{patch_layer}.hook_resid_post"
            ].clone()

            # Patch Gen5 at this layer with Gen0 activations
            def make_hook(gen0_act):
                def hook_fn(value, hook):
                    min_len = min(value.shape[1], gen0_act.shape[1])
                    value[:, :min_len, :] = gen0_act[:, :min_len, :]
                    return value
                return hook_fn

            with torch.no_grad():
                patched_logits = model_gen5.run_with_hooks(
                    prefix_tokens,
                    fwd_hooks=[(
                        f"blocks.{patch_layer}.hook_resid_post",
                        make_hook(gen0_resid)
                    )]
                )

            patched_pred = patched_logits[0, -1, :].argmax().item()
            if patched_pred == correct_token:
                restored_count += 1

            del cache_gen0, gen0_resid

        recovery = restored_count / len(failures)
        layer_results[patch_layer] = {
            "recovery": float(recovery),
            "restored_count": restored_count,
            "total_failures": len(failures),
            # Keep patched_ppl key for compatibility with plotting
            "patched_ppl": float(1.0 - recovery),
        }

        print(f"  Layer {patch_layer:2d}: restored {restored_count}/{len(failures)} "
              f"= {recovery:.3f} "
              f"({'▲' if recovery > 0.05 else ' '} significant)")

    torch.cuda.empty_cache()

    # Dummy baseline values for plotting compatibility
    gen5_ppls = [perplexity_from_logits(model_gen5, model_gen5.to_tokens(p))
                 for p in prompts[:5]]
    gen0_ppls = [perplexity_from_logits(model_gen0, model_gen0.to_tokens(p))
                 for p in prompts[:5]]

    return layer_results, float(np.mean(gen0_ppls)), float(np.mean(gen5_ppls))


def _ppl_sweep_fallback(model_gen0, model_gen5, prompts, n_layers):
    """Fallback PPL sweep if not enough failure cases found."""
    print("  Running PPL fallback sweep...")
    gen5_ppls = [perplexity_from_logits(model_gen5, model_gen5.to_tokens(p))
                 for p in prompts]
    gen0_ppls = [perplexity_from_logits(model_gen0, model_gen0.to_tokens(p))
                 for p in prompts]
    baseline_gen5_ppl = float(np.mean(gen5_ppls))
    baseline_gen0_ppl = float(np.mean(gen0_ppls))

    layer_results = {}
    for patch_layer in range(n_layers):
        patched_ppls = []
        for prompt in prompts:
            tokens = model_gen5.to_tokens(prompt)
            with torch.no_grad():
                _, cache_gen0 = model_gen0.run_with_cache(model_gen0.to_tokens(prompt))
            gen0_resid = cache_gen0[f"blocks.{patch_layer}.hook_resid_post"].clone()

            def make_hook(g0a):
                def hook_fn(v, hook):
                    ml = min(v.shape[1], g0a.shape[1])
                    v[:, :ml, :] = g0a[:, :ml, :]
                    return v
                return hook_fn

            with torch.no_grad():
                pl = model_gen5.run_with_hooks(
                    tokens,
                    fwd_hooks=[(f"blocks.{patch_layer}.hook_resid_post", make_hook(gen0_resid))]
                )
            lp = torch.nn.functional.log_softmax(pl[0, :-1], dim=-1)
            tlp = lp[torch.arange(tokens.shape[1]-1), tokens[0, 1:]]
            ppl = float(torch.exp(-tlp.mean()).item())
            patched_ppls.append(ppl)
            del cache_gen0, gen0_resid

        mppl = float(np.mean(patched_ppls))
        gap = baseline_gen5_ppl - baseline_gen0_ppl
        recovery = (baseline_gen5_ppl - mppl) / gap if gap > 0 else 0.0
        layer_results[patch_layer] = {"patched_ppl": mppl, "recovery": float(recovery)}
        print(f"  Layer {patch_layer:2d}: patched PPL={mppl:.3f}, recovery={recovery:.3f}")

    torch.cuda.empty_cache()
    return layer_results, baseline_gen0_ppl, baseline_gen5_ppl


def run_experiment_B():
    """
    Causal restoration patching: which blocks, when restored from Gen0,
    recover the most performance in the collapsed Gen5 model?
    Correlate recovery with FIM from the original paper.
    """
    print("\n" + "="*60)
    print("EXPERIMENT B: Causal Restoration Patching")
    print("="*60)
    print("Warning: This requires loading two models simultaneously.")
    print("For Pythia-1.4B this needs ~12GB VRAM. Using float16 to reduce.")

    results = {}
    output_path = os.path.join(OUTPUT_DIR, "causal_restoration_results.json")

    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        print("Loaded existing results.")

    fim_data = {
        model_name: load_fim_for_model(model_name)
        for model_name in CHECKPOINTS.keys()
    }

    for model_name, config in CHECKPOINTS.items():
        if model_name in results:
            print(f"\n{model_name}: already computed, skipping.")
            continue

        arch = config["arch"]
        n_layers = config["n_layers"]
        print(f"\n{model_name} ({arch}):")

        gen0_path = config["gens"][0]
        gen5_path = config["gens"][5]

        if not (os.path.exists(str(gen5_path)) or "EleutherAI" in str(gen5_path)):
            print(f"  Gen5 checkpoint not found: {gen5_path}")
            continue

        try:
            print("  Loading Gen0 model...")
            model_gen0 = load_tl_model(gen0_path, config["tl_name"])
            print("  Loading Gen5 model...")
            model_gen5 = load_tl_model(gen5_path, config["tl_name"])
        except Exception as e:
            print(f"  Load failed: {e}")
            continue

        layer_results, gen0_ppl, gen5_ppl = causal_restoration_sweep(
            model_gen0, model_gen5, EVAL_PROMPTS, n_layers
        )

        results[model_name] = {
            "baseline_gen0_ppl": gen0_ppl,
            "baseline_gen5_ppl": gen5_ppl,
            "layers": layer_results,
        }

        del model_gen0, model_gen5
        torch.cuda.empty_cache()
        gc.collect()

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    # Plot and FIM correlation
    _plot_and_correlate_experiment_B(results, fim_data)

    return results


def _plot_and_correlate_experiment_B(results, fim_data):
    from scipy.stats import spearmanr

    n_models = len(results)
    fig, axes = plt.subplots(2, n_models, figsize=(7 * n_models, 10))
    if n_models == 1:
        axes = axes.reshape(2, 1)

    for col, (model_name, model_results) in enumerate(results.items()):
        layers_data = {str(k): v for k, v in model_results["layers"].items()}
        n_layers = len(layers_data)
        layer_indices = sorted(int(l) for l in layers_data.keys())

        recoveries = [layers_data[str(l)]["recovery"] for l in layer_indices]
        fim_vals = [fim_data.get(model_name, {}).get(l, np.nan)
                    for l in layer_indices]

        # Top plot: recovery per layer
        ax_top = axes[0, col]
        colors = ['red' if r > 0.05 else 'steelblue' for r in recoveries]
        ax_top.bar(layer_indices, recoveries, color=colors, alpha=0.8)
        ax_top.set_xlabel("Layer", fontsize=11)
        ax_top.set_ylabel("PPL Recovery (0=none, 1=full)", fontsize=11)
        ax_top.set_title(
            f"{model_name} ({CHECKPOINTS[model_name]['arch']})\n"
            f"Gen0→Gen5 PPL gap closed by restoring each layer\n"
            f"Gen0={model_results['baseline_gen0_ppl']:.1f}, "
            f"Gen5={model_results['baseline_gen5_ppl']:.1f}",
            fontsize=10
        )
        ax_top.grid(True, alpha=0.3, axis='y')
        ax_top.axhline(y=0, color='black', linewidth=0.5)

        # Bottom plot: FIM vs recovery scatter
        ax_bot = axes[1, col]
        valid = [(f, r) for f, r in zip(fim_vals, recoveries)
                 if not np.isnan(f)]
        if valid:
            f_vals, r_vals = zip(*valid)
            ax_bot.scatter(np.log10(np.array(f_vals) + 1), r_vals,
                           alpha=0.7, s=50, color='steelblue', edgecolors='k')
            rho, p = spearmanr(f_vals, r_vals)
            ax_bot.set_title(
                f"FIM vs Recovery\nSpearman ρ={rho:+.3f} (p={p:.3f})",
                fontsize=11
            )
            # Annotate high-recovery layers
            for l, f, r in zip(layer_indices, fim_vals, recoveries):
                if not np.isnan(f) and r > 0.1:
                    ax_bot.annotate(f"L{l}", (np.log10(f + 1), r),
                                    fontsize=8, ha='center')
        ax_bot.set_xlabel("log₁₀(FIM_b) [Gen0]", fontsize=11)
        ax_bot.set_ylabel("PPL Recovery", fontsize=11)
        ax_bot.grid(True, alpha=0.3)

    plt.suptitle(
        "Causal Restoration: Which Layers Cause the Collapse?\n"
        "Red bars = significant recovery (>5% of gap closed)",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "causal_restoration_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {plot_path}")

    # Print FIM-recovery correlation
    print("\n" + "-"*60)
    print("FIM-RECOVERY CORRELATION (the key result):")
    print("If ρ > 0 for Pythia: high-FIM blocks cause collapse (FIM paper confirmed)")
    print("If ρ < 0 for SmolLM: high-FIM blocks are protected (FIM paper confirmed)")
    for model_name, model_results in results.items():
        from scipy.stats import spearmanr
        layers_data = {str(k): v for k, v in model_results["layers"].items()}
        layer_indices = sorted(int(l) for l in layers_data.keys())
        recoveries = [layers_data[str(l)]["recovery"] for l in layer_indices]
        fim_vals = [fim_data.get(model_name, {}).get(l, np.nan)
                    for l in layer_indices]
        valid = [(f, r) for f, r in zip(fim_vals, recoveries)
                 if not np.isnan(f)]
        if valid:
            f_vals, r_vals = zip(*valid)
            rho, p = spearmanr(f_vals, r_vals)
            arch = CHECKPOINTS[model_name]["arch"]
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "  ")
            print(f"  {model_name} ({arch}): rho={rho:+.3f}{sig} p={p:.4f}")
            print(f"  {'Layer':<8} {'log10FIM':>10} {'Recovery':>12}")
            fim_vals_print = [fim_data.get(model_name, {}).get(l, float('nan'))
                              for l in layer_indices]
            for l, f, r in zip(layer_indices, fim_vals_print, recoveries):
                fim_log = np.log10(f + 1e-8) if not np.isnan(f) else float('nan')
                marker = " <-- high recovery" if r > 0.1 else ""
                print(f"  {l:<8} {fim_log:>10.3f} {r:>12.4f}{marker}")
            print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["A", "B", "both"],
                        default="A",
                        help="A=logit lens, B=causal restoration, both=run both")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}")

    if args.experiment in ("A", "both"):
        run_experiment_A()

    if args.experiment in ("B", "both"):
        print("\nNote: Experiment B requires ~12GB VRAM for Pythia-1.4B.")
        print("If you run out of VRAM, try with SmolLM only by removing")
        print("'Pythia' from the CHECKPOINTS dict at the top of the script.")
        run_experiment_B()