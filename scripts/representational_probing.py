"""
Experiment E: Representational Probing Across Generations
============================================================
Based on: "WeightLens and CircuitLens" (Golimblevskaia et al., 2025)
          "Logit Lens" (nostalgebraist, 2020)

The logit lens from Experiment A showed WHERE the model commits to predictions.
This experiment asks WHAT information is linearly decodable from the residual
stream at each layer as collapse progresses.

Core idea: if a property P is linearly decodable from layer L's residual stream,
then layer L is computing something about P. If that decodability drops across
generations, that computation is being destroyed by collapse.

What properties to probe:
  1. Next-token identity: can we predict the correct next token from residual stream?
     Drop = the layer is no longer building towards the right prediction.
  2. Sentence boundary: can we predict whether the current token is end-of-sentence?
     Drop = syntactic structure is being lost.
  3. Token position: can we predict rough position in sequence?
     Stable = positional encoding is intact (basic sanity check).

Why this connects to the FIM paper:
  The FIM paper shows high-FIM blocks drift less in sequential models.
  If high-FIM blocks also maintain their representational decodability better
  than low-FIM blocks, that confirms they are functionally protected — not just
  weight-level protected.

  This is the representational confirmation of the FIM paper:
  weight protection → representational protection.

Metric: probing accuracy = linear probe (logistic regression) trained on
Gen0 activations and evaluated on Gen0. Then evaluate the SAME probe on
GenN activations. Drop in accuracy = representational drift.

This avoids training a new probe per generation — we're asking: does the
Gen0 probe still work on GenN activations? If not, the representation has moved.

Usage:
  python scripts/representational_probing.py
"""

import os
import gc
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

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
        "n_layers": 12,
    },
    "Pythia": {
        "arch": "parallel",
        "checkpoints": {
            0: "EleutherAI/pythia-1.4b",
            1: os.path.join(BASE_DIR, "models", "pythia-1.4b_treatment_gen_1"),
            3: os.path.join(BASE_DIR, "models", "pythia-1.4b_treatment_gen_3"),
            5: os.path.join(BASE_DIR, "models", "pythia-1.4b_treatment_gen_5"),
        },
        "n_layers": 24,
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
N_SAMPLES = 200   # texts to probe on
MAX_LEN   = 64    # tokens per text
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"


def get_layer_module(model, layer_idx, model_name):
    if model_name == "GPT2":
        return model.transformer.h[layer_idx]
    elif model_name == "Pythia":
        return model.gpt_neox.layers[layer_idx]
    raise ValueError(f"Unknown model: {model_name}")


def collect_residual_stream(model, tokenizer, texts, layer_idx, model_name):
    """
    Collect residual stream activations at a given layer for all texts.
    Returns: (activations [n_tokens, hidden], labels_next_token [n_tokens])
    """
    all_acts = []
    all_next_tokens = []
    all_is_eos = []

    hook_data = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        hook_data['act'] = hidden.detach().cpu().float()

    model.eval()
    layer_mod = get_layer_module(model, layer_idx, model_name)
    hook = layer_mod.register_forward_hook(hook_fn)

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=MAX_LEN, padding=False
            ).to(DEVICE)

            if enc["input_ids"].shape[1] < 3:
                continue

            _ = model(**enc)

            if "act" not in hook_data:
                continue

            acts = hook_data["act"][0]  # [seq_len, hidden]
            ids  = enc["input_ids"][0]  # [seq_len]

            # For each position except last, record:
            # - activation at this position
            # - next token (prediction target)
            # - whether this is end of sentence (simplified: token is period-like)
            n = acts.shape[0]
            for pos in range(n - 1):
                all_acts.append(acts[pos].numpy())
                all_next_tokens.append(int(ids[pos + 1].item()))

                # Sentence boundary: current token is sentence-ending punctuation
                # Use multiple indicators since TinyStories uses varied endings
                tok_id = int(ids[pos].item())
                tok_str = tokenizer.decode([tok_id]).strip()
                is_sentence_end = int(
                    tok_str in ['.', '!', '?', '..."', '."', '!"', '?"'] or
                    tok_id == tokenizer.eos_token_id
                )
                all_is_eos.append(is_sentence_end)

    hook.remove()

    if not all_acts:
        return None, None, None

    return (
        np.stack(all_acts),
        np.array(all_next_tokens),
        np.array(all_is_eos),
    )


def train_probe_gen0(acts_gen0, labels, probe_type="next_token"):
    """
    Train a linear probe on Gen0 activations.
    For next_token: too many classes — use top-100 only.
    For eos: binary classification.
    Returns: (probe, scaler, valid_mask, accuracy_gen0)
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(acts_gen0)

    if probe_type == "next_token":
        # Keep only top-100 most common next tokens
        unique, counts = np.unique(labels, return_counts=True)
        top_tokens = set(unique[np.argsort(-counts)[:100]])
        mask = np.array([l in top_tokens for l in labels])
        if mask.sum() < 50:
            return None, None, None, 0.0
        X_filtered = X[mask]
        y_filtered = labels[mask]
    elif probe_type == "eos":
        mask = np.ones(len(labels), dtype=bool)
        X_filtered = X
        y_filtered = labels
    else:
        return None, None, None, 0.0

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probe = LogisticRegression(max_iter=500, C=0.1, solver='lbfgs')
            probe.fit(X_filtered, y_filtered)
        acc = accuracy_score(y_filtered, probe.predict(X_filtered))
    except Exception as e:
        return None, None, None, 0.0

    return probe, scaler, mask, float(acc)


def eval_probe_genN(probe, scaler, acts_genN, labels, mask):
    """Evaluate a Gen0 probe on GenN activations."""
    if probe is None:
        return float('nan')
    X = scaler.transform(acts_genN)
    if mask is not None:
        X = X[mask]
        labels = labels[mask]
    try:
        acc = accuracy_score(labels, probe.predict(X))
    except Exception:
        return float('nan')
    return float(acc)


def run_analysis():
    # Load GPT-2 FIM
    import re
    for path in [
        os.path.join(BASE_DIR, "results", "gpt2_treatment_gen_0", "perblock_fim.json"),
        os.path.join(BASE_DIR, "results", "fimgpt2_gen_0.txt"),
    ]:
        if not os.path.exists(path):
            continue
        if path.endswith(".json"):
            with open(path) as f:
                data = json.load(f)
            for entry in data.get("blocks", []):
                b = entry.get("block_idx", 0)
                attn = entry.get("attention", {})
                mlp = entry.get("mlp", {})
                at = float(attn.get("top", 0)) if isinstance(attn, dict) and "error" not in attn else 0.0
                mt = float(mlp.get("top", 0)) if isinstance(mlp, dict) and "error" not in mlp else 0.0
                if at > 0 or mt > 0:
                    GEN0_FIM["GPT2"][b] = at + mt
            if GEN0_FIM["GPT2"]:
                break

    # Load texts
    print("Loading TinyStories validation set...")
    ds = load_dataset("roneneldan/TinyStories", split=f"validation[:{N_SAMPLES}]")
    texts = ds["text"]

    results = {}
    output_path = os.path.join(OUTPUT_DIR, "representational_probing.json")

    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        print("Loaded existing probing results.")

    for model_name, config in MODELS.items():
        if model_name in results:
            print(f"{model_name}: cached.")
            continue

        print(f"\n{'='*60}")
        print(f"Representational Probing: {model_name} ({config['arch']})")
        print(f"{'='*60}")

        n_layers = config["n_layers"]

        # Load Gen0 and train probes
        print("  Loading Gen0 and training probes...")
        try:
            m0 = AutoModelForCausalLM.from_pretrained(
                config["checkpoints"][0], torch_dtype=torch.float16,
                attn_implementation="eager"
            ).to(DEVICE)
            tok0 = AutoTokenizer.from_pretrained(config["checkpoints"][0])
            if tok0.pad_token is None:
                tok0.pad_token = tok0.eos_token
        except Exception as e:
            print(f"  Gen0 load failed: {e}")
            continue

        # Train probes at each layer on Gen0
        probes = {}  # layer -> (probe_next, scaler_next, mask_next, probe_eos, scaler_eos, mask_eos, acc0_next, acc0_eos)
        # Sample every 3rd layer to reduce compute and noise
        probe_layers = list(range(0, n_layers, 3)) + [n_layers - 1]
        probe_layers = sorted(set(probe_layers))
        for layer in probe_layers:
            acts, next_toks, is_eos = collect_residual_stream(
                m0, tok0, texts, layer, model_name
            )
            if acts is None:
                continue

            probe_next, scaler_next, mask_next, acc_next = train_probe_gen0(
                acts, next_toks, "next_token"
            )
            probe_eos, scaler_eos, mask_eos, acc_eos = train_probe_gen0(
                acts, is_eos, "eos"
            )

            probes[layer] = {
                "probe_next": probe_next, "scaler_next": scaler_next,
                "mask_next": mask_next, "acc0_next": acc_next,
                "probe_eos": probe_eos, "scaler_eos": scaler_eos,
                "mask_eos": mask_eos, "acc0_eos": acc_eos,
                "next_toks_gen0": next_toks,
                "is_eos_gen0": is_eos,
            }

            if layer % 4 == 0:
                print(f"  Layer {layer}: probe_next acc={acc_next:.3f}, probe_eos acc={acc_eos:.3f}")

        del m0; torch.cuda.empty_cache(); gc.collect()

        model_results = {
            "gen0": {
                str(layer): {
                    "acc_next": probes[layer]["acc0_next"],
                    "acc_eos": probes[layer]["acc0_eos"],
                }
                for layer in probes
            }
        }

        # Evaluate probes on GenN activations
        for gen, ckpt in config["checkpoints"].items():
            if gen == 0:
                continue
            if not os.path.exists(str(ckpt)):
                continue

            print(f"\n  Gen{gen}: evaluating probes...")
            try:
                mN = AutoModelForCausalLM.from_pretrained(
                    ckpt, torch_dtype=torch.float16,
                    attn_implementation="eager"
                ).to(DEVICE)
                tokN = AutoTokenizer.from_pretrained(ckpt)
                if tokN.pad_token is None:
                    tokN.pad_token = tokN.eos_token
            except Exception as e:
                print(f"  Gen{gen} load failed: {e}")
                continue

            gen_results = {}
            for layer in probe_layers:
                if layer not in probes:
                    continue
                p = probes[layer]
                acts_N, next_toks_N, is_eos_N = collect_residual_stream(
                    mN, tokN, texts, layer, model_name
                )
                if acts_N is None:
                    continue

                acc_next_N = eval_probe_genN(
                    p["probe_next"], p["scaler_next"], acts_N,
                    next_toks_N, p["mask_next"]
                )
                acc_eos_N = eval_probe_genN(
                    p["probe_eos"], p["scaler_eos"], acts_N,
                    is_eos_N, p["mask_eos"]
                )

                gen_results[str(layer)] = {
                    "acc_next": acc_next_N,
                    "acc_eos": acc_eos_N,
                    "drop_next": float(p["acc0_next"] - acc_next_N),
                    "drop_eos": float(p["acc0_eos"] - acc_eos_N),
                }

            model_results[str(gen)] = gen_results
            del mN; torch.cuda.empty_cache(); gc.collect()
            print(f"  Gen{gen}: mean drop_next={np.nanmean([v['drop_next'] for v in gen_results.values()]):.3f}")

        results[model_name] = model_results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    return results


def plot_and_correlate(results):
    from scipy.stats import spearmanr

    print("\n" + "="*60)
    print("REPRESENTATIONAL PROBING: FIM CORRELATION")
    print("="*60)
    print("Key question: do high-FIM blocks maintain probe accuracy better?")
    print("If yes at Gen5: FIM paper confirmed at representational level")
    print()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, (model_name, model_results) in enumerate(results.items()):
        arch = MODELS[model_name]["arch"]
        fim_data = GEN0_FIM.get(model_name, {})

        gen0_data = model_results.get("gen0", {})
        gen5_data = model_results.get("5", {})
        if not gen0_data or not gen5_data:
            print(f"  {model_name}: missing gen0 or gen5 data")
            continue

        layers = sorted(int(l) for l in gen5_data.keys())

        # Drop in probe accuracy at Gen5
        drops_next = [gen5_data.get(str(l), {}).get("drop_next", np.nan) for l in layers]
        drops_eos  = [gen5_data.get(str(l), {}).get("drop_eos", np.nan) for l in layers]
        fim_vals   = [fim_data.get(l, np.nan) for l in layers]

        # Trajectory: probe accuracy across gens
        ax_traj = axes[0, col]
        gens_available = sorted([int(g) for g in model_results.keys() if g.isdigit()])
        for gen in gens_available:
            gen_data = model_results.get(str(gen), {})
            accs = [gen_data.get(str(l), {}).get("acc_next", np.nan) for l in layers]
            ax_traj.plot(layers, accs, 'o-', label=f"Gen{gen}",
                        linewidth=1.5, markersize=3, alpha=0.8)

        # Gen0 accuracy
        acc0s = [gen0_data.get(str(l), {}).get("acc_next", np.nan) for l in layers]
        ax_traj.plot(layers, acc0s, 'o-', color='black', label="Gen0",
                    linewidth=2, markersize=4)

        ax_traj.set_xlabel("Layer", fontsize=11)
        ax_traj.set_ylabel("Next-Token Probe Accuracy", fontsize=11)
        ax_traj.set_title(f"{model_name} ({arch})\nProbe Accuracy Trajectory", fontsize=11)
        ax_traj.legend(fontsize=8)
        ax_traj.grid(True, alpha=0.3)

        # FIM vs accuracy drop scatter
        ax_corr = axes[1, col]
        valid = [(f, d) for f, d in zip(fim_vals, drops_next)
                 if not np.isnan(f) and not np.isnan(d)]
        if valid:
            f_vals, d_vals = zip(*valid)
            log_f = np.log10(np.array(f_vals) + 1e-8)
            ax_corr.scatter(log_f, d_vals, alpha=0.7, s=60,
                           color='steelblue' if arch == 'sequential' else 'tomato',
                           edgecolors='k')
            rho, p = spearmanr(f_vals, d_vals)
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            ax_corr.set_title(
                f"{model_name}: FIM vs Probe Accuracy Drop\n"
                f"rho={rho:+.3f}{sig} p={p:.4f}",
                fontsize=11
            )
            for l, f, d in zip(layers, fim_vals, drops_next):
                if not np.isnan(f) and not np.isnan(d) and abs(d) > 0.05:
                    ax_corr.annotate(str(l), (np.log10(f + 1e-8), d),
                                    fontsize=7, alpha=0.6)

            print(f"{model_name} ({arch}):")
            print(f"  FIM ~ Probe Drop: rho={rho:+.3f}{sig} p={p:.4f}")
            expected = "rho < 0 (high FIM = less drop = more protected)" if arch == "sequential" else "rho > 0 (high FIM = more drop)"
            print(f"  Expected: {expected}")
            confirmed = (arch == "sequential" and rho < -0.3) or (arch == "parallel" and rho > 0.3)
            print(f"  {'CONFIRMED' if confirmed else 'NOT CONFIRMED'}")
            print()

        ax_corr.set_xlabel("log10(FIM_b) Gen0", fontsize=11)
        ax_corr.set_ylabel("Probe Accuracy Drop (Gen0 - Gen5)", fontsize=11)
        ax_corr.grid(True, alpha=0.3)

    plt.suptitle(
        "Representational Probing: Is Gen0 Information Still Decodable at Gen5?\n"
        "Probe trained on Gen0, evaluated on Gen5. Drop = information lost.",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "representational_probing.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {plot_path}")


if __name__ == "__main__":
    print("Representational Probing Analysis")
    print("Tracks what information remains linearly decodable after collapse")
    print()
    results = run_analysis()
    plot_and_correlate(results)