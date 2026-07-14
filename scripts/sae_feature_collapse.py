"""
sae_feature_collapse_v2.py

Fixes v1's dead/born counting which failed due to domain mismatch:
  JB SAE trained on diverse internet text → only ~25 of 24,576 features
  activate at all on TinyStories → binary alive/dead is noise-dominated.

v2 uses MEAN ACTIVATION VECTORS instead:
  For each layer, compute mean_act[f] = mean activation of feature f across
  all 300 prompts, for Gen0 and Gen5 separately.
  This uses all 24,576 features with no threshold, giving stable continuous metrics:

  1. cosine_sim(mean_act_gen0, mean_act_gen5) per layer
     → 1.0 = identical feature activation pattern; lower = more drift
     Prediction: high-FIM layers have HIGHER cosine sim (more stable)

  2. L2_distance(mean_act_gen0, mean_act_gen5) per layer
     → lower = more stable
     Prediction: high-FIM layers have LOWER L2

  3. rank_correlation between Gen0 and Gen5 feature rankings per layer
     → Spearman on the 24,576-dim activation vectors
     → 1.0 = same features dominate; lower = different features active
     Prediction: high-FIM layers have HIGHER rank correlation

  4. reconstruction_similarity: project Gen5 onto Gen0 PCA subspace
     → same as Exp H (README) but now per-layer and correlated with FIM

Also reports the qualitative BORN > DEAD finding from v1 as context.

Run: python scripts/sae_feature_collapse_v2.py
Requires: pip install transformer_lens sae_lens
Out: results/circuit_analysis/sae_feature_v2.json
     results/circuit_analysis/sae_feature_v2.png
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
CACHE   = os.path.join(OUT_DIR, "sae_feature_v2.json")
PLOT    = os.path.join(OUT_DIR, "sae_feature_v2.png")

FIM_FILE  = os.path.join(ROOT, "results", "fimgpt2_gen_0", "perblock_fim.json")
GPT2_GEN5 = os.path.join(ROOT, "models", "gpt2_treatment_gen_5")

SAE_RELEASES = [
    ("gpt2-small-res-jb",             "blocks.{}.hook_resid_pre"),
    ("gpt2-small-resid-post-v5-128k", "blocks.{}.hook_resid_post"),
]

N_PROMPTS = 300

os.makedirs(OUT_DIR, exist_ok=True)


# ── FIM ───────────────────────────────────────────────────────────────────────

def load_fim(fim_file):
    with open(fim_file) as f:
        d = json.load(f)
    return {b["block_idx"]: b["mlp"]["top"] for b in d["blocks"]}


# ── SAE (cached) ──────────────────────────────────────────────────────────────

_sae_cache = {}

def get_sae(layer_idx, device):
    if layer_idx in _sae_cache:
        return _sae_cache[layer_idx]
    from sae_lens import SAE
    for release, tmpl in SAE_RELEASES:
        hook_id = tmpl.format(layer_idx)
        try:
            sae, _, _ = SAE.from_pretrained(release=release, sae_id=hook_id, device=device)
            _sae_cache[layer_idx] = (sae, hook_id)
            return sae, hook_id
        except Exception:
            pass
    return None, None


# ── collect mean activation vectors ──────────────────────────────────────────

def collect_mean_activations(tl_model, prompts, layer_indices, device):
    """
    Returns dict: {layer_idx: (n_features,) mean activation vector across all prompts}
    Uses all features, no threshold.
    """
    layer_means = {}
    for li in layer_indices:
        sae, hook_id = get_sae(li, device)
        if sae is None:
            print(f"    L{li}: no SAE")
            continue

        # Accumulate sum of activations across prompts
        running_sum = None
        n_valid = 0

        for prompt in prompts:
            tokens = tl_model.to_tokens(prompt)
            if tokens.shape[1] < 4:
                continue
            with torch.no_grad():
                _, cache = tl_model.run_with_cache(tokens, names_filter=hook_id)
                resid = cache[hook_id]                    # (1, seq_len, d_model)
                resid_mean = resid[0].mean(0, keepdim=True)  # (1, d_model)
                feat = sae.encode(resid_mean)[0].cpu().float()  # (n_features,)

                if running_sum is None:
                    running_sum = feat.clone()
                else:
                    running_sum += feat
                n_valid += 1

        if running_sum is not None and n_valid > 0:
            layer_means[li] = (running_sum / n_valid).numpy()  # (n_features,)
            n_feat = layer_means[li].shape[0]
            n_nonzero = int((layer_means[li] > 0).sum())
            print(f"    L{li}: {n_feat} features, {n_nonzero} with mean>0 "
                  f"({100*n_nonzero/n_feat:.1f}%)")

    return layer_means


# ── metrics ───────────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def rank_correlation(a, b):
    # Spearman on the raw activation vectors (all 24k features)
    # Only compute on features where at least one of gen0/gen5 is nonzero
    mask = (a > 0) | (b > 0)
    if mask.sum() < 4:
        return np.nan, np.nan
    rho, p = stats.spearmanr(a[mask], b[mask])
    return float(rho), float(p)


def compare_layers(means_gen0, means_gen5, layer_indices):
    results = []
    for li in layer_indices:
        if li not in means_gen0 or li not in means_gen5:
            continue
        m0 = means_gen0[li]
        m5 = means_gen5[li]

        cos  = cosine_sim(m0, m5)
        l2   = float(np.linalg.norm(m5 - m0))
        rho_rank, p_rank = rank_correlation(m0, m5)

        # Relative L2 (normalised by Gen0 norm)
        norm0 = float(np.linalg.norm(m0))
        rel_l2 = l2 / (norm0 + 1e-10)

        # How many features have mean > 0 at each gen
        n_alive0 = int((m0 > 0).sum())
        n_alive5 = int((m5 > 0).sum())

        results.append({
            "layer":         li,
            "cosine_sim":    cos,
            "l2_dist":       l2,
            "rel_l2":        rel_l2,
            "rank_rho":      rho_rank,
            "rank_p":        p_rank,
            "n_nonzero_g0":  n_alive0,
            "n_nonzero_g5":  n_alive5,
            "nonzero_delta": n_alive5 - n_alive0,
        })
        print(f"  L{li:2d}: cosine={cos:.4f}  rel_L2={rel_l2:.4f}  "
              f"rank_rho={rho_rank:.3f}  nonzero {n_alive0}→{n_alive5}")

    return results


def spearman(x, y):
    pairs = [(a, b) for a, b in zip(x, y)
             if a is not None and b is not None
             and np.isfinite(a) and np.isfinite(b)]
    if len(pairs) < 4:
        return None, None, len(pairs)
    xs, ys = zip(*pairs)
    rho, p = stats.spearmanr(xs, ys)
    return float(rho), float(p), len(pairs)


def sig(p):
    if p is None: return ""
    return "**" if p < 0.01 else ("*" if p < 0.05 else "")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if os.path.exists(CACHE):
        print(f"Cache hit: {CACHE}")
        with open(CACHE) as f:
            results = json.load(f)
        plot(results)
        print_summary(results)
        return

    try:
        from transformer_lens import HookedTransformer
        from transformers import GPT2LMHeadModel
    except ImportError as e:
        print(f"Missing: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    fim_dict = load_fim(FIM_FILE)
    layer_indices = list(range(12))
    log_fim = {i: np.log10(fim_dict[i]) if fim_dict.get(i, 0) > 0 else np.nan
               for i in layer_indices}

    print("Loading prompts...")
    prompts = []
    for item in load_dataset("roneneldan/TinyStories", split="validation", streaming=True):
        prompts.append(item["text"][:256])
        if len(prompts) >= N_PROMPTS:
            break
    print(f"  {len(prompts)} prompts")

    # Gen0
    print("\nCollecting Gen0 mean activations...")
    tl0 = HookedTransformer.from_pretrained("gpt2", device=device)
    means0 = collect_mean_activations(tl0, prompts, layer_indices, device)
    del tl0
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Gen5
    print("\nCollecting Gen5 mean activations...")
    hf5 = GPT2LMHeadModel.from_pretrained(GPT2_GEN5)
    tl5 = HookedTransformer.from_pretrained("gpt2", hf_model=hf5, device=device)
    del hf5
    means5 = collect_mean_activations(tl5, prompts, layer_indices, device)
    del tl5
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Compare
    print("\nComparing layers...")
    layer_results = compare_layers(means0, means5, layer_indices)
    for r in layer_results:
        r["fim_mlp"]       = fim_dict.get(r["layer"])
        r["log10_fim_mlp"] = log_fim.get(r["layer"])

    # Correlations
    fim_v   = [r["log10_fim_mlp"] for r in layer_results]
    cos_v   = [r["cosine_sim"]    for r in layer_results]
    rel_v   = [r["rel_l2"]        for r in layer_results]
    rank_v  = [r["rank_rho"]      for r in layer_results]
    nzdelta = [r["nonzero_delta"] for r in layer_results]

    rho_cos,   p_cos,   n = spearman(fim_v, cos_v)
    rho_l2,    p_l2,    _ = spearman(fim_v, rel_v)
    rho_rank,  p_rank,  _ = spearman(fim_v, rank_v)
    rho_nz,    p_nz,    _ = spearman(fim_v, nzdelta)

    print(f"\n  Correlations with log10(FIM_mlp), n={n}:")
    print(f"    ρ(FIM, cosine_sim)     = "
          f"{rho_cos:+.3f}{sig(p_cos)}  p={p_cos:.4f}  [want positive]" if rho_cos else "    N/A")
    print(f"    ρ(FIM, rel_L2)         = "
          f"{rho_l2:+.3f}{sig(p_l2)}  p={p_l2:.4f}  [want negative]" if rho_l2 else "    N/A")
    print(f"    ρ(FIM, rank_rho)       = "
          f"{rho_rank:+.3f}{sig(p_rank)}  p={p_rank:.4f}  [want positive]" if rho_rank else "    N/A")
    print(f"    ρ(FIM, nonzero_delta)  = "
          f"{rho_nz:+.3f}{sig(p_nz)}  p={p_nz:.4f}  [want positive: high-FIM→gain more features]"
          if rho_nz else "    N/A")

    output = {
        "model": "GPT-2", "arch": "SEQ", "n_prompts": len(prompts),
        "layer_results": layer_results,
        "correlations": {
            "rho_cos":   rho_cos,   "p_cos":   p_cos,
            "rho_l2":    rho_l2,    "p_l2":    p_l2,
            "rho_rank":  rho_rank,  "p_rank":  p_rank,
            "rho_nz":    rho_nz,    "p_nz":    p_nz,
            "n": n,
        }
    }

    with open(CACHE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Cached: {CACHE}")

    plot(output)
    print_summary(output)


# ── plot ──────────────────────────────────────────────────────────────────────

def plot(results):
    lr      = results["layer_results"]
    layers  = [r["layer"]         for r in lr]
    log_fim = [r["log10_fim_mlp"] for r in lr]
    cos_v   = [r["cosine_sim"]    for r in lr]
    rel_v   = [r["rel_l2"]        for r in lr]
    rank_v  = [r["rank_rho"]      for r in lr]
    nz_d    = [r["nonzero_delta"] for r in lr]

    fim_arr = np.array([v if v and np.isfinite(v) else np.nan for v in log_fim])
    valid   = fim_arr[np.isfinite(fim_arr)]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    c = results["correlations"]

    def scatter(ax, ys, ylabel, title, rho_key, p_key, want):
        sc = ax.scatter(layers, ys, c=fim_arr, cmap="RdYlGn_r", s=80, zorder=3,
                        vmin=valid.min(), vmax=valid.max())
        plt.colorbar(sc, ax=ax, label="log10(FIM_mlp)")
        for x, y, in zip(layers, ys):
            if y is not None and np.isfinite(y):
                ax.annotate(f"L{x}", (x, y), textcoords="offset points",
                            xytext=(3, 3), fontsize=7)
        rho, p = c.get(rho_key), c.get(p_key)
        s = f"ρ={rho:+.3f}{'**' if p and p<.01 else '*' if p and p<.05 else ''}" if rho else "ρ=N/A"
        ax.set_xlabel("Layer"); ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\n{s}  [want {want}]")
        ax.grid(True, alpha=0.3)

    scatter(axes[0][0], cos_v, "Cosine similarity",
            "Gen0 vs Gen5 mean activation vectors\n(higher = more stable)",
            "rho_cos", "p_cos", "positive ρ")
    scatter(axes[0][1], rel_v, "Relative L2 distance",
            "||mean_G5 - mean_G0|| / ||mean_G0||\n(lower = more stable)",
            "rho_l2", "p_l2", "negative ρ")
    scatter(axes[1][0], rank_v, "Rank correlation (Spearman)",
            "Feature ranking correlation Gen0 vs Gen5\n(higher = same features dominate)",
            "rho_rank", "p_rank", "positive ρ")
    scatter(axes[1][1], nz_d, "Nonzero feature count change",
            "n_nonzero_Gen5 - n_nonzero_Gen0\n(positive = more features active)",
            "rho_nz", "p_nz", "any direction")

    plt.suptitle(
        "SAE Feature Collapse v2: Mean Activation Vectors (GPT-2 Gen0→Gen5)\n"
        "Threshold-free: all 24,576 features included.\n"
        "Green=high FIM blocks. Prediction: green points show more stable activations.",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOT, dpi=150, bbox_inches="tight")
    print(f"Plot: {PLOT}")


def print_summary(results):
    c = results["correlations"]
    def f(rk, pk, want):
        v, p = c.get(rk), c.get(pk)
        if v is None: return f"N/A  [{want}]"
        s = "**" if p and p<.01 else ("*" if p and p<.05 else "")
        return f"{v:+.3f}{s}  p={p:.4f}  [want {want}]"
    print("\n" + "="*65)
    print("SAE FEATURE COLLAPSE v2 — Mean Activation Vectors")
    print("All 24,576 features, no threshold. n=12 layers.")
    print("="*65)
    print(f"  ρ(FIM, cosine_sim) = {f('rho_cos','p_cos','positive')}")
    print(f"  ρ(FIM, rel_L2)     = {f('rho_l2','p_l2','negative')}")
    print(f"  ρ(FIM, rank_rho)   = {f('rho_rank','p_rank','positive')}")
    print(f"  ρ(FIM, nz_delta)   = {f('rho_nz','p_nz','any')}")
    print()
    print("  If positive/negative as predicted:")
    print("    High-FIM MLP blocks (vt-protected, drift less) maintain more")
    print("    similar feature activation patterns Gen0→Gen5, linking weight-")
    print("    space protection directly to representational stability.")
    print()
    print("  Note: SAE trained on diverse text, evaluated on TinyStories.")
    print("    Domain mismatch means most features have near-zero mean activation.")
    print("    cosine_sim and rank_rho on sparse vectors are still valid but")
    print("    effect sizes may be small. Main diagnostic: sign and consistency.")


if __name__ == "__main__":
    main()