"""
pythia_freeze_control.py  —  Random and low-FIM freeze controls

Runs control and dynamic conditions against the FIM-targeted freeze:
  --mode random   : freeze 6 randomly chosen MLP blocks (not by FIM)
  --mode low_fim  : freeze the 6 LOWEST-FIM MLP blocks
  --mode dynamic  : recompute FIM from current model each generation,
                    unfreeze previous blocks, freeze new top-6

These controls isolate whether the Gen1 immunity in the FIM-targeted freeze
is due to FIM-targeting specifically, or just due to reducing trainable
parameter count by 14.2%.

Expected outcomes:
  random freeze:  if Gen1 PPL ≈ standard +15.6%  → FIM targeting matters
                  if Gen1 PPL ≈ freeze   -1.5%   → any 6 blocks would work
  low_fim freeze: should show LESS protection than top-FIM freeze, since
                  low-FIM blocks drift less anyway and freezing them changes
                  little about the collapse dynamics

Everything else is identical to fim_block_freeze_pythia.py:
  same training loop, same resume logic, same PPL eval, same rho computation.
  Synthetic data from the FIM-freeze run is REUSED to ensure identical training
  distribution — only the frozen block selection changes.

Run:
  python scripts/pythia_freeze_control.py --mode random
  python scripts/pythia_freeze_control.py --mode low_fim
  python scripts/pythia_freeze_control.py --mode dynamic

Out:
  results/fim_block_freeze/random_summary.json
  results/fim_block_freeze/low_fim_summary.json
"""

import argparse
import json
import os
import gc
import random
import torch
import numpy as np
from scipy import stats
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import (
    GPTNeoXForCausalLM, AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# ── config (must match fim_block_freeze_pythia.py exactly) ───────────────────
ROOT     = r"D:\Thaman\Work\hessian-spectral-analysis"
OUT_DIR  = os.path.join(ROOT, "results", "fim_block_freeze")
FIM_FILE = os.path.join(ROOT, "results", "pythia-1.4b_treatment_gen_0", "perblock_fim.json")

BASE_MODEL   = "EleutherAI/pythia-1.4b"
N_GENS       = 5
N_TRAIN      = 50_000
SEQ_LEN      = 256
FREEZE_K     = 6
BATCH_SIZE   = 8
LR           = 5e-5
WEIGHT_DECAY = 0.01
PPL_SAMPLES  = 500
RANDOM_SEED  = 42

os.makedirs(OUT_DIR, exist_ok=True)


# ── block selection ───────────────────────────────────────────────────────────

def get_top_fim_block_indices(fim_file, k):
    with open(fim_file) as f:
        d = json.load(f)
    block_fim = [(b["block_idx"], b["mlp"]["top"]) for b in d["blocks"]]
    block_fim.sort(key=lambda x: x[1], reverse=True)
    return {idx for idx, _ in block_fim[:k]}


def get_low_fim_block_indices(fim_file, k):
    with open(fim_file) as f:
        d = json.load(f)
    block_fim = [(b["block_idx"], b["mlp"]["top"]) for b in d["blocks"]]
    block_fim.sort(key=lambda x: x[1])   # ascending — lowest FIM first
    low = [(idx, fim) for idx, fim in block_fim[:k]]
    print(f"  Low-{k} FIM MLP blocks to freeze:")
    for idx, fim in sorted(low):
        print(f"    Block {idx:2d}: FIM_mlp = {fim:.1f}")
    return {idx for idx, _ in low}


def get_random_block_indices(fim_file, k, seed=RANDOM_SEED):
    with open(fim_file) as f:
        d = json.load(f)
    all_indices = [b["block_idx"] for b in d["blocks"]]
    rng = random.Random(seed)
    chosen = sorted(rng.sample(all_indices, k))
    print(f"  Random-{k} MLP blocks to freeze (seed={seed}): {chosen}")
    return set(chosen)


def get_dynamic_fim_block_indices(model, k, gen, device):
    """
    Recompute per-block FIM_mlp from the current model state.
    Uses gradient norm squared (approx trace of Fisher) as the ranking metric —
    same ordinal ordering as lambda_max for ranking purposes, much cheaper.
    Runs on 60 TinyStories validation samples (fast, ~2 min per call).
    Returns top-k block indices by current FIM_mlp approximation.
    """
    from datasets import load_dataset as _load_dataset

    print(f"  Computing dynamic FIM for Gen{gen} model...")
    n_layers = model.config.num_hidden_layers

    ds = _load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    texts = []
    for item in ds:
        texts.append(item["text"][:256])
        if len(texts) >= 60:
            break

    tokenizer_tmp = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer_tmp.pad_token is None:
        tokenizer_tmp.pad_token = tokenizer_tmp.eos_token

    block_fim_mlp = {i: 0.0 for i in range(n_layers)}

    model.eval()
    for text in texts:
        enc = tokenizer_tmp(text, return_tensors="pt", truncation=True,
                            max_length=128).to(device)
        if enc["input_ids"].shape[1] < 4:
            continue

        model.zero_grad()
        out  = model(**enc, labels=enc["input_ids"])
        out.loss.backward()

        for i in range(n_layers):
            grads = [p.grad.float().flatten()
                     for p in model.gpt_neox.layers[i].mlp.parameters()
                     if p.grad is not None]
            if grads:
                g = torch.cat(grads)
                # ||g||^2 approximates trace(Fisher) — same block ranking as lambda_max
                block_fim_mlp[i] += float((g * g).sum().item())

        model.zero_grad()

    n = max(len(texts), 1)
    for i in block_fim_mlp:
        block_fim_mlp[i] /= n

    # Normalise by number of samples
    n = max(len(texts), 1)
    for i in block_fim_mlp:
        block_fim_mlp[i] /= n

    # Sort descending, take top-k
    sorted_blocks = sorted(block_fim_mlp.items(), key=lambda x: x[1], reverse=True)
    top_k = sorted_blocks[:k]
    print(f"  Dynamic top-{k} FIM MLP blocks at Gen{gen}:")
    for idx, fim in sorted(top_k, key=lambda x: x[0]):
        print(f"    Block {idx:2d}: FIM_mlp ≈ {fim:.4f}")
    return {idx for idx, _ in top_k}, dict(block_fim_mlp)


def unfreeze_all_mlp(model):
    """Unfreeze all MLP blocks before applying new dynamic freeze."""
    for layer in model.gpt_neox.layers:
        for p in layer.mlp.parameters():
            p.requires_grad = True


def load_fim_all(fim_file):
    with open(fim_file) as f:
        d = json.load(f)
    return {b["block_idx"]: b["mlp"]["top"] for b in d["blocks"]}


# ── model setup ───────────────────────────────────────────────────────────────

def freeze_mlp_blocks(model, block_indices):
    frozen = 0
    for idx in block_indices:
        for p in model.gpt_neox.layers[idx].mlp.parameters():
            p.requires_grad = False
            frozen += p.numel()
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Frozen {frozen:,} params. Trainable: {trainable:,} / {total:,} "
          f"({100*trainable/total:.1f}%)")


# ── data: REUSE synthetic data from FIM-freeze run ────────────────────────────

def load_or_generate_synthetic(gen, model, tokenizer, device):
    """
    Reuse synthetic_gen_N.txt from the FIM-freeze run if available.
    This ensures identical training distribution across all conditions.
    If not available, generate fresh.
    """
    synth_path = os.path.join(OUT_DIR, f"synthetic_gen_{gen}.txt")
    if os.path.exists(synth_path):
        with open(synth_path, encoding="utf-8") as f:
            texts = f.read().split("\n---\n")
        texts = [t for t in texts if t.strip()][:N_TRAIN]
        print(f"  Reusing synthetic data: {synth_path} ({len(texts)} samples)")
        return texts

    # Generate fresh if not available
    print(f"  Generating {N_TRAIN} synthetic samples (no cache found)...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    prompts = []
    for item in ds:
        prompts.append(item["text"][:64])
        if len(prompts) >= N_TRAIN:
            break

    model.eval()
    generated = []
    bs = 16
    tokenizer.padding_side = "left"
    with torch.no_grad():
        pbar = tqdm(total=N_TRAIN, desc="Generating")
        for i in range(0, len(prompts), bs):
            batch = prompts[i:i+bs]
            enc = tokenizer(batch, return_tensors="pt", truncation=True,
                            max_length=64, padding=True).to(device)
            out = model.generate(
                **enc, max_new_tokens=192, do_sample=True,
                temperature=0.7, top_k=50, top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            for seq in out:
                generated.append(tokenizer.decode(seq, skip_special_tokens=True))
            pbar.update(len(batch))
        pbar.close()

    # Save so it can be reused
    with open(synth_path, "w", encoding="utf-8") as f:
        f.write("\n---\n".join(generated[:N_TRAIN]))
    return generated[:N_TRAIN]


# ── training ──────────────────────────────────────────────────────────────────

def train_one_gen(model, tokenizer, texts, device):
    class TextDS(TorchDataset):
        def __init__(self, texts, tok, maxlen):
            tok.padding_side = "right"
            self.enc = tok(texts, truncation=True, max_length=maxlen,
                           padding="max_length", return_tensors="pt")
        def __len__(self): return self.enc["input_ids"].shape[0]
        def __getitem__(self, i): return {k: v[i] for k, v in self.enc.items()}

    ds     = TextDS(texts, tokenizer, SEQ_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY,
                               betas=(0.9, 0.999))
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=100, num_training_steps=len(loader))

    model.train()
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss  = model(**batch, labels=batch["input_ids"]).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optim.step(); sched.step(); optim.zero_grad()
        pbar.set_postfix({"loss": f"{loss.item():.3f}"})

    tokenizer.padding_side = "left"


# ── evaluation ────────────────────────────────────────────────────────────────

def compute_ppl(model, tokenizer, device):
    ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    total_loss, total_tok = 0.0, 0
    model.eval()
    with torch.no_grad():
        for i, item in enumerate(ds):
            if i >= PPL_SAMPLES: break
            enc = tokenizer(item["text"], return_tensors="pt",
                            truncation=True, max_length=SEQ_LEN).to(device)
            if enc["input_ids"].shape[1] < 4: continue
            out = model(**enc, labels=enc["input_ids"])
            n = enc["input_ids"].shape[1]
            total_loss += out.loss.item() * n
            total_tok  += n
    return float(np.exp(total_loss / total_tok))


def compute_fim_drift_corr(model_g0_cpu, model_cur, fim_all, frozen_idx, device):
    log_fim, drifts = [], []
    model_cur.to("cpu")
    for idx, fim_val in fim_all.items():
        if idx in frozen_idx: continue
        w0 = model_g0_cpu.gpt_neox.layers[idx].mlp.dense_h_to_4h.weight.float()
        wn = model_cur.gpt_neox.layers[idx].mlp.dense_h_to_4h.weight.float()
        drift = (wn - w0).norm().item() / (w0.norm().item() + 1e-8)
        log_fim.append(np.log10(fim_val) if fim_val > 0 else np.nan)
        drifts.append(drift)
    model_cur.to(device)
    pairs = [(f, d) for f, d in zip(log_fim, drifts)
             if np.isfinite(f) and np.isfinite(d)]
    if len(pairs) < 4: return None, None, len(pairs)
    xs, ys = zip(*pairs)
    rho, p = stats.spearmanr(xs, ys)
    return float(rho), float(p), len(pairs)


def sig(p):
    if p is None: return ""
    return "**" if p < 0.01 else ("*" if p < 0.05 else "")


def save_results(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["random", "low_fim", "dynamic"], required=True,
                        help="random: freeze 6 random blocks; low_fim: freeze 6 lowest-FIM; "
                             "dynamic: recompute FIM each gen and refreeze top-6")
    args = parser.parse_args()

    summary_path = os.path.join(OUT_DIR, f"{args.mode}_summary.json")
    model_prefix = f"pythia_freeze_{args.mode}"

    # Resume
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            results = json.load(f)
        results["ppl"]  = {str(k): v for k, v in results["ppl"].items()}
        results["corr"] = {str(k): v for k, v in results["corr"].items()}
    else:
        results = {
            "config": {"mode": args.mode, "freeze_k": FREEZE_K,
                       "n_train": N_TRAIN, "frozen_blocks": None},
            "ppl": {}, "corr": {},
            "paper_baseline": "+1426% Pythia standard treatment Gen0→Gen5",
            "fim_freeze_baseline": "Gen1=-1.5%, Gen2=+64.0% (FIM-targeted freeze)",
        }

    completed = [int(k) for k in results["ppl"].keys()]
    last_done = max(completed) if completed else -1

    if last_done >= N_GENS:
        print("Complete.")
        print_summary(results, args.mode)
        return

    if last_done >= 0:
        print(f"Resuming {args.mode} from Gen{last_done + 1}")
    else:
        print(f"Starting {args.mode} freeze fresh.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  mode={args.mode}  |  N_TRAIN={N_TRAIN}")

    # Initial block selection (dynamic uses Gen0 FIM for first freeze,
    # then recomputes each generation inside the loop)
    if args.mode == "random":
        frozen_idx = get_random_block_indices(FIM_FILE, FREEZE_K)
    elif args.mode == "low_fim":
        frozen_idx = get_low_fim_block_indices(FIM_FILE, FREEZE_K)
    else:  # dynamic — start with Gen0 FIM-based top-k, recompute each gen
        frozen_idx = get_top_fim_block_indices(FIM_FILE, FREEZE_K)
        print(f"  Dynamic mode: initial freeze from Gen0 FIM, will recompute each gen")

    fim_all = load_fim_all(FIM_FILE)
    results["config"]["frozen_blocks"] = sorted(frozen_idx)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    if last_done >= 1:
        ckpt = os.path.join(ROOT, "models", f"{model_prefix}_gen_{last_done}")
        if os.path.exists(ckpt):
            print(f"Loading checkpoint: {ckpt}")
            model = GPTNeoXForCausalLM.from_pretrained(ckpt, torch_dtype=dtype).to(device)
        else:
            print(f"Checkpoint not found, loading base model.")
            model = GPTNeoXForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype).to(device)
    else:
        model = GPTNeoXForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype).to(device)

    model_gen0_cpu = GPTNeoXForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)

    print(f"\nFreezing {args.mode} blocks...")
    freeze_mlp_blocks(model, frozen_idx)

    if "0" not in results["ppl"]:
        ppl0 = compute_ppl(model, tokenizer, device)
        results["ppl"]["0"] = ppl0
        save_results(results, summary_path)
        print(f"  Gen0 PPL = {ppl0:.2f}")

    ppl0 = results["ppl"]["0"]

    for gen in range(max(1, last_done + 1), N_GENS + 1):
        print(f"\n{'═'*55}\n  Generation {gen} / {N_GENS}  [{args.mode}]\n{'═'*55}")

        # Dynamic: recompute FIM from current model, swap frozen blocks
        if args.mode == "dynamic":
            print(f"  Unfreezing all MLP blocks...")
            unfreeze_all_mlp(model)
            frozen_idx, current_fim = get_dynamic_fim_block_indices(
                model, FREEZE_K, gen, device)
            results["config"].setdefault("frozen_blocks_per_gen", {})
            results["config"]["frozen_blocks_per_gen"][str(gen)] = sorted(frozen_idx)
            print(f"  Refreezing top-{FREEZE_K} blocks for Gen{gen}...")
            freeze_mlp_blocks(model, frozen_idx)

        texts = load_or_generate_synthetic(gen, model, tokenizer, device)

        print("  Training...")
        train_one_gen(model, tokenizer, texts, device)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        ckpt = os.path.join(ROOT, "models", f"{model_prefix}_gen_{gen}")
        model.save_pretrained(ckpt); tokenizer.save_pretrained(ckpt)

        ppl = compute_ppl(model, tokenizer, device)
        pct = 100 * (ppl - ppl0) / ppl0
        results["ppl"][str(gen)] = ppl
        print(f"  PPL_{gen} = {ppl:.2f}  ({pct:+.1f}% from Gen0)")

        rho, p, n_free = compute_fim_drift_corr(
            model_gen0_cpu, model, fim_all, frozen_idx, device)
        results["corr"][str(gen)] = {"rho": rho, "p": p, "n_free": n_free}
        rho_str = f"{rho:+.3f}{sig(p)}" if rho is not None else "N/A"
        print(f"  Free-block rho = {rho_str}  (n_free={n_free})")

        save_results(results, summary_path)

    print_summary(results, args.mode)


def print_summary(results, mode):
    ppl0 = results["ppl"].get("0") or results["ppl"].get(0)
    print(f"\n{'='*65}")
    print(f"SUMMARY — Pythia-1.4b {mode.upper()} freeze (k={FREEZE_K})")
    print(f"Frozen blocks: {results['config']['frozen_blocks']}")
    print(f"{'='*65}")
    print(f"  {'Gen':>4}  {'PPL':>8}  {'%ΔGen0':>8}  {'free-block rho':>16}")
    print(f"  {'─'*45}")
    for g in range(0, N_GENS + 1):
        ppl  = results["ppl"].get(str(g)) or results["ppl"].get(g)
        corr = results["corr"].get(str(g)) or results["corr"].get(g, {})
        rho, p = corr.get("rho"), corr.get("p")
        if ppl is None: continue
        pct_str = f"{100*(ppl-ppl0)/ppl0:+.1f}%" if ppl0 and g > 0 else "—"
        rho_str = f"{rho:+.3f}{'**' if p and p<.01 else '*' if p and p<.05 else ''}" \
                  if rho else "—"
        print(f"  {g:>4}  {ppl:>8.2f}  {pct_str:>8}  {rho_str:>16}")
    print()
    print("COMPARISON:")
    print("  Standard Pythia:    Gen1=+15.6%  Gen2=+90.5%  Gen5=+1426%")
    print("  FIM-targeted freeze: Gen1=-1.5%  Gen2=+64.0%  Gen5=TBD")
    print()
    if mode == "random":
        print("  If random ≈ FIM-targeted: parameter reduction is what matters, not FIM selection")
        print("  If random ≈ standard:     FIM-targeting is causally specific ← expected result")
    elif mode == "low_fim":
        print("  If low_fim ≈ standard:    only HIGH-FIM blocks matter ← expected result")
        print("  If low_fim ≈ FIM-targeted: any freeze helps equally")
    else:  # dynamic
        print("  If dynamic < FIM-targeted: always freezing currently-highest FIM helps more")
        print("  If dynamic ≈ FIM-targeted: Gen0 FIM ranking is stable enough, no gain from recomputing")
        print("  If dynamic > FIM-targeted: chasing current FIM is counterproductive")
        if "frozen_blocks_per_gen" in results.get("config", {}):
            print("\n  Dynamic frozen blocks per generation:")
            for g, blocks in sorted(results["config"]["frozen_blocks_per_gen"].items(),
                                     key=lambda x: int(x[0])):
                print(f"    Gen{g}: {blocks}")


if __name__ == "__main__":
    main()