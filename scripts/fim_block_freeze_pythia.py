"""
fim_block_freeze_pythia.py  —  Causal circuit intervention

Freezes the top-k highest-FIM MLP blocks in Pythia-1.4b during recursive
self-distillation and measures whether collapse velocity decreases.

The question: does the UNPROTECTED drift of high-FIM MLP blocks (rho=+0.875**
for Pythia) causally drive its +1426% PPL collapse? Freeze those blocks and see.

Two expected outcomes (both informative):
  A) PPL increase drops substantially (e.g. +400% instead of +1426%):
     → Causal evidence that unprotected high-FIM MLP drift drives collapse.
  B) PPL stays near +1426% despite freeze:
     → The parallel residual stream architecture is the root cause, not block-level
       drift. Collapse propagates through attention regardless of MLP protection.
     → This is ALSO a strong result: proves architecture > block-level intervention.

Self-similar paradox check: the free-block FIM-drift correlation will
likely reconstitute at stronger values (as in SmolLM frozen_late).
If so, vt protection is a global optimizer property that redistributes,
not a local block-level phenomenon.

Runtime estimate (RTX 5090 laptop, 175W):
  N_TRAIN=50k: ~6-8 hours (full overnight)
  N_TRAIN=25k: ~3-4 hours  ← DEFAULT for faster causal test
  Change N_TRAIN=50000 below to match paper protocol exactly.

Run: python scripts/fim_block_freeze_pythia.py
Out: results/fim_block_freeze/summary.json
"""

import json
import os
import gc
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

# ── config ────────────────────────────────────────────────────────────────────
ROOT     = r"D:\Thaman\Work\hessian-spectral-analysis"
OUT_DIR  = os.path.join(ROOT, "results", "fim_block_freeze")
FIM_FILE = os.path.join(ROOT, "results", "pythia-1.4b_treatment_gen_0", "perblock_fim.json")

BASE_MODEL   = "EleutherAI/pythia-1.4b"
N_GENS       = 5
N_TRAIN      = 50_000   # 25k for ~3-4h; set 50_000 to match paper protocol exactly
SEQ_LEN      = 256
FREEZE_K     = 6        # top-6 of 24 blocks = top 25% by FIM_mlp
BATCH_SIZE   = 8
LR           = 5e-5
WEIGHT_DECAY = 0.01
PPL_SAMPLES  = 500

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)


# ── FIM ───────────────────────────────────────────────────────────────────────

def get_top_fim_block_indices(fim_file, k):
    with open(fim_file) as f:
        d = json.load(f)
    block_fim = [(b["block_idx"], b["mlp"]["top"]) for b in d["blocks"]]
    block_fim.sort(key=lambda x: x[1], reverse=True)
    top = [(idx, fim) for idx, fim in block_fim[:k]]
    print(f"  Top-{k} FIM MLP blocks to freeze:")
    for idx, fim in sorted(top):
        print(f"    Block {idx:2d}: FIM_mlp = {fim:.1f}")
    return {idx for idx, _ in top}


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
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Frozen {frozen:,} params. Trainable: {trainable:,} / {total:,} "
          f"({100*trainable/total:.1f}%)")


# ── data generation ───────────────────────────────────────────────────────────

def generate_synthetic(model, tokenizer, n_samples, device):
    """Mirrors treatment script generation exactly."""
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    prompts = []
    for item in ds:
        prompts.append(item["text"][:64])
        if len(prompts) >= n_samples:
            break

    model.eval()
    generated = []
    bs = 16
    with torch.no_grad():
        pbar = tqdm(total=n_samples, desc="Generating")
        for i in range(0, len(prompts), bs):
            batch = prompts[i:i+bs]
            enc = tokenizer(batch, return_tensors="pt", truncation=True,
                            max_length=64, padding=True).to(device)
            out = model.generate(
                **enc,
                max_new_tokens=192,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            for seq in out:
                generated.append(tokenizer.decode(seq, skip_special_tokens=True))
            pbar.update(len(batch))
        pbar.close()
    return generated[:n_samples]


# ── training ──────────────────────────────────────────────────────────────────

def train_one_gen(model, tokenizer, texts, device):
    """Manual training loop — no Trainer, no accelerate, no grad scaler.
    Mirrors treatment script exactly to avoid fp16/bf16 scaler issues."""

    class TextDS(TorchDataset):
        def __init__(self, texts, tok, maxlen):
            # right-padding for training (treatment script convention)
            tok.padding_side = "right"
            self.enc = tok(texts, truncation=True, max_length=maxlen,
                           padding="max_length", return_tensors="pt")
        def __len__(self): return self.enc["input_ids"].shape[0]
        def __getitem__(self, i): return {k: v[i] for k, v in self.enc.items()}

    ds     = TextDS(texts, tokenizer, SEQ_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    # Only pass trainable params to optimizer — frozen params excluded
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY,
                               betas=(0.9, 0.999))
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=100,
        num_training_steps=len(loader),
    )

    model.train()
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch, labels=batch["input_ids"]).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optim.step()
        sched.step()
        optim.zero_grad()
        pbar.set_postfix({"loss": f"{loss.item():.3f}"})

    # restore left-padding for generation
    tokenizer.padding_side = "left"


# ── evaluation ────────────────────────────────────────────────────────────────

def compute_ppl(model, tokenizer, n_samples, device):
    ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    total_loss, total_tok = 0.0, 0
    model.eval()
    with torch.no_grad():
        for i, item in enumerate(ds):
            if i >= n_samples: break
            enc = tokenizer(item["text"], return_tensors="pt",
                            truncation=True, max_length=SEQ_LEN).to(device)
            if enc["input_ids"].shape[1] < 4: continue
            out = model(**enc, labels=enc["input_ids"])
            n = enc["input_ids"].shape[1]
            total_loss += out.loss.item() * n
            total_tok  += n
    return float(np.exp(total_loss / total_tok))


def compute_fim_drift_corr(model_g0_cpu, model_cur, fim_all, frozen_idx, device):
    """Spearman rho on FREE blocks only."""
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


# ── main ──────────────────────────────────────────────────────────────────────

def save_results(results, summary_path):
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)


def print_summary(results):
    print("\n" + "="*60)
    print(f"SUMMARY — Pythia-1.4b Freeze top-{results['config']['freeze_k']} FIM MLP blocks")
    print("="*60)
    print(f"Frozen blocks: {results['config']['frozen_blocks']}")
    print()
    print(f"  {'Gen':>4}  {'PPL':>8}  {'%ΔGen0':>8}  {'free-block rho':>16}")
    print(f"  {'─'*45}")
    ppl0_val = results["ppl"].get("0") or results["ppl"].get(0)
    for g in range(0, N_GENS + 1):
        ppl  = results["ppl"].get(str(g)) or results["ppl"].get(g)
        corr = results["corr"].get(str(g)) or results["corr"].get(g, {})
        rho, p = corr.get("rho"), corr.get("p")
        pct_str = f"{100*(ppl-ppl0_val)/ppl0_val:+.1f}%" if ppl and ppl0_val and g > 0 else "—"
        rho_str = f"{rho:+.3f}{sig(p)}" if rho else "—"
        print(f"  {g:>4}  {(ppl or 0):>8.2f}  {pct_str:>8}  {rho_str:>16}")
    print()
    print("Paper baseline: Pythia standard treatment = +1426% by Gen5")
    print()
    print("INTERPRETATION:")
    print("  Freeze slows collapse substantially → high-FIM MLP drift causally drives it")
    print("  Freeze doesn't help → parallel architecture is the root cause")
    print("  Free-block rho stronger negative → self-similar paradox confirmed")


def main():
    summary_path = os.path.join(OUT_DIR, "summary.json")

    # ── resume: load existing results if present ──────────────────────────────
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            results = json.load(f)
        # normalise keys to strings for consistent lookup
        results["ppl"]  = {str(k): v for k, v in results["ppl"].items()}
        results["corr"] = {str(k): v for k, v in results["corr"].items()}
    else:
        results = {
            "config": {"freeze_k": FREEZE_K, "n_train": N_TRAIN,
                       "frozen_blocks": None},   # filled after FIM load
            "ppl":  {},
            "corr": {},
            "paper_baseline": "+1426% Pythia standard treatment Gen0→Gen5",
        }

    # figure out which gen to resume from
    completed_gens = [int(k) for k in results["ppl"].keys()]
    last_done = max(completed_gens) if completed_gens else -1

    if last_done >= N_GENS:
        print("All generations complete.")
        print_summary(results)
        return

    if last_done >= 0:
        print(f"Resuming from Gen{last_done + 1} "
              f"(gens {sorted(completed_gens)} already done)")
    else:
        print("Starting fresh.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  N_TRAIN={N_TRAIN}  |  FREEZE_K={FREEZE_K}")

    frozen_idx = get_top_fim_block_indices(FIM_FILE, FREEZE_K)
    fim_all    = load_fim_all(FIM_FILE)
    results["config"]["frozen_blocks"] = sorted(frozen_idx)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    # Load the most recent checkpoint, or base model if starting fresh
    if last_done >= 1:
        resume_ckpt = os.path.join(ROOT, "models", f"pythia_fim_freeze_gen_{last_done}")
        if os.path.exists(resume_ckpt):
            print(f"Loading checkpoint: {resume_ckpt}")
            model = GPTNeoXForCausalLM.from_pretrained(
                resume_ckpt, torch_dtype=dtype).to(device)
        else:
            print(f"WARNING: checkpoint {resume_ckpt} not found, loading base model.")
            model = GPTNeoXForCausalLM.from_pretrained(
                BASE_MODEL, torch_dtype=dtype).to(device)
    else:
        print(f"Loading base model: {BASE_MODEL}")
        model = GPTNeoXForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=dtype).to(device)

    # Gen0 reference on CPU in float32 for drift computation (needs precise subtraction)
    print("Loading Gen0 reference (CPU, float32)...")
    model_gen0_cpu = GPTNeoXForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)

    print(f"\nApplying freeze to top-{FREEZE_K} FIM MLP blocks...")
    freeze_mlp_blocks(model, frozen_idx)

    # Gen0 PPL if not already done
    if "0" not in results["ppl"]:
        print("\nGen0 PPL...")
        ppl0 = compute_ppl(model, tokenizer, PPL_SAMPLES, device)
        results["ppl"]["0"] = ppl0
        save_results(results, summary_path)
        print(f"  PPL₀ = {ppl0:.2f}")

    ppl0 = results["ppl"]["0"]

    # ── generation loop ───────────────────────────────────────────────────────
    for gen in range(last_done + 1, N_GENS + 1):
        print(f"\n{'═'*55}")
        print(f"  Generation {gen} / {N_GENS}")
        print(f"{'═'*55}")

        # Check if synthetic data was already saved for this gen
        synth_path = os.path.join(OUT_DIR, f"synthetic_gen_{gen}.txt")
        if os.path.exists(synth_path):
            print(f"  Loading cached synthetic data: {synth_path}")
            with open(synth_path, encoding="utf-8") as f:
                texts = f.read().split("\n---\n")
            texts = [t for t in texts if t.strip()]
            print(f"  Loaded {len(texts)} samples.")
        else:
            print(f"  Generating {N_TRAIN} synthetic samples...")
            texts = generate_synthetic(model, tokenizer, N_TRAIN, device)
            with open(synth_path, "w", encoding="utf-8") as f:
                f.write("\n---\n".join(texts))
            print(f"  Saved synthetic data: {synth_path}")

        print(f"  Training...")
        train_one_gen(model, tokenizer, texts, device)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        ckpt = os.path.join(ROOT, "models", f"pythia_fim_freeze_gen_{gen}")
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        print(f"  Saved checkpoint: {ckpt}")

        ppl = compute_ppl(model, tokenizer, PPL_SAMPLES, device)
        pct = 100 * (ppl - ppl0) / ppl0
        results["ppl"][str(gen)] = ppl
        print(f"  PPL_{gen} = {ppl:.2f}  ({pct:+.1f}% from Gen0)")

        rho, p, n_free = compute_fim_drift_corr(
            model_gen0_cpu, model, fim_all, frozen_idx, device)
        results["corr"][str(gen)] = {"rho": rho, "p": p, "n_free": n_free}
        rho_str = f"{rho:+.3f}{sig(p)}" if rho is not None else "N/A"
        print(f"  Free-block FIM-drift rho = {rho_str}  (n_free={n_free})")

        # Save after every gen so resume works correctly
        save_results(results, summary_path)

    print_summary(results)


if __name__ == "__main__":
    main()