import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from pyhessian import hessian
import numpy as np
import os
import json
import argparse
import gc
import sys

# --- CONFIG ---
MAX_LENGTH = 256

# Monkey Patch for PyHessian (Fixes PyTorch compatibility)
def patched_eig(input, eigenvectors=False, out=None):
    if eigenvectors:
        vals, vecs = torch.linalg.eig(input)
        return torch.stack([vals.real, vals.imag], dim=1), vecs.real
    else:
        vals = torch.linalg.eigvals(input)
        return torch.stack([vals.real, vals.imag], dim=1), None
torch.eig = patched_eig

class HessianModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, inputs):
        return self.model(
            input_ids=inputs[:, 0, :].long(),
            attention_mask=inputs[:, 1, :].long(),
            use_cache=False
        ).logits

def criterion(outputs, targets):
    return nn.CrossEntropyLoss(ignore_index=-100)(
        outputs[..., :-1, :].contiguous().view(-1, outputs.size(-1)),
        targets[..., 1:].contiguous().view(-1)
    )

def analyze_model(model_path, generation, output_dir):
    print(f"🔬 Hessian Analysis: {model_path}")
    
    # 1. Clean Start
    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load Model in FLOAT32 (Stable) but with MEMORY OPTIMIZATIONS
    try:
        print("   Loading model in float32 (for stability)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float32,   # <--- BACK TO FP32 TO FIX NANs
            attn_implementation="eager"  # <--- YOUR SDPA FIX
        ).to(device)
        model.eval()
        
        # KEY MEMORY SAVER: Gradient Checkpointing
        # This trades a little speed for WAY less VRAM
        model.gradient_checkpointing_enable()
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"❌ Load Error: {e}")
        return

    # 3. Data (Small subset for speed/memory)
    try:
        dataset = load_dataset("roneneldan/TinyStories", split="validation").select(range(32))
    except:
        print("❌ Dataset Error.")
        return

    def collate(batch):
        txt = [x["text"] for x in batch]
        tok = tokenizer(txt, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
        inputs = torch.stack([tok["input_ids"], tok["attention_mask"]], dim=1)
        targets = tok["input_ids"].clone()
        targets[tok["attention_mask"] == 0] = -100
        return inputs, targets

    # Batch size 1 is safest for Hessian
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate)
    
    # Initialize Hessian
    hessian_comp = hessian(HessianModelWrapper(model), criterion, dataloader=loader, cuda=True)
    
    # 4. Top Eigenvalues
    print("⏳ Calculating Top Eigenvalues...")
    clean_top = []
    try:
        top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=3)
        clean_top = [float(x.item()) if torch.is_tensor(x) else float(x) for x in top_eigenvalues]
        torch.cuda.empty_cache() # Flush immediately
    except RuntimeError as e:
        print(f"⚠️ Failed Top Eigs: {e}")

    # 5. Density
    print("⏳ Calculating Spectral Density...")
    density_eigen = []
    density_weight = []
    ratio = 0.0
    iqr = 0.0
    
    try:
        # iter=20 is a good balance for Control groups
        density_eigen, density_weight = hessian_comp.density(iter=20)
        
        x = np.array(density_eigen).flatten().astype(float)
        y = np.array(density_weight).flatten().astype(float)
        
        # Metrics
        sorted_idx = np.argsort(x)
        x_s, y_s = x[sorted_idx], y[sorted_idx]
        if np.sum(y_s) > 0: y_s /= np.sum(y_s)
        cum = np.cumsum(y_s)
        cum /= cum[-1] if cum[-1] > 0 else 1.0
        iqr = float(np.interp(0.75, cum, x_s) - np.interp(0.25, cum, x_s))
        max_lambda = np.max(np.abs(clean_top)) if clean_top else 0
        ratio = max_lambda / iqr if iqr > 1e-6 else 0.0
        
        print(f"📊 Gen {generation} Result: Ratio {ratio:.0f} | IQR {iqr:.4f}")

    except Exception as e:
        print(f"⚠️ Error in Density Calc: {e}")
        print("   (Saving partial results...)")
        x_s, y_s = [], []

    # 6. Save & Cleanup
    os.makedirs(output_dir, exist_ok=True)
    stats = {
        "generation": generation,
        "spectral_ratio": float(ratio),
        "bulk_width_iqr": float(iqr),
        "top_eigenvalues": clean_top,
        "plot_x": list(x_s) if len(x_s) > 0 else [],
        "plot_y": list(y_s) if len(y_s) > 0 else []
    }
    with open(f"{output_dir}/hessian_stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    # Nuke model from child process memory too
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    analyze_model(args.model_path, args.generation, args.output_dir)