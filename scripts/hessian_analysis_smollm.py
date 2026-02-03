import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from pyhessian import hessian
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
import sys
import time
import gc

# --- Configuration ---
BATCH_SIZE = 1           # Safe for 24GB VRAM
MAX_LENGTH = 512         # Matches your training
DENSITY_ITERS = 20       # Sufficient for bulk resolution
# ---------------------

# --- MONKEY PATCH: Fix PyHessian for modern PyTorch ---
def patched_eig(input, eigenvectors=False, out=None):
    if eigenvectors:
        vals, vecs = torch.linalg.eig(input)
        return torch.stack([vals.real, vals.imag], dim=1), vecs.real
    else:
        vals = torch.linalg.eigvals(input)
        return torch.stack([vals.real, vals.imag], dim=1), None
torch.eig = patched_eig
# ------------------------------------------------------

class HessianModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, inputs):
        # Unpack inputs: [batch, 2, seq_len] -> input_ids, attention_mask
        input_ids = inputs[:, 0, :].long()
        attention_mask = inputs[:, 1, :].long()
        
        # Force use_cache=False to save VRAM
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            use_cache=False
        )
        return outputs.logits

def criterion(outputs, targets):
    # Standard CrossEntropy ignoring padding (-100)
    return nn.CrossEntropyLoss(ignore_index=-100)(
        outputs[..., :-1, :].contiguous().view(-1, outputs.size(-1)),
        targets[..., 1:].contiguous().view(-1)
    )

def analyze_model(model_path, generation):
    print(f"\n🔬 STARTING HESSIAN ANALYSIS: Gen {generation}")
    print(f"   Path: {model_path}")
    
    # 1. Clean Memory
    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load Model & Tokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float32,
            device_map="auto"
        )
        model.eval()
        
        # Determine base model for tokenizer
        base_model_id = "HuggingFaceTB/SmolLM2-135M"
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 3. Load Data (Validation Subset)
    try:
        # Use existing local data if available, else download
        data_path = "data/tinystories"
        if os.path.exists(data_path):
            dataset = load_from_disk(data_path)
            if "validation" in dataset: dataset = dataset["validation"]
            elif "train" in dataset: dataset = dataset["train"]
        else:
            print("Downloading TinyStories validation set...")
            dataset = load_dataset("roneneldan/TinyStories", split="validation")

        # Select small subset for Hessian stability
        dataset = dataset.select(range(64))
        
    except Exception as e:
        print(f"❌ Data Load Error: {e}")
        return

    # 4. Collate Function (Handles Masking Correctly)
    def collate(batch):
        txt = [x["text"] for x in batch]
        tok = tokenizer(
            txt, 
            truncation=True, 
            padding="max_length", 
            max_length=MAX_LENGTH, 
            return_tensors="pt"
        )
        # Stack inputs for PyHessian: [batch, 2, seq_len]
        inputs = torch.stack([tok["input_ids"], tok["attention_mask"]], dim=1)
        
        # Create targets: Copy input_ids, but MASK PADDING with -100
        targets = tok["input_ids"].clone()
        targets[tok["attention_mask"] == 0] = -100
        
        return inputs, targets

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate)

    # 5. Initialize PyHessian
    hessian_comp = hessian(HessianModelWrapper(model), criterion, dataloader=loader, cuda=True)

    # 6. Calculate Top Eigenvalues (Outliers)
    print("\n⏳ Step 1/2: Computing Top Eigenvalues...")
    t0 = time.time()
    top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=3)
    t1 = time.time()
    print(f"   Done in {t1 - t0:.2f}s")
    
    # Clean up tensor values to floats
    clean_top_eigs = [float(x.item()) if torch.is_tensor(x) else float(x) for x in top_eigenvalues]
    print(f"   Top Eigenvalues: {clean_top_eigs}")

    # 7. Calculate Spectral Density (Bulk)
    print("\n⏳ Step 2/2: Computing Spectral Density (Lanczos)...")
    t2 = time.time()
    density_eigen, density_weight = hessian_comp.density(iter=DENSITY_ITERS)
    t3 = time.time()
    print(f"   Done in {t3 - t2:.2f}s")

    # --- CRITICAL FIX: SORTING & NORMALIZATION ---
    print("   Processing & Sorting Data...")
    
    # Flatten and ensure float type
    x = np.array(density_eigen).flatten().astype(float)
    y = np.array(density_weight).flatten().astype(float)

    # SORT by Eigenvalue (x) - Fixes the "scribble" graph
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    # Normalize Density Sum to 1 for correct stats
    if np.sum(y_sorted) > 0:
        y_sorted = y_sorted / np.sum(y_sorted)

    # Calculate IQR using Cumulative Sum
    cumulative = np.cumsum(y_sorted)
    # Normalize cumulative to 0-1 range
    if cumulative[-1] > 0:
        cumulative = cumulative / cumulative[-1]
    
    # Interpolate Quartiles
    q1 = np.interp(0.25, cumulative, x_sorted)
    q3 = np.interp(0.75, cumulative, x_sorted)
    iqr = q3 - q1
    
    # Calculate Spectral Ratio
    max_lambda = np.max(np.abs(clean_top_eigs))
    ratio = max_lambda / iqr if iqr > 1e-6 else 0.0

    print(f"\n📊 RESULTS GEN {generation}:")
    print(f"   Max λ: {max_lambda:.4f}")
    print(f"   Bulk Width (IQR): {iqr:.4f}")
    print(f"   Spectral Ratio: {ratio:.4f}")

    # 8. Save Results
    results_dir = f"results/generation_{generation}"
    os.makedirs(results_dir, exist_ok=True)

    metrics = {
        "generation": generation,
        "top_eigenvalues": clean_top_eigs,
        "bulk_width_iqr": float(iqr),
        "spectral_ratio": float(ratio),
        "plot_x": x_sorted.tolist(),
        "plot_y": y_sorted.tolist()
    }
    
    with open(f"{results_dir}/hessian_stats.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # 9. Plotting (Corrected)
    plt.figure(figsize=(10, 6))
    plt.plot(x_sorted, y_sorted, color='green', linewidth=2, label="Spectral Density")
    plt.fill_between(x_sorted, y_sorted, color='green', alpha=0.3)
    
    # Add markers for Top Eigenvalues
    for eig in clean_top_eigs:
        plt.axvline(x=eig, color='red', linestyle='--', alpha=0.5, label=f"Top λ: {eig:.0f}")

    # Remove duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title(f"SmolLM2-135M Gen {generation}\nRatio: {ratio:.2f} | Max λ: {max_lambda:.0f}")
    plt.xlabel("Eigenvalue")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"{results_dir}/spectrum.png")
    plt.close()
    
    print(f"✅ Analysis Complete! Saved to {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    
    analyze_model(args.model_path, args.generation)