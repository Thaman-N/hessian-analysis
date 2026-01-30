import torch
import sys
import os
import warnings
import time
import json
import gc
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

# --- MONKEY PATCH FOR PYHESSIAN ---
def patched_eig(input, eigenvectors=False, out=None):
    if eigenvectors:
        eigenvals_complex, eigenvecs_complex = torch.linalg.eig(input)
        eigenvecs = eigenvecs_complex.real
    else:
        eigenvals_complex = torch.linalg.eigvals(input)
        eigenvecs_complex = None
        eigenvecs = None
    
    eigenvals_real = eigenvals_complex.real
    eigenvals_imag = eigenvals_complex.imag
    eigenvals = torch.stack([eigenvals_real, eigenvals_imag], dim=1)
    
    if eigenvectors:
        return eigenvals, eigenvecs
    else:
        return eigenvals, None

torch.eig = patched_eig
# ----------------------------------

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_from_disk
from pyhessian import hessian

class HessianModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, inputs):
        input_ids = inputs[:, 0, :].long()
        attention_mask = inputs[:, 1, :].long()
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

def criterion(outputs, targets):
    shift_logits = outputs[..., :-1, :].contiguous()
    shift_labels = targets[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

def get_qwen_dataloader(data_path, tokenizer, batch_size=1, num_samples=64, max_length=128):
    print(f"Loading data from {data_path}...")
    try:
        dataset = load_from_disk(data_path)
        train_dataset = dataset["train"]
    except:
        from datasets import load_dataset
        dataset = load_dataset(data_path) if os.path.isdir(data_path) else load_from_disk(data_path)
        train_dataset = dataset["train"] if "train" in dataset else dataset

    subset = train_dataset.select(range(min(num_samples, len(train_dataset))))
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=max_length,
            return_tensors="pt"
        )
    
    print("Tokenizing...")
    tokenized = subset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    class HessianDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            item = self.data[idx]
            input_ids = torch.tensor(item["input_ids"])
            attn_mask = torch.tensor(item["attention_mask"])
            inputs = torch.stack([input_ids, attn_mask])
            targets = input_ids.clone()
            targets[attn_mask == 0] = -100
            return inputs, targets

    return DataLoader(HessianDataset(tokenized), batch_size=batch_size, shuffle=False)

def analyze_qwen(generation=0, batch_size=1, num_samples=64, model_name="Qwen/Qwen2.5-0.5B"):
    print(f"\n🚀 STARTING QWEN HESSIAN ANALYSIS - GEN {generation}")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Model
    if generation == 0:
        print(f"Loading Baseline: {model_name}")
        model_path = model_name
    else:
        model_path = f"models/qwen_generation_{generation}"
        print(f"Loading Checkpoint: {model_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.gradient_checkpointing_enable()
        print("✅ Gradient Checkpointing ENABLED")
        model.eval()
        model.train() 
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 2. Setup Data
    # SAFE MODE: Batch=1, Length=128
    dataloader = get_qwen_dataloader(
        "data/tinystories", 
        tokenizer, 
        batch_size=batch_size, 
        num_samples=num_samples,
        max_length=128 
    )
    
    hessian_model = HessianModelWrapper(model)
    hessian_comp = hessian(hessian_model, criterion, dataloader=dataloader, cuda=True)
    
    # 3. Compute Top Eigenvalues
    print("\nComputing Top Eigenvalues...")
    torch.cuda.empty_cache() # Flush memory
    gc.collect()
    
    start_eig = time.time()
    top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=3)
    
    eig_time = (time.time() - start_eig) / 60
    print(f"✅ Eigenvalues computed in {eig_time:.1f} minutes")
    print(f"Top Eigenvalues: {top_eigenvalues}")
    
    # 4. Compute Density
    print("\nComputing Spectral Density (SAFE MODE: 10 iters)...")
    torch.cuda.empty_cache() # Flush memory again
    gc.collect()
    
    start_density = time.time()
    # SUPER SAFE MODE: iter=10 (was 30)
    density_eigenvals, _ = hessian_comp.density(iter=10)
    
    density_time = (time.time() - start_density) / 60
    print(f"✅ Density computed in {density_time:.1f} minutes")

    if isinstance(density_eigenvals, list):
        full_eigenvals = np.array([float(x) for x in density_eigenvals])
    else:
        full_eigenvals = density_eigenvals

    # 5. Metrics (IQR)
    if len(full_eigenvals) > 1:
        q1 = np.percentile(full_eigenvals, 25)
        q3 = np.percentile(full_eigenvals, 75)
        iqr = q3 - q1
        bulk_center = float(np.median(full_eigenvals))
        bulk_width = float(iqr)
        
        max_abs_lambda = float(np.max(np.abs(top_eigenvalues)))
        spectral_ratio = max_abs_lambda / bulk_width if bulk_width > 0 else 0
    else:
        spectral_ratio = 0
        
    total_time = eig_time + density_time
    print(f"\n📊 RESULTS GEN {generation}:")
    print(f"   Max λ: {top_eigenvalues[0]:.4f}")
    print(f"   Bulk Width (IQR): {bulk_width:.4f}")
    print(f"   Spectral Ratio: {spectral_ratio:.4f}")
    print(f"   Total Time: {total_time:.1f} minutes")

    # 6. Save
    results_dir = f"results/qwen_generation_{generation}"
    os.makedirs(results_dir, exist_ok=True)
    
    results = {
        "generation": generation,
        "top_eigenvalues": [float(x) for x in top_eigenvalues],
        "bulk_width_iqr": bulk_width,
        "spectral_ratio": spectral_ratio,
        "computation_time": total_time
    }
    
    with open(f"{results_dir}/hessian_stats.json", "w") as f:
        json.dump(results, f, indent=2)

    plt.figure(figsize=(10, 6))
    plt.hist(full_eigenvals, bins=50, density=True, alpha=0.7, color='purple')
    plt.title(f"Qwen-0.5B Spectral Density (Gen {generation})\nRatio: {spectral_ratio:.2f}")
    plt.xlabel("Eigenvalue")
    plt.ylabel("Density")
    plt.savefig(f"{results_dir}/spectrum.png")
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", type=int, default=0)
    args = parser.parse_args()
    
    analyze_qwen(generation=args.generation)