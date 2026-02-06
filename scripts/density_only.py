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

# --- CONFIG ---
MAX_LENGTH = 256

# Monkey Patch for PyHessian
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

def clear_grad_and_reset(model):
    for param in model.parameters():
        if param.grad is not None:
            param.grad = None

def density_only(model_path, generation, output_dir):
    print(f"📈 Density ONLY: {model_path}")
    
    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        dtype=torch.float32,
        attn_implementation="eager"
    ).to(device)
    model.eval()
    model.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # Data
    dataset = load_dataset("roneneldan/TinyStories", split="validation").select(range(32))
    
    def collate(batch):
        txt = [x["text"] for x in batch]
        tok = tokenizer(txt, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
        inputs = torch.stack([tok["input_ids"], tok["attention_mask"]], dim=1)
        targets = tok["input_ids"].clone()
        targets[tok["attention_mask"] == 0] = -100
        return inputs, targets

    loader = DataLoader(dataset, batch_size=1, collate_fn=collate)
    hessian_comp = hessian(HessianModelWrapper(model), criterion, dataloader=loader, cuda=True)
    
    # DENSITY ONLY
    print("⏳ Density calculation...")
    try:
        density_eigen, density_weight = hessian_comp.density(iter=20)
        
        x = np.array(density_eigen).flatten().astype(float)
        y = np.array(density_weight).flatten().astype(float)
        
        sorted_idx = np.argsort(x)
        x_s, y_s = x[sorted_idx], y[sorted_idx]
        if np.sum(y_s) > 0: y_s /= np.sum(y_s)
        cum = np.cumsum(y_s)
        cum /= cum[-1] if cum[-1] > 0 else 1.0
        iqr = float(np.interp(0.75, cum, x_s) - np.interp(0.25, cum, x_s))
        max_lambda = float(np.max(np.abs(x_s))) if len(x_s) > 0 else 0
        ratio = max_lambda / iqr if iqr > 1e-6 else 0.0
        
        print(f"📊 Gen {generation} Density: Ratio {ratio:.0f} | IQR {iqr:.4f}")
        
        clear_grad_and_reset(model)
        torch.cuda.empty_cache()
        
        # Load existing JSON and merge
        json_path = f"{output_dir}/hessian_stats.json"
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                stats = json.load(f)
        else:
            stats = {"generation": generation, "top_eigenvalues": []}
        
        # Add density data
        stats.update({
            "spectral_ratio": float(ratio),
            "bulk_width_iqr": float(iqr),
            "plot_x": list(x_s),
            "plot_y": list(y_s)
        })
        
        os.makedirs(output_dir, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=4)
            
        print("✅ Density saved!")
        
    except Exception as e:
        print(f"❌ Density failed: {e}")

    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--model_path", type=str, required=True)  
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    density_only(args.model_path, args.generation, args.output_dir)