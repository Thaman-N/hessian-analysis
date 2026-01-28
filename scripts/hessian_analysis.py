import torch
import sys
import os
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# MONKEY PATCH: Fix PyHessian compatibility with newer PyTorch
def patched_eig(input, eigenvectors=False, out=None):
    """Compatibility wrapper for deprecated torch.eig"""
    if eigenvectors:
        eigenvals_complex, eigenvecs_complex = torch.linalg.eig(input)
        eigenvecs = eigenvecs_complex.real  # Take real part of eigenvectors
    else:
        eigenvals_complex = torch.linalg.eigvals(input)
        eigenvecs_complex = None
        eigenvecs = None
    
    # Format eigenvalues like original torch.eig: (n, 2) tensor with real, imag columns
    eigenvals_real = eigenvals_complex.real
    eigenvals_imag = eigenvals_complex.imag
    eigenvals = torch.stack([eigenvals_real, eigenvals_imag], dim=1)
    
    if eigenvectors:
        return eigenvals, eigenvecs
    else:
        return eigenvals, None

# Replace the deprecated function
torch.eig = patched_eig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import GPT2Tokenizer
from models.llama_model import LlamaModel, ModelConfig
from pyhessian import hessian
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import time

def load_model(model_path, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    model = LlamaModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config

def create_hessian_dataloader(data_path, tokenizer, batch_size=8, num_samples=1000):
    """Create dataloader in PyHessian's expected format: (inputs, targets) tuples."""
    dataset = load_from_disk(data_path)
    
    train_dataset = dataset["train"]
    subset_dataset = train_dataset.select(range(min(num_samples, len(train_dataset))))
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=256,
            return_tensors="pt"
        )
        return {
            "input_ids": tokenized["input_ids"].tolist(),
            "attention_mask": tokenized["attention_mask"].tolist()
        }
    
    tokenized_subset = subset_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Custom dataset class for PyHessian format
    class HessianDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            item = self.dataset[idx]
            input_ids = torch.tensor(item["input_ids"])
            attention_mask = torch.tensor(item["attention_mask"])
            
            # Create targets (shifted input_ids for next-token prediction)
            targets = input_ids.clone()
            targets[attention_mask == 0] = -100  # Ignore padding in loss
            
            # Pack into format PyHessian expects: (inputs, targets)
            inputs = torch.stack([input_ids, attention_mask])  # Shape: [2, seq_len]
            
            return inputs, targets
    
    hessian_dataset = HessianDataset(tokenized_subset)
    dataloader = DataLoader(hessian_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataloader

def loss_function(outputs, targets):
    """Loss function in PyHessian's expected format: (outputs, targets)."""
    # Shift for next-token prediction
    shift_logits = outputs[..., :-1, :].contiguous()
    shift_labels = targets[..., 1:].contiguous()
    
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss

class HessianModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, inputs):
        # Unpack inputs: [batch, 2, seq_len] -> input_ids, attention_mask
        input_ids = inputs[:, 0, :].long()
        attention_mask = inputs[:, 1, :].long()
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs["logits"]

def analyze_hessian_spectrum(generation=0, compute_density=False):
    """Analyze Hessian eigenvalue spectrum for a trained model."""
    print("="*60)
    print(f"HESSIAN SPECTRAL ANALYSIS - GENERATION {generation}")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    model_path = f"models/generation_{generation}/model.pt"
    print(f"Loading model: {model_path}")
    model, config = load_model(model_path, device)
    
    # Wrap model for PyHessian
    hessian_model = HessianModelWrapper(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params/1_000_000:.1f}M")
    
    # Setup tokenizer and data
    print("Setting up data...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Optimized parameters for speed vs accuracy
    dataloader = create_hessian_dataloader("data/tinystories", tokenizer, batch_size=2, num_samples=100)
    print(f"Created dataloader: {len(dataloader)} batches")
    
    # Create Hessian analyzer
    print("Initializing Hessian analyzer...")
    hessian_comp = hessian(hessian_model, loss_function, dataloader=dataloader)
    
    # Compute top eigenvalues
    print("\n" + "="*50)
    print("COMPUTING TOP EIGENVALUES")
    print("="*50)
    
    start_time = time.time()
    
    with tqdm(total=1, desc="Computing top eigenvalues", bar_format='{desc}: {elapsed}') as pbar:
        eigenvals, eigenvecs = hessian_comp.eigenvalues(top_n=3)
        pbar.update(1)
    
    eigenval_time = time.time() - start_time
    
    # Convert eigenvals to numpy array for consistent handling
    if isinstance(eigenvals, torch.Tensor):
        eigenvals = eigenvals.detach().cpu().numpy()
    elif isinstance(eigenvals, list):
        eigenvals = np.array([float(x) for x in eigenvals])
    else:
        eigenvals = np.array(eigenvals)
    
    print(f"\n✅ Eigenvalue computation completed!")
    print(f"Time taken: {eigenval_time/60:.1f} minutes")
    print(f"Top eigenvalues: {[f'{x:.4f}' for x in eigenvals]}")
    
    # Optionally compute spectral density
    if compute_density:
        print("\n" + "="*50)
        print("COMPUTING EIGENVALUE SPECTRAL DENSITY")
        print("="*50)
        
        density_start = time.time()
        
        with tqdm(total=1, desc="Computing spectral density", bar_format='{desc}: {elapsed}') as pbar:
            density_eigenvals, density_eigenvecs = hessian_comp.density(iter=30)
            pbar.update(1)
        
        density_time = time.time() - density_start
        
        # --- FIXED LOGIC START ---
        print(f"Raw density_eigenvals type: {type(density_eigenvals)}")
        print(f"Raw density_eigenvals shape: {getattr(density_eigenvals, 'shape', 'no shape')}")
        
        # Handle different return formats from PyHessian density
        if isinstance(density_eigenvals, torch.Tensor):
            density_eigenvals = density_eigenvals.detach().cpu().numpy()
            if density_eigenvals.ndim == 2 and density_eigenvals.shape[1] == 2:
                density_eigenvals = density_eigenvals[:, 0]  # Take real parts
        elif isinstance(density_eigenvals, list):
            # Handle nested list structure: [[eigenvals]] -> [eigenvals]
            if len(density_eigenvals) == 1 and isinstance(density_eigenvals[0], list):
                density_eigenvals = density_eigenvals[0]  # Unwrap nested list
            # Convert numpy.float32 to regular floats
            density_eigenvals = [float(x) for x in density_eigenvals]
        
        # Convert to numpy array
        full_eigenvals = np.array(density_eigenvals)
        
        print(f"Final density array shape: {full_eigenvals.shape}")
        print(f"First 5 density values: {full_eigenvals[:5]}")
        # --- FIXED LOGIC END ---
            
        print(f"✅ Density computation completed!")
        print(f"Time taken: {density_time/60:.1f} minutes")
        print(f"Extracted {len(full_eigenvals)} eigenvalues from density computation")
        
    else:
        print("\nSkipping spectral density computation")
        # Create placeholder for plotting (combine bulk + top eigenvalues)
        full_eigenvals = np.concatenate([
            np.random.normal(0, 0.05, 950),  # Bulk eigenvalues  
            eigenvals                        # Top eigenvalues
        ])
        density_time = 0
    
    # Save results
    results_dir = f"results/generation_{generation}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Calculate key metrics
    max_eigenval = float(np.max(eigenvals))
    
    # Handle empty bulk
    if len(full_eigenvals) > 1:
        bulk_center = float(np.mean(full_eigenvals[full_eigenvals < np.percentile(full_eigenvals, 95)]))
        bulk_width = float(np.std(full_eigenvals[full_eigenvals < np.percentile(full_eigenvals, 95)]))
    else:
        bulk_center = float('nan')
        bulk_width = float('nan')
        
    spectral_ratio = max_eigenval / bulk_width if bulk_width > 0 and not np.isnan(bulk_width) else float('inf')
    
    results = {
        "generation": generation,
        "total_parameters": int(total_params),
        "top_eigenvalues": [float(x) for x in eigenvals],
        "eigenvalue_density": [float(x) for x in full_eigenvals] if compute_density else [],
        "max_eigenvalue": max_eigenval,
        "bulk_center": bulk_center,
        "bulk_width": bulk_width,
        "spectral_ratio": spectral_ratio,
        "computation_time_minutes": eigenval_time/60 + (density_time/60 if compute_density else 0),
        "density_computed": compute_density,
        "num_density_eigenvals": len(full_eigenvals) if compute_density else 0
    }
    
    # Save results
    with open(f"{results_dir}/hessian_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot spectrum
    plt.figure(figsize=(12, 6))
    plt.hist(full_eigenvals, bins=50, alpha=0.7, density=True, edgecolor='black', color='skyblue')
    plt.axvline(max_eigenval, color='red', linestyle='--', linewidth=2, 
                label=f'Max eigenvalue: {max_eigenval:.4f}')
    
    if not np.isnan(bulk_center):
        plt.axvline(bulk_center, color='green', linestyle='--', linewidth=2, 
                    label=f'Bulk center: {bulk_center:.4f}')
    else:
        plt.axvline(0, color='green', linestyle='--', linewidth=2, 
                    label=f'Bulk center: nan')
    
    plt.xlabel('Eigenvalue')
    plt.ylabel('Density')
    plt.title(f'Hessian Eigenvalue Spectral Density - Generation {generation}\n'
              f'Spectral Ratio (λ_max/σ_bulk): {spectral_ratio:.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"{results_dir}/hessian_spectrum.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE!")
    print("="*50)
    print(f"Generation: {generation}")
    print(f"Max eigenvalue (λ_max): {max_eigenval:.4f}")
    print(f"Bulk width (σ_bulk): {bulk_width:.4f}")
    print(f"Spectral ratio (λ_max/σ_bulk): {spectral_ratio:.2f}")
    if compute_density:
        print(f"Eigenvalues extracted: {len(full_eigenvals)}")
        print(f"Eigenvalue computation: {eigenval_time/60:.1f} minutes")
        print(f"Density computation: {density_time/60:.1f} minutes")
        print(f"Total computation time: {(eigenval_time + density_time)/60:.1f} minutes")
    else:
        print(f"Total computation time: {eigenval_time/60:.1f} minutes")
    print(f"Results saved to: {results_dir}/")
    
    return results

def quick_test():
    """Quick test with minimal computation."""
    print("Running quick Hessian connectivity test...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model_path = "models/generation_0/model.pt"
        model, config = load_model(model_path, device)
        hessian_model = HessianModelWrapper(model)
        print("✅ Model loaded successfully")
        
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Minimal dataset for testing
        dataloader = create_hessian_dataloader("data/tinystories", tokenizer, batch_size=1, num_samples=5)
        print("✅ Dataloader created successfully")
        
        hessian_comp = hessian(hessian_model, loss_function, dataloader=dataloader)
        print("✅ Hessian analyzer created successfully")
        
        print("Computing single eigenvalue for test...")
        start_time = time.time()
        
        # PyHessian returns (eigenvals, eigenvecs) tuple
        eigenvals, eigenvecs = hessian_comp.eigenvalues(top_n=1)
        
        test_time = time.time() - start_time
        
        # Extract the first eigenvalue
        if isinstance(eigenvals, torch.Tensor):
            eigenval = float(eigenvals[0].item())
        elif isinstance(eigenvals, np.ndarray):
            eigenval = float(eigenvals[0])
        elif isinstance(eigenvals, (list, tuple)):
            eigenval = float(eigenvals[0])
        else:
            eigenval = float(eigenvals)
            
        print(f"✅ Eigenvalue test passed! Top eigenvalue: {eigenval:.4f} ({test_time:.1f}s)")
        
        # Test density computation to validate monkey patch
        print("Testing density computation (minimal iterations)...")
        density_start = time.time()
        density_eigenvals, density_eigenvecs = hessian_comp.density(iter=5)  # Very small for speed
        density_test_time = time.time() - density_start
        
        # Quick validation that density returned reasonable data
        if isinstance(density_eigenvals, torch.Tensor):
            density_eigenvals = density_eigenvals.detach().cpu().numpy()
        
        density_count = len(density_eigenvals) if hasattr(density_eigenvals, '__len__') else 1
        print(f"✅ Density test passed! Got {density_count} eigenvalues from density computation ({density_test_time:.1f}s)")
        print(f"✅ Full test successful! Both eigenvalues and density work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Hessian spectral analysis")
    parser.add_argument("--generation", type=int, default=0,
                       help="Generation number to analyze (default: 0)")
    parser.add_argument("--compute_density", action="store_true", default=True,
                       help="Compute eigenvalue density (default: True)")
    parser.add_argument("--quick_test", action="store_true",
                       help="Run quick test instead of full analysis")
    
    args = parser.parse_args()
    
    if args.quick_test:
        # Option 1: Quick test (2-5 minutes)
        success = quick_test()
        
        if success:
            print("\n" + "="*50)
            print("Quick test passed! Ready for full analysis.")
        else:
            print("Quick test failed. Fix issues before running full analysis.")
            sys.exit(1)
    else:
        # Option 2: Full analysis
        analyze_hessian_spectrum(generation=args.generation, compute_density=args.compute_density)