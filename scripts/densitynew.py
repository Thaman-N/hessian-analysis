import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset

# Install: pip install git+https://github.com/noahgolmant/pytorch-hessian-eigenthings.git

def efficient_hessian_analysis(model_path, generation, output_dir):
    print(f"🔬 Efficient Hessian Analysis: {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float32,
        attn_implementation="eager"
    ).to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # Minimal dataset
    dataset = load_dataset("roneneldan/TinyStories", split="validation").select(range(16))
    
    def collate(batch):
        txt = [x["text"][:100] for x in batch]
        tok = tokenizer(txt, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        return tok["input_ids"].to(device), tok["attention_mask"].to(device)
    
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate)  # Can use batch_size > 1
    
    def loss_fn(model, batch):
        input_ids, attention_mask = batch
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        return outputs.loss
    
    # This library is much more memory efficient
    from hessian_eigenthings import compute_hessian_eigenthings
    
    try:
        # Compute eigenvalues using more efficient algorithm
        eigenvals, eigenvecs = compute_hessian_eigenthings(
            model, 
            dataloader, 
            loss_fn,
            num_eigenthings=10,  # Start with 10
            mode='lanczos',      # Use Lanczos method
            max_iter=20          # This is closer to PyHessian's iterations
        )
        
        print(f"✅ Got {len(eigenvals)} eigenvalues: {eigenvals[:5]}")
        
        # Convert eigenvalues to density approximation
        eigenvals_np = eigenvals.cpu().numpy()
        
        # Simple density approximation (you can improve this)
        from scipy import stats
        density_est = stats.gaussian_kde(eigenvals_np)
        x_range = np.linspace(eigenvals_np.min(), eigenvals_np.max(), 1000)
        density_vals = density_est(x_range)
        
        # Calculate your metrics
        sorted_vals = np.sort(eigenvals_np)
        iqr = float(np.percentile(sorted_vals, 75) - np.percentile(sorted_vals, 25))
        max_lambda = float(np.max(np.abs(sorted_vals)))
        ratio = max_lambda / iqr if iqr > 1e-6 else 0.0
        
        print(f"📊 Gen {generation}: Ratio {ratio:.0f} | IQR {iqr:.4f}")
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        stats_dict = {
            "generation": generation,
            "spectral_ratio": float(ratio),
            "bulk_width_iqr": float(iqr),
            "top_eigenvalues": eigenvals_np.tolist(),
            "plot_x": x_range.tolist(),
            "plot_y": density_vals.tolist()
        }
        
        with open(f"{output_dir}/hessian_stats.json", "w") as f:
            json.dump(stats_dict, f, indent=4)
            
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    efficient_hessian_analysis(args.model_path, args.generation, args.output_dir)