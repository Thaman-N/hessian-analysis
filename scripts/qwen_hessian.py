# # import torch
# # import sys, os, time, json, gc, warnings
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import argparse
# # from pyhessian import hessian
# # from transformers import AutoModelForCausalLM, AutoTokenizer
# # from torch.utils.data import DataLoader
# # from datasets import load_from_disk

# # warnings.filterwarnings("ignore")
# # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # # --- 4090 SETTINGS ---
# # BATCH_SIZE = 1
# # MAX_LENGTH = 128
# # DENSITY_ITERS = 30    # High Res
# # PRECISION = torch.float32 # Max Science
# # # ---------------------

# # # Monkey Patch
# # def patched_eig(input, eigenvectors=False, out=None):
# #     if eigenvectors:
# #         vals, vecs = torch.linalg.eig(input)
# #         return torch.stack([vals.real, vals.imag], dim=1), vecs.real
# #     else:
# #         vals = torch.linalg.eigvals(input)
# #         return torch.stack([vals.real, vals.imag], dim=1), None
# # torch.eig = patched_eig

# # class HessianModelWrapper(torch.nn.Module):
# #     def __init__(self, model):
# #         super().__init__()
# #         self.model = model
# #     def forward(self, inputs):
# #         return self.model(input_ids=inputs[:,0,:].long(), attention_mask=inputs[:,1,:].long()).logits

# # def criterion(outputs, targets):
# #     return torch.nn.CrossEntropyLoss(ignore_index=-100)(
# #         outputs[..., :-1, :].contiguous().view(-1, outputs.size(-1)),
# #         targets[..., 1:].contiguous().view(-1)
# #     )

# # def analyze_qwen(generation):
# #     print(f"\n🔬 HESSIAN ANALYSIS: GEN {generation} (FP32 HIGH-RES)")
    
# #     # Load Model
# #     path = "Qwen/Qwen2.5-0.5B" if generation == 0 else f"models/qwen_generation_{generation}"
# #     model = AutoModelForCausalLM.from_pretrained(
# #             path, 
# #             torch_dtype=PRECISION,
# #             device_map="sequential", # <--- Forces a cleaner device placement
# #             attn_implementation="eager",
# #             low_cpu_mem_usage=True   # <--- Better memory handling
# #         )
# #     # model.gradient_checkpointing_enable()
# #     tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
# #     tokenizer.pad_token = tokenizer.eos_token
# #     model.train() # Required for gradients

# #     # Load Data (Just 64 samples is enough for Hessian, but full length)
# #     dataset = load_from_disk("data/tinystories")
# #     if "train" in dataset: dataset = dataset["train"]
# #     dataset = dataset.select(range(64))

# #     def collate(batch):
# #         txt = [x["text"] for x in batch]
# #         tok = tokenizer(txt, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
# #         inputs = torch.stack([tok["input_ids"], tok["attention_mask"]], dim=1)
# #         targets = tok["input_ids"].clone()
# #         targets[tok["attention_mask"] == 0] = -100
# #         return inputs, targets

# #     loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate)

# #     # PyHessian
# #     hessian_comp = hessian(HessianModelWrapper(model), criterion, dataloader=loader, cuda=True)
    
# #     print("Computing Top Eigenvalues...")
# #     top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=3)
# #     print(f"Top λ: {top_eigenvalues}")
    
# #     model.zero_grad()
# #     torch.cuda.empty_cache()
# #     gc.collect()
# #     time.sleep(2) # Give the driver a moment to breathe
    
# #     print(f"Computing Density ({DENSITY_ITERS} iters)...")
# #     density, _ = hessian_comp.density(iter=DENSITY_ITERS)
    
# #     # Metrics (IQR)
# #     vals = np.array(density)
# #     q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
# #     iqr = q3 - q1
# #     ratio = np.max(np.abs(top_eigenvalues)) / iqr if iqr > 0 else 0
    
# #     print(f"📊 Ratio: {ratio:.4f} | Bulk Width: {iqr:.4f}")

# #     # Save
# #     res_dir = f"results/qwen_generation_{generation}"
# #     os.makedirs(res_dir, exist_ok=True)
# #     with open(f"{res_dir}/stats.json", "w") as f:
# #         json.dump({"generation": generation, "ratio": ratio, "iqr": iqr, "top_lambda": [float(x) for x in top_eigenvalues]}, f)
    
# #     plt.figure()
# #     plt.hist(vals, bins=50, density=True, color='purple', alpha=0.7)
# #     plt.title(f"Gen {generation} Spectrum (Ratio: {ratio:.2f})")
# #     plt.savefig(f"{res_dir}/spectrum.png")

# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--generation", type=int, required=True)
# #     args = parser.parse_args()
# #     analyze_qwen(args.generation)
# import torch
# import sys, os, time, json, gc, warnings
# import numpy as np
# import matplotlib.pyplot as plt
# import argparse
# from pyhessian import hessian
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from torch.utils.data import DataLoader
# from datasets import load_from_disk

# # Prevent fragmentation
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# warnings.filterwarnings("ignore")

# # --- OPTIMIZED SETTINGS ---
# BATCH_SIZE = 1
# MAX_LENGTH = 128
# DENSITY_ITERS = 30 
# PRECISION = torch.bfloat16 # BF16 is mandatory for 24GB VRAM HVP
# DEVICE = "cuda:0"
# # ---------------------

# def patched_eig(input, eigenvectors=False, out=None):
#     if eigenvectors:
#         vals, vecs = torch.linalg.eig(input)
#         return torch.stack([vals.real, vals.imag], dim=1), vecs.real
#     else:
#         vals = torch.linalg.eigvals(input)
#         return torch.stack([vals.real, vals.imag], dim=1), None
# torch.eig = patched_eig

# class HessianModelWrapper(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#     def forward(self, inputs):
#         # inputs: [batch, 2, seq_len]
#         return self.model(
#             input_ids=inputs[:,0,:].long(), 
#             attention_mask=inputs[:,1,:].long(),
#             use_cache=False # Crucial: avoids extra memory allocation
#         ).logits

# def criterion(outputs, targets):
#     return torch.nn.CrossEntropyLoss(ignore_index=-100)(
#         outputs[..., :-1, :].contiguous().view(-1, outputs.size(-1)),
#         targets[..., 1:].contiguous().view(-1)
#     )

# def analyze_qwen(generation):
#     print(f"\n🔬 HESSIAN ANALYSIS: GEN {generation} (BF16 OPTIMIZED)")
    
#     path = "Qwen/Qwen2.5-0.5B" if generation == 0 else f"models/qwen_generation_{generation}"
    
#     model = AutoModelForCausalLM.from_pretrained(
#         path, 
#         dtype=PRECISION,
#         device_map=DEVICE,
#         attn_implementation="eager", # More memory efficient than eager
#     )
    
#     # Memory optimizations
#     # model.gradient_checkpointing_enable()
#     model.gradient_checkpointing_disable()
#     model.config.use_cache = False
#     model.train()

#     tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
#     tokenizer.pad_token = tokenizer.eos_token

#     dataset = load_from_disk("data/tinystories")
#     if "train" in dataset: dataset = dataset["train"]
#     dataset = dataset.select(range(64))

#     def collate(batch):
#         txt = [x["text"] for x in batch]
#         tok = tokenizer(txt, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
#         inputs = torch.stack([tok["input_ids"], tok["attention_mask"]], dim=1).to(DEVICE)
#         targets = tok["input_ids"].clone().to(DEVICE)
#         targets[tok["attention_mask"] == 0] = -100
#         return inputs, targets

#     loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate)

#     # Wrap model for PyHessian
#     wrapper = HessianModelWrapper(model)
#     hessian_comp = hessian(wrapper, criterion, dataloader=loader, cuda=True)
    
#     print("Computing Top Eigenvalues...")
#     top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=3)
#     print(f"Top λ: {top_eigenvalues}")
    
#     # --- HARSH CLEANUP ---
#     model.zero_grad()
#     gc.collect()
#     torch.cuda.empty_cache()
#     time.sleep(2) 
    
#     print(f"Computing Density ({DENSITY_ITERS} iters)...")
#     # Using Lanczos density estimation
#     # create_graph=True is called internally by pyhessian.density
#     density_eigenvalues, density_weight = hessian_comp.density(iter=DENSITY_ITERS)
    
#     # Calculate Metrics
#     vals = np.array(density_eigenvalues)
#     q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
#     iqr = q3 - q1
#     ratio = np.max(np.abs(top_eigenvalues)) / iqr if iqr > 0 else 0
    
#     print(f"📊 Ratio: {ratio:.4f} | Bulk Width: {iqr:.4f}")

#     # Save results
#     res_dir = f"results/qwen_generation_{generation}"
#     os.makedirs(res_dir, exist_ok=True)
#     with open(f"{res_dir}/stats.json", "w") as f:
#         json.dump({
#             "generation": generation, 
#             "ratio": ratio, 
#             "iqr": iqr, 
#             "top_lambda": [float(x) for x in top_eigenvalues]
#         }, f)
    
#     # Plotting
#     plt.figure(figsize=(10, 6))
#     plt.hist(density_eigenvalues, weights=density_weight, bins=50, color='purple', alpha=0.7)
#     plt.title(f"Gen {generation} Hessian Spectral Density")
#     plt.xlabel("Eigenvalue")
#     plt.ylabel("Density")
#     plt.savefig(f"{res_dir}/spectrum.png")
#     print(f"✅ Analysis complete. Results saved to {res_dir}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--generation", type=int, required=True)
#     args = parser.parse_args()
#     analyze_qwen(args.generation)

'''import torch
import sys, os, time, json, gc, warnings
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pyhessian import hessian
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_from_disk

# Performance & Memory Environment Variables
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")

# --- 4090 OPTIMIZED SETTINGS ---
BATCH_SIZE = 1        # Keep at 1 for Hessian stability
MAX_LENGTH = 128      # Sequence length
DENSITY_ITERS = 20    # Lanczos iterations
PRECISION = torch.bfloat16 
DEVICE = "cuda:0"
# ---------------------

# Monkey Patch for pyhessian compatibility with modern torch.linalg
def patched_eig(input, eigenvectors=False, out=None):
    if eigenvectors:
        vals, vecs = torch.linalg.eig(input)
        return torch.stack([vals.real, vals.imag], dim=1), vecs.real
    else:
        vals = torch.linalg.eigvals(input)
        return torch.stack([vals.real, vals.imag], dim=1), None
torch.eig = patched_eig

class HessianModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, inputs):
        # inputs shape: [batch, 2, seq_len] -> 0: ids, 1: mask
        return self.model(
            input_ids=inputs[:,0,:].long(), 
            attention_mask=inputs[:,1,:].long(),
            use_cache=False 
        ).logits

def criterion(outputs, targets):
    return torch.nn.CrossEntropyLoss(ignore_index=-100)(
        outputs[..., :-1, :].contiguous().view(-1, outputs.size(-1)),
        targets[..., 1:].contiguous().view(-1)
    )

def analyze_smollm(generation):
    print(f"\n🔬 HESSIAN ANALYSIS: SmolLM2-360M (GEN {generation})")
    
    # Model Path Logic
    path = "HuggingFaceTB/SmolLM2-360M" if generation == 0 else f"models/smollm_gen_{generation}"
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load Model with SDPA (Memory Efficient)
    model = AutoModelForCausalLM.from_pretrained(
        path, 
        torch_dtype=PRECISION,
        device_map=DEVICE,
        attn_implementation="eager", # Vital for 4090 VRAM headroom
        low_cpu_mem_usage=True
    )
    
    model.config.use_cache = False
    model.train() # Gradients must be enabled for Hessian

    # Load Data
    try:
        dataset = load_from_disk("data/tinystories")
        if "train" in dataset: dataset = dataset["train"]
        dataset = dataset.select(range(64)) # Representative subset
    except Exception as e:
        print(f"❌ Data Error: {e}. Ensure 'data/tinystories' exists.")
        return

    def collate(batch):
        txt = [x["text"] for x in batch]
        tok = tokenizer(txt, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
        inputs = torch.stack([tok["input_ids"], tok["attention_mask"]], dim=1).to(DEVICE)
        targets = tok["input_ids"].clone().to(DEVICE)
        targets[tok["attention_mask"] == 0] = -100
        return inputs, targets

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate)

    # Initialize PyHessian Wrapper
    wrapper = HessianModelWrapper(model)
    hessian_comp = hessian(wrapper, criterion, dataloader=loader, cuda=True)
    
    # 1. Compute Top Eigenvalues
    print("Computing Top Eigenvalues...")
    try:
        top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=3)
        print(f"Top λ: {top_eigenvalues}")
    except torch.OutOfMemoryError:
        print("💥 OOM during Eigenvalue computation. Try reducing MAX_LENGTH.")
        return

    # --- AGGRESSIVE MEMORY RESET ---
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)
    
    # 2. Compute Density (Lanczos)
    print(f"Computing Density ({DENSITY_ITERS} iters)...")
    try:
        density_eigenvalues, density_weight = hessian_comp.density(iter=DENSITY_ITERS)
        
        # Calculate Ratio Metrics
        vals = np.array(density_eigenvalues)
        q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
        iqr = q3 - q1
        # Sharpness ratio: Top Lambda vs the bulk width (IQR)
        ratio = np.max(np.abs(top_eigenvalues)) / iqr if iqr > 0 else 0
        print(f"📊 Ratio: {ratio:.4f} | Bulk Width (IQR): {iqr:.4f}")

        # Save Results
        res_dir = f"results/smollm_gen_{generation}"
        os.makedirs(res_dir, exist_ok=True)
        stats = {
            "generation": generation, 
            "ratio": float(ratio), 
            "iqr": float(iqr), 
            "top_lambda": [float(x) for x in top_eigenvalues]
        }
        with open(f"{res_dir}/stats.json", "w") as f:
            json.dump(stats, f, indent=4)
        
        # Plot Spectrum
        plt.figure(figsize=(10, 6))
        plt.hist(density_eigenvalues, weights=density_weight, bins=50, color='royalblue', alpha=0.7)
        plt.title(f"SmolLM2-360M Gen {generation} Hessian Spectral Density")
        plt.xlabel("Eigenvalue")
        plt.ylabel("Density")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(f"{res_dir}/spectrum.png")
        print(f"✅ Analysis complete. Results saved to {res_dir}")

    except torch.OutOfMemoryError as e:
        print(str(e))
        # print("💥 OOM during Density computation. Reducing DENSITY_ITERS is the only fix.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", type=int, required=True)
    args = parser.parse_args()
    analyze_smollm(args.generation)'''
    
    
    
import torch
import sys, os, time, json, gc, warnings
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pyhessian import hessian
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_from_disk

# Performance & Memory Environment Variables
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")

# --- 4090 OPTIMIZED SETTINGS FOR 135M ---
BATCH_SIZE = 1        # Hessian requires small batches for stability
MAX_LENGTH = 128      # Context window for analysis
DENSITY_ITERS = 30    # Increased for 135M to get a smoother spectral density
PRECISION = torch.bfloat16 
DEVICE = "cuda:0"
MODEL_ID = "HuggingFaceTB/SmolLM2-135M" 
# ---------------------

# Monkey Patch for pyhessian compatibility with modern torch.linalg
def patched_eig(input, eigenvectors=False, out=None):
    if eigenvectors:
        vals, vecs = torch.linalg.eig(input)
        return torch.stack([vals.real, vals.imag], dim=1), vecs.real
    else:
        vals = torch.linalg.eigvals(input)
        return torch.stack([vals.real, vals.imag], dim=1), None
torch.eig = patched_eig

class HessianModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, inputs):
        # inputs shape: [batch, 2, seq_len] -> 0: ids, 1: mask
        return self.model(
            input_ids=inputs[:,0,:].long(), 
            attention_mask=inputs[:,1,:].long(),
            use_cache=False 
        ).logits

def criterion(outputs, targets):
    return torch.nn.CrossEntropyLoss(ignore_index=-100)(
        outputs[..., :-1, :].contiguous().view(-1, outputs.size(-1)),
        targets[..., 1:].contiguous().view(-1)
    )

def analyze_smollm(generation):
    print(f"\n🔬 HESSIAN ANALYSIS: SmolLM2-135M (GEN {generation})")
    
    # Model Path Logic
    path = MODEL_ID if generation == 0 else f"models/smollm_135m_gen_{generation}"
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load Model (135M variant)
    model = AutoModelForCausalLM.from_pretrained(
        path, 
        torch_dtype=PRECISION,
        device_map=DEVICE,
        attn_implementation="eager", 
        low_cpu_mem_usage=True
    )
    
    model.config.use_cache = False
    model.train() # Must be in train mode for Hessian backprop

    # Load Data
    try:
        dataset = load_from_disk("data/tinystories")
        if "train" in dataset: dataset = dataset["train"]
        dataset = dataset.select(range(64)) # Representative subset
    except Exception as e:
        print(f"❌ Data Error: {e}. Ensure 'data/tinystories' exists.")
        return

    def collate(batch):
        txt = [x["text"] for x in batch]
        tok = tokenizer(txt, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
        inputs = torch.stack([tok["input_ids"], tok["attention_mask"]], dim=1).to(DEVICE)
        targets = tok["input_ids"].clone().to(DEVICE)
        targets[tok["attention_mask"] == 0] = -100
        return inputs, targets

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate)

    # Initialize PyHessian Wrapper
    wrapper = HessianModelWrapper(model)
    hessian_comp = hessian(wrapper, criterion, dataloader=loader, cuda=True)
    
    # 1. Compute Top Eigenvalues
    print("Computing Top Eigenvalues...")
    try:
        top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=3)
        print(f"Top λ: {top_eigenvalues}")
    except torch.OutOfMemoryError:
        print("💥 OOM during Eigenvalue computation.")
        return

    # Clear cache before heavy Lanczos iterations
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. Compute Density (Lanczos)
    print(f"Computing Density ({DENSITY_ITERS} iters)...")
    try:
        density_eigenvalues, density_weight = hessian_comp.density(iter=DENSITY_ITERS)
        
        # Calculate Ratio Metrics
        vals = np.array(density_eigenvalues)
        q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
        iqr = q3 - q1
        ratio = np.max(np.abs(top_eigenvalues)) / iqr if iqr > 0 else 0
        print(f"📊 Ratio: {ratio:.4f} | Bulk Width (IQR): {iqr:.4f}")

        # Save Results
        res_dir = f"results/smollm_135m_gen_{generation}"
        os.makedirs(res_dir, exist_ok=True)
        stats = {
            "generation": generation, 
            "ratio": float(ratio), 
            "iqr": float(iqr), 
            "top_lambda": [float(x) for x in top_eigenvalues]
        }
        with open(f"{res_dir}/stats.json", "w") as f:
            json.dump(stats, f, indent=4)
        
        # Plot Spectrum
        plt.figure(figsize=(10, 6))
        plt.hist(density_eigenvalues, weights=density_weight, bins=50, color='crimson', alpha=0.7)
        plt.title(f"SmolLM2-135M Gen {generation} Hessian Spectral Density")
        plt.xlabel("Eigenvalue")
        plt.ylabel("Density")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(f"{res_dir}/spectrum.png")
        print(f"✅ Analysis complete. Results saved to {res_dir}")

    except torch.OutOfMemoryError:
        print("💥 OOM during Density computation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", type=int, required=True)
    args = parser.parse_args()
    analyze_smollm(args.generation)