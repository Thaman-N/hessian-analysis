import os
import torch
import argparse
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import gc

# --- CONFIG ---
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MAX_LENGTH = 256
BATCH_SIZE = 8
LR = 5e-5
SAMPLES = 50000 

def train_and_analyze(generations):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load STATIC Data ONCE (0 to 50k)
    # This is the key difference from Control A (which shifts the window)
    print("📥 Loading Static Human Data (0-50k)...")
    dataset = load_dataset("roneneldan/TinyStories", split="train").select(range(SAMPLES))
    
    for gen in range(1, generations + 1):
        print(f"\n🔁 CONTROL B (Static): Gen {gen}...")
        
        # 1. Reset Model (Always start fresh from base)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

        # 2. Tokenize
        def tok_func(ex):
            return tokenizer(ex["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
        tokenized = dataset.map(tok_func, batched=True, remove_columns=["text"])
        tokenized.set_format("torch")
        loader = DataLoader(tokenized, batch_size=BATCH_SIZE, shuffle=True)

        # 3. Train
        optim = torch.optim.AdamW(model.parameters(), lr=LR)
        sched = get_linear_schedule_with_warmup(optim, 100, len(loader))
        model.train()
        
        for batch in tqdm(loader, desc="Training"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, labels=batch["input_ids"])
            outputs.loss.backward()
            optim.step()
            sched.step()
            optim.zero_grad()

        # 4. Save
        model_dir = f"models/control_b_gen_{gen}"
        result_dir = f"results/control_b_gen_{gen}"
        print(f"   Saving to {model_dir}...")
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        
        # --- MEMORY FIX: NUKE THE PARENT MODEL ---
        print("   🧹 Clearing VRAM before Analysis...")
        del model
        del optim
        del sched
        torch.cuda.empty_cache()
        gc.collect()
        # -----------------------------------------

        # 5. Analyze
        print("   Starting Analysis...")
        subprocess.run([
            "python", "scripts/hessian_analysis_generic.py",
            "--generation", str(gen),
            "--model_path", model_dir,
            "--output_dir", result_dir
        ], check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=5)
    args = parser.parse_args()
    train_and_analyze(args.generations)