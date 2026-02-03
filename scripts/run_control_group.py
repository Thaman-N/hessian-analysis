import os
import torch
import argparse
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import shutil

# --- CONFIG ---
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MAX_LENGTH = 256
BATCH_SIZE = 8
LR = 5e-5
SAMPLES_PER_GEN = 50000 

def train_control_model(generation):
    print(f"\n🚀 CONTROL A (Fresh Human): Gen {generation}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load FRESH Base Model
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # 2. Select FRESH Data Slice
    start_idx = (generation - 1) * SAMPLES_PER_GEN
    end_idx = start_idx + SAMPLES_PER_GEN
    print(f"   Slice: {start_idx} to {end_idx}")
    
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    if end_idx > len(dataset):
        dataset = dataset.select(range(start_idx, len(dataset)))
    else:
        dataset = dataset.select(range(start_idx, end_idx))

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")
    dataloader = DataLoader(tokenized, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Train
    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = get_linear_schedule_with_warmup(optim, 100, len(dataloader))

    model.train()
    for batch in tqdm(dataloader, desc=f"Training Gen {generation}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, labels=batch["input_ids"])
        outputs.loss.backward()
        optim.step()
        sched.step()
        optim.zero_grad()

    # 4. Save & Analyze
    model_dir = f"models/control_generation_{generation}"
    result_dir = f"results/control_generation_{generation}"
    
    print(f"   Saving to {model_dir}...")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    # --- MEMORY FIX: NUKE THE PARENT MODEL ---
    print("   🧹 Clearing VRAM before Analysis...")
    del model
    del optim
    del sched
    torch.cuda.empty_cache()
    
    subprocess.run([
        "python", "scripts/hessian_analysis_generic.py",
        "--generation", str(generation),
        "--model_path", model_dir,
        "--output_dir", result_dir
    ], check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=5)
    args = parser.parse_args()
    
    # Ensure Gen 0 exists (baseline)
    if not os.path.exists("models/generation_0"):
        print("⚠️ Warning: Gen 0 baseline not found. Ideally run Phase 1 first.")

    for gen in range(1, args.generations + 1):
        train_control_model(gen)