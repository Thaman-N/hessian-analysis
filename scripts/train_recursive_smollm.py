import os
import sys
import json
import torch
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from datasets import load_from_disk
import torch.nn as nn
from tqdm import tqdm

# --- Configuration ---
MODEL_ID = "HuggingFaceTB/SmolLM2-135M" 
MAX_LENGTH = 256  # Matches your Gen 0 training
BATCH_SIZE = 8    # Safe for 24GB VRAM
LR = 5e-5         # Same as Gen 0

def create_dataloader(data_path, tokenizer, batch_size=4, max_length=256):
    print(f"Loading synthetic dataset from: {data_path}")
    dataset = load_from_disk(data_path)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
    
    print("Tokenizing synthetic data...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format("torch")
    
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train_recursive(generation):
    prev_gen = generation - 1
    print(f"🚀 Phase 2: Training Generation {generation} (using Gen {prev_gen} data)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the PREVIOUS model to initialize weights (Standard Recursive Practice)
    # or Load Base SmolLM2? 
    # STANDARD: Load Base SmolLM2 every time to isolate "Data Quality" as the only variable.
    # CONTINUAL: Load Gen N-1.
    # RECOMMENDATION: Load Base SmolLM2. We want to see if the DATA is broken.
    print(f"Loading fresh SmolLM2 base model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # 2. Load Synthetic Data
    data_path = f"data/synthetic/generation_{generation}"
    if not os.path.exists(data_path):
        print(f"❌ Error: Synthetic data for Gen {generation} not found at {data_path}")
        print(f"   Run 'generate_synthetic_data.py' for Gen {prev_gen} first.")
        return

    dataloader = create_dataloader(data_path, tokenizer, BATCH_SIZE, MAX_LENGTH)

    # 3. Training Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    num_training_steps = len(dataloader) # 1 Epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=num_training_steps
    )

    # 4. Train
    model.train()
    print(f"Starting training for {num_training_steps} steps...")
    progress_bar = tqdm(range(num_training_steps))
    
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        progress_bar.set_description(f"Loss: {loss.item():.4f}")
        progress_bar.update(1)

    # 5. Save
    output_dir = f"models/generation_{generation}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving Gen {generation} model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Generation {generation} Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", type=int, required=True)
    args = parser.parse_args()
    train_recursive(args.generation)