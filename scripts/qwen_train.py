import os
import sys
import json
import torch
import argparse
import time
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader
from datasets import load_from_disk, load_dataset

# --- CONFIGURATION FOR RTX 4090 ---
BATCH_SIZE = 16          # 24GB can handle this easily for 0.5B
GRAD_ACCUMULATION = 4    # Effective Batch = 64
LEARNING_RATE = 5e-5     
NUM_EPOCHS = 1           
MAX_LENGTH = 512         
# ----------------------------------

def train_qwen(generation, data_path, output_dir):
    print(f"\n🔥 STARTING TRAINING: GEN {generation} on RTX 4090")
    device = torch.device("cuda")

    # 1. Load Tokenizer
    base_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Model
    if generation == 0:
        print(f"Loading Base: {base_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=torch.bfloat16, # 4090 loves BF16
            attn_implementation="flash_attention_2", # Use Flash Attention 2!
            device_map="auto"
        )
    else:
        prev_path = f"models/qwen_generation_{generation-1}"
        print(f"Loading Gen {generation-1}: {prev_path}")
        model = AutoModelForCausalLM.from_pretrained(
            prev_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )

    # 3. Load Data
    print(f"Loading Data: {data_path}")
    if generation == 0 and not os.path.exists(data_path):
        # First run auto-download
        dataset = load_dataset("roneneldan/TinyStories")
        dataset.save_to_disk("data/tinystories")
        dataset = dataset["train"]
    else:
        dataset = load_from_disk(data_path)
        if "train" in dataset: dataset = dataset["train"]

    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples["text"], truncation=True, padding="max_length",
            max_length=MAX_LENGTH, return_tensors="pt"
        )
    
    print("Tokenizing...")
    dataset = dataset.select(range(min(len(dataset), 50000))) # Train on 50k samples per gen
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")
    
    loader = DataLoader(tokenized, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    # 4. Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, 100, len(loader)*NUM_EPOCHS)
    
    model.train()
    progress = tqdm(total=len(loader)*NUM_EPOCHS, desc=f"Gen {generation}")
    
    start_time = time.time()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["input_ids"])
        
        loss = outputs.loss / GRAD_ACCUMULATION
        loss.backward()
        
        if (progress.n + 1) % GRAD_ACCUMULATION == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_description(f"Loss: {outputs.loss.item():.4f}")
        
        progress.update(1)

    # 5. Save
    print(f"Saving to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✅ Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--data_path", type=str, default="data/tinystories")
    args = parser.parse_args()
    
    os.makedirs(f"models/qwen_generation_{args.generation}", exist_ok=True)
    train_qwen(args.generation, args.data_path, f"models/qwen_generation_{args.generation}")