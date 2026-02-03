import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
OUTPUT_DIR = "models/generation_0"
MAX_LENGTH = 256
BATCH_SIZE = 8       # Adjusted for 24GB VRAM
LR = 5e-5            # Gentle learning rate for fine-tuning
EPOCHS = 1           # One pass is enough for domain adaptation
NUM_SAMPLES = 100000 # Subset size to keep training fast but effective

def train_gen0():
    print(f"🚀 Phase 1: Creating Gen 0 Baseline with {MODEL_ID}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Load Pre-trained Model & Tokenizer
    print("Loading SmolLM2-135M...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    
    # Fix padding for Llama-style models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # 2. Load Human Data (TinyStories)
    print(f"Loading Human TinyStories dataset (first {NUM_SAMPLES} samples)...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    dataset = dataset.select(range(NUM_SAMPLES)) 

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

    print("Tokenizing data...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format("torch")
    
    dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Optimization Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    num_training_steps = len(dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=500, num_training_steps=num_training_steps
    )

    # 4. Training Loop
    model.train()
    print(f"Starting training for {num_training_steps} steps...")
    
    progress_bar = tqdm(range(num_training_steps))
    
    for epoch in range(EPOCHS):
        for batch in dataloader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass (AutoModelForCausalLM handles loss automatically)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            progress_bar.set_description(f"Loss: {loss.item():.4f}")
            progress_bar.update(1)

    # 5. Save Gen 0
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving Gen 0 model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Gen 0 Baseline Established!")

if __name__ == "__main__":
    train_gen0()