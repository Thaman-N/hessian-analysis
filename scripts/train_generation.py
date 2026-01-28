import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import GPT2Tokenizer
from models.llama_model import LlamaModel, ModelConfig
from tqdm import tqdm
import json

def create_dataloader(data_path, tokenizer, batch_size=16, max_length=512):
    dataset = load_from_disk(data_path)
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=max_length,
            return_tensors="pt"
        )
        # Convert to lists for dataset processing
        return {
            "input_ids": tokenized["input_ids"].tolist(),
            "attention_mask": tokenized["attention_mask"].tolist()
        }
    
    train_dataset = dataset["train"].map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"]
    )
    
    # Set format to torch tensors
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )
    
    return dataloader

def train_generation_0():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model and tokenizer
    config = ModelConfig()
    model = LlamaModel(config).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1_000_000:.1f}M")
    
    # Data
    dataloader = create_dataloader("data/tinystories", tokenizer, batch_size=4)  # Smaller batch
    
    # Training setup - LOWER LEARNING RATE
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Training loop
    model.train()
    total_loss = 0
    num_steps = 1000
    valid_steps = 0
    
    progress_bar = tqdm(range(num_steps))
    data_iter = iter(dataloader)
    
    for step in progress_bar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
            
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Create labels (shift input_ids by 1 for next-token prediction)
        labels = input_ids.clone()
        # Ignore padding tokens in loss
        labels[attention_mask == 0] = -100
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"NaN loss at step {step}, skipping...")
            continue
            
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # BETTER gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        # Check gradient norm
        if grad_norm > 10.0:
            print(f"Large gradient norm: {grad_norm:.2f} at step {step}")
            
        optimizer.step()
        
        total_loss += loss.item()
        valid_steps += 1
        avg_loss = total_loss / valid_steps
        
        progress_bar.set_description(f"Loss: {loss.item():.4f}, Avg: {avg_loss:.4f}, Grad: {grad_norm:.2f}")
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}, Grad norm: {grad_norm:.2f}")
    
    # Save model
    os.makedirs("models/generation_0", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_loss': total_loss/valid_steps if valid_steps > 0 else float('inf')
    }, "models/generation_0/model.pt")
    
    print(f"Generation 0 model saved! Final avg loss: {total_loss/valid_steps:.4f}")
    return model

if __name__ == "__main__":
    train_generation_0()