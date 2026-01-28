"""
Recursive Training Script for Generations 1-10

This script trains a new model (generation N) using synthetic data generated 
from the previous generation (N-1). Uses the same architecture and hyperparameters
as the original generation 0 training for consistency.

Usage:
    python scripts/train_recursive.py --generation 1

Requirements:
    - Synthetic dataset in data/synthetic/generation_N/
    - Same model architecture and training parameters as generation 0
"""

import os
import sys
import json
import time
import torch
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.llama_model import LlamaModel, LlamaConfig
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from datasets import load_from_disk
import torch.nn as nn
from tqdm import tqdm

def create_dataloader(data_path, tokenizer, batch_size=4, max_length=512):
    """Create dataloader for synthetic dataset."""
    print(f"Loading synthetic dataset from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Synthetic dataset not found: {data_path}")
    
    # Load the synthetic dataset
    dataset = load_from_disk(data_path)
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    def tokenize_function(examples):
        # Tokenize the synthetic stories
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        return tokenized
    
    # Tokenize the dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=dataset.column_names
    )
    
    # Create custom dataset for next-token prediction
    class NextTokenDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            item = self.dataset[idx]
            input_ids = torch.tensor(item["input_ids"])
            attention_mask = torch.tensor(item["attention_mask"])
            
            # Create labels for next-token prediction (shifted input_ids)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100  # Ignore padding in loss calculation
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
    
    train_dataset = NextTokenDataset(tokenized_dataset)
    
    # Create dataloader
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader

def train_recursive_model(generation, num_steps=1000, batch_size=4, learning_rate=1e-4):
    """Train a model on synthetic data from previous generation."""
    print("="*70)
    print(f"RECURSIVE TRAINING - GENERATION {generation}")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Training generation {generation} on synthetic data from generation {generation-1}")
    
    # Verify synthetic data exists
    synthetic_data_path = f"data/synthetic/generation_{generation}"
    if not os.path.exists(synthetic_data_path):
        raise FileNotFoundError(f"Synthetic data not found: {synthetic_data_path}")
    
    # Load metadata from synthetic data
    metadata_path = f"{synthetic_data_path}/metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        print(f"Using synthetic data with {metadata['num_samples']} samples")
        print(f"Generated from generation {metadata['source_generation']}")
    
    # Initialize model with same config as generation 0
    config = LlamaConfig(
        vocab_size=50257,  # GPT-2 tokenizer
        hidden_size=512,
        num_layers=12,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=512,
        layer_norm_eps=1e-5,
        use_cache=False,  # Disable during training
        pad_token_id=50256,
        eos_token_id=50256,
    )
    
    model = LlamaModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized: {total_params/1_000_000:.1f}M parameters")
    
    # Setup tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataloader
    dataloader = create_dataloader(synthetic_data_path, tokenizer, batch_size=batch_size)
    print(f"Created dataloader: {len(dataloader)} batches")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    total_steps = min(num_steps, len(dataloader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    print(f"\nStarting training for {total_steps} steps...")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    
    model.train()
    start_time = time.time()
    
    running_loss = 0.0
    log_interval = 50
    
    progress_bar = tqdm(total=total_steps, desc=f"Training Gen {generation}")
    
    step = 0
    for epoch in range(10):  # Multiple epochs if needed
        for batch in dataloader:
            if step >= total_steps:
                break
                
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            
            # Calculate loss (next-token prediction)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            running_loss += loss.item()
            
            # Logging
            if step % log_interval == 0 and step > 0:
                avg_loss = running_loss / log_interval
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}'
                })
                running_loss = 0.0
            
            progress_bar.update(1)
            step += 1
            
            if step >= total_steps:
                break
        
        if step >= total_steps:
            break
    
    progress_bar.close()
    
    training_time = time.time() - start_time
    print(f"\n✅ Training completed!")
    print(f"Total time: {training_time/60:.1f} minutes")
    print(f"Final loss: {loss.item():.4f}")
    
    # Save the trained model
    output_dir = f"models/generation_{generation}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model checkpoint
    checkpoint = {
        'generation': generation,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'training_params': {
            'num_steps': total_steps,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'final_loss': loss.item(),
            'training_time_minutes': training_time/60,
        },
        'synthetic_data_path': synthetic_data_path,
        'timestamp': datetime.now().isoformat()
    }
    
    model_path = f"{output_dir}/model.pt"
    torch.save(checkpoint, model_path)
    
    print(f"Model saved to: {model_path}")
    
    # Save training summary
    summary = {
        "generation": generation,
        "total_parameters": int(total_params),
        "training_steps": total_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "final_loss": float(loss.item()),
        "training_time_minutes": float(training_time/60),
        "synthetic_data_samples": metadata.get('num_samples', 'unknown') if 'metadata' in locals() else 'unknown',
        "model_path": model_path,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(f"{output_dir}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Training summary saved to: {output_dir}/training_summary.json")
    
    return model_path

def main():
    parser = argparse.ArgumentParser(description="Train recursive generation model")
    parser.add_argument("--generation", type=int, required=True,
                       help="Generation number to train (must be >= 1)")
    parser.add_argument("--num_steps", type=int, default=1000,
                       help="Number of training steps (default: 1000)")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size (default: 4)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate (default: 1e-4)")
    
    args = parser.parse_args()
    
    if args.generation < 1:
        print("❌ Error: Generation must be >= 1 for recursive training")
        print("Use scripts/train_generation.py for generation 0")
        sys.exit(1)
    
    try:
        model_path = train_recursive_model(
            generation=args.generation,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        print(f"\n🎉 Success! Generation {args.generation} model trained!")
        print(f"Next steps:")
        print(f"1. Run Hessian analysis: python scripts/hessian_analysis.py --generation {args.generation}")
        print(f"2. Generate data for next generation: python scripts/generate_synthetic_data.py --generation {args.generation}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()