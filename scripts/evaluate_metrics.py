import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import numpy as np

# --- Configuration ---
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MAX_LENGTH = 512
EVAL_SAMPLES = 200 # Number of samples to test perplexity on

def calculate_perplexity(model, tokenizer, dataset, device):
    """Calculates Perplexity on HUMAN data."""
    model.eval()
    nlls = []
    
    # Select a subset of HUMAN validation data
    subset = dataset.select(range(EVAL_SAMPLES))
    
    with torch.no_grad():
        for example in tqdm(subset, desc="Calculating Perplexity"):
            encodings = tokenizer(example["text"], return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
            input_ids = encodings.input_ids.to(device)
            target_ids = input_ids.clone()
            
            # Calculate loss
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss)

    # Perplexity = exp(average_loss)
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

def measure_repetition(model, tokenizer, device):
    """Checks if the model generates repetitive loops."""
    prompt = "Once upon a time,"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    outputs = model.generate(
        **inputs, 
        max_length=100, 
        do_sample=False, # Greedy decoding reveals loops best
        pad_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Simple N-gram repetition check (4-grams)
    words = text.split()
    four_grams = [tuple(words[i:i+4]) for i in range(len(words)-3)]
    unique_4grams = len(set(four_grams))
    total_4grams = len(four_grams)
    
    # Repetition Ratio: Lower is better (1.0 = all unique)
    rep_ratio = unique_4grams / total_4grams if total_4grams > 0 else 0
    return text, rep_ratio

def evaluate_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading Human Validation Data...")
    dataset = load_dataset("roneneldan/TinyStories", split="validation")
    
    results = {}
    
    # Loop through Gen 0 to 5
    for gen in [0, 1, 2, 3, 4, 5]:
        model_path = f"models/generation_{gen}"
        if not os.path.exists(model_path):
            print(f"Skipping Gen {gen} (not found)")
            continue
            
        print(f"\n📊 Evaluating Generation {gen}...")
        
        # Load Model
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        except:
            print(f"Error loading Gen {gen}")
            continue

        # 1. Perplexity (Human Data)
        ppl = calculate_perplexity(model, tokenizer, dataset, device)
        
        # 2. Repetition Check
        sample_text, rep_ratio = measure_repetition(model, tokenizer, device)
        
        print(f"   Perplexity: {ppl:.2f}")
        print(f"   Unique 4-gram Ratio: {rep_ratio:.2f} (Lower = More Repetitive)")
        print(f"   Sample: {sample_text[:60]}...")
        
        results[gen] = {
            "perplexity": ppl,
            "repetition_ratio": rep_ratio,
            "sample": sample_text
        }

    # Save
    with open("results/summary/final_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\n✅ Evaluation Complete. Saved to results/summary/final_metrics.json")

if __name__ == "__main__":
    evaluate_all()