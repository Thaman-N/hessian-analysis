import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import mauve
import pandas as pd
import glob
import os
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SAMPLES = 200  # Number of generations to test (keep it fast)
GEN_LEN = 128      # Length of text to generate

def calculate_entropy(model, tokenizer, dataset):
    """Calculates Shannon Entropy of the model's predictive distribution"""
    # We use the first batch of validation data to check the model's 'confidence' spread
    text = "\n\n".join(dataset["text"][:5]) 
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Softmax to get probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Entropy = -sum(p * log(p))
    # We average over the sequence
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    
    return entropy.item()

def main():
    print("📉 Initializing Vanilla Metrics (Entropy & MAUVE)...")
    
    # 1. Load Human Reference Data (TinyStories Validation)
    val_dataset = load_dataset("roneneldan/TinyStories", split="validation")
    human_texts = val_dataset.select(range(MAX_SAMPLES))["text"]
    
    all_folders = sorted(glob.glob("models/*"))
    records = []

    for folder in all_folders:
        if not os.path.exists(os.path.join(folder, "config.json")): continue
        
        try:
            # Identity Parsing
            if "control_c" in folder: exp = "Control C (Qwen)"
            elif "control_b" in folder: exp = "Control B (Static)"
            elif "control" in folder: exp = "Control A (Fresh)"
            elif "generation_0" in folder: exp = "Baseline"
            else: exp = "Treatment (Recursive)"
            try: gen = int(folder.split('_')[-1])
            except: gen = 0
            
            print(f"\n🧪 Evaluating: {folder}")
            model = AutoModelForCausalLM.from_pretrained(folder, torch_dtype=torch.float16).to(DEVICE)
            tokenizer = AutoTokenizer.from_pretrained(folder)
            
            # --- METRIC 1: SHANNON ENTROPY (Vocabulary Health) ---
            entropy = calculate_entropy(model, tokenizer, val_dataset)
            print(f"   👉 Entropy: {entropy:.4f}")
            
            # --- METRIC 2: MAUVE (Distribution Quality) ---
            # Generate text first
            gen_texts = []
            print("   Generating text for MAUVE...")
            # Simple prompts for generation
            prompts = human_texts[:20] # Use first 20 human stories as prompts
            for prompt in prompts:
                inputs = tokenizer(prompt[:50], return_tensors="pt").to(DEVICE) # First 50 chars as prompt
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=GEN_LEN, do_sample=True, temperature=0.9)
                gen_texts.append(tokenizer.decode(out[0], skip_special_tokens=True))
            
            # Calculate MAUVE (Human vs Generated)
            # Note: Mauve usually needs >1000 samples for high precision, but 
            # for "obvious" collapse, 20-50 samples is enough to see the crash.
            out_mauve = mauve.compute_mauve(
                p_text=human_texts[:len(gen_texts)], 
                q_text=gen_texts, 
                device_id=0 if torch.cuda.is_available() else -1,
                max_text_length=256,
                verbose=False
            )
            mauve_score = out_mauve.mauve
            print(f"   👉 MAUVE: {mauve_score:.4f}")

            records.append({
                "Experiment": exp,
                "Generation": gen,
                "Shannon_Entropy": entropy,
                "MAUVE": mauve_score
            })
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Skipped {folder}: {e}")

    # Save Results
    df = pd.DataFrame(records).sort_values(by=["Experiment", "Generation"])
    os.makedirs("results/summary", exist_ok=True)
    df.to_csv("results/summary/vanilla_metrics.csv", index=False)
    print("\n✅ Saved Vanilla Metrics to results/summary/vanilla_metrics.csv")

if __name__ == "__main__":
    main()