import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import glob
import os
import numpy as np

# --- CONFIG ---
MAX_LENGTH = 512
STRIDE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_perplexity(model, tokenizer, dataset):
    """Calculates PPL on real human data (Validation set)"""
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    
    nlls = []
    # Loop with stride to handle long text
    for i in tqdm(range(0, encodings.input_ids.size(1), STRIDE), desc="   Calculating PPL", leave=False):
        begin_loc = max(i + STRIDE - MAX_LENGTH, 0)
        end_loc = min(i + STRIDE, encodings.input_ids.size(1))
        trg_len = end_loc - i
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100 # Mask context

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss)

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

def calculate_repetition(model, tokenizer):
    """Generates text and checks for n-gram repetition (Mode Collapse)"""
    prompts = ["Once upon a time", "The little girl", "In the dark forest", "The sun was"]
    generated_text = ""
    
    model.eval()
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            # Generate 100 tokens
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
        generated_text += tokenizer.decode(outputs[0], skip_special_tokens=True) + " "

    # Calculate Unique Bigrams / Total Bigrams
    tokens = generated_text.split()
    bigrams = list(zip(tokens, tokens[1:]))
    if len(bigrams) == 0: return 0.0
    unique_bigrams = len(set(bigrams))
    return unique_bigrams / len(bigrams)

def main():
    # 1. Load Validation Data (Real Human Stories)
    print("📚 Loading Validation Data...")
    val_data = load_dataset("roneneldan/TinyStories", split="validation").select(range(50)) # Fast check

    # 2. Find all models (Treatment, Controls, Qwen)
    # This looks for ALL folders starting with 'models/'
    all_folders = glob.glob("models/*")
    
    records = []
    print(f"🚀 Found {len(all_folders)} folders. Scanning for models...")

    for folder in sorted(all_folders):
        # Skip empty or non-model folders
        if not os.path.exists(os.path.join(folder, "config.json")):
            continue
            
        print(f"\n🔬 Evaluating: {folder}")
        try:
            # Determine Type
            if "control_c" in folder: exp_type = "Control C (Qwen)"
            elif "control_b" in folder: exp_type = "Control B (Static)"
            elif "control" in folder: exp_type = "Control A (Fresh)"
            elif "generation_0" in folder: exp_type = "Baseline"
            else: exp_type = "Treatment (Recursive)"
            
            # Determine Gen
            try: gen = int(folder.split('_')[-1])
            except: gen = 0

            # Load Model (Float16 for speed/memory)
            model = AutoModelForCausalLM.from_pretrained(folder, torch_dtype=torch.float16).to(DEVICE)
            tokenizer = AutoTokenizer.from_pretrained(folder)
            
            # Calc Metrics
            ppl = calculate_perplexity(model, tokenizer, val_data)
            rep_score = calculate_repetition(model, tokenizer)
            
            print(f"   👉 PPL: {ppl:.2f} | Uniqueness: {rep_score:.2f}")
            
            records.append({
                "Experiment": exp_type,
                "Generation": gen,
                "Perplexity": ppl,
                "Uniqueness_Score": rep_score
            })
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ⚠️ Failed: {e}")

    # Save to CSV
    df = pd.DataFrame(records).sort_values(by=["Experiment", "Generation"])
    os.makedirs("results/summary", exist_ok=True)
    df.to_csv("results/summary/text_quality_metrics.csv", index=False)
    print("\n✅ Saved Text Metrics to results/summary/text_quality_metrics.csv")

if __name__ == "__main__":
    main()