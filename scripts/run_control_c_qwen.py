import os
import torch
import argparse
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

# --- CONFIG ---
MODEL_ID = "Qwen/Qwen2.5-0.5B"
MAX_LENGTH = 256
BATCH_SIZE = 4  # Reduced for 0.5B model safety
LR = 5e-5
SAMPLES = 50000

def run_qwen_loop(generations):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for gen in range(1, generations + 1):
        print(f"\n🚀 CONTROL C (Qwen): Gen {gen}...")
        
        # --- PHASE 1: GENERATE ---
        source_path = MODEL_ID if gen == 1 else f"models/control_c_gen_{gen-1}"
        print(f"   Generating from: {source_path}")
        
        gen_model = AutoModelForCausalLM.from_pretrained(source_path, torch_dtype=torch.float16).to(device)
        gen_tok = AutoTokenizer.from_pretrained(source_path)
        gen_tok.padding_side = "left" # Critical for generation
        if gen_tok.pad_token is None: gen_tok.pad_token = gen_tok.eos_token
        
        prompts = ["Once upon a time,", "The little cat", "One sunny day,", "In a dark forest,"]
        stories = []
        gen_model.eval()
        
        with torch.no_grad():
            pbar = tqdm(total=SAMPLES, desc="Generating")
            while len(stories) < SAMPLES:
                batch_prompts = [random.choice(prompts) for _ in range(8)]
                inputs = gen_tok(batch_prompts, return_tensors="pt", padding=True).to(device)
                outputs = gen_model.generate(**inputs, max_length=MAX_LENGTH, do_sample=True, temperature=0.8, top_k=50)
                decoded = gen_tok.batch_decode(outputs, skip_special_tokens=True)
                stories.extend(decoded)
                pbar.update(len(decoded))
        
        data_path = f"data/control_c_syn_{gen}"
        Dataset.from_dict({"text": stories[:SAMPLES]}).save_to_disk(data_path)
        del gen_model
        torch.cuda.empty_cache()

        # --- PHASE 2: TRAIN ---
        print(f"   Training Gen {gen}...")
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.padding_side = "right" # Reset for training
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
        dataset = load_from_disk(data_path)
        def tok(ex): return tokenizer(ex["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
        dataset = dataset.map(tok, batched=True, remove_columns=["text"])
        dataset.set_format("torch")
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
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
            
        model_dir = f"models/control_c_gen_{gen}"
        result_dir = f"results/control_c_gen_{gen}"
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        del model
        torch.cuda.empty_cache()

        # --- PHASE 3: ANALYZE ---
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
    run_qwen_loop(args.generations)