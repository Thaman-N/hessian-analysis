import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import glob
import os
from tqdm import tqdm

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # Tiny, fast, standard for similarity

def calculate_coherence(model, tokenizer, embed_model, prompts):
    """Generates stories and measures semantic similarity to the prompt"""
    similarities = []
    
    model.eval()
    for prompt in tqdm(prompts, desc="   Generating & Embedding", leave=False):
        # 1. Generate Story
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=64,       # Short generation is enough to check logic
                do_sample=True, 
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 2. Embed Prompt vs. Output (remove the prompt from the output for fair comparison)
        # We want to know if the NEW text relates to the PROMPT
        new_text = generated_text.replace(prompt, "").strip()
        if not new_text: new_text = "empty" # Handle collapse to silence

        embeddings = embed_model.encode([prompt, new_text], convert_to_tensor=True)
        
        # 3. Calculate Cosine Similarity
        sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        similarities.append(sim.item())

    return sum(similarities) / len(similarities)

def main():
    print("🧠 Loading Sentence Transformer (Judge)...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME).to("cpu") # CPU is fast enough for embeddings
    
    # Test Prompts (Simple TinyStories style)
    prompts = [
        "Once upon a time, there was a little dog.",
        "Timmy loved to play with his red ball.",
        "The sun was shining bright in the sky.",
        "Lily went to the park with her mom.",
        "One day, a big bear came to the house."
    ]

    all_folders = sorted(glob.glob("models/*"))
    records = []
    
    print(f"🚀 Scanning {len(all_folders)} models for Semantic Drift...")

    for folder in all_folders:
        if not os.path.exists(os.path.join(folder, "config.json")): continue
        
        try:
            # Determine Experiment
            if "control_c" in folder: exp = "Control C (Qwen)"
            elif "control_b" in folder: exp = "Control B (Static)"
            elif "control" in folder: exp = "Control A (Fresh)"
            elif "generation_0" in folder: exp = "Baseline"
            else: exp = "Treatment (Recursive)"
            
            # Determine Gen
            try: gen = int(folder.split('_')[-1])
            except: gen = 0
            
            print(f"\n🔬 Checking Coherence: {folder}")
            
            # Load Generative Model
            model = AutoModelForCausalLM.from_pretrained(folder, torch_dtype=torch.float16).to(DEVICE)
            tokenizer = AutoTokenizer.from_pretrained(folder)
            
            # Run Test
            coherence_score = calculate_coherence(model, tokenizer, embed_model, prompts)
            
            print(f"   👉 Coherence Score: {coherence_score:.4f}")
            records.append({
                "Experiment": exp,
                "Generation": gen,
                "Semantic_Coherence": coherence_score
            })
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ⚠️ Failed: {e}")

    # Save
    df = pd.DataFrame(records).sort_values(by=["Experiment", "Generation"])
    os.makedirs("results/summary", exist_ok=True)
    df.to_csv("results/summary/semantic_coherence.csv", index=False)
    print("\n✅ Saved Coherence Data to results/summary/semantic_coherence.csv")

if __name__ == "__main__":
    main()