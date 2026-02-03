import torch
import os
import argparse
import json
import random
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"  # Fallback for tokenizer
MAX_LENGTH = 256  # Matches your training constraints
# ---------------------

def load_model(generation, device):
    """
    Loads the correct model for the generation.
    - Gen 0: Reads from 'models/generation_0'
    - Gen N: Reads from 'models/generation_N'
    """
    model_path = f"models/generation_{generation}"
    print(f"📂 Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}. Did you run training?")

    try:
        # Load the fine-tuned model
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float32
        ).to(device)
        
        # Load tokenizer (try local first, else fallback to hub)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except:
            print("⚠️ Local tokenizer not found, loading from Hub...")
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
            
        # --- CRITICAL FIX: LEFT PADDING FOR GENERATION ---
        tokenizer.padding_side = "left"  # <--- THIS PREVENTS THE WARNING AND BROKEN DATA
        
        # Fix padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id
            
        model.eval()
        return model, tokenizer
        
    except Exception as e:
        raise RuntimeError(f"Failed to load SmolLM model: {e}")

def generate_story_batch(model, tokenizer, device, batch_size=10, temperature=0.8, top_k=50):
    """Generates a batch of stories efficiently."""
    
    # TinyStories Prompts (Completion Style)
    prompts = [
        "Once upon a time,", "One day,", "There was a little", 
        "The sun was shining", "In a big forest,", "Alice and Bob",
        "A small cat", "The old man", "On a high mountain,",
        "Every morning,", "Tim was a", "Daisy liked to"
    ]
    
    # Select random prompts for the batch
    batch_prompts = [random.choice(prompts) for _ in range(batch_size)]
    
    # Tokenize
    inputs = tokenizer(
        batch_prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            min_length=50  # Prevent empty stories
        )
        
    # Decode
    stories = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Simple cleanup (trim incomplete sentences at the end)
    cleaned_stories = []
    for story in stories:
        # Remove prompt from the output to keep data clean (Optional, but usually good)
        # For this experiment, keeping it is fine as long as it's coherent.
        
        # Stop at the last period to ensure complete thoughts
        if '.' in story:
            story = story[:story.rindex('.')+1]
        cleaned_stories.append(story)
        
    return cleaned_stories

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", type=int, required=True, 
                       help="The generation of the MODEL to use (e.g., 0 to generate Gen 1 data)")
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Generating Synthetic Data from Gen {args.generation} Model")
    
    # 1. Load Model
    model, tokenizer = load_model(args.generation, device)
    
    # Double check padding side
    if tokenizer.padding_side != 'left':
        print("⚠️ WARNING: Tokenizer padding side is not 'left'. Forcing it now.")
        tokenizer.padding_side = 'left'

    # 2. Generation Loop
    all_stories = []
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    print(f"📝 Generating {args.num_samples} stories...")
    for _ in tqdm(range(num_batches)):
        batch_stories = generate_story_batch(
            model, tokenizer, device, 
            batch_size=args.batch_size
        )
        all_stories.extend(batch_stories)
        
        # Break if we overshoot slightly
        if len(all_stories) >= args.num_samples:
            all_stories = all_stories[:args.num_samples]
            break

    # 3. Save as HuggingFace Dataset
    target_gen = args.generation + 1
    output_dir = f"data/synthetic/generation_{target_gen}"
    
    print(f"💾 Saving to {output_dir}...")
    dataset_dict = {"text": all_stories}
    dataset = Dataset.from_dict(dataset_dict)
    dataset.save_to_disk(output_dir)
    
    # 4. Save Metadata
    metadata = {
        "source_generation": args.generation,
        "target_generation": target_gen,
        "num_samples": len(all_stories),
        "model_path": f"models/generation_{args.generation}"
    }
    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"✅ Success! Data for Gen {target_gen} is ready.")

if __name__ == "__main__":
    main()