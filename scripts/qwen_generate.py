import torch
import os
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

# --- CONFIG ---
NUM_SAMPLES = 20000     # How many stories to write? (Validation Plan says 10k+, let's do 20k)
BATCH_SIZE = 32         # 4090 can handle big batches
MAX_NEW_TOKENS = 256    # Story length
# --------------

def generate_data(generation):
    output_dir = f"data/synthetic_gen_{generation+1}" # Data FOR next gen
    model_path = f"models/qwen_generation_{generation}"
    
    print(f"\n✍️ GENERATING DATA: Model Gen {generation} -> Data for Gen {generation+1}")
    print(f"Loading: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left" # Required for batched generation
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    model.eval()

    # TinyStories Prompts
    prompts = [
        "Once upon a time", "Create a story about", "One day", "There was a little",
        "The girl wanted", "A boy found", "The dog", "In the forest"
    ] * (NUM_SAMPLES // 8 + 1)
    prompts = prompts[:NUM_SAMPLES]

    new_stories = []
    
    print("Generating...")
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
            batch_prompts = prompts[i : i + BATCH_SIZE]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to("cuda")
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            new_stories.extend(decoded)

    # Save as HuggingFace Dataset
    os.makedirs(output_dir, exist_ok=True)
    dataset = Dataset.from_dict({"text": new_stories})
    dataset.save_to_disk(output_dir)
    print(f"✅ Saved {len(new_stories)} stories to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", type=int, required=True)
    args = parser.parse_args()
    generate_data(args.generation)