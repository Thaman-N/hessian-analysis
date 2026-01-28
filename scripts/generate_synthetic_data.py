"""
Generate Synthetic Data for Recursive Training

This script generates synthetic TinyStories-style text using a trained model
from generation N, which will be used to train generation N+1.

Usage:
    python scripts/generate_synthetic_data.py --generation 0 --num_samples 100000

Requirements:
    - Trained model in models/generation_N/model.pt
    - Same tokenizer and model architecture as original training
"""

import os
import sys
import json
import torch
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset
import warnings
warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.llama_model import LlamaModel, LlamaConfig
from transformers import GPT2Tokenizer

def load_model(model_path, device):
    """Load trained model from checkpoint."""
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    model = LlamaModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config

def generate_story(model, tokenizer, device, max_length=256, temperature=0.8, top_k=50):
    """Generate a single synthetic story."""
    # Start with a random story prompt
    prompts = [
        "Once upon a time,",
        "There was a little",
        "One day,", 
        "In a small town,",
        "Long ago,",
        "A young child",
        "The sun was shining when",
        "Every morning,",
        "Deep in the forest,",
        "At the park,"
    ]
    
    prompt = random.choice(prompts)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate continuation
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
    
    # Decode generated text
    story = tokenizer.decode(generated[0], skip_special_tokens=True)
    
    # Clean up the story (remove incomplete sentences, etc.)
    sentences = story.split('. ')
    if len(sentences) > 1:
        # Keep complete sentences only
        story = '. '.join(sentences[:-1]) + '.'
    
    return story

def generate_synthetic_dataset(generation, num_samples=100000, batch_size=100):
    """Generate synthetic dataset from trained model."""
    print("="*60)
    print(f"GENERATING SYNTHETIC DATA - FROM GENERATION {generation}")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load the trained model
    model_path = f"models/generation_{generation}/model.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model, config = load_model(model_path, device)
    
    # Setup tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Generating {num_samples} synthetic stories...")
    print(f"Using temperature=0.8, top_k=50")
    
    # Generate stories in batches
    all_stories = []
    
    with tqdm(total=num_samples, desc="Generating stories") as pbar:
        for i in range(0, num_samples, batch_size):
            batch_size_actual = min(batch_size, num_samples - i)
            batch_stories = []
            
            for _ in range(batch_size_actual):
                try:
                    story = generate_story(model, tokenizer, device)
                    if len(story.strip()) > 20:  # Filter out very short stories
                        batch_stories.append(story.strip())
                except Exception as e:
                    print(f"Error generating story: {e}")
                    continue
            
            all_stories.extend(batch_stories)
            pbar.update(len(batch_stories))
    
    print(f"Generated {len(all_stories)} valid stories")
    
    if len(all_stories) == 0:
        raise RuntimeError("No valid stories were generated!")
    
    # Create output directory
    output_dir = f"data/synthetic/generation_{generation + 1}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Creating output directory: {output_dir}")
    
    # Create dataset in HuggingFace format
    dataset_dict = {
        "text": all_stories
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    # Save the dataset
    print(f"Saving dataset to: {output_dir}")
    dataset.save_to_disk(output_dir)
    
    # Verify the dataset was saved
    if not os.path.exists(output_dir):
        raise RuntimeError(f"Failed to create output directory: {output_dir}")
        
    # Check if dataset files exist
    dataset_files = os.listdir(output_dir)
    print(f"Dataset files created: {dataset_files}")
    
    # Save metadata
    metadata = {
        "source_generation": generation,
        "target_generation": generation + 1,
        "num_samples": len(all_stories),
        "generation_params": {
            "temperature": 0.8,
            "top_k": 50,
            "max_length": 256
        },
        "model_path": model_path,
        "tokenizer": "gpt2"
    }
    
    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Synthetic dataset saved to: {output_dir}")
    print(f"Dataset size: {len(all_stories)} samples")
    
    # Verify we can reload the dataset
    try:
        from datasets import load_from_disk
        test_dataset = load_from_disk(output_dir)
        print(f"✅ Dataset verification: {len(test_dataset)} samples loaded successfully")
    except Exception as e:
        print(f"❌ Dataset verification failed: {e}")
        raise RuntimeError(f"Generated dataset is corrupted: {e}")
    
    # Show some example stories
    print("\nExample generated stories:")
    print("-" * 50)
    for i, story in enumerate(all_stories[:3]):
        print(f"Story {i+1}: {story[:200]}..." if len(story) > 200 else f"Story {i+1}: {story}")
        print("-" * 50)
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data from trained model")
    parser.add_argument("--generation", type=int, required=True,
                       help="Generation number of the source model (e.g., 0 to generate data for gen 1)")
    parser.add_argument("--num_samples", type=int, default=100000,
                       help="Number of synthetic samples to generate (default: 100000)")
    parser.add_argument("--batch_size", type=int, default=100,
                       help="Batch size for generation (default: 100)")
    
    args = parser.parse_args()
    
    try:
        output_dir = generate_synthetic_dataset(
            generation=args.generation,
            num_samples=args.num_samples,
            batch_size=args.batch_size
        )
        
        print(f"\n🎉 Success! Synthetic data for generation {args.generation + 1} ready!")
        print(f"Next step: Train generation {args.generation + 1} using:")
        print(f"python scripts/train_recursive.py --generation {args.generation + 1}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()