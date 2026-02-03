import os
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIG ---
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"

def create_gen0():
    print("🚀 Creating Gen 0 Baseline (Download & Analyze)...")
    
    # 1. Download & Save Base Model as "Gen 0"
    os.makedirs("models/generation_0", exist_ok=True)
    
    print(f"   Downloading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    model.save_pretrained("models/generation_0")
    tokenizer.save_pretrained("models/generation_0")
    
    # 2. Run Analysis on it
    print("   Running Hessian Analysis on Gen 0...")
    subprocess.run([
        "python", "scripts/hessian_analysis_generic.py",
        "--generation", "0",
        "--model_path", "models/generation_0",
        "--output_dir", "results/generation_0"
    ], check=True)
    
    print("✅ Gen 0 Baseline Created!")

if __name__ == "__main__":
    create_gen0()