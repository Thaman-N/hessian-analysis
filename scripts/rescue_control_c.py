import subprocess
import os
import torch
import gc

def rescue_qwen_analysis():
    print("🚑 Starting Control C Rescue Mission...")
    
    # We need to analyze Gen 1 to 5
    generations = [1, 2, 3, 4, 5]
    
    for gen in generations:
        model_path = f"models/control_c_gen_{gen}"
        output_dir = f"results/control_c_gen_{gen}"
        
        # Check if the model actually exists
        if not os.path.exists(model_path):
            print(f"❌ Missing Model: {model_path} (Did training fail?)")
            continue
            
        print(f"\n🔄 Rescuing Gen {gen}...")
        
        # 1. CLEAN MEMORY (Just in case)
        torch.cuda.empty_cache()
        gc.collect()
        
        # 2. Run Analysis (Standalone Process = Clean VRAM)
        try:
            subprocess.run([
                "python", "scripts/hessian_analysis_generic.py",
                "--generation", str(gen),
                "--model_path", model_path,
                "--output_dir", output_dir
            ], check=True)
            print(f"✅ Gen {gen} Rescued!")
            
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to analyze Gen {gen}: {e}")

if __name__ == "__main__":
    rescue_qwen_analysis()