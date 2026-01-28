"""
Individual Generation Runner

Simple script to run a single generation of the recursive training pipeline.
Good for testing or running generations one at a time.

Usage:
    # Run Generation 1 (generate data from Gen 0, train Gen 1, analyze)
    python scripts/run_single_generation.py --generation 1

    # Just run Hessian analysis on existing model
    python scripts/run_single_generation.py --generation 1 --hessian_only

Requirements:
    - For generation N: model from generation N-1 must exist
    - All dependencies for training and Hessian analysis
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*50}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            cwd=project_root
        )
        
        print(f"✅ {description} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with return code: {e.returncode}")
        return False

def run_single_generation(generation, hessian_only=False):
    """Run a single generation of the pipeline."""
    
    print(f"🚀 RUNNING SINGLE GENERATION {generation}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if generation == 0:
        print("❌ Use scripts/train_generation.py for generation 0")
        return False
    
    if not hessian_only:
        # Check if previous generation model exists
        prev_model = f"models/generation_{generation-1}/model.pt"
        if not os.path.exists(prev_model):
            print(f"❌ Previous generation model not found: {prev_model}")
            return False
        
        print(f"✅ Previous generation model found: {prev_model}")
        
        # Step 1: Generate synthetic data
        print(f"\n📝 Step 1: Generate synthetic data for Generation {generation}")
        sync_cmd = f"python scripts/generate_synthetic_data.py --generation {generation-1}"
        if not run_command(sync_cmd, f"Generate synthetic data"):
            return False
        
        # Step 2: Train the model
        print(f"\n🏋️  Step 2: Train Generation {generation}")
        train_cmd = f"python scripts/train_recursive.py --generation {generation}"
        if not run_command(train_cmd, f"Train Generation {generation}"):
            return False
    
    # Step 3: Run Hessian analysis
    print(f"\n🔍 Step 3: Hessian Analysis for Generation {generation}")
    
    # Check if model exists
    model_path = f"models/generation_{generation}/model.pt"
    if not os.path.exists(model_path):
        print(f"❌ Generation {generation} model not found: {model_path}")
        return False
    
    # Run Hessian analysis using Python import (more reliable)
    hessian_cmd = f'python -c "import sys; sys.path.append(\\"{project_root}\\"); from scripts.hessian_analysis import analyze_hessian_spectrum; analyze_hessian_spectrum(generation={generation}, compute_density=True)"'
    if not run_command(hessian_cmd, f"Hessian Analysis Generation {generation}"):
        return False
    
    # Load and display results
    results_path = f"results/generation_{generation}/hessian_analysis.json"
    if os.path.exists(results_path):
        import json
        with open(results_path, "r") as f:
            results = json.load(f)
        
        print(f"\n📊 RESULTS FOR GENERATION {generation}:")
        print(f"   Spectral ratio: {results['spectral_ratio']:.3f}")
        print(f"   Max eigenvalue: {results['max_eigenvalue']:.3f}")
        print(f"   Bulk width: {results['bulk_width']:.3f}")
        print(f"   Analysis time: {results['computation_time_minutes']:.1f} minutes")
        print(f"   Results saved: {results_path}")
    
    print(f"\n🎉 Generation {generation} completed successfully!")
    
    # Suggest next steps
    if generation < 10:
        print(f"\nNext steps:")
        print(f"• Run Generation {generation + 1}: python scripts/run_single_generation.py --generation {generation + 1}")
        print(f"• Or run full experiment: python scripts/run_full_experiment.py --start_generation {generation + 1}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Run single generation of recursive training")
    parser.add_argument("--generation", type=int, required=True,
                       help="Generation number to run (must be >= 1)")
    parser.add_argument("--hessian_only", action="store_true",
                       help="Only run Hessian analysis (skip data generation and training)")
    
    args = parser.parse_args()
    
    if args.generation < 1:
        print("❌ Generation must be >= 1 for recursive training")
        print("Use scripts/train_generation.py for generation 0")
        sys.exit(1)
    
    try:
        success = run_single_generation(args.generation, args.hessian_only)
        if not success:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()