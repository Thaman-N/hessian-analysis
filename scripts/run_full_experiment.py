"""
Full Recursive Training Experiment - Generations 0 through 10

This script orchestrates the complete "Curse of Recursion" experiment:
1. Uses the pre-trained Generation 0 model (baseline on human data)
2. For each generation 1-10:
   - Generate synthetic data from previous generation
   - Train new model on synthetic data  
   - Run Hessian spectral analysis
   - Track spectral collapse metrics

The goal is to detect "spectral collapse" (outlier eigenvalues merging into bulk)
before standard metrics like perplexity diverge, proving Hessian analysis is a 
leading indicator of recursive training degradation.

Usage:
    python scripts/run_full_experiment.py --start_generation 1 --end_generation 10

Requirements:
    - Pre-trained Generation 0 model in models/generation_0/
    - All dependencies for training and Hessian analysis
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            cwd=project_root
        )
        
        elapsed_time = time.time() - start_time
        print(f"✅ {description} completed in {elapsed_time/60:.1f} minutes")
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout[-1000:])  # Show last 1000 chars
            
        return True, elapsed_time
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"❌ {description} failed after {elapsed_time/60:.1f} minutes")
        print(f"Return code: {e.returncode}")
        
        if e.stdout:
            print("STDOUT:")
            print(e.stdout[-1000:])
            
        if e.stderr:
            print("STDERR:")
            print(e.stderr[-1000:])
            
        return False, elapsed_time

def load_hessian_results(generation):
    """Load Hessian analysis results for a generation."""
    results_path = f"results/generation_{generation}/hessian_analysis.json"
    
    if not os.path.exists(results_path):
        print(f"⚠️  Hessian results not found for generation {generation}: {results_path}")
        return None
        
    try:
        with open(results_path, "r") as f:
            results = json.load(f)
        return results
    except Exception as e:
        print(f"❌ Error loading Hessian results for generation {generation}: {e}")
        return None

def create_summary_plots(all_results, output_dir):
    """Create comprehensive summary plots of the experiment."""
    print("\n" + "="*60)
    print("CREATING SUMMARY PLOTS")
    print("="*60)
    
    # Extract metrics across generations
    generations = []
    spectral_ratios = []
    max_eigenvalues = []
    bulk_widths = []
    computation_times = []
    
    for gen, results in all_results.items():
        if results is not None:
            generations.append(gen)
            spectral_ratios.append(results.get('spectral_ratio', float('inf')))
            max_eigenvalues.append(results.get('max_eigenvalue', 0))
            bulk_widths.append(results.get('bulk_width', 0))
            computation_times.append(results.get('computation_time_minutes', 0))
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Spectral Ratio Evolution (Main Result)
    ax1.plot(generations, spectral_ratios, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Spectral Ratio (λ_max / σ_bulk)')
    ax1.set_title('Spectral Collapse Detection\n(Lower = More Collapsed)')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Highlight the collapse trend
    if len(generations) > 1:
        z = np.polyfit(generations, np.log(spectral_ratios), 1)
        p = np.poly1d(z)
        ax1.plot(generations, np.exp(p(generations)), 'r--', alpha=0.7, 
                label=f'Trend (slope: {z[0]:.3f})')
        ax1.legend()
    
    # Plot 2: Maximum Eigenvalue Evolution
    ax2.plot(generations, max_eigenvalues, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Maximum Eigenvalue (λ_max)')
    ax2.set_title('Top Eigenvalue Evolution')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Bulk Width Evolution
    ax3.plot(generations, bulk_widths, 'ro-', linewidth=2, markersize=8)
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Bulk Width (σ_bulk)')
    ax3.set_title('Bulk Distribution Width')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Computation Time per Generation
    ax4.bar(generations, computation_times, alpha=0.7, color='purple')
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Hessian Analysis Time (minutes)')
    ax4.set_title('Computational Cost per Generation')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the summary plot
    summary_plot_path = f"{output_dir}/spectral_collapse_summary.png"
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Summary plots saved to: {summary_plot_path}")
    
    plt.show()
    
    return summary_plot_path

def save_experiment_summary(all_results, timing_data, output_dir):
    """Save comprehensive experiment summary."""
    # Calculate summary statistics
    successful_generations = [gen for gen, results in all_results.items() if results is not None]
    
    if len(successful_generations) < 2:
        print("⚠️  Not enough successful generations for trend analysis")
        return
    
    # Extract spectral ratios for trend analysis
    spectral_ratios = [all_results[gen]['spectral_ratio'] for gen in successful_generations 
                      if all_results[gen]['spectral_ratio'] != float('inf')]
    
    # Calculate spectral collapse metrics
    initial_ratio = spectral_ratios[0] if spectral_ratios else None
    final_ratio = spectral_ratios[-1] if spectral_ratios else None
    collapse_factor = initial_ratio / final_ratio if (initial_ratio and final_ratio) else None
    
    # Create comprehensive summary
    summary = {
        "experiment_info": {
            "title": "Hessian Spectral Analysis - Curse of Recursion",
            "description": "Tracking spectral collapse across recursive training generations",
            "timestamp": datetime.now().isoformat(),
            "successful_generations": successful_generations,
            "total_generations_attempted": len(all_results)
        },
        "key_findings": {
            "initial_spectral_ratio": initial_ratio,
            "final_spectral_ratio": final_ratio,
            "spectral_collapse_factor": collapse_factor,
            "generations_analyzed": len(successful_generations)
        },
        "timing_summary": {
            "total_experiment_time_hours": sum(timing_data.values()) / 3600,
            "average_generation_time_minutes": np.mean(list(timing_data.values())) / 60,
            "time_breakdown": {k: v/60 for k, v in timing_data.items()}
        },
        "detailed_results": all_results,
        "methodology": {
            "model_parameters": "102.1M",
            "hessian_iterations": 30,
            "training_steps_per_generation": 1000,
            "synthetic_samples_per_generation": 100000
        }
    }
    
    # Save summary
    summary_path = f"{output_dir}/experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"✅ Experiment summary saved to: {summary_path}")
    
    # Print key findings
    print(f"\n🔍 KEY FINDINGS:")
    print(f"   Initial spectral ratio (Gen {successful_generations[0]}): {initial_ratio:.3f}")
    print(f"   Final spectral ratio (Gen {successful_generations[-1]}): {final_ratio:.3f}")
    if collapse_factor:
        print(f"   Spectral collapse factor: {collapse_factor:.2f}x")
    print(f"   Total experiment time: {sum(timing_data.values())/3600:.1f} hours")
    
    return summary_path

def run_full_experiment(start_generation=1, end_generation=10, 
                       num_samples=100000, num_training_steps=1000):
    """Run the complete recursive training experiment."""
    
    print("🚀" + "="*60 + "🚀")
    print("HESSIAN SPECTRAL ANALYSIS - CURSE OF RECURSION EXPERIMENT")
    print("🚀" + "="*60 + "🚀")
    print(f"Generations: {start_generation} → {end_generation}")
    print(f"Synthetic samples per generation: {num_samples:,}")
    print(f"Training steps per generation: {num_training_steps:,}")
    print(f"Starting time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Verify Generation 0 exists
    gen0_model = "models/generation_0/model.pt"
    if not os.path.exists(gen0_model):
        print(f"❌ Generation 0 model not found: {gen0_model}")
        print("Please run generation 0 training first!")
        return
    
    print(f"✅ Generation 0 model found: {gen0_model}")
    
    # Load Generation 0 Hessian results for baseline
    gen0_results = load_hessian_results(0)
    if gen0_results:
        print(f"✅ Generation 0 Hessian results loaded - Spectral ratio: {gen0_results['spectral_ratio']:.3f}")
    else:
        print("⚠️  Generation 0 Hessian results not found - will proceed anyway")
    
    # Track results and timing
    all_results = {0: gen0_results}
    timing_data = {}
    experiment_start_time = time.time()
    
    # Main experiment loop
    for generation in range(start_generation, end_generation + 1):
        print(f"\n🔄 PROCESSING GENERATION {generation}")
        print(f"📊 Progress: {generation - start_generation + 1}/{end_generation - start_generation + 1}")
        
        generation_start_time = time.time()
        
        # Step 1: Generate synthetic data
        synthetic_cmd = f"python scripts/generate_synthetic_data.py --generation {generation-1} --num_samples {num_samples}"
        success, sync_time = run_command(synthetic_cmd, f"Generate synthetic data for Gen {generation}")
        
        if not success:
            print(f"❌ Synthetic data generation failed for generation {generation}")
            all_results[generation] = None
            continue
        
        # Step 2: Train recursive model
        train_cmd = f"python scripts/train_recursive.py --generation {generation} --num_steps {num_training_steps}"
        success, train_time = run_command(train_cmd, f"Train Generation {generation}")
        
        if not success:
            print(f"❌ Training failed for generation {generation}")
            all_results[generation] = None
            continue
        
        # Step 3: Run Hessian analysis
        hessian_cmd = f"cd {project_root} && python -c \"from scripts.hessian_analysis import analyze_hessian_spectrum; analyze_hessian_spectrum(generation={generation}, compute_density=True)\""
        success, hessian_time = run_command(hessian_cmd, f"Hessian analysis Gen {generation}")
        
        if not success:
            print(f"❌ Hessian analysis failed for generation {generation}")
            all_results[generation] = None
            continue
        
        # Load and display results
        results = load_hessian_results(generation)
        all_results[generation] = results
        
        if results:
            print(f"✅ Generation {generation} completed successfully!")
            print(f"   Spectral ratio: {results['spectral_ratio']:.3f}")
            print(f"   Max eigenvalue: {results['max_eigenvalue']:.3f}")
            print(f"   Analysis time: {results['computation_time_minutes']:.1f} min")
        
        # Track timing
        generation_time = time.time() - generation_start_time
        timing_data[f"generation_{generation}"] = generation_time
        
        print(f"⏱️  Generation {generation} total time: {generation_time/60:.1f} minutes")
        
        # Show progress so far
        successful_gens = [g for g, r in all_results.items() if r is not None]
        if len(successful_gens) >= 2:
            ratios = [all_results[g]['spectral_ratio'] for g in successful_gens 
                     if all_results[g]['spectral_ratio'] != float('inf')]
            if len(ratios) >= 2:
                print(f"📈 Spectral collapse trend: {ratios[0]:.3f} → {ratios[-1]:.3f} "
                      f"({ratios[0]/ratios[-1]:.2f}x collapse)")
    
    # Experiment complete
    total_time = time.time() - experiment_start_time
    timing_data["total_experiment"] = total_time
    
    print(f"\n🏁 EXPERIMENT COMPLETED!")
    print(f"Total time: {total_time/3600:.2f} hours")
    
    # Create output directory
    output_dir = f"results/full_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate summary plots and analysis
    create_summary_plots(all_results, output_dir)
    summary_path = save_experiment_summary(all_results, timing_data, output_dir)
    
    print(f"\n📁 All results saved to: {output_dir}")
    print(f"📊 Summary: {summary_path}")
    
    return all_results, output_dir

def main():
    parser = argparse.ArgumentParser(description="Run full recursive training experiment")
    parser.add_argument("--start_generation", type=int, default=1,
                       help="Starting generation number (default: 1)")
    parser.add_argument("--end_generation", type=int, default=10,
                       help="Ending generation number (default: 10)")
    parser.add_argument("--num_samples", type=int, default=100000,
                       help="Synthetic samples per generation (default: 100000)")
    parser.add_argument("--num_training_steps", type=int, default=1000,
                       help="Training steps per generation (default: 1000)")
    
    args = parser.parse_args()
    
    if args.start_generation < 1:
        print("❌ Start generation must be >= 1")
        sys.exit(1)
        
    if args.end_generation < args.start_generation:
        print("❌ End generation must be >= start generation")
        sys.exit(1)
    
    try:
        results, output_dir = run_full_experiment(
            start_generation=args.start_generation,
            end_generation=args.end_generation,
            num_samples=args.num_samples,
            num_training_steps=args.num_training_steps
        )
        
        print(f"\n🎉 SUCCESS! Complete results available in: {output_dir}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()