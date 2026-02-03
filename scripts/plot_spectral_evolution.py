import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def plot_evolution():
    # Generations to plot
    generations = [0, 1, 2, 3, 5]
    colors = ['green', 'blue', 'orange', 'red', 'purple']
    labels = ['Gen 0 (Baseline)', 'Gen 1 (Sharpening)', 'Gen 2 (Crash)', 'Gen 3 (Dead)', 'Gen 5 (Dissolved)']
    
    plt.figure(figsize=(12, 8))
    
    for gen, color, label in zip(generations, colors, labels):
        # Handle potential path variations
        path = f"results/generation_{gen}/hessian_stats.json"
        if not os.path.exists(path):
            path = f"results/smollm_generation_{gen}/hessian_stats.json"
            
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                
            # Get X and Y
            x = np.array(data["plot_x"])
            y = np.array(data["plot_y"])
            
            # Zoom in on the "Bulk" area (-10 to 10) to show the explosion
            mask = (x > -20) & (x < 20)
            
            ratio = data.get('spectral_ratio', 0)
            plt.plot(x[mask], y[mask], color=color, linewidth=2, alpha=0.8, 
                     label=f"{label} (Ratio: {ratio:.0f})")
            plt.fill_between(x[mask], y[mask], color=color, alpha=0.05)
        else:
            print(f"⚠️ Warning: Could not find results for Gen {gen}")

    plt.title("The Curse of Recursion: Evolution of Spectral Density", fontsize=16)
    plt.xlabel("Eigenvalue", fontsize=12)
    plt.ylabel("Density (Log Scale)", fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3, which="both")
    plt.legend(fontsize=11)
    
    output_dir = "results/summary"
    os.makedirs(output_dir, exist_ok=True)
    save_path = f"{output_dir}/spectral_evolution.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Comparison plot saved to {save_path}")

if __name__ == "__main__":
    plot_evolution()