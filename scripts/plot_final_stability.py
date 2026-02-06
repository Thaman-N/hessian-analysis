import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_stability():
    # Data manually extracted from your uploaded JSONs
    data = [
        # --- Control A (Fresh Human) - Healthy ---
        {"Experiment": "Control A (Fresh)", "Generation": 0, "Max_Eigenvalue": 928},
        {"Experiment": "Control A (Fresh)", "Generation": 1, "Max_Eigenvalue": 237},
        {"Experiment": "Control A (Fresh)", "Generation": 2, "Max_Eigenvalue": 287},
        {"Experiment": "Control A (Fresh)", "Generation": 3, "Max_Eigenvalue": 238},
        {"Experiment": "Control A (Fresh)", "Generation": 4, "Max_Eigenvalue": 250},
        {"Experiment": "Control A (Fresh)", "Generation": 5, "Max_Eigenvalue": 415},
        
        # --- Control B (Static) - Stagnant ---
        {"Experiment": "Control B (Static)", "Generation": 0, "Max_Eigenvalue": 928},
        {"Experiment": "Control B (Static)", "Generation": 1, "Max_Eigenvalue": 253},
        {"Experiment": "Control B (Static)", "Generation": 2, "Max_Eigenvalue": 264},
        {"Experiment": "Control B (Static)", "Generation": 3, "Max_Eigenvalue": 281},
        {"Experiment": "Control B (Static)", "Generation": 4, "Max_Eigenvalue": 301},
        {"Experiment": "Control B (Static)", "Generation": 5, "Max_Eigenvalue": 263},

        # --- Treatment (SmolLM Recursive) - The Collapse ---
        # Note: Gen 1 spiked to 600, Gen 5 crashed to 769. 
        # But for SmolLM we look at RATIO usually. Here we compare raw sharpness.
        {"Experiment": "Treatment (Recursive)", "Generation": 0, "Max_Eigenvalue": 928},
        {"Experiment": "Treatment (Recursive)", "Generation": 1, "Max_Eigenvalue": 606},
        {"Experiment": "Treatment (Recursive)", "Generation": 5, "Max_Eigenvalue": 769},

        # --- Control C (Qwen Recursive) - THE EXPLOSION ---
        # We take the absolute value of the max eigenvalue to show magnitude of instability
        {"Experiment": "Control C (Qwen)", "Generation": 1, "Max_Eigenvalue": 2007},
        {"Experiment": "Control C (Qwen)", "Generation": 2, "Max_Eigenvalue": 2609},
        {"Experiment": "Control C (Qwen)", "Generation": 3, "Max_Eigenvalue": 791}, 
        {"Experiment": "Control C (Qwen)", "Generation": 4, "Max_Eigenvalue": 1264},
        {"Experiment": "Control C (Qwen)", "Generation": 5, "Max_Eigenvalue": 6439138} # <--- !!!
    ]

    df = pd.DataFrame(data)

    # Setup Plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    colors = {
        "Control A (Fresh)": "green",
        "Control B (Static)": "blue",
        "Treatment (Recursive)": "red",
        "Control C (Qwen)": "purple"
    }

    sns.lineplot(
        data=df,
        x="Generation",
        y="Max_Eigenvalue",
        hue="Experiment",
        palette=colors,
        marker="o",
        linewidth=2.5
    )

    plt.yscale("log") # CRITICAL
    plt.title("Optimization Instability (Max Eigenvalue)", fontsize=14, fontweight='bold')
    plt.ylabel("Curvature (Log Scale) | Higher = Unstable")
    plt.xlabel("Generation")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig("results/summary/final_stability_comparison.png", dpi=300)
    print("✅ Graph saved.")

if __name__ == "__main__":
    plot_stability()