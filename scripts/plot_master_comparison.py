import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_results():
    csv_path = "results/summary/master_metrics.csv"
    if not os.path.exists(csv_path):
        print("❌ Master CSV not found. Run evaluate_all_metrics.py first!")
        return

    df = pd.read_csv(csv_path)
    
    # Setup Style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    # Define Colors
    colors = {
        "Treatment (Recursive)": "#d62728",    # Red (Danger)
        "Control A (Fresh Human)": "#2ca02c",  # Green (Healthy)
        "Control B (Static Human)": "#1f77b4", # Blue (Stable)
        "Control C (Qwen Recursive)": "#9467bd" # Purple (Complex)
    }

    # === PLOT 1: SPECTRAL RATIO (The Collapse) ===
    plt.subplot(1, 2, 1)
    sns.lineplot(
        data=df, 
        x="Generation", 
        y="Spectral_Ratio", 
        hue="Experiment", 
        palette=colors,
        marker="o", 
        linewidth=2.5
    )
    plt.title("Spectral Ratio (Higher is Better)", fontsize=12, fontweight='bold')
    plt.ylabel("Ratio (λ_max / Bulk Width)")
    plt.xlabel("Generation")
    plt.yscale("log") # Log scale is crucial because healthy models are 100x higher
    plt.legend(title=None)

    # === PLOT 2: BULK WIDTH (The Chaos) ===
    plt.subplot(1, 2, 2)
    sns.lineplot(
        data=df, 
        x="Generation", 
        y="Bulk_Width", 
        hue="Experiment", 
        palette=colors,
        marker="s", 
        linewidth=2.5
    )
    plt.title("Bulk Width (Lower is Generally Sharper)", fontsize=12, fontweight='bold')
    plt.ylabel("IQR of Eigenvalues")
    plt.xlabel("Generation")
    plt.legend().remove() # Remove legend on second plot to save space

    # Save
    output_path = "results/summary/spectral_collapse_comparison.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"🖼️  Saved Plot to: {output_path}")

if __name__ == "__main__":
    plot_results()