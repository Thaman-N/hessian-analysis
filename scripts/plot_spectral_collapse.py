import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "../results")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "summary")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define Experiments
experiments = {
    "Control A (Fresh Human)": {
        "color": "green", "marker": "o",
        "folders": {
            0: "smollmgeneration_0", 1: "control_generation_1", 2: "control_generation_2",
            3: "control_generation_3", 4: "control_generation_4", 5: "control_generation_5"
        }
    },
    "Control B (Static Human)": {
        "color": "blue", "marker": "o",
        "folders": {
            0: "smollmgeneration_0", 1: "control_b_gen_1", 2: "control_b_gen_2",
            3: "control_b_gen_3", 4: "control_b_gen_4", 5: "control_b_gen_5"
        }
    },
    "Control C (Qwen - Recursive)": {
        "color": "purple", "marker": "D", "linewidth": 2.5,
        "folders": {
            0: "control_c_gen_0", 1: "control_c_gen_1", 2: "control_c_gen_2",
            3: "control_c_gen_3", 4: "control_c_gen_4", 5: "control_c_gen_5"
        }
    }
}

def load_metrics():
    data = []
    print("📊 Loading Spectral Metrics...")
    
    for label, config in experiments.items():
        for gen, folder in config["folders"].items():
            file_path = os.path.join(RESULTS_DIR, folder, "hessian_stats.json")
            
            if not os.path.exists(file_path):
                print(f"   ⚠️ Missing: {folder}")
                continue
                
            try:
                with open(file_path, "r") as f:
                    stats = json.load(f)
                
                data.append({
                    "Experiment": label,
                    "Generation": gen,
                    "Spectral Ratio": stats.get("spectral_ratio", 0),
                    "Bulk Width": stats.get("bulk_width_iqr", 0)
                })
            except Exception as e:
                print(f"   ❌ Error reading {folder}: {e}")

    return pd.DataFrame(data)

def plot_comparison(df):
    sns.set_theme(style="whitegrid")
    
    # Create a figure with 2 subplots side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- PLOT 1: SPECTRAL RATIO ---
    sns.lineplot(
        data=df, x="Generation", y="Spectral Ratio",
        hue="Experiment", style="Experiment", markers=True, dashes=False,
        palette={k: v["color"] for k, v in experiments.items()},
        linewidth=2.5, markersize=8, ax=axes[0]
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Spectral Ratio (Higher is Better/Stabler)", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Ratio (λ_max / Bulk Width)", fontsize=12)
    axes[0].legend(title=None)
    
    # --- PLOT 2: BULK WIDTH (IQR) ---
    sns.lineplot(
        data=df, x="Generation", y="Bulk Width",
        hue="Experiment", style="Experiment", markers=True, dashes=False,
        palette={k: v["color"] for k, v in experiments.items()},
        linewidth=2.5, markersize=8, ax=axes[1]
    )
    # Use Log scale here too because Qwen's width might explode
    axes[1].set_yscale("log") 
    axes[1].set_title("Bulk Width (Lower is Sharper)", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("IQR of Eigenvalues (Log Scale)", fontsize=12)
    axes[1].legend().remove() # Remove legend from 2nd plot to save space

    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "final_spectral_collapse_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"\n✅ Plot Saved: {output_path}")

if __name__ == "__main__":
    df = load_metrics()
    if not df.empty:
        plot_comparison(df)