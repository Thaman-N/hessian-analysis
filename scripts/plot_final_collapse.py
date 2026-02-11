import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

# --- PATH CONFIGURATION ---
# 1. Determine where this script is running from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Define the path to the 'results' folder relative to 'scripts'
#    (Go up one level "..", then down into "results")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "../results")

# 3. Define the Output Directory (results/summary)
OUTPUT_DIR = os.path.join(RESULTS_DIR, "summary")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"📍 Script Location: {SCRIPT_DIR}")
print(f"📂 Reading Data from: {RESULTS_DIR}")
print(f"💾 Saving Plot to:   {OUTPUT_DIR}")

# --- EXPERIMENT MAPPING ---
# keys = Legend Labels
# values = configuration
experiments = {
    "Control A (Fresh Human)": {
        "color": "green",
        "marker": "o",
        "folders": {
            0: "smollmgeneration_0",  # Shared Baseline (SmolLM)
            1: "control_generation_1",
            2: "control_generation_2",
            3: "control_generation_3",
            4: "control_generation_4",
            5: "control_generation_5"
        }
    },
    "Control B (Static Human)": {
        "color": "blue",
        "marker": "o",
        "folders": {
            0: "smollmgeneration_0",  # Shared Baseline (SmolLM)
            1: "control_b_gen_1",
            2: "control_b_gen_2",
            3: "control_b_gen_3",
            4: "control_b_gen_4",
            5: "control_b_gen_5"
        }
    },
    "Control C (Qwen - Recursive)": {
        "color": "purple",
        "marker": "D",
        "linewidth": 3,
        "folders": {
            0: "control_c_gen_0",     # Qwen Baseline
            1: "control_c_gen_1",
            2: "control_c_gen_2",
            3: "control_c_gen_3",
            4: "control_c_gen_4",
            5: "control_c_gen_5"
        }
    }
}

def load_data():
    data = []
    
    for label, config in experiments.items():
        print(f"   Processing: {label}")
        for gen, folder_name in config["folders"].items():
            # Construct full path to the JSON
            file_path = os.path.join(RESULTS_DIR, folder_name, "hessian_stats.json")
            
            if not os.path.exists(file_path):
                print(f"      ⚠️  Missing: {folder_name}/hessian_stats.json")
                continue
                
            try:
                with open(file_path, "r") as f:
                    stats = json.load(f)
                
                # Absolute value of Top Eigenvalue (Magnitude of Curvature)
                max_eig = abs(stats["top_eigenvalues"][0])
                
                data.append({
                    "Experiment": label,
                    "Generation": gen,
                    "Max Eigenvalue": max_eig
                })
            except Exception as e:
                print(f"      ❌ Error reading {folder_name}: {e}")

    return pd.DataFrame(data)

def plot_results(df):
    # Set professional style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))
    
    # Create the plot
    sns.lineplot(
        data=df,
        x="Generation",
        y="Max Eigenvalue",
        hue="Experiment",
        style="Experiment",
        markers=True,
        dashes=False,
        palette={k: v["color"] for k, v in experiments.items()},
        linewidth=2.5,
        markersize=9
    )

    # Log scale is mandatory for this specific dataset
    plt.yscale("log")
    
    plt.title("The Topological Collapse: Optimization Instability Evolution", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("Curvature |λ_max| (Log Scale)", fontsize=14)
    
    plt.legend(title="Condition", fontsize=12, title_fontsize=12, loc="upper left", frameon=True)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Save to RESULTS/SUMMARY
    output_file = os.path.join(OUTPUT_DIR, "final_collapse_plot.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"\n✅ Plot Saved Successfully: {output_file}")
    # plt.show() # Commented out for headless servers

if __name__ == "__main__":
    df = load_data()
    if not df.empty:
        plot_results(df)
    else:
        print("❌ No data found! Check directory names in the script match your folder structure.")