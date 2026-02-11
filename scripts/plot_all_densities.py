import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "../results")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "summary")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define all 3 experiments
experiments = {
    "Control_A_Fresh": {
        "title": "Control A: Stable Evolution (Fresh Human Data)",
        "folders": {
            0: "smollmgeneration_0", 1: "control_generation_1", 
            2: "control_generation_2", 3: "control_generation_3",
            4: "control_generation_4", 5: "control_generation_5"
        }
    },
    "Control_B_Static": {
        "title": "Control B: The Stagnation (Static Data)",
        "folders": {
            0: "smollmgeneration_0", 1: "control_b_gen_1",
            2: "control_b_gen_2", 3: "control_b_gen_3",
            4: "control_b_gen_4", 5: "control_b_gen_5"
        }
    },
    "Control_C_Qwen": {
        "title": "Control C: The Topological Collapse (Qwen Recursive)",
        "folders": {
            0: "control_c_gen_0", 1: "control_c_gen_1",
            2: "control_c_gen_2", 3: "control_c_gen_3",
            4: "control_c_gen_4", 5: "control_c_gen_5"
        }
    }
}

# Consistent coloring across all plots
colors = {
    0: "#2ca02c",  # Green (Baseline)
    1: "#1f77b4",  # Blue
    2: "#ff7f0e",  # Orange
    3: "#d62728",  # Red
    4: "#9467bd",  # Purple
    5: "#8c564b"   # Brown/Dark Red
}

def plot_density_for_experiment(exp_key, config):
    print(f"🎨 Plotting {exp_key}...")
    
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    
    found_data = False
    
    for gen, folder in config["folders"].items():
        file_path = os.path.join(RESULTS_DIR, folder, "hessian_stats.json")
        
        if not os.path.exists(file_path):
            print(f"   ⚠️ Skipping Gen {gen} (File not found: {folder})")
            continue
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            x = np.array(data['plot_x'])
            y = np.array(data['plot_y'])
            
            # Normalize Y (Density) so we can compare shapes
            if np.max(y) > 0:
                y = y / np.max(y)
            
            label = f"Gen {gen}"
            if gen == 0: label += " (Baseline)"
            
            # Plot line and fill
            plt.plot(x, y, label=label, color=colors[gen], linewidth=2.5, alpha=0.9)
            plt.fill_between(x, y, color=colors[gen], alpha=0.05)
            found_data = True
            
        except Exception as e:
            print(f"   ❌ Error processing Gen {gen}: {e}")

    if not found_data:
        print(f"   ❌ No data found for {exp_key}, skipping plot.")
        plt.close()
        return

    # Use SymLog because Qwen has massive negative eigenvalues
    # Linthresh=10 means anything between -10 and 10 is linear, outside is log
    plt.xscale('symlog', linthresh=10)
    
    plt.title(config["title"], fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Eigenvalue (Curvature) | SymLog Scale", fontsize=14)
    plt.ylabel("Normalized Density", fontsize=14)
    plt.legend(fontsize=12, loc="upper right")
    plt.grid(True, which="both", alpha=0.2)
    
    # Save
    filename = f"spectral_density_{exp_key.lower()}.png"
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    for key, config in experiments.items():
        plot_density_for_experiment(key, config)