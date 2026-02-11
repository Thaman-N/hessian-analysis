import json
import os
import pandas as pd

# CONFIGURATION
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "../results")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "summary", "final_paper_table.csv")

experiments = {
    "Control A (Fresh)":  {0: "smollmgeneration_0", 1: "control_generation_1", 2: "control_generation_2", 3: "control_generation_3", 4: "control_generation_4", 5: "control_generation_5"},
    "Control B (Static)": {0: "smollmgeneration_0", 1: "control_b_gen_1", 2: "control_b_gen_2", 3: "control_b_gen_3", 4: "control_b_gen_4", 5: "control_b_gen_5"},
    "Control C (Qwen)":   {0: "control_c_gen_0", 1: "control_c_gen_1", 2: "control_c_gen_2", 3: "control_c_gen_3", 4: "control_c_gen_4", 5: "control_c_gen_5"}
}

data = []
for exp_name, folders in experiments.items():
    for gen, folder in folders.items():
        path = os.path.join(RESULTS_DIR, folder, "hessian_stats.json")
        if os.path.exists(path):
            with open(path) as f:
                stats = json.load(f)
            data.append({
                "Experiment": exp_name,
                "Generation": gen,
                "Top Eigenvalue": abs(stats["top_eigenvalues"][0]), # Absolute value for magnitude
                "Spectral Ratio": stats["spectral_ratio"],
                "Bulk Width": stats["bulk_width_iqr"]
            })

df = pd.DataFrame(data)
df.sort_values(by=["Experiment", "Generation"], inplace=True)
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Final Table Saved: {OUTPUT_FILE}")
print(df)