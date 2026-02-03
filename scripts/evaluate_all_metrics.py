import os
import json
import pandas as pd
import glob

def collect_metrics():
    print("📊 Collecting Hessian Metrics from all experiments...")
    
    # Define the experiment mapping
    experiments = {
        "Treatment (Recursive)": "results/generation_*",
        "Control A (Fresh Human)": "results/control_generation_*",
        "Control B (Static Human)": "results/control_b_gen_*",
        "Control C (Qwen Recursive)": "results/control_c_gen_*"
    }
    
    records = []

    for label, pattern in experiments.items():
        # Find all folders matching the pattern
        folders = glob.glob(pattern)
        
        for folder in folders:
            # Extract generation number from folder name
            try:
                # Assuming format ends with _X
                gen_str = folder.split('_')[-1]
                gen = int(gen_str)
            except ValueError:
                continue
                
            json_path = os.path.join(folder, "hessian_stats.json")
            
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        
                    records.append({
                        "Experiment": label,
                        "Generation": gen,
                        "Spectral_Ratio": data.get("spectral_ratio", 0),
                        "Bulk_Width": data.get("bulk_width_iqr", 0),
                        "Top_Eigenvalue": data.get("top_eigenvalues", [0])[0] if data.get("top_eigenvalues") else 0
                    })
                except Exception as e:
                    print(f"⚠️ Error reading {json_path}: {e}")

    # Create DataFrame
    df = pd.DataFrame(records)
    
    if df.empty:
        print("❌ No results found! Did you run the analysis scripts?")
        return

    # Sort nicely
    df = df.sort_values(by=["Experiment", "Generation"])
    
    # Save to CSV
    os.makedirs("results/summary", exist_ok=True)
    csv_path = "results/summary/master_metrics.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\n✅ Success! Aggregated {len(df)} data points.")
    print(f"📄 Saved Master Table to: {csv_path}")
    print("\nPreview of Data:")
    print(df.pivot(index="Generation", columns="Experiment", values="Spectral_Ratio"))

if __name__ == "__main__":
    collect_metrics()