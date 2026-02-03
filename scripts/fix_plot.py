import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Path to your stats file
stats_path = "results/smollm_generation_0/hessian_stats.json"

def fix_results():
    print(f"🔧 Fixing {stats_path}...")
    
    with open(stats_path, "r") as f:
        data = json.load(f)

    # 1. Extract raw data
    x = np.array(data["plot_x"])
    y = np.array(data["plot_y"])
    top_eigs = np.array(data["top_eigenvalues"])

    # 2. SORT the data (This fixes the Negative IQR and the Scribble Plot)
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    # 3. Recalculate Metrics correctly
    cumulative = np.cumsum(y_sorted)
    cumulative = cumulative / cumulative[-1] # Normalize 0-1
    
    q1 = np.interp(0.25, cumulative, x_sorted)
    q3 = np.interp(0.75, cumulative, x_sorted)
    iqr = q3 - q1
    
    max_lambda = np.max(np.abs(top_eigs))
    ratio = max_lambda / iqr if iqr > 1e-6 else 0.0

    print(f"✅ CORRECTION APPLIED:")
    print(f"   Old IQR: {data['bulk_width_iqr']:.4f} -> New IQR: {iqr:.4f}")
    print(f"   Old Ratio: {data['spectral_ratio']:.4f} -> New Ratio: {ratio:.4f}")

    # 4. Save Fixed JSON
    data["plot_x"] = x_sorted.tolist()
    data["plot_y"] = y_sorted.tolist()
    data["bulk_width_iqr"] = float(iqr)
    data["spectral_ratio"] = float(ratio)
    
    with open(stats_path, "w") as f:
        json.dump(data, f, indent=4)
    print("   Saved fixed JSON.")

    # 5. Re-Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_sorted, y_sorted, color='green', linewidth=2, label="Spectral Density")
    plt.fill_between(x_sorted, y_sorted, color='green', alpha=0.3)
    
    # Add markers for Top Eigenvalues
    for eig in top_eigs:
        plt.axvline(x=eig, color='red', linestyle='--', alpha=0.5)
    
    plt.title(f"SmolLM-135M Gen 0 (Corrected)\nRatio: {ratio:.2f} | Max $\lambda$: {max_lambda:.0f}")
    plt.xlabel("Eigenvalue")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_img = "results/smollm_generation_0/spectrum_fixed.png"
    plt.savefig(output_img)
    print(f"   Saved fixed plot to {output_img}")

if __name__ == "__main__":
    fix_results()