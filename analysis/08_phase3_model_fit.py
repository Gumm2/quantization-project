from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fit_power_law(x, y):
    """
    Fit y = c * x^alpha by converting it into:
    log(y) = log(c) + alpha * log(x)
    """
    log_x = np.log(x)
    log_y = np.log(y)

    alpha, log_c = np.polyfit(log_x, log_y, 1)
    c = np.exp(log_c)

    predicted_y = c * (x ** alpha)
    predicted_log_y = log_c + alpha * log_x

    residual_sum = np.sum((log_y - predicted_log_y) ** 2)
    total_sum = np.sum((log_y - np.mean(log_y)) ** 2)
    r_squared = 1 - (residual_sum / total_sum)

    return alpha, c, predicted_y, r_squared


results_dir = Path("results")
summary_path = results_dir / "group_size_sweep_summary.csv"

df = pd.read_csv(summary_path)

#remove group size 1 we dont want each weight to have its own scale, we want to see the effect of shared scales
shared_df = df[df["group_size"] > 1].copy()

group_size = shared_df["group_size"].to_numpy(dtype=float)
average_scale = shared_df["mean_average_scale"].to_numpy(dtype=float)
mean_mse = shared_df["mean_mse"].to_numpy(dtype=float)

# fit power laws
group_alpha, group_c, group_prediction, group_r2 = fit_power_law(
    group_size,
    mean_mse,
)

scale_alpha, scale_c, scale_prediction, scale_r2 = fit_power_law(
    average_scale,
    mean_mse,
)

# pearson correlations
group_correlation = np.corrcoef(
    np.log(group_size),
    np.log(mean_mse),
)[0, 1]

scale_correlation = np.corrcoef(
    np.log(average_scale),
    np.log(mean_mse),
)[0, 1]

#figure 1 group sized model

plt.figure(figsize=(8, 5))

plt.scatter(
    group_size,
    mean_mse,
    label="Observed mean MSE",
)

plt.plot(
    group_size,
    group_prediction,
    label="Power-law fit",
)

plt.xscale("log", base=2)
plt.yscale("log")

plt.xlabel("Group Size")
plt.ylabel("Mean MSE")
plt.title("Power-Law Fit: Group Size vs Quantization Error")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(results_dir / "phase3_group_size_power_law.png")


# -----------------------------
# Figure 2: Scale model
# -----------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    average_scale,
    mean_mse,
    label="Observed mean MSE",
)

plt.plot(
    average_scale,
    scale_prediction,
    label="Power-law fit",
)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Average Scale")
plt.ylabel("Mean MSE")
plt.title("Power-Law Fit: Average Scale vs Quantization Error")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(results_dir / "phase3_average_scale_power_law.png")


# save conclusions

summary_text = f"""
PHASE 3 MODEL SUMMARY
=====================

Shared-scale observations: {len(shared_df)}

Group-size power law:
MSE ≈ {group_c:.6f} × group_size^{group_alpha:.6f}
Log-space R²: {group_r2:.6f}
Log-space correlation: {group_correlation:.6f}

Average-scale power law:
MSE ≈ {scale_c:.6f} × average_scale^{scale_alpha:.6f}
Log-space R²: {scale_r2:.6f}
Log-space correlation: {scale_correlation:.6f}

Important limitation:
Group size and average scale are strongly linked by this experimental
design. A larger R² alone cannot establish that either variable causes
MSE. Future experiments must vary tensor distributions and outlier
magnitudes independently.
""".strip()

summary_output_path = results_dir / "phase3_model_summary.txt"
summary_output_path.write_text(summary_text, encoding="utf-8")

print(summary_text)
print(f"\nSaved summary to: {summary_output_path}")
print("Saved both Phase 3 figures.")

plt.show()