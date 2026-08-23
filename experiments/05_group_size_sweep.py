import os

import pandas as pd
import torch


def quantize_grouped(weights, max_int, group_size):
    reconstructed_groups = []
    scales = []
    zeroed_total = 0

    for start in range(0, len(weights), group_size):
        group = weights[start:start + group_size]

        max_value = torch.max(torch.abs(group))
        scale = max_value / max_int

        quantized = torch.round(group / scale)
        reconstructed = quantized * scale

        reconstructed_groups.append(reconstructed)
        scales.append(scale)

        zeroed_total += torch.sum((group != 0) & (quantized == 0))

    reconstructed_all = torch.cat(reconstructed_groups)

    error = weights - reconstructed_all
    mse = torch.mean(error ** 2)

    return (
        mse.item(),
        torch.mean(torch.stack(scales)).item(),
        zeroed_total.item(),
        len(scales),
    )


tensor_size = 1000
outlier_value = 1000
max_int = 7  # signed INT4 range: -7 to 7
num_trials = 100

group_sizes = [
    1, 2, 4, 8, 10, 16, 20, 25, 32,
    40, 50, 64, 100, 125, 200, 250,
    500, 1000
]

all_rows = []

for trial in range(num_trials):
    # Reproducible, but different, tensor for every trial
    torch.manual_seed(trial)

    weights = torch.randn(tensor_size)
    weights[0] = outlier_value

    for group_size in group_sizes:
        mse, average_scale, zeroed_values, num_groups = quantize_grouped(
            weights=weights,
            max_int=max_int,
            group_size=group_size,
        )

        all_rows.append({
            "trial": trial,
            "group_size": group_size,
            "num_groups": num_groups,
            "mse": mse,
            "average_scale": average_scale,
            "zeroed_values": zeroed_values,
        })


raw_df = pd.DataFrame(all_rows)

summary_df = (
    raw_df
    .groupby("group_size", as_index=False)
    .agg(
        num_groups=("num_groups", "first"),
        mean_mse=("mse", "mean"),
        std_mse=("mse", "std"),
        mean_average_scale=("average_scale", "mean"),
        std_average_scale=("average_scale", "std"),
        mean_zeroed_values=("zeroed_values", "mean"),
        std_zeroed_values=("zeroed_values", "std"),
    )
    .sort_values("group_size")
)

os.makedirs("results", exist_ok=True)

raw_path = "results/group_size_sweep_raw.csv"
summary_path = "results/group_size_sweep_summary.csv"

raw_df.to_csv(raw_path, index=False)
summary_df.to_csv(summary_path, index=False)

print("Group Size Sweep Complete")
print("-------------------------")
print(f"Trials: {num_trials}")
print(f"Tensor size: {tensor_size}")
print(f"Outlier value: {outlier_value}")
print(f"Group sizes tested: {len(group_sizes)}")

print("\nSummary:")
print(summary_df.to_string(index=False))

print(f"\nSaved raw data to: {raw_path}")
print(f"Saved summary data to: {summary_path}")