import torch
import csv
import os

def quantize_global(weights, max_int):
    max_value = torch.max(torch.abs(weights))
    scale = max_value / max_int

    quantized = torch.round(weights / scale)
    reconstructed = quantized * scale

    error = weights - reconstructed
    mse = torch.mean(error ** 2)
    num_zeroed = torch.sum((weights != 0) & (quantized == 0))

    return mse, scale, num_zeroed


def quantize_grouped(weights, max_int, group_size):
    reconstructed_groups = []
    zeroed_total = 0
    scales = []

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

    return mse, torch.stack(scales), zeroed_total


tensor_size = 1000
outlier_value = 1000
max_int = 7  # INT4

torch.manual_seed(0)
weights = torch.randn(tensor_size)
weights[0] = outlier_value

global_mse, global_scale, global_zeroed = quantize_global(weights, max_int)

group_sizes = [500, 100, 50, 10]

# ensure existence of results folder
os.makedirs("results", exist_ok=True)

csv_path = "results/group_quantization.csv"

rows = []

# first add global result
rows.append({
    "method": "global",
    "group_size": tensor_size,
    "num_groups": 1,
    "mse": global_mse.item(),
    "zeroed_values": global_zeroed.item(),
    "average_scale": global_scale.item(),
    "first_group_scale": global_scale.item(),
    "mse_improvement_vs_global": 1.0
})

print("Group Quantization Experiment")
print("-----------------------------")
print(f"Tensor size: {tensor_size}")
print(f"Outlier value: {outlier_value}")
print(f"Precision: INT4")

print("\n===== Global Quantization =====")
print(f"Global scale: {global_scale:.6f}")
print(f"Global MSE: {global_mse:.6f}")
print(f"Global zeroed values: {global_zeroed}")

print("\n===== Group Quantization =====")

for group_size in group_sizes:
    grouped_mse, grouped_scales, grouped_zeroed = quantize_grouped(
        weights,
        max_int,
        group_size
    )

    improvement = global_mse / grouped_mse

    rows.append({
        "method": "grouped",
        "group_size": group_size,
        "num_groups": len(grouped_scales),
        "mse": grouped_mse.item(),
        "zeroed_values": grouped_zeroed.item(),
        "average_scale": torch.mean(grouped_scales).item(),
        "first_group_scale": grouped_scales[0].item(),
        "mse_improvement_vs_global": improvement.item()
    })

    print(f"\nGroup size: {group_size}")
    print(f"Number of groups: {len(grouped_scales)}")
    print(f"First group scale: {grouped_scales[0]:.6f}")
    print(f"Average scale: {torch.mean(grouped_scales):.6f}")
    print(f"Grouped MSE: {grouped_mse:.6f}")
    print(f"Grouped zeroed values: {grouped_zeroed}")
    print(f"MSE improvement vs global: {improvement:.2f}x")

# Save CSV
with open(csv_path, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved results to: {csv_path}")