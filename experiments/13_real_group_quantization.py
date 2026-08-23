from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TARGET_SUFFIX = "layers.0.self_attn.q_proj.weight"

GROUP_SIZES = [
    2048,
    512,
    256,
    128,
    64,
    32,
    16,
    8,
]


def load_target_weights():
    model_directory = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            allow_patterns=["*.safetensors", "*.json"],
        )
    )

    for file in model_directory.glob("*.safetensors"):
        with safe_open(file, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if key.endswith(TARGET_SUFFIX):
                    return reader.get_tensor(key).float(), key

    raise KeyError(f"Could not find tensor ending in: {TARGET_SUFFIX}")


def grouped_symmetric_quantize(weights, group_size, qmax=7):
    rows, columns = weights.shape

    if columns % group_size != 0:
        raise ValueError(
            f"Group size {group_size} does not divide row length {columns}."
        )

    groups_per_row = columns // group_size

    grouped = weights.reshape(rows, groups_per_row, group_size)

    max_abs = grouped.abs().amax(dim=2, keepdim=True)
    scales = max_abs / qmax

    # Protect against a hypothetical all-zero group.
    safe_scales = torch.where(scales == 0, torch.ones_like(scales), scales)

    quantized = torch.round(grouped / safe_scales)
    quantized = torch.clamp(quantized, -qmax, qmax)

    reconstructed = quantized * safe_scales
    reconstructed = reconstructed.reshape_as(weights)

    error = weights - reconstructed
    mse = torch.mean(error ** 2)

    zeroed_values = ((weights != 0) & (reconstructed == 0)).sum().item()
    zeroed_percent = 100 * zeroed_values / weights.numel()

    return {
        "group_size": group_size,
        "groups_per_row": groups_per_row,
        "number_of_scales": rows * groups_per_row,
        "average_scale": scales.mean().item(),
        "maximum_scale": scales.max().item(),
        "mse": mse.item(),
        "zeroed_values": zeroed_values,
        "zeroed_percent": zeroed_percent,
    }


weights, tensor_key = load_target_weights()

print("REAL LAYER GROUP QUANTIZATION")
print("=" * 72)
print(f"Model: {MODEL_ID}")
print(f"Tensor: {tensor_key}")
print(f"Shape: {tuple(weights.shape)}")
print(f"Number of weights: {weights.numel():,}")
print("Quantization: symmetric INT4, row-wise groups")

results = []

for group_size in GROUP_SIZES:
    result = grouped_symmetric_quantize(
        weights=weights,
        group_size=group_size,
        qmax=7,
    )
    results.append(result)

baseline_mse = results[0]["mse"]
baseline_zeroed = results[0]["zeroed_percent"]

for result in results:
    result["mse_improvement_vs_2048"] = baseline_mse / result["mse"]
    result["zeroed_percentage_point_reduction"] = (
        baseline_zeroed - result["zeroed_percent"]
    )

df = pd.DataFrame(results)

print("\nRESULTS")
print("-" * 110)
print(
    f"{'Group':>8} "
    f"{'Scales':>10} "
    f"{'Avg Scale':>14} "
    f"{'MSE':>14} "
    f"{'Zeroed %':>12} "
    f"{'MSE Improve':>14}"
)
print("-" * 110)

for _, row in df.iterrows():
    print(
        f"{int(row['group_size']):>8} "
        f"{int(row['number_of_scales']):>10,} "
        f"{row['average_scale']:>14.8f} "
        f"{row['mse']:>14.8e} "
        f"{row['zeroed_percent']:>11.4f}% "
        f"{row['mse_improvement_vs_2048']:>13.2f}x"
    )

output_directory = Path("results")
output_directory.mkdir(exist_ok=True)

csv_path = output_directory / "real_group_quantization.csv"
df.to_csv(csv_path, index=False)

print(f"\nSaved results to: {csv_path}")