from pathlib import Path
import re
import time

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# grabbing the main learned linear matrices inside each transformer block
# this should give us attention + MLP instead of only q_proj like last time
TARGET_PATTERN = re.compile(
    r"model\.layers\.(\d+)\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(gate_proj|up_proj|down_proj))\.weight"
)

GROUP_SIZE = 64

RESULTS_DIR = Path("results")
OUTPUT_PATH = RESULTS_DIR / "tinyllama_all_linear_layers.csv"


def global_symmetric_quantize(weights, qmax):
    max_abs = weights.abs().max()
    scale = max_abs / qmax

    if scale.item() == 0:
        scale = torch.tensor(1.0, dtype=weights.dtype)

    quantized = torch.round(weights / scale)
    quantized = torch.clamp(quantized, -qmax, qmax)

    reconstructed = quantized * scale
    error = weights - reconstructed

    zeroed_values = (
        (weights != 0) & (quantized == 0)
    ).sum().item()

    return {
        "scale": scale.item(),
        "mse": torch.mean(error.square()).item(),
        "zeroed_values": zeroed_values,
        "zeroed_percent": 100.0 * zeroed_values / weights.numel(),
    }


def grouped_symmetric_int4(weights, group_size):
    if weights.ndim != 2:
        raise ValueError(
            f"Expected matrix but got {tuple(weights.shape)}"
        )

    rows, columns = weights.shape

    if columns % group_size != 0:
        raise ValueError(
            f"Group size {group_size} does not divide {columns}"
        )

    groups_per_row = columns // group_size

    # splitting every row into local neighborhoods so one extreme maximum
    # does not get to dictate the scale for everything else
    grouped = weights.reshape(
        rows,
        groups_per_row,
        group_size,
    )

    max_abs = grouped.abs().amax(dim=2, keepdim=True)
    scales = max_abs / 7

    safe_scales = torch.where(
        scales == 0,
        torch.ones_like(scales),
        scales,
    )

    quantized = torch.round(grouped / safe_scales)
    quantized = torch.clamp(quantized, -7, 7)

    reconstructed = quantized * safe_scales
    reconstructed = reconstructed.reshape_as(weights)

    error = weights - reconstructed

    original_grouped = weights.reshape(
        rows,
        groups_per_row,
        group_size,
    )

    zeroed_values = (
        (original_grouped != 0) & (quantized == 0)
    ).sum().item()

    return {
        "number_of_scales": rows * groups_per_row,
        "average_scale": scales.mean().item(),
        "maximum_scale": scales.max().item(),
        "mse": torch.mean(error.square()).item(),
        "zeroed_values": zeroed_values,
        "zeroed_percent": 100.0 * zeroed_values / weights.numel(),
    }


def calculate_statistics(weights):
    flat = weights.reshape(-1)

    mean = flat.mean()
    centered = flat - mean

    variance = centered.square().mean()
    std = torch.sqrt(variance)

    if std.item() == 0:
        skewness = 0.0
        pearson_kurtosis = 0.0
        excess_kurtosis = 0.0
    else:
        standardized = centered / std

        skewness = standardized.pow(3).mean().item()
        pearson_kurtosis = standardized.pow(4).mean().item()
        excess_kurtosis = pearson_kurtosis - 3.0

    abs_flat = flat.abs()
    max_abs = abs_flat.max()

    max_to_std = (
        max_abs.item() / std.item()
        if std.item() != 0
        else float("inf")
    )

    count_5 = (abs_flat > 5 * std).sum().item()
    count_10 = (abs_flat > 10 * std).sum().item()
    count_20 = (abs_flat > 20 * std).sum().item()

    # percentile information gives us something less fragile than only
    # staring at the single largest value
    quantiles = torch.quantile(
        abs_flat,
        torch.tensor(
            [0.99, 0.999, 0.9999],
            dtype=abs_flat.dtype,
        ),
    )

    return {
        "rows": weights.shape[0],
        "columns": weights.shape[1],
        "num_weights": flat.numel(),

        "minimum": flat.min().item(),
        "maximum": flat.max().item(),
        "mean": mean.item(),
        "std": std.item(),

        "max_abs": max_abs.item(),
        "max_to_std": max_to_std,

        "skewness": skewness,
        "pearson_kurtosis": pearson_kurtosis,
        "excess_kurtosis": excess_kurtosis,

        "abs_percentile_99": quantiles[0].item(),
        "abs_percentile_99_9": quantiles[1].item(),
        "abs_percentile_99_99": quantiles[2].item(),

        "percent_above_5_std":
            100.0 * count_5 / flat.numel(),

        "percent_above_10_std":
            100.0 * count_10 / flat.numel(),

        "percent_above_20_std":
            100.0 * count_20 / flat.numel(),
    }


def get_matrix_type(tensor_key):
    # converting the giant checkpoint key into something easy to group by later
    if "self_attn.q_proj" in tensor_key:
        return "q_proj"

    if "self_attn.k_proj" in tensor_key:
        return "k_proj"

    if "self_attn.v_proj" in tensor_key:
        return "v_proj"

    if "self_attn.o_proj" in tensor_key:
        return "o_proj"

    if "mlp.gate_proj" in tensor_key:
        return "gate_proj"

    if "mlp.up_proj" in tensor_key:
        return "up_proj"

    if "mlp.down_proj" in tensor_key:
        return "down_proj"

    return "unknown"


def locate_model():
    print("Locating TinyLlama checkpoint...")

    return Path(
        snapshot_download(
            repo_id=MODEL_ID,
            allow_patterns=[
                "*.safetensors",
                "*.json",
            ],
        )
    )


def find_target_matrices(model_directory):
    matches = []

    for tensor_file in sorted(
        model_directory.glob("*.safetensors")
    ):
        with safe_open(
            tensor_file,
            framework="pt",
            device="cpu",
        ) as reader:

            for key in reader.keys():
                match = TARGET_PATTERN.fullmatch(key)

                if match:
                    layer = int(match.group(1))

                    matches.append(
                        (
                            layer,
                            get_matrix_type(key),
                            tensor_file,
                            key,
                        )
                    )

    matches.sort(key=lambda x: (x[0], x[1]))

    return matches


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    model_directory = locate_model()
    targets = find_target_matrices(model_directory)

    print()
    print("FULL TINYLLAMA LINEAR-MATRIX ANALYSIS")
    print("=" * 80)
    print(f"Matrices found: {len(targets)}")
    print(f"Grouped INT4 size: {GROUP_SIZE}")
    print()

    results = []

    total_start = time.perf_counter()

    for position, (
        layer,
        matrix_type,
        tensor_file,
        tensor_key,
    ) in enumerate(targets, start=1):

        start = time.perf_counter()

        print(
            f"[{position:03d}/{len(targets):03d}] "
            f"Layer {layer:02d} {matrix_type:9s}...",
            end=" ",
            flush=True,
        )

        with safe_open(
            tensor_file,
            framework="pt",
            device="cpu",
        ) as reader:
            weights = reader.get_tensor(tensor_key).float()

        stats = calculate_statistics(weights)

        int8 = global_symmetric_quantize(
            weights,
            qmax=127,
        )

        int4 = global_symmetric_quantize(
            weights,
            qmax=7,
        )

        grouped = grouped_symmetric_int4(
            weights,
            group_size=GROUP_SIZE,
        )

        row = {
            "model": MODEL_ID,
            "layer": layer,
            "matrix_type": matrix_type,
            "tensor_key": tensor_key,

            **stats,

            "global_int8_scale": int8["scale"],
            "global_int8_mse": int8["mse"],
            "global_int8_zeroed_percent":
                int8["zeroed_percent"],

            "global_int4_scale": int4["scale"],
            "global_int4_mse": int4["mse"],
            "global_int4_zeroed_percent":
                int4["zeroed_percent"],

            "grouped_int4_group_size":
                GROUP_SIZE,

            "grouped_int4_number_of_scales":
                grouped["number_of_scales"],

            "grouped_int4_average_scale":
                grouped["average_scale"],

            "grouped_int4_maximum_scale":
                grouped["maximum_scale"],

            "grouped_int4_mse":
                grouped["mse"],

            "grouped_int4_zeroed_percent":
                grouped["zeroed_percent"],
        }

        row["int4_to_int8_mse_ratio"] = (
            int4["mse"] / int8["mse"]
        )

        row["grouped_vs_global_int4_mse_improvement"] = (
            int4["mse"] / grouped["mse"]
        )

        row["grouped_int4_zeroed_point_reduction"] = (
            int4["zeroed_percent"]
            - grouped["zeroed_percent"]
        )

        results.append(row)

        elapsed = time.perf_counter() - start

        print(
            f"{elapsed:.2f}s | "
            f"max/std={stats['max_to_std']:.1f} | "
            f"INT4 zero={int4['zeroed_percent']:.1f}% | "
            f"group64 zero={grouped['zeroed_percent']:.1f}%"
        )

        del weights

    df = pd.DataFrame(results)

    df = df.sort_values(
        ["layer", "matrix_type"]
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    total_time = time.perf_counter() - total_start

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print(f"Matrices analyzed: {len(df)}")

    print(
        f"Total learned weights analyzed: "
        f"{df['num_weights'].sum():,}"
    )

    print(
        f"Runtime: {total_time:.2f} seconds"
    )

    print(
        f"Saved dataset: {OUTPUT_PATH}"
    )

    print()
    print("MATRIX-TYPE SUMMARY")
    print("-" * 80)

    # this is our quick sanity check tonight, not the real statistical analysis yet
    summary = (
        df.groupby("matrix_type")
        .agg(
            matrices=("matrix_type", "size"),

            mean_max_to_std=(
                "max_to_std",
                "mean",
            ),

            mean_kurtosis=(
                "excess_kurtosis",
                "mean",
            ),

            mean_global_int4_zeroed=(
                "global_int4_zeroed_percent",
                "mean",
            ),

            mean_group64_zeroed=(
                "grouped_int4_zeroed_percent",
                "mean",
            ),

            mean_mse_improvement=(
                "grouped_vs_global_int4_mse_improvement",
                "mean",
            ),
        )
        .sort_index()
    )

    print(summary.to_string())

    print()
    print("WORST 10 MATRICES BY GLOBAL INT4 ZEROING")
    print("-" * 80)

    worst = (
        df.nlargest(
            10,
            "global_int4_zeroed_percent",
        )[
            [
                "layer",
                "matrix_type",
                "max_to_std",
                "excess_kurtosis",
                "global_int4_zeroed_percent",
                "grouped_int4_zeroed_percent",
                "grouped_vs_global_int4_mse_improvement",
            ]
        ]
    )

    print(worst.to_string(index=False))


if __name__ == "__main__":
    main()