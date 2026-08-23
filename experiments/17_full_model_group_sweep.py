from pathlib import Path
import re
import time

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

TARGET_PATTERN = re.compile(
    r"model\.layers\.(\d+)\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(gate_proj|up_proj|down_proj))\.weight"
)

GROUP_SIZES = [512, 256, 128, 64, 32, 16]

RESULTS_DIR = Path("results")
OUTPUT_PATH = RESULTS_DIR / "tinyllama_group_size_sweep_all_matrices.csv"


def get_matrix_type(tensor_key):
    for matrix_type in [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]:
        if matrix_type in tensor_key:
            return matrix_type

    return "unknown"


def global_int4(weights):
    max_abs = weights.abs().max()
    scale = max_abs / 7

    if scale.item() == 0:
        scale = torch.tensor(
            1.0,
            dtype=weights.dtype,
        )

    quantized = torch.round(weights / scale)
    quantized = torch.clamp(quantized, -7, 7)

    reconstructed = quantized * scale

    mse = torch.mean(
        (weights - reconstructed).square()
    ).item()

    zeroed = (
        (weights != 0) & (quantized == 0)
    ).sum().item()

    return {
        "mse": mse,
        "zeroed_percent":
            100.0 * zeroed / weights.numel(),
        "scale": scale.item(),
        "number_of_scales": 1,
    }


def grouped_int4(weights, group_size):
    rows, columns = weights.shape

    # skip group sizes that do not divide the row width cleanly
    if columns % group_size != 0:
        return None

    groups_per_row = columns // group_size

    grouped = weights.reshape(
        rows,
        groups_per_row,
        group_size,
    )

    max_abs = grouped.abs().amax(
        dim=2,
        keepdim=True,
    )

    scales = max_abs / 7

    safe_scales = torch.where(
        scales == 0,
        torch.ones_like(scales),
        scales,
    )

    quantized = torch.round(
        grouped / safe_scales
    )

    quantized = torch.clamp(
        quantized,
        -7,
        7,
    )

    reconstructed = (
        quantized * safe_scales
    ).reshape_as(weights)

    mse = torch.mean(
        (weights - reconstructed).square()
    ).item()

    zeroed = (
        (grouped != 0) & (quantized == 0)
    ).sum().item()

    return {
        "mse": mse,
        "zeroed_percent":
            100.0 * zeroed / weights.numel(),
        "average_scale":
            scales.mean().item(),
        "maximum_scale":
            scales.max().item(),
        "number_of_scales":
            rows * groups_per_row,
    }


def locate_model():
    return Path(
        snapshot_download(
            repo_id=MODEL_ID,
            allow_patterns=[
                "*.safetensors",
                "*.json",
            ],
        )
    )


def find_matrices(model_directory):
    matches = []

    for tensor_file in model_directory.glob(
        "*.safetensors"
    ):
        with safe_open(
            tensor_file,
            framework="pt",
            device="cpu",
        ) as reader:

            for key in reader.keys():
                match = TARGET_PATTERN.fullmatch(key)

                if match:
                    layer = int(
                        match.group(1)
                    )

                    matrix_type = get_matrix_type(
                        key
                    )

                    matches.append(
                        (
                            layer,
                            matrix_type,
                            tensor_file,
                            key,
                        )
                    )

    matches.sort(
        key=lambda x: (
            x[0],
            x[1],
        )
    )

    return matches


def main():
    RESULTS_DIR.mkdir(
        exist_ok=True
    )

    model_directory = locate_model()

    targets = find_matrices(
        model_directory
    )

    print()
    print("FULL-MODEL GROUP SIZE SWEEP")
    print("=" * 80)
    print(
        f"Matrices found: {len(targets)}"
    )
    print(
        f"Requested group sizes: "
        f"{GROUP_SIZES}"
    )
    print()

    rows = []

    start_total = time.perf_counter()

    for position, (
        layer,
        matrix_type,
        tensor_file,
        tensor_key,
    ) in enumerate(
        targets,
        start=1,
    ):

        print(
            f"[{position:03d}/{len(targets):03d}] "
            f"Layer {layer:02d} "
            f"{matrix_type:9s}",
            end=" ",
            flush=True,
        )

        with safe_open(
            tensor_file,
            framework="pt",
            device="cpu",
        ) as reader:
            weights = (
                reader
                .get_tensor(tensor_key)
                .float()
            )

        # true global INT4 baseline
        baseline = global_int4(
            weights
        )

        rows.append({
            "layer": layer,
            "matrix_type": matrix_type,
            "tensor_key": tensor_key,
            "method": "global",
            "group_size": weights.numel(),
            "row_width": weights.shape[1],
            "num_weights": weights.numel(),
            "number_of_scales":
                baseline["number_of_scales"],
            "mse":
                baseline["mse"],
            "zeroed_percent":
                baseline["zeroed_percent"],
            "average_scale":
                baseline["scale"],
            "maximum_scale":
                baseline["scale"],
            "mse_improvement_vs_global":
                1.0,
            "zeroed_point_reduction":
                0.0,
        })

        valid_count = 0

        for group_size in GROUP_SIZES:
            result = grouped_int4(
                weights,
                group_size,
            )

            if result is None:
                continue

            valid_count += 1

            rows.append({
                "layer": layer,
                "matrix_type": matrix_type,
                "tensor_key": tensor_key,
                "method": "grouped",
                "group_size": group_size,
                "row_width": weights.shape[1],
                "num_weights": weights.numel(),
                "number_of_scales":
                    result["number_of_scales"],
                "mse":
                    result["mse"],
                "zeroed_percent":
                    result["zeroed_percent"],
                "average_scale":
                    result["average_scale"],
                "maximum_scale":
                    result["maximum_scale"],
                "mse_improvement_vs_global":
                    baseline["mse"]
                    / result["mse"],
                "zeroed_point_reduction":
                    baseline["zeroed_percent"]
                    - result["zeroed_percent"],
            })

        print(
            f"| width={weights.shape[1]} "
            f"| valid groups={valid_count}"
        )

        del weights

    df = pd.DataFrame(rows)

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    elapsed = (
        time.perf_counter()
        - start_total
    )

    print()
    print("=" * 80)
    print("SWEEP COMPLETE")
    print("=" * 80)
    print(
        f"Rows saved: {len(df)}"
    )
    print(
        f"Runtime: {elapsed:.2f}s"
    )
    print(
        f"Saved to: {OUTPUT_PATH}"
    )

    grouped_only = df[
        df["method"] == "grouped"
    ]

    summary = (
        grouped_only
        .groupby("group_size")
        .agg(
            matrices=(
                "matrix_type",
                "size",
            ),
            mean_mse=(
                "mse",
                "mean",
            ),
            mean_zeroed_percent=(
                "zeroed_percent",
                "mean",
            ),
            mean_mse_improvement=(
                "mse_improvement_vs_global",
                "mean",
            ),
            mean_zeroed_point_reduction=(
                "zeroed_point_reduction",
                "mean",
            ),
            mean_number_of_scales=(
                "number_of_scales",
                "mean",
            ),
        )
        .sort_index(
            ascending=False
        )
    )

    print()
    print("GROUP-SIZE SUMMARY")
    print("-" * 80)
    print(
        summary.to_string()
    )


if __name__ == "__main__":
    main()