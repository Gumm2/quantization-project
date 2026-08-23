from pathlib import Path
import re
import time

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# only grabbing q_proj for now because otherwise we would be comparing
# a bunch of matrices that do different things and have different shapes
TARGET_PATTERN = re.compile(
    r"model\.layers\.(\d+)\.self_attn\.q_proj\.weight"
)

# 64 seemed like a reasonable practical group size from the earlier experiment
# small enough to protect weights but not so small that we need a scale for everything
GROUP_SIZE = 64

RESULTS_DIR = Path("results")
OUTPUT_PATH = RESULTS_DIR / "tinyllama_qproj_all_layers.csv"


def global_symmetric_quantize(
    weights: torch.Tensor,
    qmax: int,
) -> dict:
    """
    One scale for the entire matrix.

    This is basically the most aggressive/simple version and should be
    especially bad if there is one insane outlier controlling the scale.
    """

    max_abs = weights.abs().max()
    scale = max_abs / qmax

    # there probably will not be a completely zero matrix, but avoiding
    # division by zero here anyway so the script does not randomly die
    if scale.item() == 0:
        scale = torch.tensor(
            1.0,
            dtype=weights.dtype,
            device=weights.device,
        )

    quantized = torch.round(weights / scale)
    quantized = torch.clamp(quantized, -qmax, qmax)

    reconstructed = quantized * scale
    error = weights - reconstructed

    # only counting values that were originally nonzero and got destroyed
    zeroed_values = (
        (weights != 0) & (quantized == 0)
    ).sum().item()

    return {
        "scale": scale.item(),
        "mse": torch.mean(error.square()).item(),
        "zeroed_values": zeroed_values,
        "zeroed_percent": (
            100.0 * zeroed_values / weights.numel()
        ),
    }


def grouped_symmetric_int4(
    weights: torch.Tensor,
    group_size: int,
) -> dict:
    """
    Split each row into smaller groups and let every group use its own scale.

    The main idea is that one ridiculous value should only ruin its own
    neighborhood instead of poisoning the entire matrix.
    """

    if weights.ndim != 2:
        raise ValueError(
            f"Expected a matrix, received shape {tuple(weights.shape)}."
        )

    rows, columns = weights.shape

    # reshape only works cleanly if the group size divides the row length
    if columns % group_size != 0:
        raise ValueError(
            f"Group size {group_size} does not divide "
            f"the row length {columns}."
        )

    groups_per_row = columns // group_size

    # instead of thinking of this as one huge matrix, this temporarily
    # reorganizes it into rows x groups x weights inside each group
    grouped = weights.reshape(
        rows,
        groups_per_row,
        group_size,
    )

    # each individual group now gets its own local maximum and scale
    max_abs = grouped.abs().amax(
        dim=2,
        keepdim=True,
    )

    scales = max_abs / 7

    # same zero protection as before
    safe_scales = torch.where(
        scales == 0,
        torch.ones_like(scales),
        scales,
    )

    quantized = torch.round(grouped / safe_scales)
    quantized = torch.clamp(quantized, -7, 7)

    reconstructed = quantized * safe_scales

    # put everything back into the original matrix shape so MSE is easy
    # to calculate against the original weights
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
        "group_size": group_size,
        "groups_per_row": groups_per_row,
        "number_of_scales": rows * groups_per_row,
        "average_scale": scales.mean().item(),
        "maximum_scale": scales.max().item(),
        "mse": torch.mean(error.square()).item(),
        "zeroed_values": zeroed_values,
        "zeroed_percent": (
            100.0 * zeroed_values / weights.numel()
        ),
    }


def calculate_layer_statistics(
    weights: torch.Tensor,
) -> dict:
    """
    Get the basic shape of the distribution for each layer.

    Some of these might end up being useful predictors of quantization
    error and some might not, but we need the data first before guessing.
    """

    flat = weights.reshape(-1)

    mean = flat.mean()
    centered = flat - mean

    variance = centered.square().mean()
    std = torch.sqrt(variance)

    # skewness checks whether one side of the distribution has a longer tail
    # kurtosis is more about how extreme/heavy the tails are
    if std.item() == 0:
        skewness = 0.0
        pearson_kurtosis = 0.0
        excess_kurtosis = 0.0
    else:
        standardized = centered / std

        skewness = (
            standardized.pow(3).mean().item()
        )

        pearson_kurtosis = (
            standardized.pow(4).mean().item()
        )

        # normal distribution has Pearson kurtosis 3, so subtracting 3 makes
        # normal equal to zero and makes comparisons easier
        excess_kurtosis = pearson_kurtosis - 3.0

    abs_flat = flat.abs()
    max_abs = abs_flat.max()

    # this was one of the most useful numbers from layer 0 because it tells
    # us how absurd the maximum is compared to a typical weight
    max_to_std = (
        max_abs.item() / std.item()
        if std.item() != 0
        else float("inf")
    )

    count_above_5_std = (
        abs_flat > 5 * std
    ).sum().item()

    count_above_10_std = (
        abs_flat > 10 * std
    ).sum().item()

    count_above_20_std = (
        abs_flat > 20 * std
    ).sum().item()

    # these let us compare the maximum to the bulk of the extreme tail
    # instead of only comparing it to the standard deviation
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
        "count_above_5_std": count_above_5_std,
        "percent_above_5_std": (
            100.0 * count_above_5_std / flat.numel()
        ),
        "count_above_10_std": count_above_10_std,
        "percent_above_10_std": (
            100.0 * count_above_10_std / flat.numel()
        ),
        "count_above_20_std": count_above_20_std,
        "percent_above_20_std": (
            100.0 * count_above_20_std / flat.numel()
        ),
    }


def locate_model_file() -> Path:
    print("Locating TinyLlama checkpoint...")

    # should already be cached, but snapshot_download also makes sure we
    # are pointing at the correct local folder
    model_directory = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            allow_patterns=[
                "*.safetensors",
                "*.json",
            ],
        )
    )

    safetensor_files = sorted(
        model_directory.glob("*.safetensors")
    )

    if not safetensor_files:
        raise FileNotFoundError(
            f"No safetensors files found in {model_directory}."
        )

    return model_directory


def find_q_projection_tensors(
    model_directory: Path,
) -> list[tuple[int, Path, str]]:
    matched_tensors = []

    # search the checkpoint for every key that looks exactly like a q_proj
    # matrix and grab the layer number from the name
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
                    layer_number = int(match.group(1))

                    matched_tensors.append(
                        (
                            layer_number,
                            tensor_file,
                            key,
                        )
                    )

    matched_tensors.sort(key=lambda item: item[0])

    if not matched_tensors:
        raise KeyError(
            "No TinyLlama q_proj matrices were found."
        )

    return matched_tensors


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    model_directory = locate_model_file()

    targets = find_q_projection_tensors(
        model_directory
    )

    print()
    print("MODEL-WIDE Q_PROJ ANALYSIS")
    print("=" * 72)
    print(f"Model: {MODEL_ID}")
    print(f"Matrices found: {len(targets)}")
    print(f"Grouped INT4 size: {GROUP_SIZE}")
    print()

    all_rows = []
    start_time = time.perf_counter()

    # every loop is one full q_proj matrix from one transformer layer
    for position, (
        layer_number,
        tensor_file,
        tensor_key,
    ) in enumerate(targets, start=1):

        layer_start = time.perf_counter()

        print(
            f"[{position:02d}/{len(targets):02d}] "
            f"Analyzing layer {layer_number:02d}...",
            end=" ",
            flush=True,
        )

        # load only the one matrix we need instead of trying to load the
        # whole model into memory
        with safe_open(
            tensor_file,
            framework="pt",
            device="cpu",
        ) as reader:
            weights = reader.get_tensor(
                tensor_key
            ).float()

        statistics = calculate_layer_statistics(
            weights
        )

        # same layer, three different scaling strategies
        global_int8 = global_symmetric_quantize(
            weights,
            qmax=127,
        )

        global_int4 = global_symmetric_quantize(
            weights,
            qmax=7,
        )

        grouped_int4 = grouped_symmetric_int4(
            weights,
            group_size=GROUP_SIZE,
        )

        # one row in the final CSV represents one transformer layer
        row = {
            "model": MODEL_ID,
            "layer": layer_number,
            "tensor_key": tensor_key,
            **statistics,

            "global_int8_scale":
                global_int8["scale"],
            "global_int8_mse":
                global_int8["mse"],
            "global_int8_zeroed_values":
                global_int8["zeroed_values"],
            "global_int8_zeroed_percent":
                global_int8["zeroed_percent"],

            "global_int4_scale":
                global_int4["scale"],
            "global_int4_mse":
                global_int4["mse"],
            "global_int4_zeroed_values":
                global_int4["zeroed_values"],
            "global_int4_zeroed_percent":
                global_int4["zeroed_percent"],

            "grouped_int4_group_size":
                GROUP_SIZE,
            "grouped_int4_number_of_scales":
                grouped_int4["number_of_scales"],
            "grouped_int4_average_scale":
                grouped_int4["average_scale"],
            "grouped_int4_maximum_scale":
                grouped_int4["maximum_scale"],
            "grouped_int4_mse":
                grouped_int4["mse"],
            "grouped_int4_zeroed_values":
                grouped_int4["zeroed_values"],
            "grouped_int4_zeroed_percent":
                grouped_int4["zeroed_percent"],
        }

        # these ratios should make the comparisons much easier later
        row["int4_to_int8_mse_ratio"] = (
            global_int4["mse"]
            / global_int8["mse"]
        )

        row["grouped_vs_global_int4_mse_improvement"] = (
            global_int4["mse"]
            / grouped_int4["mse"]
        )

        row["grouped_int4_zeroed_point_reduction"] = (
            global_int4["zeroed_percent"]
            - grouped_int4["zeroed_percent"]
        )

        all_rows.append(row)

        elapsed = time.perf_counter() - layer_start

        print(
            f"done in {elapsed:.2f}s | "
            f"max/std={statistics['max_to_std']:.1f} | "
            f"global INT4 zeroed="
            f"{global_int4['zeroed_percent']:.2f}% | "
            f"grouped INT4 zeroed="
            f"{grouped_int4['zeroed_percent']:.2f}%"
        )

        # get rid of this matrix before loading the next one because there
        # is no reason to keep every 2048x2048 matrix sitting in RAM
        del weights

    dataframe = pd.DataFrame(all_rows)
    dataframe = dataframe.sort_values("layer")

    dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    total_elapsed = (
        time.perf_counter() - start_time
    )

    print()
    print("=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)
    print(
        f"Layers analyzed: {len(dataframe)}"
    )
    print(
        f"Total weights examined: "
        f"{dataframe['num_weights'].sum():,}"
    )
    print(
        f"Runtime: {total_elapsed:.2f} seconds"
    )
    print(
        f"Saved dataset to: {OUTPUT_PATH}"
    )

    # this is mainly to make sure the numbers look reasonable before we
    # start making graphs or drawing conclusions from a broken CSV
    print()
    print("SANITY-CHECK SUMMARY")
    print("-" * 72)

    summary_columns = [
        "layer",
        "max_to_std",
        "excess_kurtosis",
        "global_int4_mse",
        "global_int4_zeroed_percent",
        "grouped_int4_mse",
        "grouped_int4_zeroed_percent",
        "grouped_vs_global_int4_mse_improvement",
    ]

    print(
        dataframe[summary_columns].to_string(
            index=False
        )
    )

    print()
    print("MOST EXTREME LAYER BY MAX/STD")
    print("-" * 72)

    most_extreme = dataframe.loc[
        dataframe["max_to_std"].idxmax()
    ]

    print(
        f"Layer: {int(most_extreme['layer'])}"
    )
    print(
        f"Max/std: "
        f"{most_extreme['max_to_std']:.2f}"
    )
    print(
        f"Excess kurtosis: "
        f"{most_extreme['excess_kurtosis']:.2f}"
    )
    print(
        f"Global INT4 zeroed: "
        f"{most_extreme['global_int4_zeroed_percent']:.2f}%"
    )
    print(
        f"Grouped INT4 zeroed: "
        f"{most_extreme['grouped_int4_zeroed_percent']:.2f}%"
    )


if __name__ == "__main__":
    main()