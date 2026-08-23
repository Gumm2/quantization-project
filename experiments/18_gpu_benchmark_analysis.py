from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")

FILES = {
    "Q8_0": RESULTS_DIR / "tinyllama_q8_vulkan_sweep.csv",
    "Q6_K": RESULTS_DIR / "tinyllama_q6_vulkan_sweep.csv",
    "Q5_K_M": RESULTS_DIR / "tinyllama_q5_vulkan_sweep.csv",
    "Q4_K_M": RESULTS_DIR / "tinyllama_q4_vulkan_sweep.csv",
}


# --------------------------------------------------
# Load all benchmark CSVs
# --------------------------------------------------

all_rows = []

for quant_name, path in FILES.items():
    df = pd.read_csv(path)

    useful = df[
        [
            "model_type",
            "model_size",
            "n_prompt",
            "n_gen",
            "avg_ts",
            "stddev_ts",
        ]
    ].copy()

    useful["quantization"] = quant_name

    all_rows.append(useful)


combined = pd.concat(
    all_rows,
    ignore_index=True,
)


# prompt processing rows have n_prompt > 0
prompt_df = combined[
    combined["n_prompt"] > 0
].copy()

# generation rows have n_gen > 0
generation_df = combined[
    combined["n_gen"] > 0
].copy()


# --------------------------------------------------
# Save one combined benchmark table
# --------------------------------------------------

combined_path = (
    RESULTS_DIR
    / "tinyllama_gpu_benchmark_combined.csv"
)

combined.to_csv(
    combined_path,
    index=False,
)


# --------------------------------------------------
# Build a clean summary table
# --------------------------------------------------

summary_rows = []

for quant_name in FILES.keys():

    quant_data = combined[
        combined["quantization"] == quant_name
    ]

    model_size = quant_data[
        "model_size"
    ].iloc[0]

    prompt_128_row = prompt_df[
        (prompt_df["quantization"] == quant_name)
        & (prompt_df["n_prompt"] == 128)
    ].iloc[0]

    prompt_512_row = prompt_df[
        (prompt_df["quantization"] == quant_name)
        & (prompt_df["n_prompt"] == 512)
    ].iloc[0]

    prompt_1024_row = prompt_df[
        (prompt_df["quantization"] == quant_name)
        & (prompt_df["n_prompt"] == 1024)
    ].iloc[0]

    gen_64_row = generation_df[
        (generation_df["quantization"] == quant_name)
        & (generation_df["n_gen"] == 64)
    ].iloc[0]

    gen_128_row = generation_df[
        (generation_df["quantization"] == quant_name)
        & (generation_df["n_gen"] == 128)
    ].iloc[0]

    gen_256_row = generation_df[
        (generation_df["quantization"] == quant_name)
        & (generation_df["n_gen"] == 256)
    ].iloc[0]

    summary_rows.append({
        "quantization": quant_name,

        "model_size_gb":
            model_size / 1_000_000_000,

        "prompt_128_tps":
            prompt_128_row["avg_ts"],

        "prompt_128_std":
            prompt_128_row["stddev_ts"],

        "prompt_512_tps":
            prompt_512_row["avg_ts"],

        "prompt_512_std":
            prompt_512_row["stddev_ts"],

        "prompt_1024_tps":
            prompt_1024_row["avg_ts"],

        "prompt_1024_std":
            prompt_1024_row["stddev_ts"],

        "generation_64_tps":
            gen_64_row["avg_ts"],

        "generation_64_std":
            gen_64_row["stddev_ts"],

        "generation_128_tps":
            gen_128_row["avg_ts"],

        "generation_128_std":
            gen_128_row["stddev_ts"],

        "generation_256_tps":
            gen_256_row["avg_ts"],

        "generation_256_std":
            gen_256_row["stddev_ts"],
    })


summary = pd.DataFrame(summary_rows)


# --------------------------------------------------
# Calculate comparisons relative to Q8
# --------------------------------------------------

q8_generation_128 = summary.loc[
    summary["quantization"] == "Q8_0",
    "generation_128_tps",
].iloc[0]

q8_size = summary.loc[
    summary["quantization"] == "Q8_0",
    "model_size_gb",
].iloc[0]


summary[
    "generation_128_speedup_vs_q8_percent"
] = (
    (
        summary["generation_128_tps"]
        / q8_generation_128
    )
    - 1
) * 100


summary[
    "size_reduction_vs_q8_percent"
] = (
    1
    - (
        summary["model_size_gb"]
        / q8_size
    )
) * 100


summary_path = (
    RESULTS_DIR
    / "tinyllama_gpu_benchmark_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)


print("\nGPU BENCHMARK SUMMARY")
print("=" * 110)

print(
    summary.to_string(
        index=False,
    )
)


# --------------------------------------------------
# Figure 1
# quantization format vs generation throughput
#
# using the 128-token generation workload as the main comparison
# and showing run-to-run variation from the five benchmark repetitions
# --------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.errorbar(
    summary["quantization"],
    summary["generation_128_tps"],
    yerr=summary["generation_128_std"],
    marker="o",
    capsize=5,
)

plt.xlabel(
    "Quantization Format"
)

plt.ylabel(
    "Generation Throughput (tokens/sec)"
)

plt.title(
    "TinyLlama Generation Throughput on RX 9070 XT"
)

plt.grid(True)

plt.tight_layout()

generation_plot = (
    RESULTS_DIR
    / "gpu_generation_throughput_by_quantization.png"
)

plt.savefig(
    generation_plot,
    dpi=250,
)

plt.close()


# --------------------------------------------------
# Figure 2
# model size vs generation throughput
#
# this is probably the cleanest deployment tradeoff figure:
# smaller model size against actual generation performance
# --------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.errorbar(
    summary["model_size_gb"],
    summary["generation_128_tps"],
    yerr=summary["generation_128_std"],
    marker="o",
    capsize=5,
)

annotation_offsets = {
    "Q8_0": (-45, 10),
    "Q6_K": (8, 8),
    "Q5_K_M": (8, 8),
    "Q4_K_M": (8, -18),
}

for _, row in summary.iterrows():

    x_offset, y_offset = annotation_offsets[
        row["quantization"]
    ]

    plt.annotate(
        row["quantization"],
        (
            row["model_size_gb"],
            row["generation_128_tps"],
        ),
        xytext=(
            x_offset,
            y_offset,
        ),
        textcoords="offset points",
    )


# leave a little breathing room so the end labels are not jammed
# against the plot borders
x_min = summary["model_size_gb"].min()
x_max = summary["model_size_gb"].max()

x_padding = (
    x_max - x_min
) * 0.08

plt.xlim(
    x_min - x_padding,
    x_max + x_padding,
)


plt.xlabel(
    "Model Size (GB)"
)

plt.ylabel(
    "Generation Throughput (tokens/sec)"
)

plt.title(
    "Model Size vs Generation Throughput"
)

plt.grid(True)

plt.tight_layout()

tradeoff_plot = (
    RESULTS_DIR
    / "gpu_size_vs_generation_throughput.png"
)

plt.savefig(
    tradeoff_plot,
    dpi=250,
)

plt.close()


# --------------------------------------------------
# Figure 3
# prompt processing throughput
#
# prompt processing did not follow the same clean trend as generation,
# so I want to show all three prompt lengths separately
# --------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

prompt_tests = [
    (
        128,
        "prompt_128_tps",
        "prompt_128_std",
    ),
    (
        512,
        "prompt_512_tps",
        "prompt_512_std",
    ),
    (
        1024,
        "prompt_1024_tps",
        "prompt_1024_std",
    ),
]

for (
    prompt_length,
    throughput_column,
    std_column,
) in prompt_tests:

    plt.errorbar(
        summary["quantization"],
        summary[throughput_column],
        yerr=summary[std_column],
        marker="o",
        capsize=4,
        label=f"{prompt_length} tokens",
    )


plt.xlabel(
    "Quantization Format"
)

plt.ylabel(
    "Prompt Processing Throughput (tokens/sec)"
)

plt.title(
    "Prompt Processing Throughput on RX 9070 XT"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

prompt_plot = (
    RESULTS_DIR
    / "gpu_prompt_throughput_by_quantization.png"
)

plt.savefig(
    prompt_plot,
    dpi=250,
)

plt.close()


# --------------------------------------------------
# Figure 4
# generation throughput across multiple output lengths
#
# this shows whether the quantization advantage is stable instead of
# depending on the single 128-token benchmark
# --------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

generation_tests = [
    (
        64,
        "generation_64_tps",
        "generation_64_std",
    ),
    (
        128,
        "generation_128_tps",
        "generation_128_std",
    ),
    (
        256,
        "generation_256_tps",
        "generation_256_std",
    ),
]

for (
    generation_length,
    throughput_column,
    std_column,
) in generation_tests:

    plt.errorbar(
        summary["quantization"],
        summary[throughput_column],
        yerr=summary[std_column],
        marker="o",
        capsize=4,
        label=f"{generation_length} tokens",
    )


plt.xlabel(
    "Quantization Format"
)

plt.ylabel(
    "Generation Throughput (tokens/sec)"
)

plt.title(
    "Generation Throughput Across Output Lengths"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

generation_length_plot = (
    RESULTS_DIR
    / "gpu_generation_throughput_across_lengths.png"
)

plt.savefig(
    generation_length_plot,
    dpi=250,
)

plt.close()


print("\nSaved:")
print(combined_path)
print(summary_path)
print(generation_plot)
print(tradeoff_plot)
print(prompt_plot)
print(generation_length_plot)