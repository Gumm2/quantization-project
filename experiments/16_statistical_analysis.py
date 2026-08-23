from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path("results/tinyllama_all_linear_layers.csv")
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)


# variables I want to test as possible predictors of quantization difficulty
predictors = [
    "std",
    "max_abs",
    "max_to_std",
    "excess_kurtosis",
    "skewness",
    "percent_above_5_std",
    "percent_above_10_std",
    "percent_above_20_std",
    "abs_percentile_99",
    "abs_percentile_99_9",
    "abs_percentile_99_99",
]


# these are the three outcomes I care about for this analysis:
# how many weights global INT4 destroys, how much numerical error it creates,
# and how much group-64 improves that error
targets = {
    "global_int4_zeroed_percent": "Global INT4 Zeroed %",
    "global_int4_mse": "Global INT4 MSE",
    "grouped_vs_global_int4_mse_improvement": "Group-64 MSE Improvement",
}


def make_correlation_table(method):
    rows = []

    for target_column, target_name in targets.items():

        for predictor in predictors:
            correlation = df[predictor].corr(
                df[target_column],
                method=method,
            )

            rows.append({
                "method": method,
                "target": target_name,
                "predictor": predictor,
                "correlation": correlation,
                "absolute_correlation": abs(correlation),
            })

    result = pd.DataFrame(rows)

    # sorting by absolute value because a strong negative relationship
    # can be just as informative as a strong positive one
    return result.sort_values(
        ["target", "absolute_correlation"],
        ascending=[True, False],
    )


# Pearson looks for something closer to a linear relationship
# Spearman is also useful here because some of my earlier quantization
# experiments were clearly nonlinear
pearson = make_correlation_table("pearson")
spearman = make_correlation_table("spearman")

all_correlations = pd.concat(
    [pearson, spearman],
    ignore_index=True,
)

correlation_path = (
    RESULTS_DIR
    / "tinyllama_correlations.csv"
)

all_correlations.to_csv(
    correlation_path,
    index=False,
)


# print everything instead of only the winner because I want to see
# whether there are several strong predictors or one obvious standout
print("\nPEARSON CORRELATIONS")
print("=" * 90)

for target_name in targets.values():

    section = pearson[
        pearson["target"] == target_name
    ]

    print(f"\n{target_name}")
    print("-" * 90)

    print(
        section[
            [
                "predictor",
                "correlation",
            ]
        ].to_string(
            index=False,
        )
    )


print("\n\nSPEARMAN CORRELATIONS")
print("=" * 90)

for target_name in targets.values():

    section = spearman[
        spearman["target"] == target_name
    ]

    print(f"\n{target_name}")
    print("-" * 90)

    print(
        section[
            [
                "predictor",
                "correlation",
            ]
        ].to_string(
            index=False,
        )
    )


# --------------------------------------------------
# Figure 1
# max/std vs global INT4 zeroing
#
# max/std was one of my strongest candidate predictors because global
# quantization literally lets the maximum absolute value determine scale
# --------------------------------------------------

plt.figure(figsize=(8, 5))

for matrix_type, group in df.groupby("matrix_type"):

    plt.scatter(
        group["max_to_std"],
        group["global_int4_zeroed_percent"],
        label=matrix_type,
        alpha=0.75,
    )

plt.xlabel(
    "Maximum Absolute Weight / Standard Deviation"
)

plt.ylabel(
    "Global INT4 Zeroed Weights (%)"
)

plt.title(
    "Outlier Severity vs Global INT4 Weight Loss"
)

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "max_std_vs_global_int4_zeroing.png",
    dpi=250,
)

plt.close()


# --------------------------------------------------
# Figure 2
# std vs global INT4 zeroing
#
# I originally suspected standard deviation might be especially useful
# because it describes the overall spread instead of focusing on only
# the single most extreme weight. This directly tests that prediction.
# --------------------------------------------------

plt.figure(figsize=(8, 5))

for matrix_type, group in df.groupby("matrix_type"):

    plt.scatter(
        group["std"],
        group["global_int4_zeroed_percent"],
        label=matrix_type,
        alpha=0.75,
    )

plt.xlabel("Standard Deviation")

plt.ylabel(
    "Global INT4 Zeroed Weights (%)"
)

plt.title(
    "Weight Spread vs Global INT4 Weight Loss"
)

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "std_vs_global_int4_zeroing.png",
    dpi=250,
)

plt.close()


# --------------------------------------------------
# Figure 3
# max/std vs benefit from grouping
#
# if extreme global values are really driving the problem, I would
# expect matrices with worse outlier severity to benefit more when
# each group is allowed to calculate its own local scale
# --------------------------------------------------

plt.figure(figsize=(8, 5))

for matrix_type, group in df.groupby("matrix_type"):

    plt.scatter(
        group["max_to_std"],
        group[
            "grouped_vs_global_int4_mse_improvement"
        ],
        label=matrix_type,
        alpha=0.75,
    )

plt.xlabel(
    "Maximum Absolute Weight / Standard Deviation"
)

plt.ylabel(
    "Group-64 MSE Improvement (x)"
)

plt.title(
    "Outlier Severity vs Benefit from Group Quantization"
)

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "max_std_vs_grouping_improvement.png",
    dpi=250,
)

plt.close()


# --------------------------------------------------
# automatically record the strongest predictor for each outcome
# so I do not have to manually pull it out of the full table
# --------------------------------------------------

summary_lines = []

summary_lines.append(
    "TINYLLAMA STATISTICAL ANALYSIS"
)

summary_lines.append(
    "=" * 70
)

for method_name, table in [
    ("Pearson", pearson),
    ("Spearman", spearman),
]:

    summary_lines.append("")
    summary_lines.append(
        method_name.upper()
    )

    summary_lines.append(
        "-" * 70
    )

    for target_name in targets.values():

        section = table[
            table["target"] == target_name
        ]

        strongest = section.iloc[0]

        summary_lines.append("")

        summary_lines.append(
            f"Target: {target_name}"
        )

        summary_lines.append(
            f"Strongest predictor: "
            f"{strongest['predictor']}"
        )

        summary_lines.append(
            f"Correlation: "
            f"{strongest['correlation']:.6f}"
        )


# --------------------------------------------------
# I also want a matrix-family summary because the last experiment
# showed enormous differences between q/k/v/o and the MLP matrices
# --------------------------------------------------

family_summary = (
    df.groupby("matrix_type")
    .agg(
        matrices=(
            "matrix_type",
            "size",
        ),

        mean_std=(
            "std",
            "mean",
        ),

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

        mean_global_int4_mse=(
            "global_int4_mse",
            "mean",
        ),

        mean_group64_mse_improvement=(
            "grouped_vs_global_int4_mse_improvement",
            "mean",
        ),
    )
    .sort_values(
        "mean_global_int4_zeroed",
        ascending=False,
    )
)

family_summary_path = (
    RESULTS_DIR
    / "tinyllama_matrix_family_summary.csv"
)

family_summary.to_csv(
    family_summary_path,
)


# saving a plain text summary too because this will be useful later
# when I start turning the analysis into the Substack writeup
summary_lines.append("")
summary_lines.append("")
summary_lines.append(
    "MATRIX FAMILY SUMMARY"
)

summary_lines.append(
    "=" * 70
)

summary_lines.append(
    family_summary.to_string()
)

summary_text = "\n".join(
    summary_lines
)

summary_path = (
    RESULTS_DIR
    / "tinyllama_statistical_summary.txt"
)

summary_path.write_text(
    summary_text,
    encoding="utf-8",
)


print("\n\nAUTOMATIC SUMMARY")
print("=" * 90)

print(summary_text)


print("\nSaved:")

print(correlation_path)
print(family_summary_path)
print(summary_path)

print(
    RESULTS_DIR
    / "max_std_vs_global_int4_zeroing.png"
)

print(
    RESULTS_DIR
    / "std_vs_global_int4_zeroing.png"
)

print(
    RESULTS_DIR
    / "max_std_vs_grouping_improvement.png"
)