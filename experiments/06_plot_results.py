import pandas as pd
import matplotlib.pyplot as plt

# load experiment 5 summary
df = pd.read_csv("results/group_size_sweep_summary.csv")

#group size vs mse plot

plt.figure(figsize=(8, 5))

plt.plot(
    df["group_size"],
    df["mean_mse"],
    marker="o"
)

plt.xscale("log", base=2)
plt.yscale("log")

plt.xlabel("Group Size")
plt.ylabel("Mean MSE")
plt.title("Quantization Error vs Group Size")

plt.grid(True)
plt.tight_layout()

plt.savefig("results/mse_vs_group_size.png")
plt.show()


# --------------------------------------
# Plot 2: Group Size vs Zeroed Values
# --------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    df["group_size"],
    df["mean_zeroed_values"],
    marker="o"
)

plt.xscale("log", base=2)

plt.xlabel("Group Size")
plt.ylabel("Average Zeroed Values")
plt.title("Zeroed Weights vs Group Size")

plt.grid(True)
plt.tight_layout()

plt.savefig("results/zeroed_vs_group_size.png")
plt.show()


# --------------------------------------
# Plot 3: Group Size vs Average Scale
# --------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    df["group_size"],
    df["mean_average_scale"],
    marker="o"
)

plt.xscale("log", base=2)

plt.xlabel("Group Size")
plt.ylabel("Average Scale")
plt.title("Quantization Scale vs Group Size")

plt.grid(True)
plt.tight_layout()

plt.savefig("results/scale_vs_group_size.png")
plt.show()