import pandas as pd
import matplotlib.pyplot as plt

# Load experiment 5 summary
df = pd.read_csv("results/group_size_sweep_summary.csv")

# remove group size 1
df = df[df["group_size"] > 1]

plt.figure(figsize=(8,5))

plt.plot(
    df["group_size"],
    df["mean_mse"],
    marker="o",
    linewidth=2
)

plt.xscale("log", base=2)
plt.yscale("log")

plt.title("Shared-Scale Quantization Error")
plt.xlabel("Group Size")
plt.ylabel("Mean MSE")

plt.grid(True)

plt.tight_layout()

plt.savefig("results/shared_group_mse.png")

plt.show()