import pandas as pd
import matplotlib.pyplot as plt

# load experiment
df = pd.read_csv("results/group_quantization.csv")

# verify it loaded correctly
print(df)
plt.figure(figsize=(8,5))

plt.scatter(
    df["average_scale"],
    df["mse"],
    s=80
)

plt.title("Average Scale vs Mean Squared Error")
plt.xlabel("Average Scale")
plt.ylabel("Mean Squared Error (MSE)")

plt.xscale("log")

plt.grid(True)

plt.savefig("results/group_size_vs_mse.png")

plt.show()
print("\nConsecutive MSE Changes")
print("-" * 40)

for i in range(1, len(df)):
    previous = df.iloc[i - 1]
    current = df.iloc[i]

    mse_change = previous["mse"] - current["mse"]
    percent_change = (mse_change / previous["mse"]) * 100

    print(f"{int(previous['group_size'])} -> {int(current['group_size'])}")
    print(f"Absolute MSE decrease: {mse_change:.6f}")
    print(f"Percent decrease: {percent_change:.2f}%")
    print()