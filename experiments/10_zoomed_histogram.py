from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

model_directory = Path(
    snapshot_download(
        repo_id=MODEL_ID,
        allow_patterns=["*.safetensors", "*.json"],
    )
)

target_suffix = "layers.0.self_attn.q_proj.weight"

weights = None

for file in model_directory.glob("*.safetensors"):
    with safe_open(file, framework="pt", device="cpu") as reader:
        for key in reader.keys():
            if key.endswith(target_suffix):
                weights = reader.get_tensor(key).float()
                break

    if weights is not None:
        break

flat = weights.flatten().numpy()

# Keep only the central 99.8% of the distribution
lower = np.percentile(flat, 0.1)
upper = np.percentile(flat, 99.9)

zoomed = flat[(flat >= lower) & (flat <= upper)]

print("Zoomed histogram statistics")
print("---------------------------")
print(f"Lower cutoff: {lower:.6f}")
print(f"Upper cutoff: {upper:.6f}")
print(f"Remaining weights: {len(zoomed):,}")
print(f"Removed: {len(flat)-len(zoomed):,}")

plt.figure(figsize=(8,5))

plt.hist(
    zoomed,
    bins=150
)

plt.title("TinyLlama Layer 0 (Central 99.8%)")
plt.xlabel("Weight")
plt.ylabel("Frequency")

plt.grid(True)
plt.tight_layout()

plt.savefig("results/tinyllama_zoomed_histogram.png")

plt.show()