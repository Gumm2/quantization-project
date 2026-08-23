from pathlib import Path

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

flat = weights.flatten()

std = flat.std().item()

print("\nTAIL ANALYSIS")
print("-" * 40)

print(f"Standard deviation: {std:.8f}")
print()

thresholds = [1,2,3,5,10,20,50]

for n in thresholds:

    count = (flat.abs() > n * std).sum().item()
    percent = 100 * count / flat.numel()

    print(f">{n:2d} std : {count:9,d} weights ({percent:.6f}%)")

print()

percentiles = [
    90,
    95,
    99,
    99.9,
    99.99,
    99.999,
]

print("ABSOLUTE VALUE PERCENTILES")
print("-" * 40)

abs_weights = flat.abs().numpy()

for p in percentiles:

    value = np.percentile(abs_weights, p)

    print(f"{p:7}% : {value:.8f}")

print()

print(f"Maximum absolute weight: {abs_weights.max():.8f}")