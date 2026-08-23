from pathlib import Path

import matplotlib.pyplot as plt
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

print("Locating TinyLlama weight files...")
print("The 2.2 GB model is already downloaded, so it should not download again.\n")

model_directory = Path(
    snapshot_download(
        repo_id=MODEL_ID,
        allow_patterns=[
            "*.safetensors",
            "*.json",
        ],
    )
)

safetensor_files = sorted(model_directory.glob("*.safetensors"))

if not safetensor_files:
    raise FileNotFoundError(
        f"No .safetensors files were found in {model_directory}"
    )

target_suffix = "layers.0.self_attn.q_proj.weight"

selected_file = None
selected_key = None

# Use PyTorch backend because this checkpoint stores BF16 weights
for tensor_file in safetensor_files:
    with safe_open(
        tensor_file,
        framework="pt",
        device="cpu",
    ) as reader:
        for key in reader.keys():
            if key.endswith(target_suffix):
                selected_file = tensor_file
                selected_key = key
                break

    if selected_key is not None:
        break

if selected_file is None or selected_key is None:
    print("Available attention-related keys:")

    for tensor_file in safetensor_files:
        with safe_open(
            tensor_file,
            framework="pt",
            device="cpu",
        ) as reader:
            for key in reader.keys():
                if "self_attn" in key:
                    print(key)

    raise KeyError(
        f"Could not locate a tensor ending in: {target_suffix}"
    )

# Load the real BF16 weight matrix then convert to fp32 for stats
with safe_open(
    selected_file,
    framework="pt",
    device="cpu",
) as reader:
    weights = reader.get_tensor(selected_key).float()

flat_weights = weights.reshape(-1)

minimum = flat_weights.min().item()
maximum = flat_weights.max().item()
mean = flat_weights.mean().item()
standard_deviation = flat_weights.std().item()
largest_absolute_value = flat_weights.abs().max().item()

max_to_std_ratio = (
    largest_absolute_value / standard_deviation
    if standard_deviation != 0
    else float("inf")
)

print("REAL WEIGHT MATRIX")
print("------------------")
print(f"Model: {MODEL_ID}")
print(f"Tensor key: {selected_key}")
print(f"Source file: {selected_file.name}")
print(f"Original checkpoint dtype: bfloat16")
print(f"Analysis dtype: {weights.dtype}")
print(f"Shape: {tuple(weights.shape)}")
print(f"Number of weights: {flat_weights.numel():,}")
print(f"Minimum: {minimum:.8f}")
print(f"Maximum: {maximum:.8f}")
print(f"Mean: {mean:.8f}")
print(f"Standard deviation: {standard_deviation:.8f}")
print(f"Largest absolute value: {largest_absolute_value:.8f}")
print(
    "Max absolute value / standard deviation: "
    f"{max_to_std_ratio:.2f}"
)

plt.figure(figsize=(8, 5))

plt.hist(
    flat_weights.numpy(),
    bins=150,
)

plt.title("TinyLlama Layer 0 Query-Projection Weights")
plt.xlabel("Weight Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()

output_path = RESULTS_DIR / "tinyllama_layer0_qproj_histogram.png"

plt.savefig(
    output_path,
    dpi=200,
)

print(f"\nSaved histogram to: {output_path}")

plt.show()