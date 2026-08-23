from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TARGET_SUFFIX = "layers.0.self_attn.q_proj.weight"


def load_target_weights():
    model_directory = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            allow_patterns=["*.safetensors", "*.json"],
        )
    )

    for file in model_directory.glob("*.safetensors"):
        with safe_open(file, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if key.endswith(TARGET_SUFFIX):
                    return reader.get_tensor(key).float(), key

    raise KeyError(f"Could not find tensor ending in: {TARGET_SUFFIX}")


def symmetric_quantize(weights, qmax):
    max_abs = weights.abs().max()
    scale = max_abs / qmax

    quantized = torch.round(weights / scale)
    quantized = torch.clamp(quantized, -qmax, qmax)

    reconstructed = quantized * scale
    error = weights - reconstructed

    mse = torch.mean(error ** 2)
    zeroed_values = ((weights != 0) & (quantized == 0)).sum()

    return {
        "scale": scale.item(),
        "mse": mse.item(),
        "zeroed_values": zeroed_values.item(),
        "zeroed_percent": 100 * zeroed_values.item() / weights.numel(),
        "quantized": quantized,
        "reconstructed": reconstructed,
    }


weights, tensor_key = load_target_weights()

print("REAL LAYER QUANTIZATION")
print("=" * 50)
print(f"Model: {MODEL_ID}")
print(f"Tensor: {tensor_key}")
print(f"Shape: {tuple(weights.shape)}")
print(f"Number of weights: {weights.numel():,}")
print(f"Reference dtype for analysis: {weights.dtype}")

int8 = symmetric_quantize(weights, qmax=127)
int4 = symmetric_quantize(weights, qmax=7)

print("\nINT8 GLOBAL SYMMETRIC")
print("-" * 50)
print(f"Scale: {int8['scale']:.10f}")
print(f"MSE: {int8['mse']:.10e}")
print(f"Zeroed values: {int8['zeroed_values']:,}")
print(f"Zeroed percent: {int8['zeroed_percent']:.4f}%")

print("\nINT4 GLOBAL SYMMETRIC")
print("-" * 50)
print(f"Scale: {int4['scale']:.10f}")
print(f"MSE: {int4['mse']:.10e}")
print(f"Zeroed values: {int4['zeroed_values']:,}")
print(f"Zeroed percent: {int4['zeroed_percent']:.4f}%")

mse_ratio = int4["mse"] / int8["mse"]

print("\nCOMPARISON")
print("-" * 50)
print(f"INT4 / INT8 MSE ratio: {mse_ratio:.2f}x")
print(
    "INT4 minus INT8 zeroed values: "
    f"{int4['zeroed_values'] - int8['zeroed_values']:,}"
)

print("\nFirst 10 original weights:")
print(weights.flatten()[:10])

print("\nFirst 10 INT8 reconstructed weights:")
print(int8["reconstructed"].flatten()[:10])

print("\nFirst 10 INT4 reconstructed weights:")
print(int4["reconstructed"].flatten()[:10])