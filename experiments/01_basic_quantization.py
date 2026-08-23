import torch

weights = torch.randn(1000)

def quantize_and_measure(weights, max_int):
    max_value = torch.max(torch.abs(weights))
    scale = max_value / max_int

    quantized = torch.round(weights / scale)
    reconstructed = quantized * scale

    error = weights - reconstructed
    mse = torch.mean(error ** 2)

    return scale, quantized, reconstructed, error, mse

scale_8, q8, r8, e8, mse8 = quantize_and_measure(weights, 127)
scale_4, q4, r4, e4, mse4 = quantize_and_measure(weights, 7)

print("Original weights:")
print(weights)

print("\n===== INT8 =====")
print("Scale:", scale_8)
print("Quantized:", q8)
print("Reconstructed:", r8)
print("Error:", e8)
print("MSE:", mse8)

print("\n===== INT4 =====")
print("Scale:", scale_4)
print("Quantized:", q4)
print("Reconstructed:", r4)
print("Error:", e4)
print("MSE:", mse4)

mse_ratio = mse4 / mse8

print("\n===== Comparison =====")
print(f"INT4 MSE is {mse_ratio:.2f}x larger than INT8 MSE")