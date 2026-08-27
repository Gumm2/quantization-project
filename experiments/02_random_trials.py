import torch

def quantize_and_measure(weights, max_int):
    max_value = torch.max(torch.abs(weights))
    scale = max_value / max_int

    quantized = torch.round(weights / scale)
    reconstructed = quantized * scale

    error = weights - reconstructed
    mse = torch.mean(error ** 2)

    return mse

num_trials = 100
tensor_size = 1000

mse8_values = []
mse4_values = []
ratio_values = []

for trial in range(num_trials):
    torch.manual_seed(trial)
    weights = torch.randn(tensor_size)

    mse8 = quantize_and_measure(weights, 127)
    mse4 = quantize_and_measure(weights, 7)

    mse8_values.append(mse8)
    mse4_values.append(mse4)
    ratio_values.append(mse4 / mse8)

avg_mse8 = torch.mean(torch.stack(mse8_values))
avg_mse4 = torch.mean(torch.stack(mse4_values))
avg_ratio = torch.mean(torch.stack(ratio_values))

print("Number of trials:", num_trials)
print("Tensor size:", tensor_size)
print("Average INT8 MSE:", avg_mse8)
print("Average INT4 MSE:", avg_mse4)
print("Average INT4/INT8 MSE ratio:", avg_ratio)