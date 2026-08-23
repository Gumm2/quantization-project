import torch

def quantize_and_measure(weights, max_int):
    max_value = torch.max(torch.abs(weights))
    scale = max_value / max_int

    quantized = torch.round(weights / scale)
    reconstructed = quantized * scale

    error = weights - reconstructed
    mse = torch.mean(error ** 2)

    num_zeroed = torch.sum((weights != 0) & (quantized == 0))

    return mse, scale, num_zeroed, quantized, reconstructed, error


tensor_size = 1000
outlier_values = [0, 5, 10, 50, 100, 500, 1000]

print("Outlier Injection Debug Experiment")
print("----------------------------------")

for outlier in outlier_values:
    weights = torch.randn(tensor_size)

    if outlier != 0:
        weights[0] = outlier

    mse8, scale8, zeroed8, q8, r8, e8 = quantize_and_measure(weights, 127)
    mse4, scale4, zeroed4, q4, r4, e4 = quantize_and_measure(weights, 7)

    ratio = mse4 / mse8

    print("\n====================================")
    print(f"Outlier value: {outlier}")
    print("====================================")

    print("\nFirst 10 original weights:")
    print(weights[:10])

    print("\nINT8 scale:", scale8)
    print("INT8 MSE:", mse8)
    print("INT8 zeroed values:", zeroed8)
    print("INT8 first 10 quantized:")
    print(q8[:10])
    print("INT8 first 10 reconstructed:")
    print(r8[:10])
    print("INT8 first 10 errors:")
    print(e8[:10])

    print("\nINT4 scale:", scale4)
    print("INT4 MSE:", mse4)
    print("INT4 zeroed values:", zeroed4)
    print("INT4 first 10 quantized:")
    print(q4[:10])
    print("INT4 first 10 reconstructed:")
    print(r4[:10])
    print("INT4 first 10 errors:")
    print(e4[:10])

    print("\nINT4/INT8 MSE ratio:", ratio)