import torch

print("Torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.randn(5000, 5000, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("Matrix mean:", y.mean().item())