import sys

import torch
import torchvision


print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"Torchvision: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
