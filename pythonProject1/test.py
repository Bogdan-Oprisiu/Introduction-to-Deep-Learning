import torch

from torch import cuda
import numpy

# Check if CUDA is available
if cuda.is_available():
    print("CUDA is available")
else:
    print("CUDA is not available")


print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("Is CUDA available?", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("GPU name:", torch.cuda.get_device_name(0))
