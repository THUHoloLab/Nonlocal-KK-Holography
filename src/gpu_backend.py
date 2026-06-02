"""Optional PyTorch/CUDA backend used by the simulation.

The code runs on NumPy by default when CUDA or PyTorch is unavailable. Set
KK_USE_GPU=0 to force the CPU path even on a CUDA machine.
"""

import os

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def is_available():
    return torch is not None and torch.cuda.is_available()


def is_enabled():
    flag = os.environ.get("KK_USE_GPU", "1").strip().lower()
    return flag not in {"0", "false", "no", "off"} and is_available()


def device():
    return torch.device("cuda") if is_enabled() else torch.device("cpu")


def backend_name():
    if not is_available():
        return "numpy/cpu"
    if is_enabled():
        return f"torch/cuda:{torch.cuda.get_device_name(0)}"
    return "numpy/cpu (CUDA available, KK_USE_GPU=0)"


def is_torch_array(value):
    return torch is not None and isinstance(value, torch.Tensor)


def to_torch(value, dtype=None):
    if torch is None:
        raise RuntimeError("PyTorch is not available.")
    tensor = value.to(device=device()) if is_torch_array(value) else torch.as_tensor(value, device=device())
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def to_numpy(value):
    if is_torch_array(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def ft2c_tensor(x):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x)))


def ift2c_tensor(x):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x)))


def fftn_tensor(x):
    return torch.fft.fftn(x)


def ifftn_tensor(x):
    return torch.fft.ifftn(x)

