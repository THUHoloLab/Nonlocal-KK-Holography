"""Centered Fourier transform helpers."""

import numpy as np

from src import gpu_backend


def ft2c(x):
    if gpu_backend.is_torch_array(x):
        return gpu_backend.ft2c_tensor(x)
    if gpu_backend.is_enabled():
        return gpu_backend.to_numpy(gpu_backend.ft2c_tensor(gpu_backend.to_torch(x)))
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))


def ift2c(x):
    if gpu_backend.is_torch_array(x):
        return gpu_backend.ift2c_tensor(x)
    if gpu_backend.is_enabled():
        return gpu_backend.to_numpy(gpu_backend.ift2c_tensor(gpu_backend.to_torch(x)))
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))

