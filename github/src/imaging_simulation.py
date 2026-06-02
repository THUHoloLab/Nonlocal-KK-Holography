"""Forward imaging model used by the simulation."""

import numpy as np
from scipy import ndimage, signal

from src import gpu_backend
from src.fourier_ops import ft2c, ift2c


def _subsampling(image, rate, N):
    if rate <= 1:
        return image
    kernel = np.ones((rate, rate), dtype=np.float64)
    blurred = signal.convolve2d(image, kernel, mode="same", boundary="symm")
    downsampled = blurred[::rate, ::rate]
    zoom = (N / downsampled.shape[0], N / downsampled.shape[1])
    return ndimage.zoom(downsampled, zoom, order=0)[:N, :N]


def imaging_simulation(im, mask, N, subsample_rate=1, noise_intensity=0.0):
    if gpu_backend.is_enabled():
        im_t = gpu_backend.to_torch(im)
        mask_t = gpu_backend.to_torch(mask)
        with gpu_backend.torch.no_grad():
            field_t = gpu_backend.ift2c_tensor(gpu_backend.ft2c_tensor(im_t) * mask_t)
            captured_intensity = gpu_backend.to_numpy(field_t.real**2 + field_t.imag**2)
    else:
        field = ift2c(ft2c(im) * mask)
        captured_intensity = np.abs(field) ** 2

    captured_intensity = _subsampling(captured_intensity, subsample_rate, N)
    if noise_intensity > 0:
        noise = np.random.standard_normal(captured_intensity.shape)
        captured_intensity = captured_intensity + noise_intensity * np.max(captured_intensity) * noise
    return captured_intensity

