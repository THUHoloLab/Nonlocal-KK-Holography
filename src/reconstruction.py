"""Single-shot KK reconstruction in Fourier space."""

import numpy as np

from src import gpu_backend
from src.fourier_ops import ft2c, ift2c


RESPONSE_CORRECTION_GAIN_MAX = 10.0
RESPONSE_CORRECTION_MASK_FLOOR = 0.02


def _ft2c_last2_torch(x):
    return gpu_backend.torch.fft.fftshift(
        gpu_backend.torch.fft.fft2(gpu_backend.torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1)),
        dim=(-2, -1),
    )


def _ift2c_last2_torch(x):
    return gpu_backend.torch.fft.fftshift(
        gpu_backend.torch.fft.ifft2(gpu_backend.torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1)),
        dim=(-2, -1),
    )


def _regularized_mask_deconvolution(spectrum, mask, signum_fun):
    h = np.asarray(mask, dtype=np.float64)
    analytic_side = np.asarray(signum_fun, dtype=np.float64) >= 0
    valid = (h > RESPONSE_CORRECTION_MASK_FLOOR) & analytic_side
    eps = 1.0 / (RESPONSE_CORRECTION_GAIN_MAX**2)
    corrected = np.zeros_like(spectrum, dtype=np.complex128)
    corrected[valid] = spectrum[valid] * h[valid] / (h[valid] ** 2 + eps)
    return corrected


def _regularized_mask_deconvolution_torch(spectrum, mask, signum_fun):
    h = mask.real.to(dtype=spectrum.real.dtype)
    valid = (h > RESPONSE_CORRECTION_MASK_FLOOR) & (signum_fun.real >= 0)
    eps = 1.0 / (RESPONSE_CORRECTION_GAIN_MAX**2)
    corrected = gpu_backend.torch.zeros_like(spectrum)
    corrected[valid] = spectrum[valid] * h[valid] / (h[valid] ** 2 + eps)
    return corrected


def reconstruction(intensity, signum_fun, mask, N):
    _ = N
    if gpu_backend.is_enabled():
        with gpu_backend.torch.no_grad():
            intensity_t = gpu_backend.to_torch(intensity)
            signum_t = gpu_backend.to_torch(signum_fun)
            mask_t = gpu_backend.to_torch(mask)
            intensity_t = gpu_backend.torch.clamp(intensity_t.real, min=1e-12)
            real_part_t = gpu_backend.torch.log(intensity_t) / 2.0
            imag_part_t = _ift2c_last2_torch(-1j * signum_t * _ft2c_last2_torch(real_part_t)).real
            kk_field_t = gpu_backend.torch.sqrt(intensity_t) * gpu_backend.torch.exp(1j * imag_part_t)
            return gpu_backend.to_numpy(
                _regularized_mask_deconvolution_torch(_ft2c_last2_torch(kk_field_t), mask_t, signum_t)
            )

    intensity = np.clip(np.real(intensity), 1e-12, None)
    real_part = np.log(intensity) / 2.0
    imag_part = np.real(ift2c(-1j * signum_fun * ft2c(real_part)))
    kk_field = np.sqrt(intensity) * np.exp(1j * imag_part)
    return _regularized_mask_deconvolution(ft2c(kk_field), mask, signum_fun)


def reconstruction_batch(intensities, signum_funs, masks, N=None, return_torch=False, real_dtype=None):
    _ = N
    if gpu_backend.is_enabled():
        with gpu_backend.torch.no_grad():
            real_dtype = gpu_backend.torch.float64 if real_dtype is None else real_dtype
            intensity_t = gpu_backend.to_torch(intensities, dtype=real_dtype)
            signum_t = gpu_backend.to_torch(signum_funs, dtype=real_dtype)
            mask_t = gpu_backend.to_torch(masks, dtype=real_dtype)
            intensity_t = gpu_backend.torch.clamp(intensity_t.real, min=1e-12)
            real_part_t = gpu_backend.torch.log(intensity_t) / 2.0
            imag_part_t = _ift2c_last2_torch(-1j * signum_t * _ft2c_last2_torch(real_part_t)).real
            kk_field_t = gpu_backend.torch.sqrt(intensity_t) * gpu_backend.torch.exp(1j * imag_part_t)
            spectra_t = _regularized_mask_deconvolution_torch(_ft2c_last2_torch(kk_field_t), mask_t, signum_t)
            return spectra_t if return_torch else gpu_backend.to_numpy(spectra_t)

    intensities = np.asarray(intensities)
    return np.stack(
        [reconstruction(i, s, m, intensities.shape[-1]) for i, s, m in zip(intensities, signum_funs, masks)],
        axis=0,
    )


def ift2c_batch(spectra, return_torch=False):
    if gpu_backend.is_enabled():
        field_t = _ift2c_last2_torch(gpu_backend.to_torch(spectra))
        return field_t if return_torch else gpu_backend.to_numpy(field_t)
    spectra = np.asarray(spectra)
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spectra, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))

