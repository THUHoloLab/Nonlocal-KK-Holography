"""Weighted Fourier synthesis with no phase alignment or constant phase alignment."""

import numpy as np

from src.fourier_ops import ift2c


SYNTHESIS_WEIGHT_POWER = 2.0
SYNTHESIS_WEIGHT_FLOOR = 0.02


def _synthesis_weight(mask):
    h = np.clip(np.asarray(mask, dtype=np.float64), 0.0, None)
    weight = h**SYNTHESIS_WEIGHT_POWER
    weight[h <= SYNTHESIS_WEIGHT_FLOOR] = 0.0
    return weight


def _empty_alignment_result(M, reference_index=0, phase_align_mode="none"):
    return {
        "phase_align_mode": phase_align_mode,
        "reference_index": int(reference_index),
        "coefficient_columns": ["phi0_rad", "slope_x_rad_per_px", "slope_y_rad_per_px"],
        "pixel_shift_columns": ["shift_x_px", "shift_y_px"],
        "coefficients": np.zeros((M, 3), dtype=np.float64),
        "pairwise_coefficients": np.zeros((M, 3), dtype=np.float64),
        "pixel_shifts": np.zeros((M, 2), dtype=np.float64),
        "pairwise_pixel_shifts": np.zeros((M, 2), dtype=np.float64),
        "metadata": [],
    }


def _constant_phase_offset(previous, current, overlap):
    usable = overlap & (np.abs(previous) > 0) & (np.abs(current) > 0)
    if not np.any(usable):
        return 0.0, float("nan"), 0
    corr = np.sum(current[usable] * np.conj(previous[usable]))
    norm = np.sqrt(np.sum(np.abs(current[usable]) ** 2) * np.sum(np.abs(previous[usable]) ** 2))
    return float(np.angle(corr)), float(np.abs(corr) / norm) if norm > 0 else float("nan"), int(np.count_nonzero(usable))


def _phase_model(shape, coefficients):
    phi0, slope_x, slope_y = [float(v) for v in coefficients]
    yy, xx = np.indices(shape, dtype=np.float64)
    return phi0 + slope_x * xx + slope_y * yy


def synthesize(F_kk_fields, synthesis_masks, M, phase_align=None, phase_align_mode="none", reference_index=0):
    if M <= 0:
        raise ValueError("M must be positive.")
    if phase_align is not None:
        phase_align_mode = "constant" if bool(phase_align) else "none"
    phase_align_mode = str(phase_align_mode).lower()
    if phase_align_mode not in {"none", "constant"}:
        raise ValueError('This GitHub version supports phase_align_mode="none" or "constant" only.')
    if int(reference_index) != 0:
        raise NotImplementedError("Only reference_index=0 is supported.")

    alignment = _empty_alignment_result(M, reference_index=0, phase_align_mode=phase_align_mode)
    coefficients = alignment["coefficients"]
    pairwise_coefficients = alignment["pairwise_coefficients"]

    if phase_align_mode == "constant":
        for idx in range(1, M):
            prev_w = _synthesis_weight(synthesis_masks[idx - 1])
            cur_w = _synthesis_weight(synthesis_masks[idx])
            overlap = (prev_w * cur_w) > 1e-12
            phi0, corr, pixels = _constant_phase_offset(F_kk_fields[idx - 1], F_kk_fields[idx], overlap)
            pairwise_coefficients[idx, 0] = phi0
            coefficients[idx, 0] = coefficients[idx - 1, 0] + phi0
            alignment["metadata"].append(
                {
                    "index": idx,
                    "pairwise_phi0_rad": phi0,
                    "accumulated_phi0_rad": coefficients[idx, 0],
                    "normalized_correlation": corr,
                    "fit_pixels": pixels,
                }
            )

    filtered = np.zeros_like(F_kk_fields[0], dtype=np.complex128)
    average = np.zeros_like(synthesis_masks[0], dtype=np.float64)
    for idx in range(M):
        weight = _synthesis_weight(synthesis_masks[idx])
        corrected = np.asarray(F_kk_fields[idx], dtype=np.complex128) * np.exp(-1j * _phase_model(F_kk_fields[idx].shape, coefficients[idx]))
        filtered += corrected * weight
        average += weight

    syn_f = np.zeros_like(filtered)
    valid = average != 0
    syn_f[valid] = filtered[valid] / average[valid]
    return ift2c(syn_f), syn_f, alignment

