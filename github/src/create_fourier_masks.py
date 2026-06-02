"""Fourier mask generation for ideal and measured-transmission models.

The measurement mode first looks for local CSV files under
data/measurement. When those files are present, it computes an s-polar
transmission curve and broadens it by the configured illumination FWHM. When
the local files are absent, it falls back to an analytic broadened cutoff.
"""

from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent
LOCAL_MEASUREMENT_DIR = PROJECT_DIR / "data" / "measurement"
PLOT_LAMBDA_NM = np.arange(500.0, 900.0 + 0.25, 0.5)
PLOT_K_OVER_K0 = np.arange(0.0, 0.8 + 0.0005, 0.001)
MEASUREMENT_ANGLE_DEG = np.arange(0.0, 70.0 + 0.25, 0.5)


def _gaussian_spectrum(lambda_um, fwhm_nm, sample_count=121, span_sigma=4.0):
    if fwhm_nm is None or float(fwhm_nm) <= 0:
        return np.array([float(lambda_um)]), np.array([1.0])
    sigma_um = float(fwhm_nm) * 1e-3 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    offsets = np.linspace(-span_sigma * sigma_um, span_sigma * sigma_um, int(sample_count))
    wavelengths = float(lambda_um) + offsets
    wavelengths = wavelengths[wavelengths > 0]
    weights = np.exp(-0.5 * ((wavelengths - float(lambda_um)) / sigma_um) ** 2)
    weights /= np.sum(weights)
    return wavelengths, weights


def _read_spectrum_data(path):
    data = np.genfromtxt(path, delimiter=",")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Unexpected spectrum CSV format: {path}")
    return data[:, 0], np.maximum(data[:, 1:], 0.0)


def _interp_along_wavelength(source_wavelength_nm, values, target_wavelength_nm):
    interpolated = np.empty((target_wavelength_nm.size, values.shape[1]), dtype=np.float64)
    for col_idx in range(values.shape[1]):
        interpolated[:, col_idx] = np.interp(
            target_wavelength_nm,
            source_wavelength_nm,
            values[:, col_idx],
            left=np.nan,
            right=np.nan,
        )
    return interpolated


def _interp_transmission_map(wavelength_nm, angle_deg, transmission, target_wavelength_nm, target_theta_deg):
    by_wavelength = _interp_along_wavelength(wavelength_nm, transmission, target_wavelength_nm)
    transmission_map = np.empty((target_wavelength_nm.size, target_theta_deg.size), dtype=np.float64)
    for row_idx in range(by_wavelength.shape[0]):
        row = by_wavelength[row_idx, :]
        valid = np.isfinite(row)
        if np.count_nonzero(valid) < 2:
            transmission_map[row_idx, :] = np.nan
        else:
            transmission_map[row_idx, :] = np.interp(
                target_theta_deg,
                angle_deg[valid],
                row[valid],
                left=np.nan,
                right=np.nan,
            )
    return transmission_map


def _local_measurement_files_exist(measurement_dir):
    required = [
        "dark_spectrum_20ms.csv",
        "source_spectrum_20ms_sPol.csv",
        "Thorlabs_FESH0700_20ms_0_0.5_70_sPolar.csv",
    ]
    return all((Path(measurement_dir) / name).exists() for name in required)


def _load_local_s_polar_transmission_curve(lambda_um, illumination_fwhm_nm, measurement_dir=LOCAL_MEASUREMENT_DIR):
    measurement_dir = Path(measurement_dir)
    if not _local_measurement_files_exist(measurement_dir):
        return None

    dark_wavelength_nm, dark_data = _read_spectrum_data(measurement_dir / "dark_spectrum_20ms.csv")
    source_wavelength_nm, source_data = _read_spectrum_data(measurement_dir / "source_spectrum_20ms_sPol.csv")
    filter_wavelength_nm, filter_data = _read_spectrum_data(
        measurement_dir / "Thorlabs_FESH0700_20ms_0_0.5_70_sPolar.csv"
    )

    dark_for_source = np.interp(source_wavelength_nm, dark_wavelength_nm, np.mean(dark_data, axis=1))
    dark_for_filter = np.interp(filter_wavelength_nm, dark_wavelength_nm, np.mean(dark_data, axis=1))
    source_spectrum = np.mean(source_data, axis=1) - dark_for_source
    filter_signal = filter_data - dark_for_filter[:, np.newaxis]
    source_on_filter_grid = np.interp(filter_wavelength_nm, source_wavelength_nm, source_spectrum, left=np.nan, right=np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        transmission_s = filter_signal / source_on_filter_grid[:, np.newaxis]
    transmission_s[~np.isfinite(transmission_s)] = np.nan

    theta_deg = np.rad2deg(np.arcsin(PLOT_K_OVER_K0))
    ts_map = _interp_transmission_map(
        filter_wavelength_nm,
        MEASUREMENT_ANGLE_DEG,
        transmission_s,
        PLOT_LAMBDA_NM,
        theta_deg,
    )
    baseline_band = (PLOT_LAMBDA_NM >= 750.0) & (PLOT_LAMBDA_NM <= 800.0)
    baseline = np.nanmean(ts_map[baseline_band, :], axis=0)
    ts_map = np.maximum(ts_map - baseline[np.newaxis, :], 0.0)

    wavelengths, weights = _gaussian_spectrum(lambda_um, illumination_fwhm_nm)
    wavelengths_nm = wavelengths * 1000.0
    local_weights = np.interp(PLOT_LAMBDA_NM, wavelengths_nm, weights, left=0.0, right=0.0)
    if np.sum(local_weights) <= 0:
        local_weights = np.zeros_like(PLOT_LAMBDA_NM)
        local_weights[np.argmin(np.abs(PLOT_LAMBDA_NM - float(lambda_um) * 1000.0))] = 1.0
    local_weights = local_weights / np.sum(local_weights)

    weighted_ts = np.nansum(ts_map * local_weights[:, np.newaxis], axis=0)
    valid_weight = np.sum(np.isfinite(ts_map) * local_weights[:, np.newaxis], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        broadened_ts = weighted_ts / valid_weight
    broadened_ts[~np.isfinite(broadened_ts)] = 0.0
    broadened_ts = np.maximum(broadened_ts, 0.0)
    if np.max(broadened_ts) > 0:
        broadened_ts = broadened_ts / np.max(broadened_ts)
    return PLOT_K_OVER_K0, np.clip(broadened_ts, 0.0, 1.0)


def _measurement_intensity_transmission(
    f_dist,
    lambda_um,
    na_threshold,
    illumination_fwhm_nm,
    edge_softness=0.006,
    sample_count=121,
):
    """Approximate spectral broadening of a measured NA cutoff.

    For each wavelength sample, the radial coordinate maps to sin(theta) as
    lambda*f_dist. Averaging over the source spectrum broadens the mask edge.
    """
    wavelengths, weights = _gaussian_spectrum(lambda_um, illumination_fwhm_nm, sample_count=sample_count)
    intensity = np.zeros_like(f_dist, dtype=np.float64)
    softness = max(float(edge_softness), 1e-6)
    for wavelength, weight in zip(wavelengths, weights):
        sin_theta = wavelength * f_dist
        intensity += weight / (1.0 + np.exp((sin_theta - float(na_threshold)) / softness))
    return np.clip(intensity, 0.0, 1.0)


def create_fourier_masks(
    lambda_um,
    M,
    fx,
    fy,
    NA_threshold,
    NA_illumination,
    mask_type="ideal",
    rotation_direction="cw",
    initial_angle=0.0,
    illumination_fwhm_nm=None,
    measurement_edge_softness=0.006,
    measurement_sample_count=121,
    measurement_dir=LOCAL_MEASUREMENT_DIR,
):
    r_threshold = float(NA_threshold) / float(lambda_um)
    r_illumination = float(NA_illumination) / float(lambda_um)
    df = abs(float(fx[0, 1] - fx[0, 0])) if fx.shape[1] > 1 else 0.0
    tol = 0.5 * df

    mask_mode = str(mask_type).lower()
    if mask_mode not in {"ideal", "measurement"}:
        raise ValueError('mask_type must be either "ideal" or "measurement".')

    direction_sign = -1.0 if str(rotation_direction).lower() == "cw" else 1.0
    angle_step = 360.0 / M if M else 0.0
    masks = []

    local_measurement_curve = None
    if mask_mode == "measurement":
        local_measurement_curve = _load_local_s_polar_transmission_curve(
            lambda_um,
            illumination_fwhm_nm,
            measurement_dir=measurement_dir,
        )

    for idx in range(M):
        theta = np.deg2rad(float(initial_angle) + direction_sign * idx * angle_step)
        center_fx = r_illumination * np.cos(theta)
        center_fy = r_illumination * np.sin(theta)
        f_dist = np.sqrt((fx - center_fx) ** 2 + (fy - center_fy) ** 2)

        if mask_mode == "ideal":
            intensity_transmission = (f_dist <= (r_threshold + tol)).astype(np.float64)
        elif local_measurement_curve is not None:
            k_over_k0, transmission = local_measurement_curve
            intensity_transmission = np.interp(
                (f_dist * float(lambda_um)).ravel(),
                k_over_k0,
                transmission,
                left=0.0,
                right=0.0,
            ).reshape(fx.shape)
        else:
            intensity_transmission = _measurement_intensity_transmission(
                f_dist,
                lambda_um=lambda_um,
                na_threshold=NA_threshold,
                illumination_fwhm_nm=illumination_fwhm_nm,
                edge_softness=measurement_edge_softness,
                sample_count=measurement_sample_count,
            )

        # Downstream filtering is applied to complex field amplitude.
        masks.append(np.sqrt(np.clip(intensity_transmission, 0.0, None)))

    return masks
