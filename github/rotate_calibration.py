"""Rotation calibration and image alignment utility.

The script estimates the rotation center from images named angle_000.bmp,
angle_010.bmp, ... and writes aligned center crops plus a preview GIF.
"""

from pathlib import Path
import re

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from utils.crop_center import crop_center


ANGLE_FILENAME_RE = re.compile(r"angle_(\d+)\.bmp$", re.IGNORECASE)


def read_gray_image(image_path):
    image = imageio.imread(image_path)
    if image.ndim == 3:
        image = image[..., :3].mean(axis=2)
    return image.astype(np.float64)


def parse_angle_from_name(filename):
    match = ANGLE_FILENAME_RE.match(filename)
    return None if match is None else int(match.group(1))


def discover_angle_image_paths(data_dir):
    angle_to_path = {}
    for image_path in sorted(Path(data_dir).glob("*.bmp")):
        angle = parse_angle_from_name(image_path.name)
        if angle is not None:
            angle_to_path[angle] = image_path
    if not angle_to_path:
        raise FileNotFoundError(f"No angle_XXX.bmp images found in {data_dir}")
    return angle_to_path


def estimate_image_scale(gray_image, eps=1e-6):
    valid = gray_image[np.isfinite(gray_image) & (gray_image > eps)]
    return 1.0 if valid.size == 0 else float(np.median(valid))


def make_hann_window(shape):
    return np.maximum(np.outer(np.hanning(shape[0]), np.hanning(shape[1])), 1e-6)


def quadratic_peak_offset(prev_value, center_value, next_value):
    denominator = prev_value - 2.0 * center_value + next_value
    if abs(denominator) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (prev_value - next_value) / denominator, -1.0, 1.0))


def phase_correlation_translation(reference, moving):
    """Estimate moving-to-reference translation with subpixel peak fitting."""
    if reference.shape != moving.shape:
        raise ValueError("reference and moving must have the same shape.")
    window = make_hann_window(reference.shape)
    reference_work = (reference - np.mean(reference)) * window
    moving_work = (moving - np.mean(moving)) * window

    ref_ft = np.fft.fftn(reference_work)
    mov_ft = np.fft.fftn(moving_work)
    cross_power = ref_ft * np.conj(mov_ft)
    cross_power /= np.maximum(np.abs(cross_power), 1e-12)
    corr_abs = np.abs(np.fft.ifftn(cross_power))

    peak = np.unravel_index(np.argmax(corr_abs), corr_abs.shape)
    shifts = np.array(peak, dtype=np.float64)
    midpoints = np.fix(np.array(corr_abs.shape) / 2.0)
    shifts[shifts > midpoints] -= np.array(corr_abs.shape)[shifts > midpoints]

    py, px = peak
    peak_value = corr_abs[py, px]
    shifts += np.array(
        [
            quadratic_peak_offset(corr_abs[(py - 1) % corr_abs.shape[0], px], peak_value, corr_abs[(py + 1) % corr_abs.shape[0], px]),
            quadratic_peak_offset(corr_abs[py, (px - 1) % corr_abs.shape[1]], peak_value, corr_abs[py, (px + 1) % corr_abs.shape[1]]),
        ],
        dtype=np.float64,
    )
    dy, dx = shifts
    return float(dx), float(dy)


def estimate_multiplicative_background(image_paths, block_rows=256, smooth_sigma=0.0, eps=1e-6):
    """Estimate a shared flat-field background from the median normalized stack."""
    image_paths = [Path(path) for path in image_paths]
    reference = read_gray_image(image_paths[0])
    height, width = reference.shape
    scales = [max(estimate_image_scale(read_gray_image(path), eps=eps), eps) for path in image_paths]
    background = np.empty((height, width), dtype=np.float64)

    for row_start in range(0, height, block_rows):
        row_end = min(row_start + block_rows, height)
        stack = np.empty((len(image_paths), row_end - row_start, width), dtype=np.float32)
        for idx, (path, scale) in enumerate(zip(image_paths, scales)):
            stack[idx] = (read_gray_image(path)[row_start:row_end, :] / scale).astype(np.float32)
        background[row_start:row_end, :] = np.median(stack, axis=0)

    if smooth_sigma and smooth_sigma > 0:
        background = ndimage.gaussian_filter(background, sigma=float(smooth_sigma), mode="nearest")
    background = np.maximum(background, eps)
    return background / max(float(np.median(background)), eps)


def apply_multiplicative_background_correction(image, background, eps=1e-6):
    corrected = image.astype(np.float64) / np.maximum(background, eps)
    return corrected


def derotate_image(image, angle_deg, fill_mode="nearest"):
    return ndimage.rotate(image, -float(angle_deg), reshape=False, order=3, mode=fill_mode)


def estimate_absolute_shifts(derotated_by_angle, reference_angle):
    """Use neighbor registration edges and least squares to get absolute shifts."""
    angles = sorted(derotated_by_angle)
    edges = []
    for idx, angle_i in enumerate(angles):
        for step in (1, 2):
            if idx + step >= len(angles):
                continue
            angle_j = angles[idx + step]
            dx, dy = phase_correlation_translation(derotated_by_angle[angle_i], derotated_by_angle[angle_j])
            edges.append((angle_i, angle_j, dx, dy))

    index = {angle: idx for idx, angle in enumerate(angles)}
    matrix = np.zeros((len(edges) + 1, len(angles)), dtype=np.float64)
    rhs_x = np.zeros(len(edges) + 1, dtype=np.float64)
    rhs_y = np.zeros(len(edges) + 1, dtype=np.float64)
    for row, (angle_i, angle_j, dx, dy) in enumerate(edges):
        matrix[row, index[angle_i]] = -1.0
        matrix[row, index[angle_j]] = 1.0
        rhs_x[row] = dx
        rhs_y[row] = dy
    matrix[-1, index[int(reference_angle)]] = 1.0

    abs_x = np.linalg.lstsq(matrix, rhs_x, rcond=None)[0]
    abs_y = np.linalg.lstsq(matrix, rhs_y, rcond=None)[0]
    return {angle: (float(abs_x[index[angle]]), float(abs_y[index[angle]])) for angle in angles}


def fit_circle_through_origin(shift_records):
    coords = np.array([[record["dx"], record["dy"]] for record in shift_records], dtype=np.float64)
    lhs = coords
    rhs = -(coords[:, 0] ** 2 + coords[:, 1] ** 2)
    ab = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    center = -0.5 * ab
    radius = float(np.linalg.norm(center))
    residuals = np.abs(np.linalg.norm(coords - center[None, :], axis=1) - radius)
    return center, radius, residuals


def estimate_rotation_center(data_dir, reference_angle=0, remove_background=True):
    angle_to_path = discover_angle_image_paths(data_dir)
    angles = sorted(angle_to_path)
    reference_angle = int(reference_angle) if int(reference_angle) in angle_to_path else angles[0]
    image_paths = [angle_to_path[angle] for angle in angles]
    background = estimate_multiplicative_background(image_paths) if remove_background else None

    reference = read_gray_image(angle_to_path[reference_angle])
    if background is not None:
        reference = apply_multiplicative_background_correction(reference, background)
    height, width = reference.shape
    geometric_center = np.array([(width - 1) / 2.0, (height - 1) / 2.0], dtype=np.float64)

    derotated_by_angle = {}
    for angle in angles:
        image = read_gray_image(angle_to_path[angle])
        if background is not None:
            image = apply_multiplicative_background_correction(image, background)
        derotated_by_angle[angle] = derotate_image(image, angle - reference_angle)

    absolute_shifts = estimate_absolute_shifts(derotated_by_angle, reference_angle)
    shift_records = [
        {"angle": angle, "dx": absolute_shifts[angle][0], "dy": absolute_shifts[angle][1]}
        for angle in angles
        if angle != reference_angle
    ]
    circle_center, circle_radius, residuals = fit_circle_through_origin(shift_records)
    center_xy = geometric_center + circle_center
    diagnostics = [
        {**record, "residual_px": float(residual)}
        for record, residual in zip(shift_records, residuals)
    ]
    return (float(center_xy[0]), float(center_xy[1])), diagnostics, background


def align_image(image, angle_deg, center_xy, fill_mode="nearest"):
    if image.ndim == 3:
        return np.stack([align_image(image[..., c], angle_deg, center_xy, fill_mode) for c in range(image.shape[2])], axis=2)
    height, width = image.shape
    geometric_center = np.array([(width - 1) / 2.0, (height - 1) / 2.0], dtype=np.float64)
    shift_xy = geometric_center - np.asarray(center_xy, dtype=np.float64)
    shifted = ndimage.shift(image, shift=(shift_xy[1], shift_xy[0]), order=3, mode=fill_mode)
    return ndimage.rotate(shifted, -float(angle_deg), reshape=False, order=3, mode=fill_mode)


def run_rotate_calibration(data_dir, output_dir, crop_size=1300, reference_angle=0, remove_background=True):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    angle_to_path = discover_angle_image_paths(data_dir)
    angles = sorted(angle_to_path)
    center_xy, diagnostics, background = estimate_rotation_center(data_dir, reference_angle, remove_background)
    print(f"Estimated rotation center: x={center_xy[0]:.2f}, y={center_xy[1]:.2f}")

    if background is not None:
        preview = np.clip(255.0 * background / max(float(np.max(background)), 1e-12), 0, 255).astype(np.uint8)
        imageio.imwrite(output_dir / "estimated_background.bmp", preview)

    frames = []
    for angle in angles:
        image = imageio.imread(angle_to_path[angle])
        if background is not None:
            image = apply_multiplicative_background_correction(image, background)
        aligned = crop_center(align_image(image, angle - reference_angle, center_xy), (crop_size, crop_size))
        aligned_u8 = np.clip(np.rint(aligned), 0, 255).astype(np.uint8)
        imageio.imwrite(output_dir / f"crop_aligned_{angle:03d}.bmp", aligned_u8)
        frames.append(aligned_u8)

    imageio.mimsave(output_dir / "aligned_sequence.gif", frames, duration=0.15, loop=0)
    save_shift_plot(diagnostics, output_dir / "estimated_shifts_xy.png")
    return center_xy


def save_shift_plot(diagnostics, output_path):
    if not diagnostics:
        return
    xs = [item["dx"] for item in diagnostics]
    ys = [item["dy"] for item in diagnostics]
    labels = [item["angle"] for item in diagnostics]
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.scatter(xs, ys, c=labels, cmap="turbo")
    for x, y, label in zip(xs, ys, labels):
        ax.text(x, y, str(label), fontsize=8)
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.axvline(0, color="0.5", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("dx (pixels)")
    ax.set_ylabel("dy (pixels)")
    ax.set_title("Estimated shifts")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "rotation_images"
    output_dir = base_dir / "outputs" / "rotate_calibration"
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Put angle_XXX.bmp images in {data_dir} before running rotate_calibration.py."
        )
    run_rotate_calibration(data_dir, output_dir)
