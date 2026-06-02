"""Nonlocal KK holography simulation.

The script builds a complex phase object from a USAF target image, propagates it
to the sensor plane, simulates Fourier-filtered intensity measurements, performs
single-shot KK reconstruction, back propagates each recovered spectrum, and
combines the results by weighted Fourier synthesis.

Only PNG figures are saved. The only phase-alignment modes included here are
"none" and "constant".
"""

from pathlib import Path
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from src import gpu_backend
from src.create_fourier_masks import create_fourier_masks
from src.create_signum_funs import create_signum_funs
from src.fourier_ops import ft2c, ift2c
from src.imaging_simulation import imaging_simulation
from src.load_initial_image import load_initial_image
from src.phase_utils import unwrap_centered_phase
from src.reconstruction import ift2c_batch, reconstruction_batch
from src.synthesize import synthesize


# ----------------------------- User parameters -----------------------------
LAMBDA_UM = 0.675
M = 12
N = 256
DMIN_UM = 3.45
Z_UM = 3000.0

OBJECT_NA_CONSTRAINT = 0.20
NA_THRESHOLD = 0.41
NA_ILLUMINATION = 0.41
MASK_TYPE = "measurement"  # "ideal" or "measurement"

# Used only when MASK_TYPE="measurement". If measured CSV files are present in
# data/measurement, they are used to build the transmission curve. Otherwise an
# analytic broadened cutoff is used.
ILLUMINATION_FWHM_NM = 2.0
MEASUREMENT_EDGE_SOFTNESS = 0.006
MEASUREMENT_SAMPLE_COUNT = 121

SUBSAMPLING_RATE = 1
NOISE_INTENSITY = 0.002
USE_PHASE_UNWRAP = True
USE_BATCH_RECONSTRUCTION = True
USE_SINGLE_PRECISION = os.environ.get("KK_SINGLE_PRECISION", "1").lower() not in {
    "0",
    "false",
    "no",
    "off",
}

# Available synthesis alignment modes for this script.
SYNTHESIS_PHASE_ALIGN_MODES = ("none", "constant")


def synchronize_backend():
    if not gpu_backend.is_enabled():
        return
    try:
        if gpu_backend.torch.cuda.is_available():
            gpu_backend.torch.cuda.synchronize()
    except Exception:
        pass


def build_propagator(lambda_um, z_um, fx, fy):
    """Angular-spectrum free-space propagation kernel."""
    return np.exp(
        1j
        * 2.0
        * np.pi
        * z_um
        / lambda_um
        * np.sqrt(np.maximum(0.0, 1.0 - (lambda_um * fx) ** 2 - (lambda_um * fy) ** 2))
    )


def phase_limits(*phase_maps):
    max_abs = max(float(np.max(np.abs(phase))) for phase in phase_maps)
    return -max_abs, max_abs


def get_display_phase(field, use_phase_unwrap):
    return unwrap_centered_phase(field) if use_phase_unwrap else np.angle(field)


def make_measurement_masks(lambda_um, m, fx, fy):
    return create_fourier_masks(
        lambda_um,
        m,
        fx,
        fy,
        NA_THRESHOLD,
        NA_ILLUMINATION,
        mask_type=MASK_TYPE,
        illumination_fwhm_nm=ILLUMINATION_FWHM_NM,
        measurement_edge_softness=MEASUREMENT_EDGE_SOFTNESS,
        measurement_sample_count=MEASUREMENT_SAMPLE_COUNT,
    )


def save_single_shot_figures(output_dir, measurement_masks, captured_intensities, object_phase_maps, f_kk_fields_obj):
    fig1, axes1 = plt.subplots(3, M, figsize=(4 * M, 9))
    axes1 = np.atleast_2d(axes1)
    for idx in range(M):
        axes1[0, idx].imshow(measurement_masks[idx], cmap="gray")
        axes1[0, idx].set_title(f"Fourier mask {idx + 1}")
        axes1[0, idx].axis("off")

        im_intensity = axes1[1, idx].imshow(captured_intensities[idx], cmap="gray")
        axes1[1, idx].set_title(f"Captured intensity {idx + 1}")
        axes1[1, idx].axis("off")
        fig1.colorbar(im_intensity, ax=axes1[1, idx], fraction=0.046, pad=0.04)

        phase_map = object_phase_maps[idx]
        vmin, vmax = phase_limits(phase_map)
        im_phase = axes1[2, idx].imshow(phase_map, cmap="magma", vmin=vmin, vmax=vmax)
        axes1[2, idx].set_title(f"Back-propagated phase {idx + 1}")
        axes1[2, idx].axis("off")
        fig1.colorbar(im_phase, ax=axes1[2, idx], fraction=0.046, pad=0.04)
    fig1.tight_layout()
    fig1.savefig(output_dir / "01_masks_captured_backprop.png", dpi=200)

    fig2, axes2 = plt.subplots(2, M, figsize=(4 * M, 6))
    axes2 = np.atleast_2d(axes2)
    for idx in range(M):
        phase_map = object_phase_maps[idx]
        vmin, vmax = phase_limits(phase_map)
        im_phase = axes2[0, idx].imshow(phase_map, cmap="magma", vmin=vmin, vmax=vmax)
        axes2[0, idx].set_title(f"Object phase {idx + 1}")
        axes2[0, idx].axis("off")
        fig2.colorbar(im_phase, ax=axes2[0, idx], fraction=0.046, pad=0.04)

        axes2[1, idx].imshow(np.log10(np.abs(f_kk_fields_obj[idx]) + 1e-12), cmap="viridis")
        axes2[1, idx].set_title(f"Object spectrum {idx + 1}")
        axes2[1, idx].axis("off")
    fig2.tight_layout()
    fig2.savefig(output_dir / "02_single_shot_reconstruction.png", dpi=200)


def save_synthesis_figures(output_dir, mode, syn_obj, syn_f_obj, obj):
    syn_phase = get_display_phase(syn_obj, USE_PHASE_UNWRAP)
    gt_phase = get_display_phase(obj, USE_PHASE_UNWRAP)
    final_phase_vmin, final_phase_vmax = phase_limits(syn_phase, gt_phase)

    fig3, axes3 = plt.subplots(2, 3, figsize=(12, 8))
    im_rec_phase = axes3[0, 0].imshow(syn_phase, cmap="magma", vmin=final_phase_vmin, vmax=final_phase_vmax)
    axes3[0, 0].set_title(f"Synthesized phase ({mode})")
    fig3.colorbar(im_rec_phase, ax=axes3[0, 0], fraction=0.046, pad=0.04)

    im_rec_amp = axes3[0, 1].imshow(np.abs(syn_obj), cmap="gray")
    axes3[0, 1].set_title("Synthesized amplitude")
    fig3.colorbar(im_rec_amp, ax=axes3[0, 1], fraction=0.046, pad=0.04)

    axes3[0, 2].imshow(np.log10(np.abs(syn_f_obj) + 1e-12), cmap="viridis")
    axes3[0, 2].set_title("Synthesized Fourier spectrum")

    im_gt_phase = axes3[1, 0].imshow(gt_phase, cmap="magma", vmin=final_phase_vmin, vmax=final_phase_vmax)
    axes3[1, 0].set_title("Ground-truth phase")
    fig3.colorbar(im_gt_phase, ax=axes3[1, 0], fraction=0.046, pad=0.04)

    im_gt_amp = axes3[1, 1].imshow(np.abs(obj), cmap="gray")
    axes3[1, 1].set_title("Ground-truth amplitude")
    fig3.colorbar(im_gt_amp, ax=axes3[1, 1], fraction=0.046, pad=0.04)

    axes3[1, 2].imshow(np.log10(np.abs(ft2c(obj)) + 1e-12), cmap="viridis")
    axes3[1, 2].set_title("Ground-truth Fourier spectrum")

    for ax in axes3.ravel():
        ax.axis("off")
    fig3.tight_layout()
    fig3.savefig(output_dir / f"03_final_comparison_{mode}.png", dpi=200)

    fig4 = plt.figure(figsize=(10, 4))
    plt.plot(syn_phase[N // 2, :], linewidth=2.0, label=f"Synthesized phase ({mode})")
    plt.plot(gt_phase[N // 2, :], linewidth=2.0, label="Ground-truth phase")
    plt.title("Center-line phase comparison at object plane")
    plt.xlabel("y")
    plt.ylabel("Phase")
    plt.legend()
    plt.tight_layout()
    fig4.savefig(output_dir / f"04_center_line_phase_{mode}.png", dpi=200)


def main():
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "outputs" / "main_simu"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Array backend: {gpu_backend.backend_name()}")

    fmax = 1.0 / DMIN_UM
    fmin = fmax / N
    f = np.arange(-fmax / 2.0, fmax / 2.0, fmin)
    fx, fy = np.meshgrid(f, f)

    obj = load_initial_image(N, phase_scale=0.5 * np.pi, image_path=base_dir / "USAF-1951.png")
    obj_na_constraint = create_fourier_masks(LAMBDA_UM, 1, fx, fy, OBJECT_NA_CONSTRAINT, 0.0, mask_type="ideal")[0]
    obj = ift2c(obj_na_constraint * ft2c(obj))

    propagator_forward = build_propagator(LAMBDA_UM, Z_UM, fx, fy)
    propagator_backward = np.conj(propagator_forward)
    sensor_field = ift2c(ft2c(obj) * propagator_forward)

    measurement_masks = make_measurement_masks(LAMBDA_UM, M, fx, fy)
    captured_intensities = [
        imaging_simulation(sensor_field, mask, N, SUBSAMPLING_RATE, NOISE_INTENSITY)
        for mask in measurement_masks
    ]

    signum_funs = create_signum_funs(N, M)
    synchronize_backend()
    kk_start_s = time.perf_counter()
    if USE_BATCH_RECONSTRUCTION:
        f_kk_fields_sensor_stack = reconstruction_batch(
            np.stack(captured_intensities, axis=0),
            np.stack(signum_funs, axis=0),
            np.stack(measurement_masks, axis=0),
            N,
        )
        f_kk_fields_sensor = [f_kk_fields_sensor_stack[idx] for idx in range(M)]
    else:
        from src.reconstruction import reconstruction

        f_kk_fields_sensor = [
            reconstruction(captured, signum_fun, mask, N)
            for captured, signum_fun, mask in zip(captured_intensities, signum_funs, measurement_masks)
        ]
    synchronize_backend()
    kk_reconstruction_time_s = time.perf_counter() - kk_start_s

    synchronize_backend()
    backprop_start_s = time.perf_counter()
    f_kk_fields_obj = [field * propagator_backward for field in f_kk_fields_sensor]
    kk_fields_obj = ift2c_batch(np.stack(f_kk_fields_obj, axis=0))
    kk_fields_obj = [kk_fields_obj[idx] for idx in range(M)]
    synchronize_backend()
    backprop_time_s = time.perf_counter() - backprop_start_s

    object_phase_maps = [get_display_phase(field, USE_PHASE_UNWRAP) for field in kk_fields_obj]
    save_single_shot_figures(output_dir, measurement_masks, captured_intensities, object_phase_maps, f_kk_fields_obj)

    print(f"Mask type: {MASK_TYPE}")
    print(f"NA threshold: {NA_THRESHOLD}")
    print(f"NA illumination: {NA_ILLUMINATION}")
    print(f"Illumination FWHM (nm): {ILLUMINATION_FWHM_NM}")
    print(f"KK reconstruction time for {M} frames: {kk_reconstruction_time_s:.4f} s")
    print(f"Back propagation / IFFT time for {M} frames: {backprop_time_s:.4f} s")

    for mode in SYNTHESIS_PHASE_ALIGN_MODES:
        synchronize_backend()
        synthesis_start_s = time.perf_counter()
        syn_obj, syn_f_obj, alignment = synthesize(
            f_kk_fields_obj,
            measurement_masks,
            M,
            phase_align_mode=mode,
        )
        synchronize_backend()
        synthesis_time_s = time.perf_counter() - synthesis_start_s
        save_synthesis_figures(output_dir, mode, syn_obj, syn_f_obj, obj)
        print(f"Synthesis phase alignment mode: {mode}")
        print(f"Synthesis time ({mode}): {synthesis_time_s:.4f} s")
        print(f"Phase coefficients ({mode}) [phi0, sx, sy]: {np.round(alignment['coefficients'], 6).tolist()}")

    if os.environ.get("SHOW_PLOTS", "0") == "1":
        plt.show()
    else:
        plt.close("all")
    print(f"Saved PNG figures to: {output_dir}")


if __name__ == "__main__":
    main()
