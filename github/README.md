# Nonlocal KK Holography Simulation

This folder contains a runnable simulation workflow, numerical helper modules,
local input data, and a rotation-calibration utility.

## Scripts

- `main_simu.py` builds a complex phase object from `USAF-1951.png`, simulates
  Fourier-filtered intensity measurements, reconstructs single-shot KK spectra,
  back propagates them to the object plane, and saves synthesized PNG figures.
- `rotate_calibration.py` estimates the rotation center from angle-indexed BMP
  images and writes aligned center crops plus a GIF preview.

## Simulation Parameters

Edit the parameter block at the top of `main_simu.py` to choose:

- `MASK_TYPE`: `ideal` or `measurement`
- `NA_THRESHOLD`
- `NA_ILLUMINATION`
- `ILLUMINATION_FWHM_NM`
- `SYNTHESIS_PHASE_ALIGN_MODES`: `none` and/or `constant`

## Data

- `USAF-1951.png` is used as the target image for the simulated phase object.
- `data/measurement/` stores optional measured transmission CSV files for
  `MASK_TYPE="measurement"`.

Expected measured CSV filenames:

- `dark_spectrum_20ms.csv`
- `source_spectrum_20ms_sPol.csv`
- `Thorlabs_FESH0700_20ms_0_0.5_70_sPolar.csv`
- `transmission_results_Center675nm_FWHM2nm.csv`

If the measured CSV files are available, the measurement mask uses the local
s-polar transmission data and the configured illumination FWHM. If they are not
available, the script uses an analytic broadened cutoff model.

## Run

```bash
python main_simu.py
```

PNG figures are written to `outputs/main_simu/`.
