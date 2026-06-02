"""Load the USAF image and convert it into a complex phase object."""

from pathlib import Path

import numpy as np
from PIL import Image


Image.MAX_IMAGE_PIXELS = None


def load_initial_image(N, image_path=None, phase_scale=0.3 * np.pi, amplitude_mode="ones"):
    if image_path is None:
        image_path = Path(__file__).resolve().parent.parent / "USAF-1951.png"

    with Image.open(image_path) as pil_image:
        image = pil_image.convert("F").resize((N, N), Image.Resampling.BILINEAR)
        image = np.asarray(image, dtype=np.float64)

    if image.max() > 1.0:
        image = image / 255.0

    phase = image - image.min()
    phase = phase / max(float(phase.max()), 1e-12)
    phase = phase * float(phase_scale)

    if amplitude_mode == "ones":
        amplitude = np.ones((N, N), dtype=np.float64)
    elif amplitude_mode == "half_offset":
        amplitude = 1.0 - image / 2.0
    else:
        raise ValueError(f"Unsupported amplitude_mode: {amplitude_mode}")

    return amplitude * np.exp(1j * phase)

