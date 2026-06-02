"""Phase display utilities."""

import numpy as np

try:
    from skimage.restoration import unwrap_phase as skimage_unwrap_phase
except ImportError:  # pragma: no cover
    skimage_unwrap_phase = None


def unwrap_phase_2d(phase):
    phase = np.asarray(phase, dtype=np.float64)
    if skimage_unwrap_phase is not None:
        return skimage_unwrap_phase(phase)
    return np.unwrap(np.unwrap(phase, axis=0), axis=1)


def unwrap_centered_phase(field_or_phase):
    phase = np.angle(field_or_phase) if np.iscomplexobj(field_or_phase) else np.asarray(field_or_phase)
    unwrapped = unwrap_phase_2d(phase)
    center = unwrapped[unwrapped.shape[0] // 2, unwrapped.shape[1] // 2]
    return unwrapped - center

