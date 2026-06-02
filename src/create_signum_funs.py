"""Generate Hilbert-transform sign functions for rotated illumination angles."""

import numpy as np


def create_signum_funs(N, M, rotation_direction="cw", initial_angle=0.0):
    half = N // 2
    x, y = np.meshgrid(np.arange(-half, -half + N), np.arange(-half, -half + N))
    signum_funs = []

    angle_step = 360.0 / M if M else 0.0
    direction_sign = -1.0 if str(rotation_direction).lower() == "cw" else 1.0
    for idx in range(M):
        theta = np.deg2rad(float(initial_angle) + direction_sign * idx * angle_step)
        signum_funs.append(np.sign(x * np.cos(theta) + y * np.sin(theta)))
    return signum_funs

