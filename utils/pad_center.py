"""Center pad an array with zeros."""

import numpy as np


def pad_center(input_array, output_size):
    out_h, out_w = int(output_size[0]), int(output_size[1])
    output_shape = (out_h, out_w) + tuple(input_array.shape[2:])
    output_array = np.zeros(output_shape, dtype=input_array.dtype)
    in_h, in_w = input_array.shape[:2]
    start_h = (out_h - in_h) // 2
    start_w = (out_w - in_w) // 2
    output_array[start_h : start_h + in_h, start_w : start_w + in_w, ...] = input_array
    return output_array
