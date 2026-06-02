"""Center crop an array."""


def crop_center(input_array, output_size):
    out_h, out_w = int(output_size[0]), int(output_size[1])
    in_h, in_w = input_array.shape[:2]
    start_h = (in_h - out_h) // 2
    start_w = (in_w - out_w) // 2
    return input_array[start_h : start_h + out_h, start_w : start_w + out_w, ...]

