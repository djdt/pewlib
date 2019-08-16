import numpy as np

from typing import List, Tuple


def subpixel_offset(
    x: np.ndarray, offsets: List[Tuple[int, int]], pixelsize: Tuple[int, int]
) -> np.ndarray:
    """Takes a 3d array and stretches and offsets each layer.

    Given an offset of (1,1) and pixelsize of (2,2) each layer will be streched by 2
    and every even layer will be shifted by 1 pixel.

    Args:
        offsets: The pixel offsets in (x, y).
        pixelsize: Final size to stretch to.

    Returns:
        The offset array.
    """
    # Offset for first layer must be zero
    if offsets[0] != (0, 0):
        offsets.insert(0, (0, 0))
    overlap = np.max(offsets, axis=0)

    if x.ndim != 3:
        raise ValueError("Data must be three dimensional!")

    # Calculate new shape
    new_shape = np.array(x.shape[:2]) * pixelsize + overlap
    # Create empty array to store data in
    data = np.zeros((*new_shape, x.shape[2]), dtype=x.dtype)

    for i in range(0, x.shape[2]):
        # Cycle through offsets
        start = offsets[i % len(offsets)]
        end = -(overlap[0] - start[0]) or None, -(overlap[1] - start[1]) or None
        # Stretch arrays as required
        data[start[0] : end[0], start[1] : end[1], i] = np.repeat(
            x[:, :, i], pixelsize[0], axis=0
        ).repeat(pixelsize[1], axis=1)

    return data


def subpixel_offset_equal(
    x: np.ndarray, offsets: List[int], pixelsize: int
) -> np.ndarray:
    return subpixel_offset(x, [(o, o) for o in offsets], (pixelsize, pixelsize))
