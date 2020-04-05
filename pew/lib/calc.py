import numpy as np

from typing import Callable, List, Tuple


def local_maxima(x: np.ndarray) -> np.ndarray:
    return np.nonzero(
        np.logical_and(np.r_[True, x[1:] > x[:-1]], np.r_[x[:-1] > x[1:], True])
    )[0]


# def windowed_extrema(
#     x: np.ndarray, window: int, step: int = 1, mode: str = "maxima"
# ) -> np.ndarray:
#     windows = sliding_window_centered(x, window, step)
#     if mode == "minima":
#         extrema = np.argmin(windows, axis=1)
#     else:
#         extrema = np.argmax(windows, axis=1)
#     return np.nonzero(extrema == (window // 2))[0]


def cwt(
    x: np.ndarray, windows: np.ndarray, wavelet: Callable[..., np.ndarray]
) -> np.ndarray:
    cwt = np.empty((windows.shape[0], x.size), dtype=x.dtype)
    for i in range(cwt.shape[0]):
        n = np.amin((x.size, windows[i] * 10))
        cwt[i] = np.convolve(
            np.pad(x, (n // 2 - 1, n // 2), mode="edge"),
            wavelet(n, windows[i]),
            mode="valid",
        )
    return cwt


def ricker_wavelet(size: int, sigma: float) -> np.ndarray:
    x = np.linspace(-size / 2.0, size / 2.0, size)
    a = 2.0 / (np.sqrt(3.0 * sigma) * np.power(np.pi, 0.25))
    kernel = np.exp(-((0.5 * np.square(x) / np.square(sigma))))
    kernel = a * (1.0 - np.square(x) / np.square(sigma)) * kernel
    return kernel


def sliding_window(x: np.ndarray, window: int, step: int = 1) -> np.ndarray:
    shape = ((x.size - window) // step + 1, window)
    strides = (step * x.strides[0], x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def sliding_window_centered(
    x: np.ndarray, window: int, step: int = 1, mode: str = "edge"
) -> np.ndarray:
    x_pad = np.pad(x, (window // 2, window - window // 2 - 1), mode=mode)
    return sliding_window(x_pad, window, step)


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
