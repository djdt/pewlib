import numpy as np

from pew.calc import moving_average_filter

from typing import Tuple


def find_peak(
    x: np.ndarray, gradient: float = None, height: float = None
) -> Tuple[int, int]:
    gradient = gradient or np.max(x) / 10.0
    height = height or np.min(x)

    gradients = np.gradient(x)

    start = np.argmax(np.logical_and(gradients > gradient, x >= height))
    end = x.size - np.argmax(
        np.logical_and(gradients[::-1] < -gradient, x[::-1] >= height)
    )

    if start == 0 or end == x.size:
        return 0, 0
    return start, end


def smooth_find_and_integrate(
    x: np.ndarray, smoothing: int = None, find_peak_kws: dict = None
) -> float:
    if smoothing is not None and smoothing > 1:
        x = moving_average_filter(x, smoothing)

    if find_peak_kws is None:
        find_peak_kws = {}

    start, end = find_peak(x, **find_peak_kws)

    return np.trapz(x[start:end])


def lines_to_spots(
    data: np.ndarray,
    shape: Tuple[int, ...],
    smoothing: int = None,
    gradient: float = None,
    height: float = None,
) -> np.ndarray:
    spot_data = np.empty(np.prod(shape), dtype=data.dtype)

    peak_kws = dict(gradient=gradient, height=height)

    for name in data.dtype.names:
        spot_data[name] = np.apply_along_axis(
            smooth_find_and_integrate, 1, data[name], smoothing, peak_kws
        )

    return spot_data.reshape(shape)
