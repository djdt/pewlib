import numpy as np

from pew.lib.peakfinding import find_peaks_cwt, bin_and_bound_peaks

from typing import Tuple


def lines_to_spots(
    lines: np.ndarray,
    shape: Tuple[int, ...],
    min_width: int,
    max_width: int,
    find_peak_kws: dict = None,
) -> np.ndarray:
    """Convert lines with x 'peaks_per_line'to a 2d array with 'shape'."""
    assert np.prod(shape) == lines.shape[0]

    # Unstructured
    if lines.dtype.names is None:
        peaks = find_peaks_cwt(lines.ravel(), min_width, max_width, **find_peak_kws)
        peaks = bin_and_bound_peaks(
            peaks, lines.size, lines.shape[1], offset=lines.shape[1] // 2
        )
        return peaks["area"].reshape(shape)

    # Structured
    spots = np.empty(shape, lines.dtype)
    for name in lines.dtype.names:
        peaks = find_peaks_cwt(lines[name].ravel(), min_width, max_width, **find_peak_kws)
        peaks = bin_and_bound_peaks(
            peaks, lines.size, lines.shape[1], offset=lines.shape[1] // 2
        )
        spots[name] = peaks["area"].reshape(shape)
    return spots
