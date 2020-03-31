import numpy as np

from pew.calc import cwt, local_maxima, ricker_wavelet, sliding_window_centered

from typing import Tuple


PEAK_DTYPE = np.dtype(
    {
        "names": ["height", "width", "area", "base", "top", "bottom", "left", "right"],
        "formats": [float, float, float, float, int, int, int, int],
    }
)


def _identify_ridges(
    cwt_coef: np.ndarray, windows: np.ndarray, gap_threshold: int = None
) -> np.ndarray:
    if gap_threshold is None:
        gap_threshold = len(windows) // 4

    maxima = local_maxima(cwt_coef[-1])
    ridges = np.full((cwt_coef.shape[0], maxima.size), -1, dtype=int)
    ridges[-1] = maxima

    for i in np.arange(cwt_coef.shape[0] - 2, -1, -1):  # Start from second last row
        maxima = local_maxima(cwt_coef[i])

        idx = np.searchsorted(maxima, ridges[i + 1])
        idx1 = np.clip(idx, 0, maxima.size - 1)
        idx2 = np.clip(idx - 1, 0, maxima.size - 1)

        diff1 = maxima[idx1] - ridges[i + 1]
        diff2 = ridges[i + 1] - maxima[idx2]

        min_diffs = np.where(diff1 <= diff2, idx1, idx2)

        ridges[i] = np.where(
            np.abs(ridges[i + 1] - maxima[min_diffs]) <= windows[i] // 4,
            maxima[min_diffs],
            -1,
        )
        maxima[min_diffs] = -1

        remaining_maxima = maxima[maxima > -1]
        if remaining_maxima.size != 0:
            new_ridges = np.full(
                (cwt_coef.shape[0], remaining_maxima.shape[0]), -1, dtype=int
            )
            new_ridges[i] = remaining_maxima
            ridges = np.hstack((ridges, new_ridges))

    return ridges


def _filter_ridges(
    ridges: np.ndarray,
    cwt_coef: np.ndarray,
    min_length: int = None,
    noise_window: int = 100,
    min_noise: float = None,
    min_snr: float = 10.0,
) -> np.ndarray:
    if min_noise is None:
        min_noise = np.amax(np.abs(cwt_coef[0])) / 100.0
    if min_length is None:
        min_length = ridges.shape[0] // 2

    # Trim ridges that are to short
    ridge_lengths = np.count_nonzero(ridges > -1, axis=0)
    ridges = ridges[:, ridge_lengths > min_length]

    # Build array of ridge values, filter out non valid ridges
    values = np.take_along_axis(cwt_coef, ridges, axis=1)
    max_rows = np.argmax(np.where(ridges > -1, values, 0), axis=0)
    max_cols = np.take_along_axis(ridges, max_rows[np.newaxis, :], axis=0)[0]

    col_order = np.argsort(max_cols)

    max_coords = np.vstack((max_rows[col_order], max_cols[col_order]))

    # Reducing number of windows here improves performance
    windows = sliding_window_centered(cwt_coef[0], noise_window, 1)[max_coords[1]]
    signals = cwt_coef[max_coords[0], max_coords[1]]
    noises = np.percentile(np.abs(windows), 10, axis=1)
    noises[noises < min_noise] = min_noise

    snrs = signals / noises

    ridges = ridges[:, col_order][:, snrs > min_snr]
    max_coords = max_coords[:, snrs > min_snr]

    return ridges, max_coords


def _peak_data_from_ridges(
    x: np.ndarray,
    ridges: np.ndarray,
    maxima_coords: np.ndarray,
    cwt_windows: np.ndarray,
    base_method: str = "baseline",
    height_method: str = "cwt",
    width_factor: float = 2.5,
) -> np.ndarray:
    widths = (np.take(cwt_windows, maxima_coords[0]) * width_factor).astype(int)

    lefts = np.clip(maxima_coords[1] - widths // 2, 0, x.size - 1)
    rights = np.clip(maxima_coords[1] + widths // 2, 0, x.size - 1)

    indicies = lefts + np.arange(np.amax(widths) + 1)[:, None]
    indicies = np.where(indicies - lefts < widths, indicies, rights)

    if height_method == "cwt":  # Height at maxima cwt ridge
        tops = maxima_coords[1]
    elif height_method == "maxima":
        tops = np.argmax(x[indicies], axis=0) + lefts
    else:
        raise ValueError("Valid values for height_method are 'cwt', 'maxima'.")

    if base_method == "baseline":
        bottoms = tops  # Default to tops
        windows = sliding_window_centered(x, np.amax(widths) * 4)
        bases = np.percentile(windows[bottoms], 25, axis=1)
    elif base_method == "edge":
        bottoms = np.minimum(lefts, rights)
        bases = x[bottoms]
    elif base_method == "minima":
        bottoms = np.argmin(x[indicies], axis=0) + lefts
        bases = x[bottoms]
    elif base_method == "prominence":
        bottoms = np.maximum(lefts, rights)
        bases = x[bottoms]
    elif base_method == "zero":
        bottoms = tops  # Default to tops
        bases = 0.0
    else:
        raise ValueError(
            "Valid values for base_method are 'baseline', "
            "'edge', 'prominence', 'minima', 'zero'."
        )

    area = np.trapz(x[indicies] - bases, indicies, axis=0)

    peaks = np.empty(tops.shape, dtype=PEAK_DTYPE)
    peaks["area"] = area
    peaks["height"] = x[tops] - bases
    peaks["width"] = widths
    peaks["base"] = bases
    peaks["top"] = tops
    peaks["bottom"] = bottoms
    peaks["left"] = lefts
    peaks["right"] = rights
    return peaks


def _filter_peaks(
    peaks: np.ndarray,
    min_area: float = 0.0,
    min_height: float = 0.0,
    min_width: float = 0.0,
) -> np.ndarray:
    bad_area = peaks["area"] < min_area
    bad_heights = peaks["height"] < min_height
    bad_widths = peaks["width"] < min_width
    bad_peaks = np.logical_or.reduce((bad_area, bad_heights, bad_widths))

    return peaks[~bad_peaks]


def find_peaks(
    x: np.ndarray,
    min_midth: int,
    max_width: int,
    peak_base_method: str = "baseline",
    peak_height_method: str = "cwt",
    peak_width_factor: float = 2.5,
    peak_min_area: float = 0.0,
    peak_min_height: float = 0.0,
    peak_min_width: float = 0.0,
    ridge_gap_threshold: int = None,
    ridge_min_snr: float = 9.0,
) -> np.ndarray:

    windows = np.arange(min_midth, max_width)
    cwt_coef = cwt(x, windows, ricker_wavelet)
    ridges = _identify_ridges(cwt_coef, windows, gap_threshold=ridge_gap_threshold)
    ridges, ridge_maxima = _filter_ridges(
        ridges, cwt_coef, noise_window=windows[-1] * 4, min_snr=ridge_min_snr
    )

    peaks = _peak_data_from_ridges(
        x,
        ridges,
        ridge_maxima,
        windows,
        base_method=peak_base_method,
        height_method=peak_height_method,
        width_factor=peak_width_factor,
    )

    peaks = _filter_peaks(
        peaks,
        min_area=peak_min_area,
        min_height=peak_min_height,
        min_width=peak_min_width,
    )

    return peaks


def bin_and_bound_peaks(
    peaks: np.ndarray,
    data_size: int,
    bin_size: int,
    peaks_per_bin: int = 1,
    offset: int = 0,
) -> np.ndarray:
    """Bins peaks and ensures that there is 1 peak per bin. If less a zero
     area peak is added, if more the largest peak in the bin is used."""
    bins = np.arange(0, data_size, bin_size)
    idx = np.searchsorted(bins, peaks["top"]) - 1

    bound_peaks = np.zeros((data_size // bin_size) * peaks_per_bin, dtype=peaks.dtype)
    bound_peaks["top"] = np.arange(
        offset, data_size + offset, bin_size // peaks_per_bin
    )
    for i in np.arange(data_size // bin_size):
        bin_peaks = peaks[idx == i]
        if bin_peaks.size > peaks_per_bin:
            bin_peaks = bin_peaks[np.argpartition(bin_peaks["area"], -peaks_per_bin)][
                -peaks_per_bin:
            ]
            bin_peaks.sort(order="top")

        n = i * peaks_per_bin
        bound_peaks[n : n + bin_peaks.size] = bin_peaks

    return bound_peaks


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
        peaks = find_peaks(lines.ravel(), min_width, max_width, **find_peak_kws)
        peaks = bin_and_bound_peaks(
            peaks, lines.size, lines.shape[1], offset=lines.shape[1] // 2
        )
        return peaks["area"].reshape(shape)

    # Structured
    spots = np.empty(shape, lines.dtype)
    for name in lines.dtype.names:
        peaks = find_peaks(lines[name].ravel(), min_width, max_width, **find_peak_kws)
        peaks = bin_and_bound_peaks(
            peaks, lines.size, lines.shape[1], offset=lines.shape[1] // 2
        )
        spots[name] = peaks["area"].reshape(shape)
    return spots
