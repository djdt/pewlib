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
    cwt_coef: np.ndarray, windows: np.ndarray, gap_threshold: int = None, mode="maxima",
) -> np.ndarray:
    if gap_threshold is None:
        gap_threshold = len(windows) // 4

    extrema = local_extrema(cwt_coef[-1], windows[-1] * 2, mode=mode)
    ridges = np.full((cwt_coef.shape[0], extrema.size), -1, dtype=int)
    ridges[-1] = extrema

    for i in np.arange(cwt_coef.shape[0] - 2, -1, -1):  # Start from second last row
        extrema = local_extrema(cwt_coef[i], windows[i] * 2, mode=mode)

        idx = np.searchsorted(extrema, ridges[i + 1])
        idx1 = np.clip(idx, 0, extrema.size - 1)
        idx2 = np.clip(idx - 1, 0, extrema.size - 1)

        diff1 = extrema[idx1] - ridges[i + 1]
        diff2 = ridges[i + 1] - extrema[idx2]

        min_diffs = np.where(diff1 <= diff2, idx, idx2)

        ridges[i] = np.where(
            np.abs(ridges[i + 1] - extrema[min_diffs]) <= windows[i] // 4,
            extrema[min_diffs],
            -1,
        )
        extrema[min_diffs] = -1

        remaining_extrema = extrema[extrema > -1]
        if remaining_extrema.size != 0:
            new_ridges = np.full(
                (cwt_coef.shape[0], remaining_extrema.shape[0]), -1, dtype=int
            )
            new_ridges[i] = remaining_extrema
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
    max_coords = np.vstack((max_rows, max_cols))

    # Reducing number of windows here improves performance
    windows = sliding_window_centered(cwt_coef[0], noise_window, 1)[max_coords[1]]
    signals = cwt_coef[max_coords[0], max_coords[1]]
    noises = np.percentile(np.abs(windows), 10, axis=1)
    noises[noises < min_noise] = min_noise

    snrs = signals / noises

    return ridges[:, snrs > min_snr], max_coords[:, snrs > min_snr]


# TODO Change integ method to peak_base_method
def _peak_data_from_ridges(
    x: np.ndarray,
    ridges: np.ndarray,
    maxima_coords: np.ndarray,
    cwt_windows: np.ndarray,
    peak_base_method: str = "baseline",
    peak_height_method: str = "maxima",
) -> np.ndarray:
    widths = np.take(cwt_windows, maxima_coords[0])

    lefts = np.clip(maxima_coords[1] - widths, 0, x.size - 1)
    rights = np.clip(maxima_coords[1] + widths, 0, x.size - 1)
    bases = np.minimum(lefts, rights)

    if peak_height_method == "cwt":  # Height at maxima cwt ridge
        tops = maxima_coords[1]
    elif peak_height_method == "maxima":  # Max data height inside peak width
        ranges = np.vstack((lefts, rights)).T
        # PYTHON LOOP HERE
        tops = np.array([np.argmax(x[r[0] : r[1]]) + r[0] for r in ranges], dtype=int)
    else:
        raise ValueError("Valid peak_height_method are 'cwt', 'maxima'.")

    if peak_base_method == "minima":
        bottoms = ()
    elif peak_base_method == "lowest_edge":
        bottoms = np.minimum(lefts, rights)
        bases = x[bottoms]
    elif peak_base_method == "zero":
        bottoms = np.zeros(tops.shape, dtype=int)  # Uninitialised
        bases = np.zeros(bottoms.shape, dtype=float)
    elif peak_base_method == "prominence":
        bottoms = np.maximum(lefts, rights)
        bases = x[bottoms]
    elif peak_base_method == "baseline":
        # TODO Percentile method here
        pass
    else:
        raise ValueError(
            "Valid values for peak_base_method are 'baseline', "
            "'edge', 'prominence', 'minima', 'zero'."
        )

    if peak_integration_method == "base":
        ibases = x[bases]
    elif peak_integration_method == "prominence":
        ibases = np.maximum(x[lefts], x[rights])
    else:
        raise ValueError("Valid peak_integration_method are 'base', 'prominence'.")

    # TODO possible improvement here
    area = np.array([np.trapz(x[r[0] : r[1]] - ib) for r, ib in zip(ranges, ibases)])

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


# TODO Add a way of setting intergration to baseline, probably looking a a large region low percentile around each peak
def find_peaks(
    x: np.ndarray,
    min_midth: int,
    max_width: int,
    distance: int = None,
    min_number: int = None,
    max_number: int = None,
    peak_integration_method: str = "base",
    peak_min_area: float = 0.0,
    peak_min_height: float = 0.0,
    peak_min_prominence: float = 0.0,
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
        peak_height_method="maxima",
        peak_integration_method=peak_integration_method,
    )

    # Filter the peaks based on final criteria
    bad_area = peaks["area"] < peak_min_area
    bad_heights = peaks["height"] < peak_min_height
    bad_widths = peaks["width"] < peak_min_width
    bad_prominences = (
        peaks["height"] - np.maximum(x[peaks["left"]], x[peaks["right"]])
        < peak_min_prominence
    )
    bad_peaks = np.logical_or.reduce(
        (bad_area, bad_heights, bad_widths, bad_prominences)
    )

    return peaks[~bad_peaks]


# TODO Generalise this function so that it can take multiple peaks per bin, etc
def bin_and_bound_peaks(
    peaks: np.ndarray, data_size: int, bin_size: int, offset: int = 0,
) -> np.ndarray:
    """Bins peaks and ensures that there is 1 peak per bin. If less a zero
     area peak is added, if more the largest peak in the bin is used."""
    bins = np.arange(0, data_size, bin_size)
    idx = np.searchsorted(bins, peaks["top"]) - 1

    bound_peaks = np.zeros(data_size // bin_size, dtype=peaks.dtype)
    bound_peaks["top"] = np.arange(offset, data_size + offset, bin_size)
    for i in np.arange(data_size // bin_size):
        n = np.count_nonzero(idx == i)
        if n != 0:
            bin_peaks = peaks[idx == i]
            bound_peaks[i] = bin_peaks[np.argmax(bin_peaks["area"])]

    return bound_peaks


def _lines_to_spots(
    lines: np.ndarray,
    shape: Tuple[int, ...],
    min_width: int,
    max_width: int,
    find_peak_kws: dict = None,
) -> np.ndarray:
    assert np.prod(shape) == lines.shape[0]

    if find_peak_kws is None:
        find_peak_kws = {}
    peaks = find_peaks(lines.ravel(), min_width, max_width, **find_peak_kws)
    peaks = bin_and_bound_peaks(
        peaks, lines.size, lines.shape[1], offset=lines.shape[1] // 2
    )
    return peaks["area"].reshape(shape)


def lines_to_spots(
    lines: np.ndarray,
    shape: Tuple[int, ...],
    min_width: int,
    max_width: int,
    find_peak_kws: dict = None,
) -> np.ndarray:
    if lines.dtype.names is None:
        return _lines_to_spots(lines, shape, min_width, max_width, find_peak_kws)

    spots = np.empty(shape, lines.dtype)
    for name in lines.dtype.names:
        spots[name] = _lines_to_spots(
            lines[name], shape, min_width, max_width, find_peak_kws
        )
    return spots
