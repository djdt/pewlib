import numpy as np

from pew.calc import cwt, local_extrema, ricker_wavelet, sliding_window_centered


PEAK_DTYPE = np.dtype(
    {
        "names": ["height", "width", "area", "top", "base", "left", "right"],
        "formats": [float, float, float, int, int, int, int],
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
    peak_height_method: str = "maxima",
    peak_integration_method: str = "base",
) -> np.ndarray:
    widths = np.take(cwt_windows, maxima_coords[0]) * 2

    lefts = np.clip(maxima_coords[1] - widths // 2, 0, x.size - 1)
    rights = np.clip(maxima_coords[1] + widths // 2, 0, x.size - 1)

    indicies = lefts + np.arange(np.amax(widths) + 1)[:, None]
    indicies = np.where(indicies - lefts < widths, indicies, rights)
    bases = np.argmin(x[indicies], axis=0) + lefts

    if peak_height_method == "cwt":  # Height at maxima cwt ridge
        tops = maxima_coords[1]
    elif peak_height_method == "maxima":  # Max data height inside peak width
        ranges = np.vstack((lefts, rights)).T
        # PYTHON LOOP HERE
        tops = np.array([np.argmax(x[r[0] : r[1]]) + r[0] for r in ranges], dtype=int)
    else:
        raise ValueError("Valid peak_height_method are 'cwt', 'maxima'.")

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
    peaks["height"] = x[tops] - x[bases]
    peaks["width"] = widths
    peaks["top"] = tops
    peaks["base"] = bases
    peaks["left"] = lefts
    peaks["right"] = rights
    return peaks


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

    peaks = peaks[~bad_peaks]

    if max_number is not None and peaks.size > max_number:
        peaks = peaks[np.argsort(peaks["area"])][-max_number:]
    elif min_number is not None and peaks.size < min_number:
        if distance is None:
            distance = np.median(np.diff(peaks["top"]))
        peak_idx = (peaks["top"] // distance).astype(int) - 1
        new_peaks = np.zeros(min_number, dtype=peaks.dtype)
        new_peaks["top"] = np.arange(distance, distance * (min_number + 1), distance)

        new_peaks[peak_idx] = peaks
        peaks = new_peaks

    return peaks


def find_peaks_structured(
    x: np.ndarray,
    min_width: int,
    max_width: int,
    size: int,
    distance: int = None,
    peak_integration_method: str = "base",
    peak_min_area: float = 0.0,
    peak_min_height: float = 0.0,
    peak_min_prominence: float = 0.0,
    peak_min_width: float = 0.0,
    ridge_gap_threshold: int = None,
    ridge_min_snr: float = 9.0,
) -> np.ndarray:
    dtype = [(name, PEAK_DTYPE) for name in x.dtype.names]
    peaks = np.empty(size, dtype=dtype)

    for name in peaks.dtype.names:
        peaks[name] = find_peaks(
            x[name],
            min_width,
            max_width,
            distance=distance,
            min_number=size,
            max_number=size,
            peak_integration_method=peak_integration_method,
            peak_min_area=peak_min_area,
            peak_min_height=peak_min_height,
            peak_min_prominence=peak_min_prominence,
            peak_min_width=peak_min_width,
            ridge_gap_threshold=ridge_gap_threshold,
            ridge_min_snr=ridge_min_snr,
        )

    return peaks
