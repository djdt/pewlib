import numpy as np

from pewlib.process.calc import local_maxima, reset_cumsum, view_as_blocks

from typing import Callable, Tuple


PEAK_DTYPE = np.dtype(
    {
        "names": ["height", "width", "area", "base", "top", "bottom", "left", "right"],
        "formats": [float, float, float, float, int, int, int, int],
    }
)


def cwt(
    x: np.ndarray, windows: np.ndarray, wavelet: Callable[..., np.ndarray]
) -> np.ndarray:
    cwt = np.empty((windows.shape[0], x.size), dtype=x.dtype)
    for i in range(cwt.shape[0]):
        n = np.amin((x.size, windows[i] * 10))
        cwt[i] = np.convolve(x, wavelet(n, windows[i]), mode="same")
    return cwt


def ricker_wavelet(size: int, sigma: float) -> np.ndarray:
    x = np.linspace(-size / 2.0, size / 2.0, size)
    a = 2.0 / (np.sqrt(3.0 * sigma) * np.pi ** 0.25)
    return a * (1.0 - (x / sigma) ** 2) * np.exp(-(x ** 2 / (2.0 * sigma ** 2)))


def _cwt_identify_ridges(
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


def _cwt_filter_ridges(
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
    cwt_pad = np.pad(
        cwt_coef[0],
        (noise_window // 2, noise_window - noise_window // 2 - 1),
        mode="edge",
    )
    windows = view_as_blocks(cwt_pad, (noise_window,), (1,))[max_coords[1]]
    signals = cwt_coef[max_coords[0], max_coords[1]]
    noises = np.percentile(np.abs(windows), 10, axis=1)
    noises[noises < min_noise] = min_noise

    snrs = signals / noises

    ridges = ridges[:, col_order][:, snrs > min_snr]
    max_coords = max_coords[:, snrs > min_snr]

    return ridges, max_coords


def _zscore_peaks(
    x: np.ndarray, lag: int, threshold: float = 3.3, influence: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    signal = np.zeros(x.size, dtype=np.int8)
    filtered = x.copy()
    means = np.empty_like(x)
    means[:lag] = x[:lag]
    stds = np.zeros_like(x)

    for i in range(lag, x.shape[0]):
        means[i] = np.mean(filtered[i - lag : i])
        stds[i] = np.std(filtered[i - lag : i])

        if np.abs(x[i] - means[i]) > stds[i] * threshold:
            signal[i] = 1 if x[i] > means[i] else -1
            filtered[i] = influence * x[i] + (1.0 - influence) * filtered[i - 1]

    return signal, filtered, means, stds


def find_peaks_cwt(
    x: np.ndarray,
    min_midth: int,
    max_width: int,
    ridge_gap_threshold: int = None,
    ridge_min_snr: float = 9.0,
    width_factor: float = 2.5,
    peak_base_method: str = "baseline",
    peak_height_method: str = "maxima",
    peak_min_area: float = 0.0,
    peak_min_height: float = 0.0,
    peak_min_width: float = 0.0,
) -> np.ndarray:

    windows = np.arange(min_midth, max_width)
    cwt_coef = cwt(x, windows, ricker_wavelet)
    ridges = _cwt_identify_ridges(cwt_coef, windows, gap_threshold=ridge_gap_threshold)
    ridges, ridge_maxima = _cwt_filter_ridges(
        ridges, cwt_coef, noise_window=windows[-1] * 4, min_snr=ridge_min_snr
    )

    if ridges.size == 0:
        return np.array([], dtype=PEAK_DTYPE)

    widths = (np.take(windows, ridge_maxima[0]) * width_factor).astype(int)
    lefts = np.clip(ridge_maxima[1] - widths // 2, 0, x.size - 1)
    rights = np.clip(ridge_maxima[1] + widths // 2, 1, x.size)

    peaks = peaks_from_edges(
        x, lefts, rights, base_method=peak_base_method, height_method=peak_height_method
    )

    peaks = filter_peaks(
        peaks,
        min_area=peak_min_area,
        min_height=peak_min_height,
        min_width=peak_min_width,
    )

    return peaks


def find_peaks_zscore(
    x: np.ndarray,
    lag: int = 10,
    threshold: float = 3.3,
    influence: float = 0.5,
    peak_base_method: str = "baseline",
    peak_height_method: str = "maxima",
    peak_min_area: float = 0.0,
    peak_min_height: float = 0.0,
    peak_min_width: float = 0.0,
    use_cython: bool = False,
) -> np.ndarray:

    if use_cython:
        from pewlib.process.zscore import zscore_peaks

        signal, _ = zscore_peaks(x, lag, threshold, influence)
    else:
        signal, _ = _zscore_peaks(x, lag, threshold, influence)
    signal[signal < 0] = 0  # Only look at positive peaks

    lefts = np.nonzero(np.logical_and(signal[:-1] == 0, signal[1:] == 1))[0]
    rights = np.nonzero(np.logical_and(signal[1:] == 0, signal[:-1] == 1))[0] + 1
    lefts = lefts[: rights.size]  # In case peak overlaps end

    peaks = peaks_from_edges(
        x, lefts, rights, base_method=peak_base_method, height_method=peak_height_method
    )

    peaks = filter_peaks(
        peaks,
        min_area=peak_min_area,
        min_height=peak_min_height,
        min_width=peak_min_width,
    )

    return peaks


# def find_regularly_spaced_peaks(
#     x: np.ndarray,
#     domain: Tuple[int, int],
#     peak_spacing: float,
#     peak_max_width: int = None,
#     peak_search_dist: int = None,
#     peak_base_method: str = "baseline",
#     peak_height_method: str = "maxima",
#     peak_width_method: str = "minima",
# ) -> np.ndarray:
#     if peak_search_dist is None:
#         peak_search_dist = int(peak_spacing / 2.0)
#     if peak_max_width is None:
#         peak_max_width = peak_search_dist

#     ideal_locations = np.round(np.arange(domain[0], domain[1], peak_spacing)).astype(
#         int
#     )

#     if peak_height_method == "maxima":
#         search_lefts = ideal_locations - peak_search_dist // 2
#         search_indicies = search_lefts + np.arange(peak_search_dist)[:, None]
#         tops = np.argmax(x[search_indicies], axis=0) + search_lefts
#     elif peak_height_method == "none":
#         tops = ideal_locations
#     else:
#         raise ValueError("Valid values for height_method are 'maxima', 'none'.")

#     if peak_width_method == "minima":
#         indicies = (tops - peak_max_width // 2) + np.arange(peak_max_width // 2)[
#             :, None
#         ]
#         lefts = np.argmin(x[indicies], axis=0) + (tops - peak_max_width // 2)
#         print(lefts)
#         indicies = tops + np.arange(peak_max_width // 2)[:, None]
#         rights = np.argmin(x[indicies], axis=0) + tops
#     elif peak_width_method == "full":
#         lefts = tops - peak_max_width // 2
#         rights = tops + peak_max_width // 2
#     else:
#         raise ValueError("Valid values for width_method are 'minima', 'full'.")

#     if peak_base_method == "baseline":
#         bottoms = tops  # Default to tops
#         windows = sliding_window_centered(x, peak_search_dist * 2)
#         bases = np.percentile(windows[bottoms], 25, axis=1)
#     elif peak_base_method == "edge":
#         bottoms = np.minimum(lefts, rights)
#         bases = x[bottoms]
#     elif peak_base_method == "minima":
#         bottoms = np.argmin(x[indicies], axis=0) + lefts
#         bases = x[bottoms]
#     elif peak_base_method == "prominence":
#         bottoms = np.maximum(lefts, rights)
#         bases = x[bottoms]
#     elif peak_base_method == "zero":
#         bottoms = tops  # Default to tops
#         bases = 0.0
#     else:
#         raise ValueError(
#             "Valid values for base_method are 'baseline', "
#             "'edge', 'prominence', 'minima', 'zero'."
#         )

#     widths = rights - lefts
#     indicies = lefts + np.arange(peak_max_width + 1)[:, None]
#     indicies = np.clip(indicies, 0, x.size - 1)
#     indicies = np.where(indicies - lefts < widths, indicies, rights)
#     area = np.trapz(x[indicies] - bases, indicies, axis=0)

#     peaks = np.empty(tops.shape, dtype=PEAK_DTYPE)
#     peaks["area"] = area
#     peaks["height"] = x[tops] - bases
#     peaks["width"] = widths
#     peaks["base"] = bases
#     peaks["top"] = tops
#     peaks["bottom"] = bottoms
#     peaks["left"] = lefts
#     peaks["right"] = rights
#     return peaks


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
    )[: bound_peaks.size]
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


def insert_missing_peaks(
    peaks: np.ndarray,
    distance: float = None,
    param: str = "top",
    missing_peak_area: float = 0.0,
) -> np.ndarray:
    """Looks for gaps greater than 'distance' in peaks using the given parameter.
    If a gap is found then enough peaks are insert to ensure no gap remains.
    These peaks will have area 'missing_peak_area' and their 'param' set to
    the previous peak + 'distance'.
    If 'distance' is None then the median distance * 1.1 is used.
    """
    assert peaks.ndim == 1

    if distance is None:
        distance = np.median(np.diff(peaks[param])) * 1.1

    diffs = np.diff(peaks[param])
    idx = np.flatnonzero(diffs > distance)
    missing_peak_counts = (diffs[idx] // distance).astype(int)

    # Insert more peaks were required
    idx = np.repeat(idx, missing_peak_counts)

    # Get the distance new peaks are offset
    distances = np.concatenate((np.diff(idx) == 0, [0]))
    distances = (reset_cumsum(distances, 0) + 1) * distance

    missing_peaks = np.zeros(np.sum(missing_peak_counts), dtype=peaks.dtype)
    missing_peaks["area"] = missing_peak_area
    missing_peaks[param] = peaks[param][idx] + distances

    return np.insert(peaks, idx + 1, missing_peaks)


def filter_peaks(
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


def peaks_from_edges(
    x: np.ndarray,
    lefts: np.ndarray,
    rights: np.ndarray,
    base_method: str = "baseline",
    height_method: str = "maxima",
) -> np.ndarray:

    widths = rights - lefts
    indicies = lefts + np.arange(np.amax(widths) + 1)[:, None]
    indicies = np.clip(indicies, 0, x.size - 1)
    indicies = np.where(indicies - lefts < widths, indicies, rights)

    if height_method == "center":
        tops = (lefts + rights) // 2
    elif height_method == "maxima":
        tops = np.argmax(x[indicies], axis=0) + lefts
    else:
        raise ValueError("Valid values for height_method are 'center', 'maxima'.")

    if base_method == "baseline":
        bottoms = tops  # Default to tops
        bwin = np.amax(widths) * 4
        x_pad = np.pad(x, (bwin // 2, bwin - bwin // 2 - 1), mode="edge")
        windows = view_as_blocks(x_pad, (bwin,), (1,))
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
