import numpy as np

from pew.calc import cwt, local_extrema, ricker_wavelet, sliding_window_centered

import logging

logger = logging.getLogger(__name__)


def local_extrema_raw(
    x: np.ndarray, window: int, step: int = 1, mode: str = "maxima"
) -> np.ndarray:
    windows = sliding_window_centered(x, window, step)
    if mode == "minima":
        extrema = np.argmin(windows, axis=1)
    else:
        extrema = np.argmax(windows, axis=1)
    return extrema == (window // 2)


def _identify_ridges(
    cwt_coef: np.ndarray, windows: np.ndarray, gap_threshold: int = None, mode="maxima",
) -> np.ndarray:
    if gap_threshold is None:
        gap_threshold = len(windows) // 4

    extrema = local_extrema(cwt_coef[-1], windows[-1] * 2, mode=mode)
    ridges = np.array([], dtype=int).reshape(cwt_coef.shape[0], 0)

    for i in np.arange(cwt_coef.shape[0] - 1, -1, -1):
        extrema = local_extrema(cwt_coef[i], windows[i] * 2, mode=mode)

        if ridges.size != 0:
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

        # for ridge in ridges:
        #     # Skip already gapped ridges
        #     if np.count_nonzero(ridge[i + 1 :] == -1) > gap_threshold:
        #         continue

        #     idx = np.amin((np.searchsorted(extrema, ridge[i + 1]), extrema.size - 1))
        #     idx2 = np.amax((idx - 1, 0))
        #     d1 = extrema[idx] - ridge[i + 1]
        #     d2 = ridge[i + 1] - extrema[idx2]
        #     # idx2 = np.amin((idx + 1, extrema.size - 1))
        #     min_diff = np.where(d1 <= d2, idx, idx2)
        #     if np.abs(extrema[min_diff] - ridge[i + 1]) <= windows[i] // 4:
        #         ridge[i] = extrema[min_diff]
        #         extrema[min_diff] = -1  # Skip later

        remaining_extrema = extrema[extrema > -1]
        if remaining_extrema.size != 0:
            new_ridges = np.full(
                (cwt_coef.shape[0], remaining_extrema.shape[0]), -1, dtype=int
            )
            new_ridges[i] = remaining_extrema
            ridges = np.hstack((ridges, new_ridges))

    return ridges

    # if gap_threshold is None:
    #     gap_threshold = len(windows) // 4

    # ridges = np.array([], dtype=int).reshape(0, cwt_coef.shape[0])

    # for i in np.arange(cwt_coef.shape[0] - 1, -1, -1):
    #     extrema = local_extrema(cwt_coef[i], windows[i] * 2, mode=mode)

    #     for ridge in ridges:
    #         # Skip already gapped ridges
    #         if np.count_nonzero(ridge[i + 1 :] == -1) > gap_threshold:
    #             continue

    #         idx = np.amin((np.searchsorted(extrema, ridge[i + 1]), extrema.size - 1))
    #         idx2 = np.amax((idx - 1, 0))
    #         d1 = extrema[idx] - ridge[i + 1]
    #         d2 = ridge[i + 1] - extrema[idx2]
    #         # idx2 = np.amin((idx + 1, extrema.size - 1))
    #         min_diff = np.where(d1 <= d2, idx, idx2)
    #         if np.abs(extrema[min_diff] - ridge[i + 1]) <= windows[i] // 4:
    #             ridge[i] = extrema[min_diff]
    #             extrema[min_diff] = -1  # Skip later

    #     remaining_extrema = extrema[extrema > -1]
    #     if remaining_extrema.size != 0:
    #         new_ridges = np.full(
    #             (remaining_extrema.shape[0], cwt_coef.shape[0]), -1, dtype=int
    #         )
    #         new_ridges[:, i] = remaining_extrema
    #         ridges = np.vstack((ridges, new_ridges))

    # return ridges.T


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


def _peak_data_from_ridges(
    x: np.ndarray,
    ridges: np.ndarray,
    maxima_coords: np.ndarray,
    cwt_windows: np.ndarray,
    peak_height_method: str = "maxima",
    peak_integration_method: str = "base",
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

    if peak_integration_method == "base":
        ibases = x[bases]
    elif peak_integration_method == "prominence":
        ibases = np.maximum(x[lefts], x[rights])
    else:
        raise ValueError("Valid peak_integration_method are 'base', 'prominence'.")
    area = np.array([np.trapz(x[r[0] : r[1]] - ib) for r, ib in zip(ranges, ibases)])

    dtype = np.dtype(
        {
            "names": ["height", "width", "area", "top", "base", "left", "right"],
            "formats": [float, float, float, int, int, int, int],
        }
    )
    peaks = np.empty(tops.shape, dtype=dtype)
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
    ridge_gap_threshold: int = None,
    ridge_min_snr: float = 10.0,
    peak_integration_method: str = "base",
    peak_min_area: float = 0.0,
    peak_min_height: float = 0.0,
    peak_min_width: float = 0.0,
    peak_min_prominence: float = 0.0,
    peak_max_number: int = None,
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

    if peak_max_number is not None and peaks.size > peak_max_number:
        peaks = peaks[np.argsort(peaks["area"])][-peak_max_number:]

    return peaks


# def lines_to_spots(
#     data: np.ndarray,
#     shape: Tuple[int, ...],

# ) -> np.ndarray:
#     spot_data = np.empty(np.prod(shape), dtype=data.dtype)

#     peak_kws = dict(areas_only=True, gradient=gradient, height=height)

#     for name in data.dtype.names:
#         spot_data[name] = np.apply_along_axis(
#             find_peaks, 1, data[name], , peak_kws
#         )

#     return spot_data.reshape(shape)
