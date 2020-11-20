import numpy as np

from pew.lib.calc import view_as_blocks

from typing import Tuple, Union


def rolling_mean(
    x: np.ndarray, block: Union[int, Tuple[int, ...]], threshold: float = 3.0
) -> np.ndarray:
    """Filter an array using rolling mean.

    Each value of `x` is compared to the mean of its `block`, the values arround it.
    If it is `threshold` times the standard deviation *without the central value* then
    it is considered an outlier. This prevents the value from impacting the stddev.
    The mean of each block is recalculated outliers set to the new local mean.

    Args:
        x: array
        block: size of window, int or same dims as `x`
        threshold: number of stddevs away from mean to consider outlier

    Returns:
        array with outliers set to local means
    """
    if isinstance(block, int):
        block = tuple([block])
    assert len(block) == x.ndim

    # Prepare array by padding with nan
    pads = [(b // 2, b // 2) for b in block]
    x_pad = np.pad(x, pads, constant_values=np.nan)

    blocks = view_as_blocks(x_pad, block, tuple([1 for b in block]))

    # Calculate means and stds
    axes = tuple(np.arange(x.ndim, x.ndim * 2))
    means = np.nanmean(blocks, axis=axes)

    # Don't include the central point in the std calculation
    nancenter = np.ones(block, dtype=np.float64)
    nancenter[np.array(nancenter.shape) // 2] = np.nan
    stds = np.nanstd(blocks * nancenter, axis=axes)

    # Check for outlying values and set as nan
    outliers = np.abs(x - means) > threshold * stds

    unpads = tuple([slice(p[0], -p[1]) for p in pads])
    x_pad[unpads][outliers] = np.nan

    # As the mean is sensitive to outliers reclaculate it
    means = np.nanmean(blocks, axis=axes)

    return np.where(np.logical_and(outliers, means), means, x)


def rolling_median(
    x: np.ndarray, block: Union[int, Tuple[int, ...]], threshold: float = 3.0
) -> np.ndarray:
    """Filter an array using rolling median.

    Each value of `x` is compared to the median of its `block`, the values arround it.
    If it is `threshold` times the median distance from the median then
    it is considered an outlier.
    The mean of each block is recalculated outliers set to the local median.

    Args:
        x: array
        block: size of window, int or same dims as `x`
        threshold: number of median distances away from medians to consider outlier

    Returns:
        array with outliers set to local means
    """
    if isinstance(block, int):  # pragma: no cover
        block = tuple([block])
    assert len(block) == x.ndim

    # Prepare array by padding with nan
    pads = [(b // 2, b // 2) for b in block]
    x_pad = np.pad(x, pads, constant_values=np.nan)

    blocks = view_as_blocks(x_pad, block, tuple([1 for b in block]))

    # Calculate median and differences
    axes = tuple(np.arange(x.ndim, x.ndim * 2))
    medians = np.nanmedian(blocks, axis=axes)

    # Remove padding
    unpads = tuple([slice(p[0], -p[1]) for p in pads])
    x_pad[unpads] = np.abs(x - medians)

    # Median of differences
    median_medians = np.nanmedian(blocks, axis=axes)

    # Outliers are n medians from data
    outliers = np.abs(x - medians) > threshold * median_medians

    return np.where(np.logical_and(outliers, medians), medians, x)
