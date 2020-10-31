import numpy as np

from pew.lib.calc import view_as_blocks

from typing import Tuple, Union


def rolling_mean(
    x: np.ndarray, block: Union[int, Tuple[int, ...]], threshold: float = 3.0
) -> np.ndarray:
    """Rolling filter of size 'block'.
    If the value of x is 'threshold' stddevs from the local mean it is considered an outlier.
    Outliers are replaced with the local mean (excluding outliers).
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
    stds = np.nanstd(blocks, axis=axes)
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
    """Rolling filter of size 'block'.
    If the value of x is 'threshold' medians from the local median it is considered an outlier.
    Outliers are replaced with the local median.
    """
    if isinstance(block, int):
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
