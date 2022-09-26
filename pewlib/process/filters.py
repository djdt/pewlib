"""Filtering can be used to remove artifacts, such as spikes, from images.
Care must be taken when using filtering to ensure that legitmate data is not
also altered.
"""
import numpy as np

from pewlib.process.calc import view_as_blocks

from typing import Tuple, Union


def rolling_mean(
    x: np.ndarray, block: Union[int, Tuple[int, ...]], threshold: float = 3.0
) -> np.ndarray:
    """Filter an array using rolling mean.

    Each value of `x` is compared to the mean of its `block`, the values arround it.
    If it is `threshold` times the standard deviation then it is considered an outlier.
    Outliers are set to the local mean.

    Args:
        x: array
        block: size of window, int or same dims as `x`
        threshold: number of stddevs away from mean to consider outlier

    Returns:
        array with outliers set to local means

    Example
    -------

    Removing spikes from 1d data.

    >>> import numpy as np
    >>> from pewlib.process import filters
    >>> a = np.sin(np.linspace(0, 1, 50))
    >>> a[5::10] +=np.random.choice([-1, 1], size=10)
    >>> b = filters.rolling_mean(a, 3, threshold=1.0)

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from pewlib.process import filters
        a = np.sin(np.linspace(0, 10, 50))
        a[5::10] +=np.random.choice([-1, 1], size=5)
        b = filters.rolling_mean(a, 3, threshold=1.0)

        plt.plot(a, c="black")
        plt.plot(b, ls=":", c="red", label="filtered")
        plt.legend()
        plt.show()
    """
    if isinstance(block, int):
        block = tuple([block] * x.ndim)
    assert len(block) == x.ndim

    # Prepare array by padding with nan
    pads = [(b // 2, b // 2) for b in block]
    x_pad = np.pad(x, pads, mode="mean", stat_length=pads)

    blocks = view_as_blocks(x_pad, block, tuple([1] * x.ndim))

    # Calculate means and stds
    axes = tuple(np.arange(x.ndim, x.ndim * 2))

    means = np.mean(blocks, axis=axes)
    stds = np.std(blocks, axis=axes)

    # Check for outlying values and set as nan
    outliers = np.abs(x - means) > threshold * stds

    return np.where(outliers, means, x)


def rolling_median(
    x: np.ndarray,
    block: Union[int, Tuple[int, ...]],
    threshold: float = 3.0,
) -> np.ndarray:
    """Filter an array using rolling median.

    Each value of `x` is compared to the median of its `block`, the values arround it.
    If it is `threshold` times the stdev from the median then it is considered an outlier.
    Outliers are set to the local median.

    Args:
        x: array
        block: size of window, int or same dims as `x`
        threshold: number of SDs (via MAD) away from median to consider outlier

    Returns:
        array with outliers set to local means

    Example
    -------

    Removing poisson noise from an image.

    >>> import numpy as np
    >>> from pewlib.process import filters
    >>> a = np.sin(np.linspace(0, 1, 2500).reshape((50, 50)))
    >>> a += np.random.poisson(lam=0.01, size=(50, 50))
    >>> b = filters.rolling_median(a, (5, 5), threshold=3.0)

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from pewlib.process import filters
        a = np.sin(np.linspace(0, 1, 2500).reshape((50, 50)))
        a += np.random.poisson(lam=0.01, size=(50, 50))
        b = filters.rolling_median(a, (5, 5), threshold=3.0)

        f, ax = plt.subplots(1, 2)
        ax[0].imshow(a, vmax=1.0)
        ax[0].set_title("raw image 'a'")
        ax[1].imshow(b, vmax=1.0)
        ax[1].set_title("filtered image 'b'")
        plt.show()

    """
    if isinstance(block, int):  # pragma: no cover
        block = tuple([block] * x.ndim)
    assert len(block) == x.ndim

    # Prepare array by padding with nan
    pads = [(b // 2, b // 2) for b in block]
    y = np.pad(x, pads, mode="median", stat_length=pads)

    blocks = view_as_blocks(y, block, tuple([1] * x.ndim))

    # Calculate median and differences
    axes = tuple(np.arange(x.ndim, x.ndim * 2))
    medians = np.median(blocks, axis=axes)

    # Remove padding
    unpads = tuple([slice(p[0], -p[1]) for p in pads])
    y[unpads] = np.abs(x - medians)

    # Median of differences
    mad = np.median(blocks, axis=axes) * 1.4826  # estimate stddev

    # Outliers are n medians from data
    outliers = y[unpads] > threshold * mad

    return np.where(outliers, medians, x)
