"""Module for registering and merging images."""

import numpy as np

from pewlib.laser import Laser

from typing import Iterable, Union, Tuple


def fft_register_images(a: np.ndarray, b: np.ndarray) -> Tuple[int, ...]:
    """Register two images using FFT correlation.

    Arrays are zero-padded to `a.shape` + `b.shape` - 1

    Args:
        a: nd array
        b: nd array

    Returns:
        offset of 'b' from 'a' in pixels
    """
    s = np.array(a.shape) + b.shape - 1

    a = np.pad(a, np.stack((np.zeros(a.ndim, dtype=int), s - a.shape), axis=1))
    b = np.pad(b, np.stack((np.zeros(b.ndim, dtype=int), s - b.shape), axis=1))

    xcorr = np.fft.irfftn(np.fft.rfftn(a) * np.fft.rfftn(b).conj())
    xcorr = np.fft.fftshift(xcorr)

    return np.unravel_index(np.argmax(xcorr), xcorr.shape) - np.array(xcorr.shape) // 2


def overlap_arrays(
    a: np.ndarray,
    b: np.ndarray,
    offset: Iterable[int],
    fill: float = np.nan,
    mode: str = "replace",
) -> np.ndarray:
    """Merges two arrays by enlarging.

    Coordinates (0, 0) of array `b` will be at `offset` in the new array.
    Non-overlapping areas are filled with `fill`.
    Overlapped values are calculated using `mode`. Mode 'replace' will use values in `b`,
    'mean' and 'sum' uses the mean and sum of `a` and `b` respectively.

    Args:
        a: ndim array
        b: ndim array
        offset: offset for 'b', any int
        fill: value for non-overlapping areas
        mode: value for overlapped areas, {'replace', 'mean', 'sum'}

    Returns:
        new overlapped array, same dtype as 'a'

    Example
    -------

    >>> import numpy as np
    >>> from pewlib.process import register
    >>> a = np.arange(9.0).reshape(3, 3)
    >>> b = np.arange(9.0).reshape(3, 3)
    >>> c = register.overlap_arrays(a, b, fill=0, mode="sum")


    .. plot::

        import numpy as np
        from pewlib.process import register
        import matplotlib.pyplot as plt
        c = register.overlap_arrays(np.arange(9.0).reshape(3, 3), np.arange(9.0).reshape(3, 3), fill=0, mode="sum")

        plt.imshow(c)
        plt.show()
    """

    if a.ndim != b.ndim:  # pragma: no cover
        raise ValueError(
            f"Arrays 'a' and 'b' must have the same dimensions, {a.ndim} != {b.ndim}."
        )

    offset = np.array(offset, dtype=int)
    if offset.size != a.ndim:  # pragma: no cover
        raise ValueError("Offset must have same dimensions as arrays.")

    if mode not in ["replace", "mean", "sum"]:  # pragma: no cover
        raise ValueError("'mode' must be 'replace', 'mean' or 'sum'.")

    # find the max / min extents of the new array
    new_shape = np.maximum(a.shape, offset + b.shape) - np.minimum(0, offset)
    c = np.full(new_shape, fill, dtype=a.dtype)

    # offsets for each array
    aoffset = np.where(offset < 0, -offset, 0)
    boffset = np.where(offset >= 0, offset, 0)
    aslice = tuple(slice(o, o + s) for o, s in zip(aoffset, a.shape))
    bslice = tuple(slice(o, o + s) for o, s in zip(boffset, b.shape))

    # size and slice of overlapping area
    abslice = tuple(slice(max(a.start, b.start), min(a.stop, b.stop), None) for a, b in zip(aslice, bslice))
    absize = tuple(s.stop - s.start for s in abslice)
    hasoverlap = all(x > 0 for x in absize)

    # sections of overlap from a, b
    asec = tuple(slice(o, o + s) for o, s in zip(aoffset, absize))
    bsec = tuple(slice(o, o + s) for o, s in zip(boffset, absize))

    c[aslice] = a
    c[bslice] = b

    if mode == "replace":
        pass
    elif mode == "mean" and hasoverlap:
        c[abslice] = np.mean([a[asec], b[bsec]], axis=0)
    elif mode == "sum" and hasoverlap:
        c[abslice] = np.sum([a[asec], b[bsec]], axis=0)

    return c


def overlap_structured_arrays(
    a: np.ndarray,
    b: np.ndarray,
    offset: Iterable[int],
    fill: float = np.nan,
    mode: str = "replace",
) -> np.ndarray:
    """Merges two structured arrays by enlarging.

    Coordinates (0, 0) of array `b` will be at `offset` in the new array.
    Shared names in 'a' and 'b' are calculated using `mode` where overlap occurs.

    Args:
        a: ndim array
        b: ndim array
        offset: offset for 'b'
        fill: value for non-overlapping areas
        mode: value for overlapped areas, {'replace', 'mean', 'sum'}

    Returns:
        new overlapped array

    See Also:
        :func:`pewlib.process.register.overlap_arrays`
    """
    offset = np.array(offset, dtype=int)
    if offset.size != a.ndim:  # pragma: no cover
        raise ValueError("Anchor must have same dimensions as arrays.")

    # find the max / min extents of the new array
    new_shape = np.maximum(a.shape, offset + b.shape) - np.minimum(0, offset)
    new_dtype = list(set(a.dtype.descr + b.dtype.descr))
    c = np.empty(new_shape, dtype=new_dtype)

    for name in c.dtype.names:
        if name not in b.dtype.names:
            c[name] = overlap_arrays(
                np.full(b.shape, fill), a[name], offset=-offset, fill=fill, mode=mode
            )
        elif name not in a.dtype.names:
            c[name] = overlap_arrays(
                np.full(a.shape, fill), b[name], offset=offset, fill=fill, mode=mode
            )
        else:
            c[name] = overlap_arrays(
                a[name], b[name], offset=offset, fill=fill, mode=mode
            )

    return c


def overlap_lasers(
    a: Laser, b: Laser, anchor: Union[str, Tuple[int, int]] = "top left"
) -> Laser:
    """Merges two arrays by enlarging.

    Coordinates (0, 0) of array 'b' will be at 'anchor' in the new array.
    Shared names and calibrations in 'a' are overwritten by 'b' where overlap occurs.
    Valid anchors are 'top left', 'top right', 'bottom left', 'bottom right', 'center' or
    a tuple of ints sepcififying the coordinates of overlap.

    Args:
        a: Laser
        b: Laser
        anchor: offset

    Returns:
        new overlapped array
    """
    if isinstance(anchor, str):
        if anchor == "top left":
            anchor = (0, 0)
        elif anchor == "top right":
            anchor = (0, a.shape[1] - b.shape[1])
        elif anchor == "bottom left":
            anchor = (a.shape[0] - b.shape[0], 0)
        elif anchor == "bottom right":
            anchor = (a.shape[0] - b.shape[0], a.shape[1] - b.shape[1])
        elif anchor == "center":
            anchor = (np.array(a.shape, dtype=int) + b.shape) // 2 - b.shape
        else:
            raise ValueError("Unknown anchor string.")

    data = overlap_structured_arrays(a.data, b.data, anchor=anchor)

    calibration = {}
    for name in data.dtype.names:
        if name in b.elements:
            calibration[name] = b.calibration[name]
        elif name in a.elements:
            calibration[name] = a.calibration[name]

    info = {"Overlap": f"{a.info['Name']}:{b.info['Name']}@{anchor[0],anchor[1]}"}
    info.update(a.info)

    return Laser(data=data, calibration=calibration, config=a.config, info=info)
