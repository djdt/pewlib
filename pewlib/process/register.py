"""Module for registering and merging images."""

from typing import Iterable, List, Tuple

import numpy as np


def anchor_offset(a: np.ndarray, b: np.ndarray, anchor: str) -> Tuple[int, int]:
    """Return offset of `b` from `a` given a common achor point.

    Both `a` and `b` must be 2d arrays.
    Valid anchors are 'top left', 'top right', 'bottom left', 'bottom right' or 'center'.

    Args:
        anchor: anchor poisition
        a: nd array
        b: nd array

    Returns:
        offset of 'b' from 'a' in pixels
    """
    assert a.ndim == 2 and b.ndim == 2

    if anchor == "top left":
        return (0, 0)
    elif anchor == "top right":
        return (0, a.shape[1] - b.shape[1])
    elif anchor == "bottom left":
        return (a.shape[0] - b.shape[0], 0)
    elif anchor == "bottom right":
        return (a.shape[0] - b.shape[0], a.shape[1] - b.shape[1])
    elif anchor == "center":
        return tuple((np.array(a.shape, dtype=int) + b.shape) // 2 - b.shape)
    else:  # pragma: no cover
        raise ValueError("Unknown anchor string.")


def fft_register_offset(a: np.ndarray, b: np.ndarray) -> Tuple[int, ...]:
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
    arrays: List[np.ndarray],
    offsets: List[Iterable[int]],
    fill: float = np.nan,
    mode: str = "replace",
) -> np.ndarray:
    """Merges two arrays by enlarging.

    Coordinates (0, 0) of array `b` will be at `offset` in the new array.
    Non-overlapping areas are filled with `fill`.
    Overlapped values are calculated using `mode`. Mode 'replace' will replace
    values with those last in `array`,
    'mean' and 'sum' uses the mean and sum of `arrays` respectively.

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
    >>> c = register.overlap_arrays([a, b], [(0, 0), (2, 2)], fill=0, mode="sum")


    .. plot::

        import numpy as np
        from pewlib.process import register
        import matplotlib.pyplot as plt
        c = register.overlap_arrays(
            [np.arange(9.0).reshape(3, 3), np.arange(9.0).reshape(3, 3)],
            [(0, 0), (2, 2)], fill=0, mode="sum"
        )

        plt.imshow(c)
        plt.show()
    """

    if not all(a.ndim == arrays[0].ndim for a in arrays):  # pragma: no cover
        raise ValueError("Arrays must have the same dimensions.")

    min_offset = np.amin(offsets, axis=0)
    offsets = [(offset - min_offset) for offset in offsets]
    new_shape = np.amax(
        [np.array(offset) + a.shape for offset, a in zip(offsets, arrays)], axis=0
    )

    overlap = np.full(new_shape, fill, dtype=arrays[0].dtype)
    visits = np.zeros(new_shape, dtype=int)  # Track idx for mean calc
    for i, (offset, array) in enumerate(zip(offsets, arrays)):
        slice_idx = tuple(slice(o, o + s) for o, s in zip(offset, array.shape))
        if mode == "replace":
            overlap[slice_idx] = array
        elif mode == "mean" or mode == "sum":
            overlap[slice_idx] = np.nansum([overlap[slice_idx], array], axis=0)
            visits[slice_idx] += 1
    if mode == "mean":
        overlap[visits > 1] /= visits[visits > 1]

    return overlap


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
    new_dtype = list(a.dtype.descr)
    new_dtype.extend([x for x in b.dtype.descr if x not in new_dtype])
    c = np.empty(new_shape, dtype=new_dtype)

    for name in c.dtype.names:
        if name not in b.dtype.names:
            c[name] = overlap_arrays(
                np.full(b.shape, fill),
                a[name],
                offset=-offset,
                fill=fill,
                mode="replace",
            )
        elif name not in a.dtype.names:
            c[name] = overlap_arrays(
                np.full(a.shape, fill),
                b[name],
                offset=offset,
                fill=fill,
                mode="replace",
            )
        else:
            c[name] = overlap_arrays(
                a[name], b[name], offset=offset, fill=fill, mode=mode
            )

    return c
