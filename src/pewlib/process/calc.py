"""This module contains functions used by other processing modules."""

import numpy as np


def local_maxima(x: np.ndarray) -> np.ndarray:
    """Indicies of local maxima.

    Maxima are values greater than the 2 values surrounding them.

    Args:
        x: 1d array

    Returns:
        indicies of maxima
    """
    return np.nonzero(
        np.logical_and(np.r_[True, x[1:] > x[:-1]], np.r_[x[:-1] > x[1:], True])
    )[0]


def normalise(x: np.ndarray, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    """Normalise an array.

    Args:
        x: array
        vmin: new minimum
        vmax: new maxmimum

    Raises:
        ValueError if `x` is a simgle value"""
    xmax, xmin = np.amax(x), np.amin(x)
    if xmax == xmin:  # pragma: no cover
        raise ValueError("Cannot normalise array, min == max.")

    x = (x - xmin) / (xmax - xmin)
    x *= vmax - vmin
    x += vmin
    return x


def reset_cumsum(x: np.ndarray, reset_value: float = 0.0) -> np.ndarray:
    """Cumulative sum that resets at the given value.

    Args:
        x: array
        reset_value: Value where the cumsum resets to 0"""
    c = np.cumsum(x)
    n = x == reset_value
    oc = np.maximum.accumulate(c * n)
    return c - oc


def shuffle_blocks(
    x: np.ndarray,
    block: tuple[int, ...],
    mask: np.ndarray | None = None,
    mode: str = "pad",
    shuffle_partial: bool = False,
) -> np.ndarray:
    """Shuffle an ndim array as tiles of a certain size.

    If a `mask` is passed then only the region within the mask is shuffled.
    If `shuffle_partial` then partially masked blocks will be shuffled otherwise
    only fully masked blocks are. The inplace `mode` is much faster but cannot shuffle
    array edges.

    Args:
        x: array
        block: block shape, same dims as `x`
        mask: mask, same shape as `x`, optional
        mode: method, {'pad', 'inplace'}
        shuffle_partial: shuffle partially masked blocks

    Returns:
        new array if pad, view if inplace
    """
    shape = x.shape
    if mask is None:  # pragma: no cover
        mask = np.ones(x.shape, dtype=bool)
    # Pad the array to fit the blocksize
    if mode == "pad":
        pads = [(0, p) for p in (block - (np.array(x.shape) % block)) % block]
        x = np.pad(x, pads, mode="edge")
        mask = np.pad(mask, pads, mode="edge")
    elif mode == "inplace":
        # Use mask to prevent shuffling of blocks out of bounds
        trim = x.shape - (np.array(x.shape) % block)
        for axis, t in enumerate(trim):
            np.swapaxes(mask, 0, axis)[slice(t, None)] = False
    else:  # pragma: no cover
        raise ValueError("Mode must be 'pad' or 'inplace'.")

    blocks = view_as_blocks(x, block)
    mask = view_as_blocks(mask, block)

    # Mask only in blocks with all (mask_all) or some mask
    axes = tuple(np.arange(x.ndim, x.ndim + len(block)))
    mask = np.any(mask, axis=axes) if shuffle_partial else np.all(mask, axis=axes)

    # Create flat index then shuffle
    idx = np.nonzero(mask)
    nidx = np.random.permutation(np.ravel_multi_index(idx, mask.shape))
    nidx = np.unravel_index(nidx, mask.shape)
    blocks[idx] = blocks[nidx]

    if mode == "pad":
        unpads = tuple([slice(0, s) for s in shape])
        x = x[unpads]
    return x


def subpixel_offset(
    x: np.ndarray, offsets: list[tuple[int, int]], pixelsize: tuple[int, int]
) -> np.ndarray:
    """Offsets layers in a 3d array.

    First `x` is enlarged by `pixelsize` then list of `offsets` are applied
    across axis 2 of `x`. If the first offset is not (0, 0) then it is prepended.
    Given `offsets` of [(0, 0), (1, 1)] and pixelsize of (2, 2) each layer
    will be streched by 2 and every 2nd layer will be shifted by 1 pixel.

    Args:
        offsets: pixel offsets in (x, y)
        pixelsize: enlargement (x, y)

    Returns:
        array
    """
    # Offset for first layer must be zero
    if offsets[0] != (0, 0):
        offsets.insert(0, (0, 0))  # pragma: no cover
    overlap = np.max(offsets, axis=0)

    if x.ndim != 3:  # pragma: no cover
        raise ValueError("Data must be three dimensional!")

    # Calculate new shape
    new_shape = np.array(x.shape[:2]) * pixelsize + overlap
    # Create empty array to store data in
    data = np.zeros((*new_shape, x.shape[2]), dtype=x.dtype)

    for i in range(0, x.shape[2]):
        # Cycle through offsets
        start = offsets[i % len(offsets)]
        end = -(overlap[0] - start[0]) or None, -(overlap[1] - start[1]) or None
        # Stretch arrays as required
        data[start[0] : end[0], start[1] : end[1], i] = np.repeat(
            x[:, :, i], pixelsize[0], axis=0
        ).repeat(pixelsize[1], axis=1)

    return data


def subpixel_offset_equal(
    x: np.ndarray, offsets: list[int], pixelsize: int
) -> np.ndarray:
    """Offsets layers in a 3d array.

    Special case of 'subpixel_offset' where x==y for `offsets` and `pixelsize`.

    Args:
        offsets: pixel offsets in (x, y)
        pixelsize: enlargement (x, y)

    Returns:
        array

    See Also:
        :func:`pewlib.process.calc.subpixel_offset`
    """
    return subpixel_offset(x, [(o, o) for o in offsets], (pixelsize, pixelsize))


def view_as_blocks(
    x: np.ndarray, block: tuple[int, ...], step: tuple[int, ...] | None = None
) -> np.ndarray:
    """Block view of array

    Can be overlapping if `step` < `block`.

    Args:
        x: array
        block: block size, same dims as `x`
        step: step, same dims as `x`, defaults to `block`

    Returns:
        view into array

    See Also:
        :func:`skimage.util.shape.view_as_blocks`
    """
    assert len(block) == x.ndim
    if step is None:
        step = block
    x = np.ascontiguousarray(x)
    shape = tuple((np.array(x.shape) - block) // np.array(step) + 1) + block
    strides = tuple(x.strides * np.array(step)) + x.strides
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
