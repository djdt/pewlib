import numpy as np

from typing import List, Tuple


def greyscale_to_rgb(array: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Convert a gret scale image to a single color rgb image.

    The image is clipped to 0.0 to 1.0.

    Args:
        array: Image
        rgb: 3 or 4 color array (rgb / rgba)
"""
    array = np.clip(array, 0.0, 1.0)
    return array[..., None] * np.array(rgb, dtype=float)


# def ida(
#     a: np.ndarray,
#     b: np.ndarray,
#     Ct: float,
#     ms: float,
#     mt: float,
#     Ms: float,
#     Mt: float,
#     Aas: float,
#     Abs: float,
#     Aat: float,
#     Abt: float,
# ) -> np.ndarray:
#     Rm = a / b
#     Rs = Abs / Aas
#     Rt = Aat / Abt
#     Cs = Ct * (mt / ms) * (Ms / Mt) * (Abt / Aas) * ((Rm - Rt) / (1.0 - Rm * Rs))
#     return Cs


def kmeans(
    x: np.ndarray, k: int, init: str = "kmeans++", max_iterations: int = 1000,
) -> np.ndarray:
    """K-means clustering. Returns an array mapping objects to their clusters.
     Raises a ValueError if the loop exceeds max_iterations.

     Args:
        x: Data. Shape is (n, m) for n objects with m attributes.
        k: Number of clusters.
        init: Method to determine initial cluster centers. Can be 'kmeans++' or 'random'.
"""
    # Ensure at least 1 dim for variables
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if init == "kmeans++":
        centers = kmeans_plus_plus(x, k)
    elif init == "random":
        ix = np.random.choice(np.arange(x.shape[0]), k)
        centers = x[ix].copy()
    else:  # pragma: no cover
        raise ValueError("'init' must be 'kmeans++' or 'random'.")

    # Sort centers by the first attribute
    centers = centers[np.argsort((centers[:, 0]))]

    while max_iterations > 0:
        max_iterations -= 1

        distances = np.sqrt(np.sum((centers[:, None] - x) ** 2, axis=2))
        idx = np.argmin(distances, axis=0)

        new_centers = centers.copy()
        for i in np.unique(idx):
            new_centers[i] = np.mean(x[idx == i], axis=0)

        if np.allclose(centers, new_centers):
            return idx
        centers = new_centers

    raise ValueError("No convergance in allowed iterations.")  # pragma: no cover


def kmeans_plus_plus(x: np.ndarray, k: int) -> np.ndarray:
    """Selects inital cluster positions using K-means++ algorithm.
"""
    ix = np.arange(x.shape[0])
    centers = np.empty((k, *x.shape[1:]))
    centers[0] = x[np.random.choice(ix, 1)]

    for i in range(1, k):
        distances = np.sqrt(np.sum((centers[:i, None] - x) ** 2, axis=2))
        distances = np.amin(distances, axis=0) ** 2
        centers[i] = x[np.random.choice(ix, 1, p=distances / distances.sum())]

    return centers.copy()


def local_maxima(x: np.ndarray) -> np.ndarray:
    return np.nonzero(
        np.logical_and(np.r_[True, x[1:] > x[:-1]], np.r_[x[:-1] > x[1:], True])
    )[0]


def normalise(x: np.ndarray, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    """Normalise an array.

    Args:
        x: Array
        vmin: New minimum
        vmax: New maxmimum
"""
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
        x: Input array.
        reset_value: Value where the cumsum resets to 0.
"""
    c = np.cumsum(x)
    n = x == reset_value
    oc = np.maximum.accumulate(c * n)
    return c - oc


def shuffle_blocks(
    x: np.ndarray,
    block: Tuple[int, ...],
    mask: np.ndarray = None,
    mask_all: bool = True,
) -> np.ndarray:
    """Shuffle an ndim array as tiles of a certain size.
    If a mask is passed then only the region within the mask is shuffled.
    If mask_all is True then only entirely masked blocks are shuffled otherwise
    even partially masked blocks will be shuffled.

    Args:
        x: Input array.
        block: Shape of blocks, ndim must be the same as x.
        mask: Optional mask data, shape must be the same as x.
        mask_all: Only shuffle entirely masked blocks.
"""
    # Pad the array to fit the blocksize
    pads = [(0, (b - (shape % b)) % b) for b, shape in zip(block, x.shape)]
    xpad = np.pad(x, pads, mode="edge")
    if mask is None:
        mask = np.ones(xpad.shape, dtype=bool)

    blocks = view_as_blocks(xpad, block)
    mask = view_as_blocks(np.pad(mask, pads, mode="edge"), block)
    # Mask only in blocks with all (mask_all) or some mask
    axis = tuple(np.arange(x.ndim, x.ndim + len(block)))
    mask = np.all(mask, axis=axis) if mask_all else np.any(mask, axis=axis)

    # Create flat index then shuffle
    idx = np.nonzero(mask)
    nidx = np.random.permutation(np.ravel_multi_index(idx, mask.shape))
    nidx = np.unravel_index(nidx, mask.shape)
    blocks[idx] = blocks[nidx]

    slices = [slice(0, s) for s in x.shape]
    return xpad[tuple(slices)]


def sliding_window(x: np.ndarray, window: int, step: int = 1) -> np.ndarray:
    """1D version of view_as_blocks."""
    shape = ((x.size - window) // step + 1, window)
    strides = (step * x.strides[0], x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def sliding_window_centered(
    x: np.ndarray, window: int, step: int = 1, mode: str = "edge"
) -> np.ndarray:
    x_pad = np.pad(x, (window // 2, window - window // 2 - 1), mode=mode)
    return sliding_window(x_pad, window, step)


def subpixel_offset(
    x: np.ndarray, offsets: List[Tuple[int, int]], pixelsize: Tuple[int, int]
) -> np.ndarray:
    """Takes a 3d array and stretches and offsets each layer.

    Given an offset of (1,1) and pixelsize of (2,2) each layer will be streched by 2
    and every even layer will be shifted by 1 pixel.

    Args:
        offsets: The pixel offsets in (x, y).
        pixelsize: Final size to stretch to.

    Returns:
        The offset array.
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
    x: np.ndarray, offsets: List[int], pixelsize: int
) -> np.ndarray:
    return subpixel_offset(x, [(o, o) for o in offsets], (pixelsize, pixelsize))


def view_as_blocks(
    x: np.ndarray, block: Tuple[int, ...], step: Tuple[int, ...] = None
) -> np.ndarray:
    """Create block sized views into a array, offset by step amount.
    https://github.com/scikit-image/scikit-image/blob/master/skimage/util/shape.py

    Args:
        x: The array.
        block: The size of the view.
        step: Size of step, defaults to block.

    Returns:
        An array of views.
    """
    assert len(block) == x.ndim
    if step is None:
        step = block
    x = np.ascontiguousarray(x)
    shape = tuple((np.array(x.shape) - block) // np.array(step) + 1) + block
    strides = tuple(x.strides * np.array(step)) + x.strides
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


# def windowed_extrema(
#     x: np.ndarray, window: int, step: int = 1, mode: str = "maxima"
# ) -> np.ndarray:
#     windows = sliding_window_centered(x, window, step)
#     if mode == "minima":
#         extrema = np.argmin(windows, axis=1)
#     else:
#         extrema = np.argmax(windows, axis=1)
#     return np.nonzero(extrema == (window // 2))[0]
