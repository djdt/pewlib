import numpy as np

from pewlib.laser import Laser

from typing import Union, Tuple


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


def overlap_structured_arrays(
    a: np.ndarray, b: np.ndarray, anchor: Tuple[int, ...], fill: float = np.nan
) -> np.ndarray:
    """Merges two arrays by enlarging. Non-overlapping areas are filled with 'fill'.

    Coordinates (0, 0) of array 'b' will be at 'anchor' in the new array.
    Shared names in 'a' are overwritten by 'b' where overlap occurs.

    Args:
        a: ndim array
        b: ndim array
        anchor: offset for 'b', any int
        fill: value to fill non-overlapping areas

    Returns:
        new overlapped array
    """
    if a.ndim != b.ndim:
        raise ValueError(
            f"Arrays 'a' and 'b' must have the same dimensions, {a.ndim} != {b.ndim}."
        )

    offset = np.array(anchor, dtype=int)
    if offset.size != a.ndim:
        raise ValueError("Anchor must have same dimensions as 'a'.")

    # find the max / min extents of the new array
    new_shape = np.maximum(a.shape, offset + b.shape) - np.minimum(0, offset)
    new_dtype = list(set(a.dtype.descr + b.dtype.descr))
    c = np.full(new_shape, fill, dtype=new_dtype)

    # offsets for each array
    aoffset = np.where(offset < 0, -offset, 0)
    boffset = np.where(offset >= 0, offset, 0)
    aslice = tuple(slice(o, o + s) for o, s in zip(aoffset, a.shape))
    bslice = tuple(slice(o, o + s) for o, s in zip(boffset, b.shape))

    for name in a.dtype.names:
        c[name][aslice] = a[name]
    for name in b.dtype.names:
        c[name][bslice] = b[name]
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
