import numpy as np

from typing import Callable, Tuple


def local_extrema(
    x: np.ndarray, window: int, step: int = 1, mode: str = "maxima"
) -> np.ndarray:
    windows = sliding_window_centered(x, window, step)
    if mode == "minima":
        extrema = np.argmin(windows, axis=1)
    else:
        extrema = np.argmax(windows, axis=1)
    return np.nonzero(extrema == (window // 2))[0]


def cwt(
    x: np.ndarray, windows: np.ndarray, wavelet: Callable[..., np.ndarray]
) -> np.ndarray:
    cwt = np.empty((windows.shape[0], x.shape[0]), dtype=x.dtype)
    for i in range(cwt.shape[0]):
        n = np.amin((x.size, windows[i] * 10))
        cwt[i] = np.convolve(x, wavelet(n, windows[i]), mode="same")
    return cwt


def ricker_wavelet(size: int, sigma: float) -> np.ndarray:
    x = np.linspace(-size / 2.0, size / 2.0, size)
    a = 2.0 / (np.sqrt(3.0 * sigma) * np.power(np.pi, 0.25))
    kernel = np.exp(-((0.5 * np.square(x) / np.square(sigma))))
    kernel = a * (1.0 - np.square(x) / np.square(sigma)) * kernel
    return kernel


def sliding_window(x: np.ndarray, window: int, step: int = 1) -> np.ndarray:
    shape = ((x.size - window) // step + 1, window)
    strides = (step * x.strides[0], x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def sliding_window_centered(
    x: np.ndarray, window: int, step: int = 1, mode: str = "edge"
) -> np.ndarray:
    x_pad = np.pad(x, (window // 2, window - window // 2 - 1), mode=mode)
    return sliding_window(x_pad, window, step)


def weighting(x: np.ndarray, weighting: str, safe: bool = True) -> np.ndarray:
    if safe and x.size > 0:  # Avoid div 0 problems
        x = x.copy()
        x[x == 0] = np.amin(x[x != 0])

    if weighting is None or weighting == "None":
        return None
    elif weighting == "x":
        return x
    elif weighting == "1/x":
        return 1.0 / x
    elif weighting in ["1/(x^2)", "1/x²"]:
        return 1.0 / (x ** 2.0)
    else:
        raise ValueError(f"Unknown weighting {weighting}.")


def weighted_rsq(x: np.ndarray, y: np.ndarray, w: np.ndarray = None) -> float:
    c = np.cov(x, y, aweights=w)
    d = np.diag(c)
    stddev = np.sqrt(d.real)
    c /= stddev[:, None]
    c /= stddev[None, :]

    np.clip(c.real, -1, 1, out=c.real)
    if np.iscomplexobj(c):
        np.clip(c.imag, -1, 1, out=c.imag)

    return c[0, 1] ** 2.0


def weighted_linreg(
    x: np.ndarray, y: np.ndarray, w: np.ndarray = None
) -> Tuple[float, float, float, float]:
    coef, stats = np.polynomial.polynomial.polyfit(
        x, y, 1, w=w if w is None else np.sqrt(w), full=True
    )
    r2 = weighted_rsq(x, y, w)
    yerr = np.sqrt(np.sum(stats[0]) / (x.size - 2))

    return coef[1], coef[0], r2, yerr
