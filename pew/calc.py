import numpy as np

from typing import Tuple


def get_weights(x: np.ndarray, weighting: str, safe: bool = True) -> np.ndarray:
    if safe and x.size > 0:  # Avoid div 0 problems
        x = x.copy()
        x[x == 0] = np.min(x[x != 0])

    if weighting is None or weighting == "None":
        return None
    elif weighting == "x":
        return x
    elif weighting == "1/x":
        return 1.0 / x
    elif weighting in ["1/(x^2)", "1/x²"]:
        return 1.0 / (x ** 2.0)
    else:  # Default is no weighting
        raise ValueError(f"Unknown weighting {weighting}.")


def moving_average_filter(x: np.ndarray, n: int = 3) -> np.ndarray:
    x_pad = np.pad(x, (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(x_pad, np.ones((n,)) / n, mode="valid")


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
    yerr = np.sqrt(stats[0] / (x.size - 2))

    return coef[1], coef[0], r2, yerr


def sliding_window(x: np.ndarray, window: int, step: int = 1) -> np.ndarray:
    shape = ((x.size - window) // step + 1, window)
    strides = (step * x.strides[0], x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def centered_sliding_window(
    x: np.ndarray, window: int, step: int = 1, pad_value: float = np.nan
) -> np.ndarray:
    x_pad = np.pad(
        x, (window // 2, window - window // 2 - 1), constant_values=pad_value
    )
    return sliding_window(x_pad, window, step)
