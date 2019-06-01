import numpy as np

from typing import Tuple


def get_weights(x: np.ndarray, weighting: str, safe: bool = True) -> np.ndarray:
    if safe:  # Avoid div 0 problems
        x = x.copy()
        x[x == 0] = np.min(np.nonzero(x))

    if weighting == "x":
        return x
    if weighting == "1/x":
        return 1.0 / x
    elif weighting == "1/(x^2)":
        return 1.0 / (x ** 2.0)
    else:  # Default is no weighting
        return None


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
) -> Tuple[float, float, float]:
    m, b = np.polyfit(x, y, 1, w=w)
    r2 = weighted_rsq(x, y, w)
    return m, b, r2
