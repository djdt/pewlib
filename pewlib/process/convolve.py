"""Conv- and deconvolution have many applications in image processing such as
adding or removing blur. This module contains functions for performing 1-dimensional
convolutions as well as functions for creating various distributions.

"""

import numpy as np

_s2 = np.sqrt(2.0)
_s2pi = np.sqrt(2.0 * np.pi)


def convolve(x: np.ndarray, psf: np.ndarray, mode: str = "pad") -> np.ndarray:
    """Convolve with 'pad' mode.

    If `mode` is 'pad' then `x` is edge padded to converse size on convolution.
    Other modes are passed directly to :func:`numpy.convolve`.

    Args:
        x: array
        psf: point spread function
        mode: convolution mode {'pad', 'full', 'valid', 'same'}

    See Also:
        :func:`numpy.convolve`
    """
    # Pad array with edge
    if mode == "pad":
        x_pad = np.pad(
            x, (psf.size // 2, psf.size // 2 - 1 + psf.size % 2), mode="edge"
        )
        return np.convolve(x_pad, psf, mode="valid")
    else:  # pragma: no cover
        return np.convolve(x, psf, mode=mode)


def deconvolve(x: np.ndarray, psf: np.ndarray, mode: str = "valid"):
    """Inverse of convolution.

    Deconvolution is performed in frequency domain.

    Args:
        x: array
        psf: point spread function
        mode: if same, return same size as `x` {'valid', 'same'}

    Notes:
        Based on https://rosettacode.org/wiki/Deconvolution/1D
    """

    def shift_bit_length(x: int) -> int:
        return 1 << (x - 1).bit_length()

    r = shift_bit_length(max(x.size, psf.size))
    y = np.fft.irfft(np.fft.rfft(x, r) / np.fft.rfft(psf, r), r)
    rec = np.trim_zeros(np.real(y))[: x.size - psf.size - 1]
    if mode == "valid":
        return rec
    elif mode == "same":
        return np.hstack((rec, x[rec.size :]))
    else:  # pragma: no cover
        raise ValueError("Valid modes are 'valid', 'same'.")


def erf(x: float | np.ndarray) -> float | np.ndarray:
    """Error function approximation.

    The maximum error is 1.5e-7 [1].

    Args:
        x: value

    Returns:
        approximation of error function

    References:
        .. [1] Abramowitz, Milton, and Irene A. Stegun, eds. Handbook of mathematical
            functions with formulas, graphs, and mathematical tables.
            Vol. 55. US Government printing office, 1970.
    """
    sign = np.sign(x)
    a = np.array([0.278393, 0.230389, 0.000972, 0.078108])
    p = np.array([[1, 2, 3, 4]]).T
    sum = np.sum(a * np.power(x, p).T, axis=1)
    return sign * (1.0 - 1.0 / (1.0 + sum) ** 4)


def erfinv(x: float) -> float:
    """Inverse error function approximation.

    The maximum error is 6e-3 [2].

    Args:
        x: value

    Returns:
        approximation of inverse error function

    References:
        .. [2] Winitzki, S. A handy approximation for the error function and its inverse
            2008
    """
    sign = np.sign(x)
    x = np.log((1.0 - x) * (1.0 + x))

    tt1 = 2.0 / (np.pi * 0.14) + 0.5 * x
    tt2 = 1.0 / 0.14 * x

    return sign * np.sqrt(-tt1 + np.sqrt(tt1 * tt1 - tt2))


def gamma(x: float) -> float:
    """Gamma function approximation.

    Maximum error of 3e-7 [3].

    Args:
        x: value

    Returns:
        approximation of error function

    References:

    .. [3] Abramowitz, Milton, and Irene A. Stegun, eds. Handbook of mathematical
        functions with formulas, graphs, and mathematical tables.
        Vol. 55. US Government printing office, 1970.
    """
    assert x >= 0.0
    # Use recursion
    b = np.array(
        [
            1.0,
            -0.577191652,
            0.988205891,
            -0.897056937,
            0.918206857,
            -0.756704078,
            0.482199394,
            -0.193527818,
            0.035868343,
        ]
    )
    z = x % 1.0
    n = 1.0 / x if x < 1.0 else np.prod(z + np.arange(1, int(x - z)))
    return n * np.sum(b * np.power(z, np.arange(9)))


def beta_pdf(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    B = (gamma(alpha) * gamma(beta)) / gamma(alpha + beta)
    return x ** (alpha - 1.0) * (1.0 - x) ** (beta - 1.0) / B


def beta(
    size: int, alpha: float, beta: float, scale: float = 1.0, shift: float = 0.0
) -> np.ndarray:
    """Beta distribution.

    Range of 0 to 1. The `scale` and `shift` arguments can be used to change the range.

    Args:
        size: size of distribution
        alpha: alpha term, > 0
        beta: beta term, > 0
        scale: scale x
        shift: shift x

    Returns:
        array of (x, y) points
    """
    x = np.linspace(shift, 1.0 * scale + shift, size)
    y = beta_pdf(x, alpha, beta)
    return np.stack((x, y / y.sum()), axis=1)


# def cauchy_pdf(x: np.ndarray, gamma: float, x0: float) -> np.ndarray:
#     return 1.0 / (np.pi * gamma * (1.0 + np.power((x - x0) / gamma, 2)))


# def cauchy(
#     size: int, gamma: float, x0: float, scale: float = 1.0, shift: float = 0.0
# ) -> np.ndarray:
#     x = np.linspace(-size * 0.5 * scale + shift, size * 0.5 * scale + shift, size,)
#     y = cauchy_pdf(x, gamma, x0)
#     return np.stack((x, y / y.sum()), axis=1)


def exponential_pdf(x: np.ndarray, _lambda: float) -> np.ndarray:
    return _lambda * np.exp(-_lambda * x)


def exponential(
    size: int, _lambda: float, scale: float = 1.0, shift: float = 1e-6
) -> np.ndarray:
    """Exponential distribution.

    Range of 0 to `size`, the `scale` and `shift` arguments can be used to change
    the range.

    Args:
        size: size of distribution
        _lambda: lambda term, > 0
        scale: scale x
        shift: shift x

    Returns:
        array of (x, y) points
    """
    x = np.linspace(shift, size * scale + shift, size)
    y = exponential_pdf(x, _lambda)
    return np.stack((x, y / y.sum()), axis=1)


def inversegamma_pdf(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return ((beta**alpha) / gamma(alpha)) * x ** (-alpha - 1.0) * np.exp(-beta / x)


def inversegamma(
    size: int, alpha: float, beta: float, scale: float = 1.0, shift: float = 1e-6
) -> np.ndarray:
    """Inverse Gamma distribution.

    Range of 0 to `size`, the `scale` and `shift` arguments can be used to change
    the range.

    Args:
        size: size of distribution
        alpha: alpha term, > 0
        beta: beta term, > 0
        scale: scale x
        shift: shift x

    Returns:
        array of (x, y) points
    """
    x = np.linspace(shift, size * scale + shift, size)
    y = inversegamma_pdf(x, alpha, beta)
    return np.stack((x, y / y.sum()), axis=1)


def laplace_pdf(x: np.ndarray, b: float, mu: float) -> np.ndarray:
    return (1.0 / (2.0 * b)) * np.exp(-np.abs(x - mu) / b)


def laplace(
    size: int, b: float, mu: float, scale: float = 1.0, shift: float = 0.0
) -> np.ndarray:
    """Laplace distribution.

    Range of -0.5 * `size` to 0.5 * `size`, the `scale` and `shift` arguments
    can be used to change the range.

    Args:
        size: size of distribution
        b: scale term, > 0
        mu: location
        scale: scale x
        shift: shift x

    Returns:
        array of (x, y) points
    """
    x = np.linspace(-size * 0.5 * scale + shift, size * 0.5 * scale + shift, size)
    y = laplace_pdf(x, b, mu)
    return np.stack((x, y / y.sum()), axis=1)


# def logcauchy_pdf(x: np.ndarray, sigma: float, mu: float) -> np.ndarray:
# return (1.0 / (x * np.pi)) * (sigma / (np.power(np.log(x) - mu, 2) + sigma * sigma))


# def logcauchy(
#     size: int, sigma: float, mu: float, scale: float = 1.0, shift: float = 2e-1
# ) -> np.ndarray:
#     x = np.linspace(shift, size * scale + shift, size)
#     y = logcauchy_pdf(x, sigma, mu)
#     return np.stack((x, y / y.sum()), axis=1)


def loglaplace_pdf(x: np.ndarray, b: float, mu: float) -> np.ndarray:
    return 1.0 / (2.0 * b * x) * np.exp(-np.abs(np.log(x) - mu) / b)


def loglaplace(
    size: int, b: float, mu: float, scale: float = 1.0, shift: float = 1e-6
) -> np.ndarray:
    """Log-Laplace distribution.

    Range of 0 to `size`, the `scale` and `shift` arguments can be used to change
    the range.

    Args:
        size: size of distribution
        b: scale term, > 0
        mu: location
        scale: scale x
        shift: shift x

    Returns:
        array of (x, y) points
    """
    x = np.linspace(shift, size * scale + shift, size)
    y = loglaplace_pdf(x, b, mu)
    return np.stack((x, y / y.sum()), axis=1)


def lognormal_pdf(x: np.ndarray, sigma: float, mu: float) -> np.ndarray:
    return 1.0 / (x * sigma * _s2pi) * np.exp(-0.5 * ((np.log(x) - mu) / sigma) ** 2)


def lognormal(
    size: int, sigma: float, mu: float, scale: float = 1.0, shift: float = 1e-6
) -> np.ndarray:
    """Log-normal distribution.

    Range of 0 to `size`, the `scale` and `shift` arguments can be used to change
    the range.

    Args:
        size: size of distribution
        sigma: sigma term, > 0
        mu: location
        scale: scale x
        shift: shift x

    Returns:
        array of (x, y) points
    """
    x = np.linspace(shift, size * scale + shift, size)
    y = lognormal_pdf(x, sigma, mu)
    return np.stack((x, y / y.sum()), axis=1)


def normal_pdf(x: np.ndarray, sigma: float, mu: float) -> np.ndarray:
    return 1.0 / (sigma * _s2pi) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def normal(
    size: int, sigma: float, mu: float, scale: float = 1.0, shift: float = 0.0
) -> np.ndarray:
    """Normal distribution.

    Range of -0.5 * `size` to 0.5 * `size`, the `scale` and `shift` arguments
    can be used to change the range.

    Args:
        size: size of distribution
        sigma: sqrt variance, > 0
        mu: mean
        scale: scale x
        shift: shift x

    Returns:
        array of (x, y) points
    """
    x = np.linspace(-size * 0.5 * scale + shift, size * 0.5 * scale + shift, size)
    y = normal_pdf(x, sigma, mu)
    return np.stack((x, y / y.sum()), axis=1)


def super_gaussian_pdf(
    x: np.ndarray, sigma: float, mu: float, power: float
) -> np.ndarray:
    return 1.0 / (sigma * _s2pi) * np.exp(-0.5 * ((x - mu) / sigma) ** (2 * power))


def super_gaussian(
    size: int,
    sigma: float,
    mu: float,
    power: float,
    scale: float = 1.0,
    shift: float = 0.0,
) -> np.ndarray:
    """Super-Gaussian distribution.

    Range of -0.5 * `size` to 0.5 * `size`, the `scale` and `shift` arguments
    can be used to change the range.

    Args:
        size: size of distribution
        sigma: sqrt variance, > 0
        mu: mean
        power: exponent
        scale: scale x
        shift: shift x

    Returns:
        array of (x, y) points
    """
    x = np.linspace(-size * 0.5 * scale + shift, size * 0.5 * scale + shift, size)
    y = super_gaussian_pdf(x, sigma, mu, power)
    return np.stack((x, y / y.sum()), axis=1)


def triangular_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    y = np.where(
        x < 0.0, (2.0 * (x - a)) / (a * (a - b)), (2.0 * (b - x)) / (b * (b - a))
    )
    y[x == 0.0] = 2.0 / (b - a)
    y[np.logical_or(x < a, x > b)] = 0.0
    return y


def triangular(
    size: int, a: float, b: float, scale: float = 1.0, shift: float = 0.0
) -> np.ndarray:
    """Triangular distribution.

    Range of -0.5 * `size` to 0.5 * `size`, the `scale` and `shift` arguments
    can be used to change the range.

    Args:
        size: size of distribution
        a: a <= x <= b
        b: a < b
        c: a <= c <= b
        scale: scale x
        shift: shift x

    Returns:
        array of (x, y) points
    """
    x = np.linspace(-size * 0.5 * scale + shift, size * 0.5 * scale + shift, size)
    y = triangular_pdf(x, a, b)
    return np.stack((x, y / y.sum()), axis=1)
