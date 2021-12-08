import numpy as np

from typing import Tuple, Union


def weights_from_weighting(
    x: np.ndarray, weighting: str, safe: bool = True
) -> np.ndarray:
    """Get weighting for `x`.

    Conveience function for calculating simple weightings. If `safe` then any
    zeros in `x` are replace with the minimum non-zero value.

    Args:
        x: 1d-array
        weighting: weighting string {'Equal', 'x', '1/x', '1/(x^2)'}
        safe: replace zeros with minimum

    Returns:
        weights, same size as x
    """
    if x.size == 0:
        return np.empty(0, dtype=x.dtype)

    if safe:
        if np.all(x == 0):  # Impossible weighting
            return np.ones_like(x)
        x = x.copy()
        x[x == 0] = np.nanmin(x[x != 0])

    if weighting == "Equal":
        return np.ones_like(x)
    elif weighting == "x":
        return x
    elif weighting == "1/x":
        return 1.0 / x
    elif weighting == "1/(x^2)":
        return 1.0 / (x ** 2.0)
    else:
        raise ValueError(f"Unknown weighting {weighting}.")


def weighted_rsq(x: np.ndarray, y: np.ndarray, w: np.ndarray = None) -> float:
    """Calculate r² for weighted linear regression.

    Args:
        x: 1d-array
        y: array, same size as `x`
        w: weights, same size as `x`
    """
    c = np.cov(x, y, aweights=w)
    d = np.diag(c)
    stddev = np.sqrt(d.real)
    c /= stddev[:, None]
    c /= stddev[None, :]

    np.clip(c.real, -1, 1, out=c.real)
    if np.iscomplexobj(c):  # pragma: no cover
        np.clip(c.imag, -1, 1, out=c.imag)

    return c[0, 1] ** 2.0


def weighted_linreg(
    x: np.ndarray, y: np.ndarray, w: np.ndarray = None
) -> Tuple[float, float, float, float]:
    """Weighted linear regression.

    Uses polyfit with sqrt(weights) for intercept and gradient.

    Args:
        x: 1d-array
        y: array, same size as `x`
        w: weights, same size as `x`

    Returns:
       gradient
       intercept
       r²
       error, S(y,x) the (unweighted) residual standard deviation

    See Also:
        :func:`pewlib.calibration.weighted_rsq`
    """
    coef = np.polynomial.polynomial.polyfit(x, y, 1, w=w if w is None else np.sqrt(w))
    r2 = weighted_rsq(x, y, w)
    error = np.sqrt(np.sum(((coef[0] + x * coef[1]) - y) ** 2) / (x.size - 2))

    return coef[1], coef[0], r2, error


class Calibration(object):
    """Class for calibration storage and calculations.

    Weights can be automatically generated by passing weighting string to weights.

    Args:
        intercept: of line-of-best-fit
        gradient: of line-of-best-fit
        unit: calibration units, eg. 'ppm'
        rsq: r² of line-of-best-fit
        error: error in line-of-best-fit
        points: array of (x, y)
        weights: weighting {'Equal', 'x', '1/x', '1/(x^2)', 'y', '1/y', '1/(y^2)'}
            or (name, array) of weights for linear-regression, same length as `points`
    """

    KNOWN_WEIGHTING = ["Equal", "x", "1/x", "1/(x^2)", "y", "1/y", "1/(y^2)"]

    def __init__(
        self,
        intercept: float = 0.0,
        gradient: float = 1.0,
        unit: str = "",
        rsq: float = None,
        error: float = None,
        points: np.ndarray = None,
        weights: Union[str, Tuple[str, np.ndarray]] = "Equal",
    ):
        self.intercept = intercept
        self.gradient = gradient
        self.unit = unit

        self.rsq = rsq
        self.error = error

        self._points: np.ndarray = np.empty((0, 2), dtype=np.float64)
        self._weights: np.ndarray = np.empty(0, dtype=np.float64)
        self.weighting: str = ""

        if points is not None:
            self.points = points

        self.weights = weights

    @property
    def x(self) -> np.ndarray:
        return self._points[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self._points[:, 1]

    @property
    def points(self) -> np.ndarray:
        return self._points

    @points.setter
    def points(self, points: np.ndarray) -> None:
        points = np.array(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Points must have shape (n, 2).")
        self._points = points

    @property
    def weights(self) -> np.ndarray:
        if self.weighting in Calibration.KNOWN_WEIGHTING:
            if "y" in self.weighting:
                return weights_from_weighting(self.y, self.weighting.replace("y", "x"))
            else:
                return weights_from_weighting(self.x, self.weighting)
        return self._weights

    @weights.setter
    def weights(self, weights: Union[str, Tuple[str, np.ndarray]]) -> None:
        if isinstance(weights, str):
            self.weighting = weights
            self._weights = np.empty(0, dtype=np.float64)
        else:
            self.weighting = weights[0]
            w = np.array(weights[1], dtype=np.float64)
            if w.size != self.x.size:
                raise ValueError("Weights must be same length as points.")
            self._weights = w

    def __str__(self) -> str:
        s = f"y = {self.gradient:.4g} · x - {self.intercept:.4g}"
        if self.rsq is not None:
            s += f"\nr² = {self.rsq:.4f}"
        return s

    def calibrate(self, data: np.ndarray) -> np.ndarray:
        if self.intercept == 0.0 and self.gradient == 1.0:
            return data
        return (data - self.intercept) / self.gradient

    def update_linreg(self) -> None:
        if self.points.size == 0:
            self.gradient, self.intercept, self.rsq, self.error = 1.0, 0.0, None, None
        else:
            no_nans = ~(np.isnan(self.points).any(axis=1))
            if np.count_nonzero(no_nans) < 2:
                self.gradient, self.intercept, self.rsq, self.error = (
                    1.0,
                    0.0,
                    None,
                    None,
                )
            else:
                x, y, w = self.x[no_nans], self.y[no_nans], self.weights[no_nans]
                self.gradient, self.intercept, self.rsq, self.error = weighted_linreg(
                    x, y, w
                )

    def to_array(self) -> np.ndarray:
        points = self.points
        unit = np.array(self.unit)
        weights = np.array(self.weights)
        weighting = np.array(self.weighting)
        return np.array(
            (
                self.intercept,
                self.gradient,
                unit,
                self.rsq,
                self.error,
                points,
                weights,
                weighting,
            ),
            dtype=[
                ("intercept", np.float64),
                ("gradient", np.float64),
                ("unit", unit.dtype),
                ("rsq", np.float64),
                ("error", np.float64),
                ("points", points.dtype, points.shape),
                ("weights", weights.dtype, weights.shape),
                ("weighting", weighting.dtype),
            ],
        )

    @classmethod
    def from_array(cls, array: np.ndarray) -> "Calibration":
        if array["weighting"] in Calibration.KNOWN_WEIGHTING:
            weights = str(array["weighting"])
        else:
            weights = (str(array["weighting"]), array["weights"])  # type: ignore
        return cls(
            intercept=float(array["intercept"]),
            gradient=float(array["gradient"]),
            unit=str(array["unit"]),
            rsq=None if np.isnan(array["rsq"]) else float(array["rsq"]),
            error=None if np.isnan(array["error"]) else float(array["error"]),
            points=array["points"],
            weights=weights,
        )

    @classmethod
    def from_points(
        cls,
        points: np.ndarray,
        unit: str = "",
        weights: Union[str, Tuple[str, np.ndarray]] = "Equal",
    ) -> "Calibration":
        """Create a :class:`Calibration` from points.

        Calulates linear-regression params from `points`."""
        calibration = cls(points=points, weights=weights, unit=unit)
        calibration.update_linreg()
        return calibration
