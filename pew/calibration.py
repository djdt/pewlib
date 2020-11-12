import numpy as np

from typing import Tuple, Union


def weighting(x: np.ndarray, weighting: str, safe: bool = True) -> np.ndarray:
    if safe and x.size > 0:  # Avoid div 0 problems
        x = x.copy()
        x[x == 0] = np.nanmin(x[x != 0])

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
    if np.iscomplexobj(c):  # pragma: no cover
        np.clip(c.imag, -1, 1, out=c.imag)

    return c[0, 1] ** 2.0


def weighted_linreg(
    x: np.ndarray, y: np.ndarray, w: np.ndarray = None
) -> Tuple[float, float, float, float]:
    coef, stats = np.polynomial.polynomial.polyfit(
        x, y, 1, w=w if w is None else np.sqrt(w), full=True
    )
    r2 = weighted_rsq(x, y, w)
    error = np.sqrt(np.sum(stats[0]) / (x.size - 2)) if x.size > 2 else 0.0

    return coef[1], coef[0], r2, error


class Calibration(object):
    def __init__(
        self,
        intercept: float = 0.0,
        gradient: float = 1.0,
        rsq: float = None,
        error: float = None,
        points: np.ndarray = None,
        weights: Union[np.ndarray, str] = None,
        weighting: str = "None",
        unit: str = "",
    ):
        """Weights can be automatically generated by passing
        x, 1/x, or 1/(x^2) to weights."""

        self.intercept = intercept
        self.gradient = gradient
        self.unit = unit

        self.rsq = rsq
        self.error = error

        self._points: np.ndarray = np.array([[], []], dtype=np.float64)
        if points is not None:
            self.points = points

        self._weights: np.ndarray = np.array([], dtype=np.float64)
        if weights is not None:
            self.weights = weights
        self.weighting = weighting

    @property
    def points(self) -> np.ndarray:
        return self._points

    @points.setter
    def points(self, points: np.ndarray) -> None:
        points = np.array(points, dtype=np.float64)
        if points.ndim != 2:
            raise ValueError("Points must have 2 dimensions.")
        self._points = points

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @weights.setter
    def weights(self, weights: Union[np.ndarray, str]) -> None:
        if isinstance(weights, str):
            self._weights = weighting(self._points[:, 0], weights)
        else:
            weights = np.array(weights, dtype=np.float64)
            if weights.ndim != 1 or weights.size != self.points.size // 2:
                raise ValueError("Weights must have same length as points.")
            self._weights = weights

    def __str__(self) -> str:
        s = f"y = {self.gradient:.4g} · x - {self.intercept:.4g}"
        if self.rsq is not None:
            s += f"\nr² = {self.rsq:.4f}"
        return s

    def concentrations(self) -> np.ndarray:
        return self.points[:, 0]

    def counts(self) -> np.ndarray:
        return self.points[:, 1]

    def update_linreg(self) -> None:
        if self.points.size > 0:
            self.gradient, self.intercept, self.rsq = 1.0, 0.0, None
        else:
            no_nans = ~np.isnan(self.points).any(axis=1)
            if np.count_nonzero(no_nans) == 0:
                self.gradient, self.intercept, self.rsq = 1.0, 0.0, None
            else:
                x, y = self.points[no_nans, 0], self.points[no_nans, 1]
                w = self._weights
                if w.size > 0:
                    w = w[no_nans]
                self.gradient, self.intercept, self.rsq, self.error = weighted_linreg(
                    x, y, w
                )

    def calibrate(self, data: np.ndarray) -> np.ndarray:
        if self.intercept == 0.0 and self.gradient == 1.0:
            return data
        return (data - self.intercept) / self.gradient

    def to_array(self) -> np.ndarray:
        points = self.points
        weights = self.weights
        return np.array(
            (
                self.intercept,
                self.gradient,
                self.unit,
                self.rsq,
                self.error,
                points,
                weights,
                self.weighting,
            ),
            dtype=[
                ("intercept", np.float64),
                ("gradient", np.float64),
                ("unit", "U32"),
                ("rsq", np.float64),
                ("error", np.float64),
                ("points", points.dtype, points.shape),
                ("weights", weights.dtype, weights.shape),
                ("weighting", "U32"),
            ],
        )

    @classmethod
    def from_array(cls, array: np.ndarray) -> "Calibration":
        cdict = {name: array[name] for name in array.dtype.names}
        # Swap out nan for None
        if cdict["rsq"] == np.nan:
            cdict["rsq"] = None
        if cdict["error"] == np.nan:
            cdict["error"] = None
        return cls(**cdict)

    @classmethod
    def from_points(
        cls,
        points: np.ndarray,
        weights: Union[np.ndarray, str] = None,
        weighting: str = "None",
        unit: str = "",
    ) -> "Calibration":
        calibration = cls(
            points=points, weights=weights, weighting=weighting, unit=unit
        )
        calibration.update_linreg()
        return calibration
