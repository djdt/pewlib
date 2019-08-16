import numpy as np

from .calc import get_weights, weighted_linreg


class LaserCalibration(object):
    def __init__(
        self,
        intercept: float = 0.0,
        gradient: float = 1.0,
        rsq: float = None,
        points: np.ndarray = None,
        weighting: str = None,
        unit: str = "",
    ):
        self.intercept = intercept
        self.gradient = gradient
        self.unit = unit

        self.rsq = rsq
        self._points = None
        if points is not None:
            self.points = points
        self.weighting = weighting

    @property
    def points(self) -> np.ndarray:
        return self._points

    @points.setter
    def points(self, points: np.ndarray) -> None:
        points = np.array(points, dtype=float)
        if points.ndim != 2:
            raise ValueError("Points must have 2 dimensions.")
        if points.shape[0] < 2 or points.shape[1] != 2:
            raise ValueError("A minimum of 2 points required.")
        self._points = points

    def __str__(self) -> str:
        s = f"y = {self.gradient:.4g} · x - {self.intercept:.4g}"
        if self.rsq is not None:
            s += f"\nr² = {self.rsq:.4f}"
        return s

    def concentrations(self) -> np.ndarray:
        if self.points is None:
            return np.array([], dtype=np.float64)
        return self.points[:, 0]

    def counts(self) -> np.ndarray:
        if self.points is None:
            return np.array([], dtype=np.float64)
        return self.points[:, 1]

    def update_linreg(self) -> None:
        if self.points is None:
            self.gradient, self.intercept, self.rsq = 1.0, 0.0, None
        else:
            no_nans = self.points[~np.isnan(self.points).any(axis=1)]
            if no_nans.size == 0 or no_nans.ndim != 2:
                self.gradient, self.intercept, self.rsq = 1.0, 0.0, None
            else:
                x, y = no_nans[:, 0], no_nans[:, 1]
                self.gradient, self.intercept, self.rsq = weighted_linreg(
                    x, y, get_weights(x, self.weighting)
                )

    def calibrate(self, data: np.ndarray) -> np.ndarray:
        if self.intercept == 0.0 and self.gradient == 1.0:
            return data
        return (data - self.intercept) / self.gradient

    @classmethod
    def from_points(
        cls, points: np.ndarray, weighting: str = None, unit: str = ""
    ):  # type: ignore
        calibration = cls(points=points, weighting=weighting, unit=unit)
        calibration.update_linreg()
        return calibration
