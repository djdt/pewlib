import numpy as np

from .calc import get_weights, weighted_linreg


class LaserCalibration(object):
    DEFAULT_UNIT = ""

    def __init__(
        self,
        intercept: float = 0.0,
        gradient: float = 1.0,
        unit: str = None,
        rsq: float = None,
        points: np.ndarray = None,
        weighting: str = None,
    ):
        self.intercept = intercept
        self.gradient = gradient
        self.unit = unit if unit is not None else LaserCalibration.DEFAULT_UNIT

        self.rsq = rsq
        self.points = points
        self.weighting = weighting

    def __str__(self) -> str:
        return f"{self.gradient:.4g}*x{self.intercept:+.4g}"

    def update_from_points(self) -> None:
        x = self.points[:, 0]
        y = self.points[:, 1]
        self.intercept, self.gradient, self.rsq = weighted_linreg(
            x, y, get_weights(x, self.weighting)
        )

    def calibrate(self, data: np.ndarray) -> np.ndarray:
        if self.intercept == 0.0 and self.gradient == 1.0:
            return data
        return (data - self.intercept) / self.gradient
