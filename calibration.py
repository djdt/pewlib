import numpy as np


class LaserCalibration(object):
    DEFAULT_UNIT = ""

    def __init__(self, intercept: float = 0.0, gradient: float = 1.0, unit: str = None):
        self.gradient = gradient
        self.intercept = intercept
        self.unit = unit if unit is not None else LaserCalibration.DEFAULT_UNIT

    def calibrate(self, data: np.ndarray) -> np.ndarray:
        if self.intercept == 0.0 and self.gradient == 1.0:
            return data
        return (data - self.intercept) / self.gradient
