import numpy as np

from .calibration import LaserCalibration
from .config import LaserConfig


class LaserData(object):
    def __init__(self, data: np.ndarray, calibration: LaserCalibration = None):
        self.data = data
        self.calibration = (
            calibration if calibration is not None else LaserCalibration()
        )

    def get(self, config: LaserConfig, **kwargs) -> np.ndarray:
        data = self.data.copy()

        if "extent" in kwargs:
            x0, x1, y0, y1 = kwargs.get("extent", (0.0, 0.0, 0.0, 0.0))
            px, py = config.pixel_size()
            x0, x1 = int(x0 / px), int(x1 / px)
            y0, y1 = int(y0 / py), int(y1 / py)
            # We have to invert the extent, as mpl use bottom left y coords
            data = data[-y1:-y0, x0:x1]

        if kwargs.get("calibrate", False):
            data = self.calibration.calibrate(data)

        return data
