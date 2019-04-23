import numpy as np
import copy

from .calibration import LaserCalibration
from .config import LaserConfig

from typing import Dict, List, Tuple


class Laser(object):
    def __init__(
        self,
        data: np.ndarray = np.array([], dtype=float),
        calibration: Dict[str, LaserCalibration] = None,
        config: LaserConfig = None,
        name: str = "",
        filepath: str = "",
    ):
        self.data = data

        self.config = copy.copy(config) if config is not None else LaserConfig()
        if calibration is not None:
            self.calibration = calibration
        else:
            self.calibration = {n: LaserCalibration() for n in self.data.dtype.names}

        self.name = name
        self.filepath = filepath

    def isotopes(self) -> List[str]:
        return self.data.dtype.names()

    def get(
        self,
        name: str,
        calibrate: bool = False,
        extent: Tuple[float, float, float, float] = None,
    ) -> np.ndarray:
        # Calibration
        data = self.data[name]

        if extent is not None:
            px, py = self.config.pixel_size()
            x1, x2 = int(extent[0] / px), int(extent[1] / px)
            y1, y2 = int(extent[2] / py), int(extent[3] / py)
            # We have to invert the extent, as mpl use bottom left y coords
            yshape = data.shape[0]
            data = data[yshape - y2 : yshape - y1, x1:x2]

        if calibrate:
            data = self.calibration[name].calibrate(data)

        return data

    def convert(self, x: float, unit_from: str, unit_to: str) -> float:
        # Convert into rows
        if unit_from in ["s", "seconds"]:
            x = x / self.config.scantime
        elif unit_from in ["um", "μm", "micro meters"]:
            x = x / self.config.pixel_width()
        # Convert to desired unit
        if unit_to in ["s", "seconds"]:
            x = x * self.config.scantime
        elif unit_to in ["um", "μm", "micro meters"]:
            x = x * self.config.pixel_width()
        return x

    # def extent(self) -> Tuple[float, float, float, float]:
    #     # Image data is stored [rows][cols]
    #     x = self.width() * self.config.pixel_width()
    #     y = self.height() * self.config.pixel_height()
    #     return (0.0, x, 0.0, y)

    @staticmethod
    def formatName(name: str) -> str:
        pass
