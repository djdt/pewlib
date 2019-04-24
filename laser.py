import numpy as np
import copy

from .calibration import LaserCalibration
from .config import LaserConfig

from typing import Dict, List, Tuple


class Laser(object):
    def __init__(
        self,
        data: np.ndarray = None,
        calibration: Dict[str, LaserCalibration] = None,
        config: LaserConfig = None,
        name: str = "",
        filepath: str = "",
    ):
        self.data = data if data is not None else np.zeros((1, 1), dtype=float)

        self.config = copy.copy(config) if config is not None else LaserConfig()
        if calibration is not None:
            self.calibration = copy.deepcopy(calibration)
        else:
            self.calibration = {n: LaserCalibration() for n in self.data.dtype.names}

        self.name = name
        self.filepath = filepath

    def isotopes(self) -> List[str]:
        return self.data.dtype.names

    def get(
        self,
        name: str = None,
        calibrate: bool = False,
        extent: Tuple[float, float, float, float] = None,
    ) -> np.ndarray:
        # Calibration
        if name is None:
            data = self.data.copy()
        else:
            data = self.data[name]

        if extent is not None:
            px, py = self.config.pixel_size()
            x1, x2 = int(extent[0] / px), int(extent[1] / px)
            y1, y2 = int(extent[2] / py), int(extent[3] / py)
            # We have to invert the extent, as mpl use bottom left y coords
            yshape = data.shape[0]
            data = data[yshape - y2 : yshape - y1, x1:x2]

        if calibrate:
            if name is None:
                for name in data.dtype.names:
                    data[name] = self.calibration[name].calibrate(data[name])
            else:
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

    def add_isotope(
        self, name: str, data: np.ndarray, calibration: LaserCalibration = None
    ) -> bool:
        if name in self.data.dtype.names:
            return False
        np.lib.recfunctions.append_fields(self.data, name, data)
        self.calibration[name] = (
            calibration if calibration is not None else LaserCalibration()
        )
        return True

    def remove_isotope(self, name: str) -> bool:
        if name not in self.data.data.names:
            return False
        np.lib.recfunctions.drop_fields(self.data, name)
        self.calibration.pop(name)
        return True

    # def extent(self) -> Tuple[float, float, float, float]:
    #     # Image data is stored [rows][cols]
    #     x = self.width() * self.config.pixel_width()
    #     y = self.height() * self.config.pixel_height()
    #     return (0.0, x, 0.0, y)

    @staticmethod
    def formatName(name: str) -> str:
        pass
