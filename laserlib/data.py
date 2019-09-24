import numpy as np
import numpy.lib.recfunctions as recfunctions

from .calibration import Calibration
from .config import Config

from typing import List


class Data(object):
    def __init__(self, array: np.ndarray, calibrations: dict = None):
        self.array = array
        self.calibrations = {name: Calibration() for name in self.isotopes}
        if calibrations is not None:
            self.calibrations.update(calibrations)

    @property
    def isotopes(self) -> List[str]:
        return self.array.dtype.names

    @property
    def shape(self) -> List[int]:
        return self.array.shape

#     def __add__(self, data: "Data") -> None:
#         recfunctions.merge_arrays((self.array, data.array))
#         self.calibrations.update(data.calibrations)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.array[key]

    # def get(self, isotope: str, **kwargs) -> np.ndarray:
    #     data = self.data[isotope]

    #     if kwargs.get("calibrate", False):
    #         data = self.calibrations[isotope].calibrate(data)

    #     return data


class IsotopeData(object):
    def __init__(self, isotope: str, data: np.ndarray, calibration: Calibration = None):
        self.isotope = isotope
        self.data = data
        self.calibration = calibration if calibration is not None else Calibration()

    @property
    def shape(self) -> List[int]:
        return self.data.shape

    def get(self, config: Config, **kwargs) -> np.ndarray:
        data = self.data

        if "extent" in kwargs:
            x0, x1, y0, y1 = kwargs.get("extent", (0.0, 0.0, 0.0, 0.0))
            px, py = config.get_pixel_width(), config.get_pixel_height()
            x0, x1 = int(x0 / px), int(x1 / px)
            y0, y1 = int(y0 / py), int(y1 / py)
            # We have to invert the extent, as mpl use bottom left y coords
            ymax = data.shape[0]
            data = data[ymax - y1 : ymax - y0, x0:x1]

        if kwargs.get("calibrate", False):
            data = self.calibration.calibrate(data)

        return data
