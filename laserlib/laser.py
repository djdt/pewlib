import numpy as np
import numpy.lib.recfunctions as rfn
import copy

from .calibration import Calibration
from .config import Config

from typing import Dict, List, Tuple


class Laser(object):
    def __init__(
        self,
        data: np.ndarray,
        calibration: Dict[str, Calibration] = None,
        config: Config = None,
        name: str = "",
        path: str = "",
    ):
        assert data.dtype.names is not None

        self.data = data
        self.calibration = {name: Calibration() for name in data.dtype.names}
        if calibration is not None:
            self.calibration.update(calibration)

        self.config = copy.copy(config) if config is not None else Config()

        self.name = name
        self.path = path

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        return self.config.data_extent(self.array.shape[:2])

    @property
    def isotopes(self) -> List[str]:
        return self.data.dtype.names

    @property
    def shape(self) -> List[int]:
        return self.data.shape

    @property
    def layers(self) -> int:
        return 1

    def add(self, isotope: str, data: np.ndarray) -> None:
        assert data.shape == self.data.shape
        rfn.append_fields(self.data, isotope, data, usemask=False)

    def remove(self, isotope: str) -> None:
        rfn.drop_fields(self.data, isotope, usemask=False)

    def get(self, isotope: str = None, **kwargs) -> np.ndarray:
        """Valid kwargs are calibrate, extent, flat."""
        if isotope is None:
            data = self.data.copy()
        else:
            data = self.data[isotope]

        if "extent" in kwargs:
            x0, x1, y0, y1 = kwargs["extent"]
            px, py = self.config.get_pixel_width(), self.config.get_pixel_height()
            x0, x1 = int(x0 / px), int(x1 / px)
            y0, y1 = int(y0 / py), int(y1 / py)
            # We have to invert the extent, as mpl use bottom left y coords
            ymax = data.shape[0]
            data = data[ymax - y1 : ymax - y0, x0:x1]

        if kwargs.get("calibrate", False):
            data = self.calibration[isotope].calibrate(data)

        return data

    @classmethod
    def from_list(
        cls,
        isotopes: List[str],
        datas: List[np.ndarray],
        config: Config = None,
        name: str = "",
        path: str = "",
    ):  # type: ignore
        dtype = [(isotope, float) for isotope in isotopes]
        data = np.empty(datas[0].shape, dtype=dtype)
        return cls(data=data, config=config, name=name, path=path)
