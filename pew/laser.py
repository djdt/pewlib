import numpy as np
import copy

from pew.calibration import Calibration
from pew.config import Config

from typing import Dict, List, Tuple, Union


class _Laser:
    data: Union[np.ndarray, List[np.ndarray]] = None
    calibration: Dict[str, Calibration] = None
    config: Config = None
    name = ""
    path = ""

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        return (0.0, 0.0, 0.0, 0.0)

    @property
    def isotopes(self) -> Tuple[str, ...]:
        return ()

    @property
    def layers(self) -> int:
        return 1

    @property
    def shape(self) -> Tuple[int, ...]:
        return (0, 0)

    def add(self, isotope: str, data: Union[np.ndarray, List[np.ndarray]]) -> None:
        raise NotImplementedError

    def remove(self, isotope: str) -> None:
        raise NotImplementedError

    def get(self, isotope: str, **kwargs) -> np.ndarray:
        raise NotImplementedError


class Laser(_Laser):
    def __init__(
        self,
        data: np.ndarray,
        calibration: Dict[str, Calibration] = None,
        config: Config = None,
        name: str = "",
        path: str = "",
    ):
        self.data: np.ndarray = data
        self.calibration = {name: Calibration() for name in self.isotopes}
        if calibration is not None:
            self.calibration.update(copy.deepcopy(calibration))

        self.config = copy.copy(config) if config is not None else Config()

        self.name = name
        self.path = path

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        return self.config.data_extent(self.shape[:2])

    @property
    def isotopes(self) -> Tuple[str, ...]:
        return self.data.dtype.names

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def add(
        self, isotope: str, data: np.ndarray, calibration: Calibration = None
    ) -> None:
        assert data.shape == self.data.shape
        new_dtype = self.data.dtype.descr + [(isotope, data.dtype.str)]

        new_data = np.empty(self.data.shape, dtype=new_dtype)
        for name in self.data.dtype.names:
            new_data[name] = self.data[name]
        new_data[isotope] = data
        self.data = new_data

        if calibration is None:
            calibration = Calibration()
        self.calibration[isotope] = calibration

    def remove(self, isotope: str) -> None:
        new_dtype = [descr for descr in self.data.dtype.descr if descr[0] != isotope]

        new_data = np.empty(self.data.shape, dtype=new_dtype)
        for name in self.data.dtype.names:
            if name != isotope:
                new_data[name] = self.data[name]
        self.data = new_data

        self.calibration.pop(isotope)

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
            if isotope is None:  # Perform calibration on all data
                for name in data.dtype.names:
                    data[name] = self.calibration[name].calibrate(data[name])
            else:
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
    ) -> "Laser":
        assert len(isotopes) == len(datas)
        dtype = [(isotope, float) for isotope in isotopes]

        structured = np.empty(datas[0].shape, dtype=dtype)
        for isotope, data in zip(isotopes, datas):
            structured[isotope] = data

        return cls(data=structured, config=config, name=name, path=path)
