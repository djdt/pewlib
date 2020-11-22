"""Class for data collected line-by-line.
Line-by-line data is collected in multiple lines, with each line performed as a
continuous ablation in one direction. The lines are then stacked to form an image.
"""
import numpy as np
import numpy.lib.recfunctions as rfn
from pathlib import Path
import copy

from pewlib.calibration import Calibration
from pewlib.config import Config

from typing import Any, Dict, List, Tuple, Union


class _Laser:
    """Base class for other laser classes."""

    data: Union[np.ndarray, List[np.ndarray]] = None
    calibration: Dict[str, Calibration] = None
    config: Config = None
    name: str = ""
    path: Path = None

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

    def add(
        self, isotope: str, data: Any, calibration: Calibration = None
    ) -> None:  # pragma: no cover
        raise NotImplementedError

    def remove(self, names: Union[str, List[str]]) -> None:  # pragma: no cover
        raise NotImplementedError

    def rename(self, names: Dict[str, str]) -> None:  # pragma: no cover
        raise NotImplementedError

    def get(self, isotope: str, **kwargs) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class Laser(_Laser):
    """Class for line-by-line laser data.

    Args:
        data: structured array of elemental data
        calibration: dict mapping elements to calibrations, optional
        config: laser parameters
        name: name of image
        path: path to file

    Todo:
        Support rastered collection.
    """

    def __init__(
        self,
        data: np.ndarray,
        calibration: Dict[str, Calibration] = None,
        config: Config = None,
        name: str = "",
        path: Path = None,
    ):
        self.data: np.ndarray = data
        self.calibration = {name: Calibration() for name in self.isotopes}
        if calibration is not None:
            self.calibration.update(copy.deepcopy(calibration))

        self.config = copy.copy(config) if config is not None else Config()

        self.name = name
        self.path = path or Path()

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """Image extent in μm"""
        return self.config.data_extent(self.shape[:2])

    @property
    def isotopes(self) -> Tuple[str, ...]:
        """Elements stored."""
        return self.data.dtype.names

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def add(
        self, isotope: str, data: np.ndarray, calibration: Calibration = None
    ) -> None:
        """Adds a new element.

        Args:
            isotope: element name
            data: array
            calibration: calibration for data, optional
        """
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

    def remove(self, names: Union[str, List[str]]) -> None:
        """Remove element(s)."""
        if isinstance(names, str):
            names = [names]
        self.data = rfn.drop_fields(self.data, names, usemask=False)
        for name in names:
            self.calibration.pop(name)

    def rename(self, names: Dict[str, str]) -> None:
        """Change the name of element(s).

        Args:
            names: dict mapping old to new names
        """
        self.data = rfn.rename_fields(self.data, names)
        for old, new in names.items():
            self.calibration[new] = self.calibration.pop(old)

    def get(
        self,
        isotope: str = None,
        calibrate: bool = False,
        extent: Tuple[float, float, float, float] = None,
        **kwargs,
    ) -> np.ndarray:
        """Get elemental data.

        If `isotope` is None then all elements are returned in a structured array.

        Args:
            isotope: element name, optional
            calibrate: apply calibration
            extent: trim to extent, μm

        Returns:
            structured if isotope is None else unstructured
        """
        if isotope is None:
            data = self.data.copy()
        else:
            data = self.data[isotope]

        if extent is not None:
            x0, x1, y0, y1 = extent
            px, py = self.config.get_pixel_width(), self.config.get_pixel_height()
            x0, x1 = int(x0 / px), int(x1 / px)
            y0, y1 = int(y0 / py), int(y1 / py)
            data = data[y0:y1, x0:x1]

        if calibrate:
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
        path: Path = None,
    ) -> "Laser":
        """Creates class from a list of names and unstructured arrays."""
        assert len(isotopes) == len(datas)
        dtype = [(isotope, float) for isotope in isotopes]

        structured = np.empty(datas[0].shape, dtype=dtype)
        for isotope, data in zip(isotopes, datas):
            structured[isotope] = data

        return cls(data=structured, config=config, name=name, path=path)
