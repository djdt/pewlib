"""Class for data collected line-by-line.
Line-by-line data is collected in multiple lines, with each line performed as a
continuous ablation in one direction. The lines are then stacked to form an image.
"""

import copy

import numpy as np
import numpy.lib.recfunctions as rfn

from pewlib.calibration import Calibration
from pewlib.config import Config


class Laser(object):
    """Class for line-by-line laser data.

    Args:
        data: structured array of elemental data
        calibration: dict mapping elements to calibrations, optional
        config: laser parameters
        info: dict (str, str) of additional info

    Todo:
        Support rastered collection.
    """

    def __init__(
        self,
        data: np.ndarray,
        calibration: dict[str, Calibration] | None = None,
        config: Config | None = None,
        info: dict[str, str] | None = None,
    ):
        self.data: np.ndarray = data
        self.calibration = {name: Calibration() for name in self.elements}
        if calibration is not None:
            self.calibration.update(copy.deepcopy(calibration))

        if config is None:
            self.config = Config()
        else:
            self.config = copy.copy(config)

        self.info = info or {}

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Image extent in μm"""
        return self.config.data_extent(self.shape[:2])

    @property
    def elements(self) -> tuple[str, ...]:
        """Elements stored."""
        return self.data.dtype.names

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def layers(self) -> int:
        return 1

    def add(
        self, element: str, data: np.ndarray, calibration: Calibration | None = None
    ) -> None:
        """Adds a new element.

        Args:
            element: element name
            data: array
            calibration: calibration for data, optional
        """
        assert data.shape == self.data.shape
        new_dtype = self.data.dtype.descr + [(element, data.dtype.str)]

        new_data = np.empty(self.data.shape, dtype=new_dtype)
        for name in self.data.dtype.names:
            new_data[name] = self.data[name]
        new_data[element] = data
        self.data = new_data

        if calibration is None:
            calibration = Calibration()
        self.calibration[element] = calibration

    def remove(self, names: str | list[str]) -> None:
        """Remove element(s)."""
        if isinstance(names, str):
            names = [names]
        self.data = rfn.drop_fields(self.data, names, usemask=False)
        for name in names:
            self.calibration.pop(name)

    def rename(self, names: dict[str, str]) -> None:
        """Change the name of element(s).

        Args:
            names: dict mapping old to new names
        """
        self.data = rfn.rename_fields(self.data, names)
        for old, new in names.items():
            self.calibration[new] = self.calibration.pop(old)

    def get(
        self,
        element: str | None = None,
        calibrate: bool | None = False,
        extent: tuple[float, float, float, float] | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Get elemental data.

        If `element` is None then all elements are returned in a structured array.

        Args:
            element: element name, optional
            calibrate: apply calibration
            extent: trim to extent, μm

        Returns:
            structured if element is None else unstructured
        """
        if element is None:
            data = self.data.copy()
        else:
            data = self.data[element]

        if extent is not None:
            x0, x1, y0, y1 = extent
            px, py = self.config.get_pixel_width(), self.config.get_pixel_height()
            x0, x1 = int(x0 / px), int(x1 / px)
            y0, y1 = int(y0 / py), int(y1 / py)
            data = data[y0:y1, x0:x1]

        if calibrate:
            if element is None:  # Perform calibration on all data
                for name in data.dtype.names:
                    data[name] = self.calibration[name].calibrate(data[name])
            else:
                data = self.calibration[element].calibrate(data)

        return data

    @classmethod
    def from_list(
        cls,
        elements: list[str],
        datas: list[np.ndarray],
        config: Config | None = None,
        info: dict[str, str] = {},
    ) -> "Laser":
        """Creates class from a list of names and unstructured arrays."""
        assert len(elements) == len(datas)
        dtype = [(element, float) for element in elements]

        structured = np.empty(datas[0].shape, dtype=dtype)
        for element, data in zip(elements, datas):
            structured[element] = data

        return cls(data=structured, config=config, info=info)
