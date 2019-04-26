import numpy as np
import copy

from .config import LaserConfig
from .data import LaserData

from typing import Any, Dict, List


class Laser(object):
    def __init__(
        self,
        data: Dict[str, LaserData] = None,
        config: LaserConfig = None,
        name: str = "",
        filepath: str = "",
    ):
        self.data = data if data is not None else {}
        self.config = copy.copy(config) if config is not None else LaserConfig()

        self.name = name
        self.filepath = filepath

    def isotopes(self) -> List[str]:
        return list(self.data.keys())

    def get(self, isotope: str, **kwargs: Any) -> np.ndarray:
        """Valid kwargs are calibrate, extent."""
        if isotope not in self.data:
            return np.zeros((1, 1), dtype=float)

        return self.data[isotope].get(self.config, **kwargs)

    def get_structured(self, **kwargs: Any) -> np.ndarray:
        data = []
        for isotope in self.isotopes():
            data.append(self.data[isotope].get(self.config, **kwargs))

        dtype = [(isotope, float) for isotope in self.isotopes()]
        structured = np.empty(data[0].shape, dtype)
        for isotope, d in zip(self.isotopes(), data):
            structured[isotope] = d
        return structured

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

    @staticmethod
    def formatName(name: str) -> str:
        pass
