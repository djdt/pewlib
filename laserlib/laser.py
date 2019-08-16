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
        self.layers = 1
        self.config = copy.copy(config) if config is not None else LaserConfig()

        self.name = name
        self.filepath = filepath

    @property
    def isotopes(self) -> List[str]:
        return list(self.data.keys())

    def get(self, isotope: str, **kwargs: Any) -> np.ndarray:
        """Valid kwargs are calibrate, extent."""
        if isotope not in self.data:
            return np.zeros((1, 1), dtype=float)

        return self.data[isotope].get(self.config, **kwargs)

    def get_structured(self, **kwargs: Any) -> np.ndarray:
        dtype = [(isotope, float) for isotope in self.data]
        structured = np.empty(next(iter(self.data.values())).data.shape, dtype)
        for isotope, _ in dtype:
            structured[isotope] = self.data[isotope].get(self.config, **kwargs)
        return structured

    @classmethod
    def from_structured(
        cls,
        data: np.ndarray,
        config: LaserConfig = None,
        name: str = "",
        filepath: str = "",
    ):  # type: ignore
        data = {k: LaserData(data[k]) for k in data.dtype.names}
        return cls(data=data, config=config, name=name, filepath=filepath)
