import numpy as np
import copy

from .config import LaserConfig
from .data import LaserData

from typing import Any, Dict, List, Tuple


class Laser(object):
    def __init__(
        self,
        data: Dict[str, LaserData] = None,
        config: LaserConfig = None,
        name: str = "",
        path: str = "",
    ):
        self.shape = None
        self.data = {}
        if data is not None:
            for v in data.values():
                self.check_shape(v.shape)
            self.data.update(data)

        self.config = copy.copy(config) if config is not None else LaserConfig()

        self.name = name
        self.path = path

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        if self.shape is None:
            return (0, 0, 0, 0)
        return self.config.data_extent(self.shape[:2])

    @property
    def isotopes(self) -> List[str]:
        return list(self.data.keys())

    @property
    def layers(self) -> int:
        return 1

    def add(self, isotope: str, data: np.ndarray) -> None:
        self.check_shape(data.shape)
        self.data[isotope] = LaserData(data)

    def check_shape(self, shape: List[int]) -> None:
        if self.shape is None:
            self.shape = shape
        assert self.shape == shape

    def get(self, isotope: str, **kwargs: Any) -> np.ndarray:
        """Valid kwargs are calibrate, extent, flat."""
        return self.data[isotope].get(self.config, **kwargs)

    def get_any(self, **kwargs: Any) -> np.ndarray:
        return next(iter(self.data.values())).get(self.config, **kwargs)

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
        path: str = "",
    ):  # type: ignore
        data = {k: LaserData(data[k]) for k in data.dtype.names}
        return cls(data=data, config=config, name=name, path=path)
