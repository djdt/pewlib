import numpy as np

from ..laser import Laser

from .config import KrissKrossConfig
from .data import KrissKrossData

from typing import Dict, List, Tuple


class KrissKross(Laser):
    def __init__(
        self,
        data: Dict[str, KrissKrossData] = None,
        config: KrissKrossConfig = None,
        name: str = "",
        path: str = "",
    ):
        if config is None:
            config = KrissKrossConfig()
        else:
            assert isinstance(config, KrissKrossConfig)

        super().__init__(data, config, name, path)

    @property
    def layers(self) -> int:
        return len(next(iter(self.data.values())).data)

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """Calculates the extent of the array POST krisskross."""
        pixelsize = self.config.subpixels_per_pixel
        offset = np.max(self.config._subpixel_offsets)
        new_shape = np.array(self.shape[:2]) * self.config.magnification
        new_shape = new_shape * pixelsize + offset
        return self.config.data_extent(new_shape)

    def check_config_valid(self, config: KrissKrossConfig) -> bool:
        if config.warmup < 0:
            return False
        data = next(iter(self.data.values())).data
        if config.magnification * data[1].shape[0] + config._warmup > data[0].shape[1]:
            return False
        if config.magnification * data[0].shape[0] + config._warmup > data[1].shape[1]:
            return False

        return True

    @classmethod
    def from_structured(
        cls,
        data: List[np.ndarray],
        config: KrissKrossConfig = None,
        name: str = "",
        path: str = "",
    ):  # type: ignore
        data = {k: KrissKrossData([d[k] for d in data]) for k in data[0].dtype.names}
        return cls(data=data, config=config, name=name, path=path)
