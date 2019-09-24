import numpy as np

from ..laser import Laser

from .config import SRRConfig
from .data import SRRData

from typing import Dict, List, Tuple


class SRR(Laser):
    def __init__(
        self,
        data: Dict[str, SRRData] = None,
        config: SRRConfig = None,
        name: str = "",
        path: str = "",
    ):
        if config is None:
            config = SRRConfig()
        else:
            assert isinstance(config, SRRConfig)

        super().__init__(data, config, name, path)

    @property
    def layers(self) -> int:
        return len(next(iter(self.data.values())).data)

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """Calculates the extent of the array POST srr."""
        pixelsize = self.config.subpixels_per_pixel
        offset = np.max(self.config._subpixel_offsets)
        new_shape = np.array(self.shape[:2]) * self.config.magnification
        new_shape = new_shape * pixelsize + offset
        return self.config.data_extent(new_shape)

    def check_config_valid(self, config: SRRConfig) -> bool:
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
        config: SRRConfig = None,
        name: str = "",
        path: str = "",
    ):  # type: ignore
        data = {k: SRRData([d[k] for d in data]) for k in data[0].dtype.names}
        return cls(data=data, config=config, name=name, path=path)
