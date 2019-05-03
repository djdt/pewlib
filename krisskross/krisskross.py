import numpy as np

from ..laser import Laser

from .config import KrissKrossConfig
from .data import KrissKrossData

from typing import Dict, List


class KrissKross(Laser):
    def __init__(
        self,
        data: Dict[str, KrissKrossData] = None,
        config: KrissKrossConfig = None,
        name: str = "",
        filepath: str = "",
    ):
        self.data: Dict[str, KrissKrossData] = data if data is not None else {}

        self.config = config if config is not None else KrissKrossConfig()

        self.name = name
        self.filepath = filepath

    @classmethod
    def from_structured(
        cls,
        data: List[np.ndarray],
        config: KrissKrossConfig = None,
        name: str = "",
        filepath: str = "",
    ):  # type: ignore
        data = {k: KrissKrossData([d[k] for d in data]) for k in data[0].dtype.names}
        return cls(data=data, config=config, name=name, filepath=filepath)

    def check_config_valid(self, config: KrissKrossConfig) -> bool:
        mfactor = config.magnification_factor()
        warmup = config.warmup_lines()
        data = self.data[self.isotopes()[0]].data
        if mfactor * data[1].shape[0] + warmup > data[0].shape[1]:
            return False
        if mfactor * data[0].shape[0] + warmup > data[1].shape[1]:
            return False

        return True
