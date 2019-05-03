import numpy as np

from ..laser import Laser

from .config import KrissKrossConfig
from .data import KrissKrossData

from typing import Dict


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
