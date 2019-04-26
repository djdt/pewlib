import numpy as np

from ..laser import Laser

from .config import KrissKrossConfig
from .data import KrissKrossData

from typing import Dict


class KrissKross(Laser):
    def __init__(
        self,
        data: Dict[str, KrissKrossData],
        config: KrissKrossConfig = None,
        name: str = "",
        filepath: str = "",
    ):
        self.data = data  # type: ignore

        self.config = config if config is not None else KrissKrossConfig()

        self.name = name
        self.filepath = filepath
