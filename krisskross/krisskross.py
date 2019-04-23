import numpy as np

from ..laser import Laser
from ..calibration import LaserCalibration

from .calc import subpixel_offset_equal
from .config import KrissKrossConfig

from typing import Dict, List, Tuple


# KKType = TypeVar("KKType", bound="KrissKross")  # For typing


class KrissKross(Laser):
    def __init__(
        self,
        data: List[np.ndarray] = [],
        calibration: Dict[str, LaserCalibration] = None,
        config: KrissKrossConfig = None,
        name: str = "",
        filepath: str = "",
    ):
        for layer in data:
            assert(layer.dtype == data[0].dtype)
        self.data = data

        self.config = config if config is not None else KrissKrossConfig()
        if calibration is not None:
            self.calibration = calibration
        else:
            self.calibration = {n: LaserCalibration() for n in self.data[0].dtype.names}

        self.name = name
        self.filepath = filepath

    def isotopes(self) -> List[str]:
        return self.data[0].dtype.names()

    def _krisskross(self) -> np.ndarray:
        assert isinstance(self.config, KrissKrossConfig)
        warmup = self.config.warmup_lines()
        mfactor = self.config.magnification_factor()

        # Calculate the line lengths
        length = (self.data[1].shape[0] * mfactor, self.data[0].shape[0] * mfactor)
        # Reshape the layers and stack into matrix
        aligned = np.empty(
            (length[1], length[0], len(self.data)), dtype=self.data[0].dtype
        )
        for i, layer in enumerate(self.data):
            # Trim data of warmup time and excess
            layer = layer[:, warmup : warmup + length[i % 2]]
            # Stretch array
            layer = np.repeat(layer, mfactor, axis=0)
            # Flip vertical layers
            if i % 2 == 1:
                layer = layer.T
            aligned[:, :, i] = layer

        return subpixel_offset_equal(
            aligned, self.config.subpixel_offsets(), self.config.subpixel_per_pixel[0]
        )

    def get(
        self,
        name: str,
        calibrate: bool = False,
        extent: Tuple[float, float, float, float] = None,
        flat: bool = True,
        layer: int = None,
    ) -> np.ndarray:
        assert isinstance(self.config, KrissKrossConfig)
        data = self._krisskross() if layer is None else self.data[layer]

        data = super().get(name, calibrate=calibrate, extent=extent)

        if layer is None and flat:
            data = np.mean(data, axis=2)

        return data
