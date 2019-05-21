import numpy as np

from ..calibration import LaserCalibration
from ..data import LaserData

from .calc import subpixel_offset_equal
from .config import KrissKrossConfig

from typing import List
from .config import LaserConfig


class KrissKrossData(LaserData):
    def __init__(self, data: List[np.ndarray], calibration: LaserCalibration = None):
        self.data = data
        self.calibration = (
            calibration if calibration is not None else LaserCalibration()
        )

    def _krisskross(self, config: KrissKrossConfig) -> np.ndarray:
        warmup = config.warmup_lines()
        mfactor = config.magnification_factor()

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
            aligned, config.subpixel_offsets(), config.subpixel_per_pixel[0]
        )

    def get(self, config: LaserConfig, **kwargs) -> np.ndarray:
        assert isinstance(config, KrissKrossConfig)

        if hasattr(kwargs, "layer"):
            data = self.data[kwargs["layer"]].copy()
        else:
            data = self._krisskross(config)

        if "extent" in kwargs:
            x0, x1, y0, y1 = kwargs.get("extent", (0.0, 0.0, 0.0, 0.0))
            px, py = config.pixel_size()
            x0, x1 = int(x0 / px), int(x1 / px)
            y0, y1 = int(y0 / py), int(y1 / py)
            # We have to invert the extent, as mpl use bottom left y coords
            ymax = data.shape[0]
            data = data[ymax - y1:ymax - y0, x0:x1]

        if kwargs.get("calibrate", False):
            data = self.calibration.calibrate(data)

        if kwargs.get("flat", False):
            data = np.add(data, axis=2)

        return data
