import numpy as np

from ..calibration import LaserCalibration
from ..data import LaserData

from .calc import subpixel_offset_equal
from .config import KrissKrossConfig

from typing import List
from .config import LaserConfig


def krisskross_layers(data: List[np.ndarray], config: KrissKrossConfig) -> np.ndarray:
    warmup = config.warmup_lines()
    mfactor = config.magnification_factor()

    # Calculate the line lengths
    length = (data[1].shape[0] * mfactor, data[0].shape[0] * mfactor)
    # Reshape the layers and stack into matrix
    aligned = np.empty(
        (length[1], length[0], len(data)), dtype=data[0].dtype
    )
    for i, layer in enumerate(data):
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


class KrissKrossData(LaserData):
    def __init__(self, data: List[np.ndarray], calibration: LaserCalibration = None):
        self.data = data
        self.calibration = (
            calibration if calibration is not None else LaserCalibration()
        )

    def get(self, config: LaserConfig, **kwargs) -> np.ndarray:
        assert isinstance(config, KrissKrossConfig)

        layer = kwargs.get("layer", None)
        if layer is not None:
            data = self.data[layer].copy()
        else:
            data = krisskross_layers(self.data, config)

        if "extent" in kwargs:
            x0, x1, y0, y1 = kwargs.get("extent", (0.0, 0.0, 0.0, 0.0))
            if layer is not None:  # Use normal laser width and height
                px, py = (config.layer_pixel_width(), config.layer_pixel_height())
            else:
                px, py = config.pixel_width(), config.pixel_height()
            x0, x1 = int(x0 / px), int(x1 / px)
            y0, y1 = int(y0 / py), int(y1 / py)
            # We have to invert the extent, as mpl use bottom left y coords
            ymax = data.shape[0]
            data = data[ymax - y1 : ymax - y0, x0:x1]

        if kwargs.get("calibrate", False):
            data = self.calibration.calibrate(data)

        if kwargs.get("flat", False) and data.ndim > 2:
            data = np.mean(data, axis=2)

        return data
