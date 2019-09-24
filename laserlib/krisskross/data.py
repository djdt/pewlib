import numpy as np

from ..calibration import Calibration
from ..data import IsotopeData

from .calc import subpixel_offset_equal
from .config import SRRConfig

from typing import List
from .config import Config


def krisskross_layers(data: List[np.ndarray], config: SRRConfig) -> np.ndarray:
    # Calculate the line lengths
    length = (
        data[1].shape[0] * config.magnification,
        data[0].shape[0] * config.magnification,
    )
    # Reshape the layers and stack into matrix
    aligned = np.empty((length[1], length[0], len(data)), dtype=data[0].dtype)
    for i, layer in enumerate(data):
        # Trim data of warmup time and excess
        layer = layer[:, config._warmup : config._warmup + length[i % 2]]
        # Stretch array
        layer = np.repeat(layer, config.magnification, axis=0)
        # Flip vertical layers
        if i % 2 == 1:
            layer = layer.T
        aligned[:, :, i] = layer

    return subpixel_offset_equal(
        aligned, config._subpixel_offsets, config.subpixels_per_pixel
    )


class SRRData(IsotopeData):
    def __init__(self, data: List[np.ndarray], calibration: Calibration = None):
        self.data = data
        self.calibration = (
            calibration if calibration is not None else Calibration()
        )

    @property
    def shape(self) -> List[int]:
        return (self.data[1].shape[0], self.data[0].shape[0], len(self.data))

    def get(self, config: Config, **kwargs) -> np.ndarray:
        assert isinstance(config, SRRConfig)

        layer = kwargs.get("layer", None)
        if layer is not None:
            data = self.data[layer].copy()
            if layer % 2 == 1:
                data = data.T
        else:
            data = krisskross_layers(self.data, config)

        if "extent" in kwargs:
            x0, x1, y0, y1 = kwargs.get("extent", (0.0, 0.0, 0.0, 0.0))
            px, py = config.get_pixel_width(layer), config.get_pixel_height(layer)
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
