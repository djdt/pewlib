from typing import Tuple
import numpy as np


class LaserConfig(object):
    def __init__(
        self, spotsize: float = 35.0, speed: float = 140.0, scantime: float = 0.25
    ):
        self.spotsize = spotsize
        self.speed = speed
        self.scantime = scantime

    def get_pixel_width(self) -> float:
        return self.speed * self.scantime

    def get_pixel_height(self) -> float:
        return self.spotsize

    def data_extent(self, data: np.ndarray) -> Tuple[float, float, float, float]:
        return (
            0.0,
            self.get_pixel_width() * data.shape[1],
            0.0,
            self.get_pixel_height() * data.shape[0],
        )
