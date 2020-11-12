import numpy as np

from typing import Tuple


class Config(object):
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

    def data_extent(
        self, shape: Tuple[int, ...], **kwargs
    ) -> Tuple[float, float, float, float]:
        px, py = self.get_pixel_width(), self.get_pixel_height()
        return (0.0, px * shape[1], 0.0, py * shape[0])

    def to_array(self) -> np.ndarray:
        return np.array(
            (self.spotsize, self.speed, self.scantime),
            dtype=[
                ("spotsize", np.float64),
                ("speed", np.float64),
                ("scantime", np.float64),
            ],
        )

    @classmethod
    def from_array(cls, array: np.ndarray) -> "Config":
        return cls(
            spotsize=float(array["spotsize"]),
            speed=float(array["speed"]),
            scantime=float(array["scantime"]),
        )
