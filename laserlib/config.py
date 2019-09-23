from typing import Tuple


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

    def data_extent(
        self, shape: Tuple[int, int], **kwargs
    ) -> Tuple[float, float, float, float]:
        px, py = self.get_pixel_width(), self.get_pixel_height()
        return (0.0, px * shape[1], 0.0, py * shape[0])
