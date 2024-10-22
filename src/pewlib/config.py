import numpy as np


class Config(object):
    """Class for the rastered parameters of image.

    Args:
        spotsize: laser-spot diameter, μm
        speed: laser movement speed, μm/s
        scantime: MS acquisition time, s
    """

    _class = "Raster"

    def __init__(
        self, spotsize: float = 35.0, speed: float = 140.0, scantime: float = 0.25
    ):
        self.spotsize = spotsize
        self.speed = speed
        self.scantime = scantime

    def data_extent(self, shape: tuple[int, ...]) -> tuple[float, float, float, float]:
        """Extent of data in μm."""
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

    def get_pixel_width(self) -> float:
        """Pixel width in μm."""
        return self.speed * self.scantime

    def get_pixel_height(self) -> float:
        """Pixel height in μm."""
        return self.spotsize

    @classmethod
    def from_array(cls, array: np.ndarray) -> "Config":
        return cls(
            spotsize=float(array["spotsize"]),
            speed=float(array["speed"]),
            scantime=float(array["scantime"]),
        )


class SpotConfig(Config):
    """Class for the spotwise parameters of image.

    Args:
        spotsize: x, y distance between laser-spots, μm
    """

    _class = "Spot"

    def __init__(self, spotsize: float = 100.0, spotsize_y: float | None = None):
        super().__init__(spotsize=spotsize, speed=0.0, scantime=0.0)
        if spotsize_y is None:
            spotsize_y = spotsize
        self.spotsize_y = spotsize_y

    def to_array(self) -> np.ndarray:
        return np.array(
            [self.spotsize, self.spotsize_y], dtype=[("spotsize", np.float64)]
        )

    def get_pixel_width(self) -> float:
        """Pixel width in μm."""
        return self.spotsize

    def get_pixel_height(self) -> float:
        """Pixel height in μm."""
        return self.spotsize_y

    @classmethod
    def from_array(cls, array: np.ndarray) -> "Config":
        return cls(spotsize=array["spotsize"][0], spotsize_y=array["spotsize"][1])
