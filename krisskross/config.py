import numpy as np

from ..laser import LaserConfig

from typing import Tuple


class KrissKrossConfig(LaserConfig):
    def __init__(
        self,
        spotsize: float = 35.0,
        speed: float = 140.0,
        scantime: float = 0.25,
        warmup: float = 12.5,
        subpixel_offsets: np.ndarray = [[0, 2], [1, 2]],
    ):
        super().__init__(spotsize=spotsize, speed=speed, scantime=scantime)
        self._warmup = np.round(warmup / self.scantime).astype(int)

        offsets = np.array(subpixel_offsets, dtype=int)
        assert offsets.ndim == 2
        self._subpixel_size = np.lcm.reduce(offsets[:, 1])
        self._subpixel_offsets = offsets[:, 0] * self._subpixel_size // offsets[:, 1]

    @property
    def warmup(self) -> float:
        """Returns the laser warmup (time before data recorded) in seconds."""
        return self._warmup * self.scantime

    @warmup.setter
    def warmup(self, seconds: float) -> None:
        self._warmup = np.round(seconds / self.scantime).astype(int)

    @property
    def magnification(self) -> float:
        return np.round(self.spotsize / (self.speed * self.scantime)).astype(int)

    @property
    def subpixel_offsets(self) -> np.ndarray:
        return np.array(
            [[offset, self._subpixel_size] for offset in self._subpixel_offsets]
        )

    @subpixel_offsets.setter
    def subpixel_offsets(self, offsets: np.ndarray) -> None:
        offsets = np.array(offsets, dtype=int)
        assert offsets.ndim == 2
        self._subpixel_size = np.lcm.reduce(offsets[:, 1])
        self._subpixel_offsets = offsets[:, 0] * self._subpixel_size // offsets[:, 1]

    @property
    def subpixels_per_pixel(self) -> int:
        return np.lcm(self._subpixel_size, self.magnification) // self.magnification

    def set_equal_subpixel_offsets(self, width: int) -> None:
        self._subpixel_offsets = np.arange(0, width, dtype=int)
        self._subpixel_size = width

    def get_pixel_width(self, layer: int = None) -> float:
        if layer is None:
            return super().get_pixel_width() / self.subpixels_per_pixel
        elif layer % 2 == 0:
            return super().get_pixel_width()
        else:
            return super().get_pixel_height()

    def get_pixel_height(self, layer: int = None) -> float:
        if layer is None:
            return super().get_pixel_width() / self.subpixels_per_pixel
        elif layer % 2 == 0:
            return super().get_pixel_height()
        else:
            return super().get_pixel_width()

    # Return without the washout included
    def data_extent(
        self, data: np.ndarray, layer: int = None
    ) -> Tuple[float, float, float, float]:
        if layer is None:
            return (
                self.get_pixel_width() * self._warmup,
                self.get_pixel_width() * (self._warmup + data.shape[1]),
                self.get_pixel_height() * self._warmup,
                self.get_pixel_height() * (self._warmup + data.shape[0]),
            )
        else:
            return (
                0.0,
                self.get_pixel_width(layer) * data.shape[1],
                0.0,
                self.get_pixel_height(layer) * data.shape[0],
            )
