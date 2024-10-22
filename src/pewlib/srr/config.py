from collections.abc import Iterable

import numpy as np

from pewlib.config import Config


class SRRConfig(Config):
    """Class for the super-resolution-reconstruction image parameters.

    Args:
        spotsize: laser-spot diameter, μm
        speed: laser movement speed, μm/s
        scantime: MS acquisition time, s
        warmup: warmup time in s
        subpixel_offsets: list of offsets of layers, (offset, pixelsize)

    See Also:
        :class:`pewlib.config.Config`
    """

    _class = "SRR"

    def __init__(
        self,
        spotsize: float = 35.0,
        speed: float = 140.0,
        scantime: float = 0.25,
        warmup: float = 12.5,
        subpixel_offsets: Iterable[tuple[int, int]] = ((0, 2), (1, 2)),
    ):
        super().__init__(spotsize=spotsize, speed=speed, scantime=scantime)
        self._warmup = 0
        self.warmup = warmup

        self._subpixel_size = 0
        self._subpixel_offsets = np.array([], dtype=np.int32)
        self.subpixel_offsets = np.array(subpixel_offsets)

    @property
    def warmup(self) -> float:
        """Laser warmup (time before data recorded) in seconds."""
        return self._warmup * self.scantime

    @warmup.setter
    def warmup(self, seconds: float) -> None:
        self._warmup = np.round(seconds / self.scantime).astype(int)

    @property
    def magnification(self) -> float:
        """Magnification due to non-equal aspect."""
        return self.spotsize / (self.speed * self.scantime)

    @property
    def subpixel_offsets(self) -> np.ndarray:
        """Layer offsets."""
        return np.array(
            [[offset, self._subpixel_size] for offset in self._subpixel_offsets]
        )

    @subpixel_offsets.setter
    def subpixel_offsets(self, offsets: np.ndarray) -> None:
        offsets = np.array(offsets, dtype=int)
        if offsets.ndim != 2:
            raise ValueError("Offsets must have 2 dimensions.")
        self._subpixel_size = np.lcm.reduce(offsets[:, 1])
        self._subpixel_offsets = offsets[:, 0] * self._subpixel_size // offsets[:, 1]

    @property
    def subpixels_per_pixel(self) -> int:
        """Pixel width in subpixels."""
        mag = (
            1.0 / self.magnification if self.magnification < 1.0 else self.magnification
        )
        mag = np.round(mag).astype(int)
        return np.lcm(self._subpixel_size, mag) // mag

    def set_equal_subpixel_offsets(self, width: int) -> None:
        self._subpixel_offsets = np.arange(0, width, dtype=int)
        self._subpixel_size = width

    def get_pixel_width(self, layer: int | None = None) -> float:
        """Pixel width in μm.

        Args:
            layer: limit to layer
        """
        if layer is None:
            return super().get_pixel_width() / self.subpixels_per_pixel
        elif layer % 2 == 0:
            return super().get_pixel_width()
        else:
            return super().get_pixel_height()

    def get_pixel_height(self, layer: int | None = None) -> float:
        """Pixel height in μm.

        Args:
            layer: limit to layer
        """
        if layer is None:
            return super().get_pixel_width() / self.subpixels_per_pixel
        elif layer % 2 == 0:
            return super().get_pixel_height()
        else:
            return super().get_pixel_width()

    # Return without the washout included
    def data_extent(
        self, shape: tuple[int, ...], layer: int | None = None
    ) -> tuple[float, float, float, float]:
        """Extent of data in μm.

        Args:
            shape: data shape
            layer: limit calculation to layer
        """
        px, py = self.get_pixel_width(layer), self.get_pixel_height(layer)
        warmup = self._warmup
        if layer is None:
            return (
                px * warmup,
                px * (warmup + shape[1]),
                py * warmup,
                py * (warmup + shape[0]),
            )
        else:
            return (0.0, px * shape[1], 0.0, py * shape[0])

    def valid_for_data(self, data: list[np.ndarray]) -> bool:
        """Checks if this config is valid for data."""
        if self.warmup < 0:
            return False

        mag = self.magnification
        mag = np.round(1.0 / mag if mag < 1.0 else mag).astype(int)
        mag_axis = 0 if self.magnification > 1.0 else 1

        limit = (
            data[1].shape[mag_axis] * mag,
            data[0].shape[mag_axis] * mag,
        )

        if data[0].shape[1] < self._warmup + limit[0]:
            return False
        if data[1].shape[1] < self._warmup + limit[1]:  # pragma: no cover
            return False
        return True

    def to_array(self) -> np.ndarray:
        offsets = self.subpixel_offsets
        return np.array(
            (self.spotsize, self.speed, self.scantime, self.warmup, offsets),
            dtype=[
                ("spotsize", np.float64),
                ("speed", np.float64),
                ("scantime", np.float64),
                ("warmup", np.float64),
                ("subpixel_offsets", offsets.dtype, offsets.shape),
            ],
        )

    @classmethod
    def from_array(cls, array: np.ndarray) -> "SRRConfig":
        return cls(**{str(name): array[name] for name in array.dtype.names})
