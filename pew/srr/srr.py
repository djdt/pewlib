import numpy as np
import copy

from ..laser import _Laser, Laser
from ..calibration import Calibration

from .calc import subpixel_offset_equal
from .config import SRRConfig

from typing import Dict, List, Tuple


class SRRLaser(_Laser):
    def __init__(
        self,
        data: List[np.ndarray],
        calibration: Dict[str, Calibration] = None,
        config: SRRConfig = None,
        name: str = "",
        path: str = "",
    ):
        assert len(data) > 1
        self.data: List[np.ndarray] = data
        self.calibration = {name: Calibration() for name in self.isotopes}
        if calibration is not None:
            self.calibration.update(copy.deepcopy(calibration))

        self.config: SRRConfig = copy.copy(
            config
        ) if config is not None else SRRConfig()

        self.name = name
        self.path = path

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """Calculates the extent of the array POST srr."""
        pixelsize = self.config.subpixels_per_pixel
        offset = np.max(self.config._subpixel_offsets)
        new_shape = np.array(self.shape[:2]) * self.config.magnification
        new_shape = new_shape * pixelsize + offset
        return self.config.data_extent(new_shape)

    @property
    def isotopes(self) -> Tuple[str, ...]:
        return self.data[0].dtype.names

    @property
    def layers(self) -> int:
        return len(self.data)

    @property
    def shape(self) -> Tuple[int, ...]:
        return (self.data[1].shape[0], self.data[0].shape[0], len(self.data))

    def add(
        self, isotope: str, data: List[np.ndarray], calibration: Calibration = None
    ) -> None:
        assert len(data) == len(self.data)
        for i in range(0, len(self.data)):
            assert data[i].shape == self.data[i].shape
            dtype = self.data[i].dtype
            new_dtype = dtype.descr + [(isotope, data[i].dtype.str)]

            new_data = np.empty(self.data[i].shape, dtype=new_dtype)
            for name in dtype.names:
                new_data[name] = self.data[name]
            new_data[isotope] = data
            self.data[i] = new_data

        if calibration is None:
            calibration = Calibration()
        self.calibration[isotope] = calibration

    def remove(self, isotope: str) -> None:
        for i in range(0, len(self.data)):
            dtype = self.data[i].dtype
            new_dtype = [descr for descr in dtype.descr if descr[0] != isotope]

            new_data = np.empty(self.data[i].shape, dtype=new_dtype)
            for name in dtype.names:
                if name != isotope:
                    new_data[name] = self.data[name]
            self.data[i] = new_data

        self.calibration.pop(isotope)

    def get(self, isotope: str = None, **kwargs) -> np.ndarray:
        layer = kwargs.get("layer", None)
        if layer is not None:
            data = self.data[layer].copy()
            # Flip alternate layers
            if layer % 2 == 1:
                data = data.T
        else:
            data = self.krisskross()

        if isotope is not None:
            data = data[isotope]

        if "extent" in kwargs:
            x0, x1, y0, y1 = kwargs["extent"]
            px, py = (
                self.config.get_pixel_width(layer),
                self.config.get_pixel_height(layer),
            )
            x0, x1 = int(x0 / px), int(x1 / px)
            y0, y1 = int(y0 / py), int(y1 / py)
            # We have to invert the extent, as mpl use bottom left y coords
            ymax = data.shape[0]
            data = data[ymax - y1 : ymax - y0, x0:x1]

        if kwargs.get("calibrate", False):
            if isotope is None:  # Perform calibration on all data
                for name in data.dtype.names:
                    data[name] = self.calibration[name].calibrate(data[name])
            else:
                data = self.calibration[isotope].calibrate(data)

        if kwargs.get("flat", False) and data.ndim > 2:
            if isotope is not None:
                data = np.mean(data, axis=2)
            else:
                structured = np.empty(data.shape[:2], data.dtype)
                for name in data.dtype.names:
                    structured[name] = np.mean(data[name], axis=2)
                data = structured

        return data

    def check_config_valid(self, config: SRRConfig) -> bool:
        if config.warmup < 0:
            return False
        shape = self.data[1].shape[0], self.data[0].shape[1]
        limit = self.data[0].shape[0], self.data[1].shape[1]
        if config.magnification * shape[0] + config._warmup > limit[0]:
            return False
        if config.magnification * shape[1] + config._warmup > limit[1]:
            return False
        return True

    def krisskross(self) -> np.ndarray:
        # Calculate the line lengths
        length = (
            self.data[1].shape[0] * self.config.magnification,
            self.data[0].shape[0] * self.config.magnification,
        )
        # Reshape the layers and stack into matrix
        aligned = np.empty(
            (length[1], length[0], self.layers), dtype=self.data[0].dtype
        )
        for i, layer in enumerate(self.data):
            # Trim data of warmup time and excess
            layer = layer[:, self.config._warmup : self.config._warmup + length[i % 2]]
            # Stretch array
            layer = np.repeat(layer, self.config.magnification, axis=0)
            # Flip vertical layers
            if i % 2 == 1:
                layer = layer.T
            aligned[:, :, i] = layer

        return subpixel_offset_equal(
            aligned, self.config._subpixel_offsets, self.config.subpixels_per_pixel
        )

    @classmethod
    def from_list(
        cls,
        isotopes: List[str],
        layers: List[List[np.ndarray]],
        config: SRRConfig = None,
        name: str = "",
        path: str = "",
    ) -> "SRRLaser":
        dtype = [(isotope, float) for isotope in isotopes]

        structured_layers = []
        for datas in layers:
            assert len(isotopes) == len(datas)
            structured = np.empty(datas[0].shape, dtype=dtype)
            for isotope, data in zip(isotopes, datas):
                structured[isotope] = data
            structured_layers.append(structured)

        return cls(data=structured_layers, config=config, name=name, path=path)

    @classmethod
    def from_lasers(cls, lasers: List[Laser]) -> "SRRLaser":
        assert all(lasers[0].isotopes == laser.isotopes for laser in lasers[1:])

        config = SRRConfig(
            lasers[0].config.spotsize, lasers[0].config.speed, lasers[0].config.scantime
        )
        calibration = lasers[0].calibration
        data = [laser.data for laser in lasers]

        return cls(
            data=data,
            calibration=calibration,
            config=config,
            name=lasers[0].name,
            path=lasers[0].path,
        )
