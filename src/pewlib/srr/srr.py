"""
Class for Super-Resolution-Reconstruction (SRR) LA-ICP-MS.
SRR is performed by ablating layers of a sample in a line-by-line fashion.
Layers are offset by a faction of the spotsize to increase image resolution.
The resulting image is 3d, but can be flattened to a higher resolution 2d image.

References:
    Westerhausen, M. T.; Bishop, D. P.; Dowd, A.; Wanagat, J.; Cole, N.
    & Doble, P. A. Super-Resolution Reconstruction for Two- and Three-Dimensional
    LA-ICP-MS Bioimaging Analytical Chemistry, American Chemical Society (ACS), 2019
"""

import copy

import numpy as np
import numpy.lib.recfunctions as rfn

from pewlib.calibration import Calibration
from pewlib.laser import Laser
from pewlib.process.calc import subpixel_offset_equal
from pewlib.srr.config import SRRConfig


class SRRLaser(Laser):
    """Class for SRR laser data.

    Args:
        data: list of structured arrays
        calibration: dict mapping elements to calibrations, optional
        config: SRR laser parameters
        info: dict(str, str) of additional info

    See Also:
        :class:`pewlib.laser.Laser`
    """

    def __init__(
        self,
        data: list[np.ndarray],
        calibration: dict[str, Calibration] | None = None,
        config: SRRConfig | None = None,
        info: dict[str, str] | None = None,
    ):
        assert len(data) > 1
        self.data: list[np.ndarray] = data
        self.calibration = {name: Calibration() for name in self.elements}
        if calibration is not None:
            self.calibration.update(copy.deepcopy(calibration))

        self.config: SRRConfig = (
            copy.copy(config) if config is not None else SRRConfig()
        )
        self.info = info or {}

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Data extent in μm

        This is calculated *post* SRR.
        """
        pixelsize = self.config.subpixels_per_pixel
        offset = np.max(self.config._subpixel_offsets)
        new_shape = np.array(self.shape[:2]) * self.config.magnification
        new_shape = new_shape * pixelsize + offset
        return self.config.data_extent(new_shape)

    @property
    def elements(self) -> tuple[str, ...]:
        if len(self.data) == 0:  # pragma: no cover
            return ()
        return self.data[0].dtype.names

    @property
    def layers(self) -> int:
        return len(self.data)

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.data[1].shape[0], self.data[0].shape[0], len(self.data))

    def add(
        self,
        element: str,
        data: list[np.ndarray],
        calibration: Calibration | None = None,
    ) -> None:
        """Add an element."""
        assert len(data) == len(self.data)
        for i in range(0, len(self.data)):
            assert data[i].shape == self.data[i].shape
            dtype = self.data[i].dtype
            new_dtype = dtype.descr + [(element, data[i].dtype.str)]

            new_data = np.empty(self.data[i].shape, dtype=new_dtype)
            for name in dtype.names:
                new_data[name] = self.data[i][name]
            new_data[element] = data[i]
            self.data[i] = new_data

        if calibration is None:
            calibration = Calibration()
        self.calibration[element] = calibration

    def remove(self, names: str | list[str]) -> None:
        """Remove element(s)."""
        if isinstance(names, str):
            names = [names]
        for i in range(len(self.data)):
            self.data[i] = rfn.drop_fields(self.data[i], names, usemask=False)
        for name in names:
            self.calibration.pop(name)

    def rename(self, names: dict[str, str]) -> None:
        """Rename element(s).

        Args:
            names: dict mapping old to new name
        """
        for i in range(len(self.data)):
            self.data[i] = rfn.rename_fields(self.data[i], names)
        for old, new in names.items():
            self.calibration[(new)] = self.calibration.pop(old)

    def get(
        self,
        element: str | None = None,
        calibrate: bool = False,
        extent: tuple[float, float, float, float] | None = None,
        flat: bool = False,
        layer: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Get elemental data.

        If `element` is None then all elements are returned in a structured array.
        If a 2d array is required then `flat` will flatten the array by calculating
        the mean across the 2nd axis. If a `layer` is given then the layer is extracted
        otherwise SRR is performed and the resulting array returned.

        Args:
            element: element name, optional
            calibrate: apply calibration
            extent: trim to extent, μm
            flat: flatten to 2d
            layer: extract layer, optional

        Returns:
            structured if element is None else unstructured
            2d if layer or flat, else 3d
        """
        if layer is not None:
            data = self.data[layer].copy()
            # Flip alternate layers
            if layer % 2 == 1:
                data = data.T
        else:
            data = self.krisskross()

        if element is not None:
            data = data[element]

        if extent is not None:
            x0, x1, y0, y1 = extent
            px, py = (
                self.config.get_pixel_width(layer),
                self.config.get_pixel_height(layer),
            )
            x0, x1 = int(x0 / px), int(x1 / px)
            y0, y1 = int(y0 / py), int(y1 / py)
            # We have to invert the extent, as mpl use bottom left y coords
            ymax = data.shape[0]
            data = data[ymax - y1 : ymax - y0, x0:x1]

        if calibrate:  # pragma: no cover, covered in laser
            if element is None:  # Perform calibration on all data
                for name in data.dtype.names:
                    data[name] = self.calibration[name].calibrate(data[name])
            else:
                data = self.calibration[element].calibrate(data)

        if flat and data.ndim > 2:
            if element is not None:
                data = np.mean(data, axis=2)
            else:
                structured = np.empty(data.shape[:2], data.dtype)
                for name in data.dtype.names:
                    structured[name] = np.mean(data[name], axis=2)
                data = structured

        return data

    def check_config_valid(self, config: SRRConfig) -> bool:
        """Checks if SRRConfig is valid for data."""
        return config.valid_for_data(self.data)

    def krisskross(self) -> np.ndarray:
        """Perform SRR."""
        # Calculate the line lengths
        mag = self.config.magnification
        mag = np.round(1.0 / mag if mag < 1.0 else mag).astype(int)
        mag_axis = 0 if self.config.magnification > 1.0 else 1

        length = (
            self.data[1].shape[mag_axis] * mag,
            self.data[0].shape[mag_axis] * mag,
        )
        # Reshape the layers and stack into matrix
        aligned = np.empty(
            (length[1], length[0], self.layers), dtype=self.data[0].dtype
        )
        for i, layer in enumerate(self.data):
            # Trim data of warmup time and excess
            layer = layer[:, self.config._warmup : self.config._warmup + length[i % 2]]
            # Stretch array
            layer = np.repeat(layer, mag, axis=mag_axis)
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
        elements: list[str],
        layers: list[list[np.ndarray]],
        config: SRRConfig | None = None,
        info: dict[str, str] | None = None,
    ) -> "SRRLaser":
        """Creates class from a list of names and lists of unstructured arrays."""
        dtype = [(element, float) for element in elements]

        structured_layers = []
        for datas in layers:
            assert len(elements) == len(datas)
            structured = np.empty(datas[0].shape, dtype=dtype)
            for element, data in zip(elements, datas):
                structured[element] = data
            structured_layers.append(structured)

        return cls(data=structured_layers, config=config, info=info)

    @classmethod
    def from_lasers(cls, lasers: list[Laser]) -> "SRRLaser":
        """Stacks :class:`Laser` to form SRR.

        Calibration and config are taken from the first :class:`Laser`.
        """
        assert all(lasers[0].elements == laser.elements for laser in lasers[1:])

        config = SRRConfig(
            lasers[0].config.spotsize, lasers[0].config.speed, lasers[0].config.scantime
        )
        calibration = lasers[0].calibration
        data = [laser.data for laser in lasers]

        return cls(
            data=data,
            calibration=calibration,
            config=config,
            info=lasers[0].info,
        )
