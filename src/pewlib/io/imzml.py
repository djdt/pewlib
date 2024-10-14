import logging
from io import BufferedReader
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

logger = logging.getLogger(__name__)

CV_PARAMGROUP = {
    "MZ_ARRAY": "MS:1000514",
    "INTENSITY_ARRAY": "MS:1000515",
    "NO_COMPRESSION": "MS:1000568",
    "EXTERNAL_DATA": "IMS:1000101",
}
CV_SCANSETTINGS = {
    "LINE_SCAN_BOTTOM_UP": "IMS:1000492",
    "LINE_SCAN_LEFT_RIGHT": "IMS:1000491",
    "LINE_SCAN_RIGHT_LEFT": "IMS:1000490",
    "LINE_SCAN_TOP_DOWN": "IMS:1000493",
    "BOTTOM_UP": "IMS:1000400",
    "LEFT_RIGHT": "IMS:1000402",
    "RIGHT_LEFT": "IMS:1000403",
    "TOP_DOWN": "IMS:1000401",
    "ONE_WAY": "IMS:1000411",
    "RANDOM_ACCESS": "IMS:1000412",
    "FLY_BACK": "IMS:1000413",
    "HORIZONTAL_LINE_SCAN": "IMS:1000480",
    "VERTICAL_LINE_SCAN": "IMS:1000481",
    "MAX_DIMENSION_X": "IMS:1000044",
    "MAX_DIMENSION_Y": "IMS:1000045",
    "MAX_COUNT_OF_PIXEL_X": "IMS:1000042",
    "MAX_COUNT_OF_PIXEL_Y": "IMS:1000043",
    "PIXEL_SIZE_X": "IMS:1000046",
    "PIXEL_SIZE_Y": "IMS:1000047",
}
CV_BINARYDATA = {
    "BINARY_TYPE_8BIT_INTEGER": "IMS:1100000",
    "BINARY_TYPE_16BIT_INTEGER": "IMS:1100001",
    "BINARY_TYPE_32BIT_INTEGER": "MS:1000519",
    "BINARY_TYPE_64BIT_INTEGER": "MS:1000522",
    "BINARY_TYPE_32BIT_FLOAT": "MS:1000521",
    "BINARY_TYPE_64BIT_FLOAT": "MS:1000523",
}
CV_SPECTRUM = {
    "POSITION_X": "IMS:1000050",
    "POSITION_Y": "IMS:1000051",
    "TOTAL_ION_CURRENT": "MS:1000285",
    "EXTERNAL_OFFSET": "IMS:1000102",
    "EXTERNAL_ARRAY_LENGTH": "IMS:1000103",
    "EXTERNAL_ENCODED_LENGTH": "IMS:1000104",
    "LOWEST_OBSERVED_MZ": "MS:1000528",
    "HIGHEST_OBSERVED_MZ": "MS:1000527",
}

MZML_NS = {"mz": "http://psi.hupo.org/ms/mzml"}


def is_imzml(path: Path | str) -> bool:
    path = Path(path)
    if path.suffix.lower() in [".imzml", ".imzxml"]:
        return True
    return False


def is_imzml_binary_data(path: Path | str) -> bool:
    path = Path(path)
    if path.suffix.lower() == ".ibd":
        return True
    return False


class ScanSettings(object):
    def __init__(
        self, image_size: tuple[int, int] | None, pixel_size: tuple[float, float]
    ):
        self.image_size = image_size
        self.pixel_size = pixel_size

    @classmethod
    def from_xml_element(cls, element: ElementTree.Element) -> "ScanSettings":
        x = element.find(
            f"mz:cvParam[@accession='{CV_SCANSETTINGS['MAX_COUNT_OF_PIXEL_X']}']",
            MZML_NS,
        )
        y = element.find(
            f"mz:cvParam[@accession='{CV_SCANSETTINGS['MAX_COUNT_OF_PIXEL_Y']}']",
            MZML_NS,
        )
        if x is None or y is None:  # pragma: no cover
            # we can calculate this from the max x, y pos
            image_size = None
        else:
            image_size = int(x.attrib["value"]), int(y.attrib["value"])

        px = element.find(
            f"mz:cvParam[@accession='{CV_SCANSETTINGS['PIXEL_SIZE_X']}']",
            MZML_NS,
        )
        py = element.find(
            f"mz:cvParam[@accession='{CV_SCANSETTINGS['PIXEL_SIZE_Y']}']",
            MZML_NS,
        )
        if px is None or py is None:  # pragma: no cover
            raise ValueError("unable to read image pixel size")
        pixel_size = float(px.attrib["value"]), float(py.attrib["value"])

        return cls(image_size, pixel_size)


class Spectrum(object):
    def __init__(
        self,
        pos: tuple[int, int],
        tic: float | None,
        offsets: dict[str, int],
        lengths: dict[str, int],
    ):
        self.pos = pos
        self.tic = tic
        self.offsets = offsets
        self.lengths = lengths

    @property
    def x(self) -> int:
        return self.pos[0]

    @property
    def y(self) -> int:
        return self.pos[1]

    @classmethod
    def from_xml_element(
        cls, element: ElementTree.Element, scan_number: int = 1
    ) -> "Spectrum":
        scans = element.findall("mz:scanList/mz:scan", MZML_NS)
        if len(scans) < scan_number:  # pragma: no cover
            raise ValueError(f"unable to find scan {scan_number}")

        px = scans[scan_number - 1].find(
            f"mz:cvParam[@accession='{CV_SPECTRUM['POSITION_X']}']", MZML_NS
        )
        py = scans[scan_number - 1].find(
            f"mz:cvParam[@accession='{CV_SPECTRUM['POSITION_Y']}']", MZML_NS
        )
        if px is None or py is None:  # pragma: no cover
            raise ValueError("unable to read position from spectrum")

        pos = int(px.attrib["value"]), int(py.attrib["value"])

        tic_element = element.find(
            f"mz:cvParam[@accession='{CV_SPECTRUM['TOTAL_ION_CURRENT']}']", MZML_NS
        )
        if tic_element is None:  # pragma: no cover, tic None
            tic = None
        else:
            tic = float(tic_element.get("value", 0.0))

        offsets: dict[str, int] = {}
        lengths: dict[str, int] = {}

        for array in element.iterfind(
            "mz:binaryDataArrayList/mz:binaryDataArray", MZML_NS
        ):
            key = array.find("mz:referenceableParamGroupRef", MZML_NS)
            offset = array.find(
                f"mz:cvParam[@accession='{CV_SPECTRUM['EXTERNAL_OFFSET']}']", MZML_NS
            )
            length = array.find(
                f"mz:cvParam[@accession='{CV_SPECTRUM['EXTERNAL_ENCODED_LENGTH']}']",
                MZML_NS,
            )
            if key is None or offset is None or length is None:  # pragma: no cover
                raise ValueError("invalid binary data array")
            offsets[key.attrib["ref"]] = int(offset.attrib["value"])
            lengths[key.attrib["ref"]] = int(length.attrib["value"])

        return cls(pos, tic, offsets, lengths)

    def get_binary_data(
        self,
        reference_id: str,
        dtype: type,
        external_binary: Path | BufferedReader | None = None,
    ) -> np.ndarray:
        if external_binary is None:  # pragma: no cover
            raise NotImplementedError("direct read of binary data not supported")

        if not isinstance(external_binary, BufferedReader):
            external_binary = external_binary.open("rb")

        external_binary.seek(self.offsets[reference_id])
        buffer = external_binary.read(self.lengths[reference_id])
        return np.frombuffer(buffer, dtype=dtype)


class ParamGroup(object):
    type_names = {
        "BINARY_TYPE_8BIT_INTEGER": np.uint8,
        "BINARY_TYPE_16BIT_INTEGER": np.uint16,
        "BINARY_TYPE_32BIT_INTEGER": np.uint32,
        "BINARY_TYPE_64BIT_INTEGER": np.uint64,
        "BINARY_TYPE_32BIT_FLOAT": np.float32,
        "BINARY_TYPE_64BIT_FLOAT": np.float64,
    }
    mz_array_cv = CV_PARAMGROUP["MZ_ARRAY"]
    intensity_array_cv = CV_PARAMGROUP["INTENSITY_ARRAY"]

    def __init__(
        self, id: str, dtype: type, compressed: bool = False, external: bool = False
    ):
        self.id = id
        self.dtype = dtype
        self.compressed = compressed
        self.external = external

    @classmethod
    def from_xml_element(cls, element: ElementTree.Element) -> "ParamGroup":
        id = element.get("id", None)
        if id is None:  # pragma: no cover
            raise ValueError(f"invalid param group element {element}")

        dtype = None
        for key, val in CV_BINARYDATA.items():
            if element.find(f"mz:cvParam[@accession='{val}']", MZML_NS) is not None:
                dtype = ParamGroup.type_names[key]
                break

        if dtype is None:  # pragma: no cover
            raise ValueError(f"cannot read dtype for param group {id}")

        compressed, external = False, False

        if (
            element.find(
                f"mz:cvParam[@accession='{CV_PARAMGROUP['NO_COMPRESSION']}']", MZML_NS
            )
            is not None
        ):  # pragma: no cover
            raise NotImplementedError("reading compressed data not implemented")

        if (
            element.find(
                f"mz:cvParam[@accession='{CV_PARAMGROUP['EXTERNAL_DATA']}']", MZML_NS
            )
            is not None
        ):
            external = True

        return cls(id, dtype, compressed=compressed, external=external)


class ImzML(object):
    def __init__(
        self,
        scan_settings: ScanSettings,
        mz_params: ParamGroup,
        intensity_params: ParamGroup,
        spectra: list[Spectrum] | dict[tuple[int, int], Spectrum],
        external_binary: Path | str,
    ):

        self.scan_settings = scan_settings
        self.mz_params = mz_params
        self.intensity_params = intensity_params

        if isinstance(spectra, list):  # pragma: no cover, just building dict
            self.spectra = {(s.x, s.y): s for s in spectra}
        else:
            self.spectra = spectra

        self.external_binary = Path(external_binary)

    @property
    def image_size(self) -> tuple[int, int]:
        if self.scan_settings.image_size is not None:
            return self.scan_settings.image_size
        else:
            positions = np.array(self.spectra.keys())
            return np.amax(positions[:, 0]), np.amax(positions[:, 1])

    @classmethod
    def from_etree(
        cls,
        et: ElementTree.ElementTree,
        external_binary: Path | str,
        scan_number: int = 1,
    ) -> "ImzML":
        params_list = et.find("mz:referenceableParamGroupList", MZML_NS)
        if params_list is None:  # pragma: no cover
            raise ValueError("parameter list not found")

        mz_params = params_list.find(
            "mz:referenceableParamGroup/"
            f"mz:cvParam[@accession='{ParamGroup.mz_array_cv}']/..",
            MZML_NS,
        )
        if mz_params is None:  # pragma: no cover
            raise ValueError("unable to find m/z array parameters")
        intensity_params = params_list.find(
            "mz:referenceableParamGroup/"
            f"mz:cvParam[@accession='{ParamGroup.intensity_array_cv}']/..",
            MZML_NS,
        )
        if intensity_params is None:  # pragma: no cover
            raise ValueError("unable to find intensity array parameters")

        spectra = {}
        for element in et.iterfind("mz:run/mz:spectrumList/mz:spectrum", MZML_NS):
            spectrum = Spectrum.from_xml_element(element)
            spectra[(spectrum.x, spectrum.y)] = spectrum

        scans = et.findall("mz:scanSettingsList/mz:scanSettings", MZML_NS)
        if len(scans) < scan_number:  # pragma: no cover
            raise ValueError(f"unable to find scan settings for scan {scan_number}")

        return cls(
            ScanSettings.from_xml_element(scans[scan_number - 1]),
            ParamGroup.from_xml_element(mz_params),
            ParamGroup.from_xml_element(intensity_params),
            spectra,
            external_binary=external_binary,
        )

    @classmethod
    def from_file(
        cls, path: Path | str, external_binary: Path | str | None = None
    ) -> "ImzML":
        """Create an ImzML object from a file path.
        If `external_binary` is None, the imzML path with suffix '.ibd' is used.

        Args:
            path: path to imzML file
            external_binary: path to .ibd file
        """

        path = Path(path)
        if not path.exists():  # pragma: no cover, bad file
            raise FileNotFoundError("imzML file not found")

        if external_binary is None:
            external_binary = path.with_suffix(".ibd")
        else:  # pragma: no cover, supplied file
            external_binary = Path(external_binary)

        if not external_binary.exists():  # pragma: no cover, bad file
            raise FileNotFoundError("external binary file not found")

        et = ElementTree.parse(path)
        return ImzML.from_etree(et, external_binary)

    def mass_range(self) -> tuple[float, float]:
        """Maximum mass range."""

        fp = self.external_binary.open("rb")

        low, high = np.inf, -np.inf
        for spectra in self.spectra.values():
            mz_array = spectra.get_binary_data(
                self.mz_params.id,
                self.mz_params.dtype,
                external_binary=fp,
            )
            low = min(low, mz_array[0])
            high = max(high, mz_array[-1])
        return low, high

    def extract_tic(self) -> np.ndarray:
        """The total-ion-chromatogram image.

        Extracted from the cvParam MS:1000285.
        """
        tic = np.full(self.image_size, np.nan, dtype=float)
        for (x, y), spec in self.spectra.items():
            tic[x - 1, y - 1] = spec.tic
        return np.rot90(tic, 1)

    def extract_masses(
        self,
        target_masses: np.ndarray | float,
        mass_width_ppm: float | None = 10.0,
        mass_width_mz: float | None = None,
    ) -> np.ndarray:
        """Extracts image of one or more m/z.

        Data within +/- 0.5 `mass_width_ppm` or `mass_width_mz` is summed.

        Args:
            target_masses: m/z to extract
            mass_width_ppm: extraction width in ppm
            mass_width_mz: extraction width in m/z (Da)

        Returns:
            array of intensities, shape (X, Y, N)
        """
        target_masses = np.atleast_1d(target_masses)
        if mass_width_ppm is not None:
            target_widths: np.ndarray | float = (
                target_masses * mass_width_ppm / 1e6 / 2.0
            )
        elif mass_width_mz is not None:
            target_widths = mass_width_mz / 2.0
        else:  # pragma: no cover
            raise ValueError(
                "either 'mass_width_ppm' or 'mass_width_mz' must be supplied."
            )

        target_windows = np.stack(
            (target_masses - target_widths, target_masses + target_widths), axis=1
        )

        fp = self.external_binary.open("rb")

        data: np.ndarray = np.full(
            (*self.image_size, len(target_masses)),
            np.nan,
            dtype=self.intensity_params.dtype,
        )
        for spectra in self.spectra.values():
            mz_array = spectra.get_binary_data(
                self.mz_params.id, self.mz_params.dtype, external_binary=fp
            )
            intensity_array = spectra.get_binary_data(
                self.intensity_params.id,
                self.intensity_params.dtype,
                external_binary=fp,
            )
            idx = np.searchsorted(mz_array, target_windows.flat)
            # faster than np.clip
            idx[idx > intensity_array.size - 1] = intensity_array.size - 1
            data[spectra.x - 1, spectra.y - 1] = np.add.reduceat(intensity_array, idx)[
                ::2
            ]
        return np.rot90(data, 1)

    def binned_masses(
        self, mass_width_mz: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Summed intensities within a certain width.

        Bins data across the entire mass range, with a bin width of `mass_width_mz`.

        Args:
            mass_width_mz: width of each bin
        Returns:
            array of bins, binned intensity data
        """
        mass_min, mass_max = self.mass_range()
        bins = np.arange(mass_min, mass_max + mass_width_mz, mass_width_mz)

        fp = self.external_binary.open("rb")

        data: np.ndarray = np.full(
            (*self.image_size, len(bins)),
            np.nan,
            dtype=self.intensity_params.dtype,
        )
        for spectra in self.spectra.values():
            mz_array = spectra.get_binary_data(
                self.mz_params.id,
                self.mz_params.dtype,
                external_binary=fp,
            )
            intensity_array = spectra.get_binary_data(
                self.intensity_params.id,
                self.intensity_params.dtype,
                external_binary=fp,
            )
            idx = np.searchsorted(mz_array, bins.flat)
            # faster than np.clip
            idx[idx > intensity_array.size - 1] = intensity_array.size - 1
            data[spectra.x - 1, spectra.y - 1] = np.add.reduceat(intensity_array, idx)
        return bins, np.rot90(data, 1)

    def untargeted_extraction(
        self,
        num: int = 10,
        precision_mz: float = 0.1,
        min_pixel_count: int = 10,
        min_height_fraction: float = 0.1,
        min_height_absolute: float = 100.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extracts the `num` most abundant masses for each spectra.
        The `precision` specifies the width to use for grouping similar masses.

        Args:
            num: number of peaks per spectra to test
            precision: number of  decimals for grouping m/z
            min_pixel_count: minimum number of pixels a mass must occour in
            min_height_fraction: minimum peak height as fraction of maximum image signal
            min_height_absolute: minimum peak height in counts

        Returns:
            array of (average) masses len N, image of size (X, Y, N)
        """

        mzs: np.ndarray = np.full(
            (*self.image_size, num),
            np.nan,
            dtype=self.mz_params.dtype,
        )
        intensities: np.ndarray = np.full(
            (*self.image_size, num),
            np.nan,
            dtype=self.intensity_params.dtype,
        )

        fp = self.external_binary.open("rb")
        for spectra in self.spectra.values():
            mz_array = spectra.get_binary_data(
                self.mz_params.id, self.mz_params.dtype, external_binary=fp
            )
            intensity_array = spectra.get_binary_data(
                self.intensity_params.id,
                self.intensity_params.dtype,
                external_binary=fp,
            )
            idx = np.argpartition(intensity_array, -num)[-num:]
            mzs[spectra.x - 1, spectra.y - 1] = mz_array[idx]
            intensities[spectra.x - 1, spectra.y - 1] = intensity_array[idx]

        # filter peaks below height minimums
        max_signal = np.unravel_index(np.nanargmax(intensities), intensities.shape)
        valid = np.logical_and(
            intensities > (intensities[max_signal] * min_height_fraction),
            intensities > min_height_absolute,
        )
        mzs[~valid] = np.nan
        intensities[~valid] = np.nan

        # find unique grouped masses
        valid = ~np.isnan(mzs)
        # round to precision_mz
        rounded_masses = precision_mz * np.round(mzs[valid] / precision_mz)
        unique_masses, idx, counts = np.unique(
            rounded_masses, return_inverse=True, return_counts=True
        )
        # filter below count minimum
        valid_idx = np.flatnonzero(counts > min_pixel_count)

        # compute average masses and re-extract
        avg_masses = np.empty(valid_idx.size)
        for i in range(len(valid_idx)):
            avg_masses[i] = np.mean(mzs[valid][idx == valid_idx[i]])
        data = self.extract_masses(avg_masses, mass_width_mz=precision_mz)
        return avg_masses, data


def load(
    imzml: Path | str | ImzML,
    external_binary: Path | str,
    target_masses: float | np.ndarray,
    mass_width_ppm: float = 10.0,
) -> tuple[np.ndarray, dict]:  # pragma: no cover, tested elsewehere
    """Load data from an imzML.

    Args:
        imzml: path to imzML, or pre-parsed tree
        external_binary: path to binary data, usually .ibd file
        target_masses: masses to import
        mass_width_ppm: width of imported regions

    Returmz:
        image data, dict of parameters
    """

    if not isinstance(imzml, ImzML):
        imzml = ImzML.from_file(imzml, external_binary)

    return imzml.extract_masses(target_masses, mass_width_ppm), {
        "spotsize": imzml.scan_settings.pixel_size
    }
