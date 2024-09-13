from pathlib import Path
from xml.etree import ElementTree

import numpy as np

CV_PARAMGROUP = {
    "MZ_ARRAY": "MS:1000514",
    "INTENSITY_ARRAY": "MS:1000515",
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
    "LOWEST_OBSERVED_MZ": "MS:1000528",
    "HIGHEST_OBSERVED_MZ": "MS:1000527",
}

MZML_NS = {"mz": "http://psi.hupo.org/ms/mzml"}


def get_image_size(
    imzml: ElementTree.ElementTree, scan_number: int = 1
) -> tuple[int, int]:
    scans = imzml.findall("mz:scanSettingsList/mz:scanSettings", MZML_NS)
    if len(scans) < scan_number:
        raise ValueError(f"unable to read image size for scan {scan_number}")
    x = scans[scan_number - 1].find(
        f"mz:cvParam[@accession='{CV_SCANSETTINGS['MAX_COUNT_OF_PIXEL_X']}']",
        MZML_NS,
    )
    y = scans[scan_number - 1].find(
        f"mz:cvParam[@accession='{CV_SCANSETTINGS['MAX_COUNT_OF_PIXEL_Y']}']",
        MZML_NS,
    )
    if x is None or y is None:
        raise ValueError("unable to read image size")
    return int(x.get("value", -1)), int(y.get("value", -1))


def get_image_pixel_size(
    imzml: ElementTree.ElementTree, scan_number: int = 1
) -> tuple[int, int]:
    scans = imzml.findall("mz:scanSettingsList/mz:scanSettings", MZML_NS)
    if len(scans) < scan_number:
        raise ValueError(f"unable to read pixel size for scan {scan_number}")
    x = scans[scan_number - 1].find(
        f"mz:cvParam[@accession='{CV_SCANSETTINGS['PIXEL_SIZE_X']}']",
        MZML_NS,
    )
    y = scans[scan_number - 1].find(
        f"mz:cvParam[@accession='{CV_SCANSETTINGS['PIXEL_SIZE_Y']}']",
        MZML_NS,
    )
    if x is None or y is None:
        raise ValueError("unable to read image pixel size")
    return int(x.get("value", -1)), int(y.get("value", -1))


def get_reference_id(imzml: ElementTree.ElementTree, cv: str) -> str:
    for group in imzml.iterfind(
        "mz:referenceableParamGroupList/mz:referenceableParamGroup", MZML_NS
    ):
        if (
            group.find(f"mz:cvParam[@accession='{CV_PARAMGROUP[cv]}']", MZML_NS)
            is not None
        ):
            id = group.get("id", None)
            if id is None:
                raise ValueError(f"unable to find id for '{cv}' reference group")
            return id
    raise KeyError(f"unable to find group for '{cv}'")


def get_data_type_for_reference(
    imzml: ElementTree.ElementTree, reference_id: str
) -> type:
    type_names = {
        "BINARY_TYPE_8BIT_INTEGER": np.uint8,
        "BINARY_TYPE_16BIT_INTEGER": np.uint16,
        "BINARY_TYPE_32BIT_INTEGER": np.uint32,
        "BINARY_TYPE_64BIT_INTEGER": np.uint64,
        "BINARY_TYPE_32BIT_FLOAT": np.float32,
        "BINARY_TYPE_64BIT_FLOAT": np.float64,
    }
    group = imzml.find(
        "mz:referenceableParamGroupList/"
        f"mz:referenceableParamGroup[@id='{reference_id}']",
        MZML_NS,
    )
    if group is None:
        raise ValueError(f"cannot find group '{reference_id}'")
    for key, val in CV_BINARYDATA.items():
        if group.find(f"mz:cvParam[@accession='{val}']", MZML_NS) is not None:
            return type_names[key]
    raise KeyError(f"cannot find data type for '{reference_id}'")


class Spectrum(object):
    def __init__(self, spectrum: ElementTree.Element, scan_number: int = 1):
        scans = spectrum.findall("mz:scanList/mz:scan", MZML_NS)
        if len(scans) < scan_number:
            raise ValueError(f"unable to find scan {scan_number}")

        px = scans[scan_number - 1].find(
            f"mz:cvParam[@accession='{CV_SPECTRUM['POSITION_X']}']", MZML_NS
        )
        py = scans[scan_number - 1].find(
            f"mz:cvParam[@accession='{CV_SPECTRUM['POSITION_Y']}']", MZML_NS
        )
        if px is None or py is None:
            raise ValueError("unable to read position from spectrum")

        self.pos = int(px.get("value", -1)), int(py.get("value", -1))

        tic = spectrum.find(
            f"mz:cvParam[@accession='{CV_SPECTRUM['TOTAL_ION_CURRENT']}']", MZML_NS
        )
        if tic is None:
            raise ValueError("unable to read TIC from spectrum")
        self.tic = float(tic.get("value", 0.0))

        self.offsets: dict[str, int] = {}
        self.lengths: dict[str, int] = {}

        for array in spectrum.iterfind(
            "mz:binaryDataArrayList/mz:binaryDataArray", MZML_NS
        ):
            key = array.find("mz:referenceableParamGroupRef", MZML_NS)
            offset = array.find(
                f"mz:cvParam[@accession='{CV_SPECTRUM['EXTERNAL_OFFSET']}']", MZML_NS
            )
            length = array.find(
                f"mz:cvParam[@accession='{CV_SPECTRUM['EXTERNAL_ARRAY_LENGTH']}']",
                MZML_NS,
            )
            if key is None or offset is None or length is None:
                raise ValueError("invalid binary data array")
            self.offsets[key.get("ref", "")] = int(offset.get("value", -1))
            self.lengths[key.get("ref", "")] = int(length.get("value", -1))

    @property
    def x(self) -> int:
        return self.pos[0]

    @property
    def y(self) -> int:
        return self.pos[1]

    def get_binary_data(
        self, reference_id: str, dtype: type, external_binary: Path | str | None = None
    ) -> np.ndarray:
        if external_binary is None:
            raise NotImplementedError("direct read of binary data not supported")

        return np.fromfile(
            external_binary,
            count=self.lengths[reference_id],
            offset=self.offsets[reference_id],
            dtype=dtype,
        )


class ImzML(object):
    def __init__(self, imzml: Path | str, external_binary: Path | str | None = None):
        et = ElementTree.parse(imzml)

        self.external_binary = external_binary

        self.image_size = get_image_size(et)
        self.pixel_size = get_image_pixel_size(et)

        self.mz_reference = get_reference_id(et, "MZ_ARRAY")
        self.mz_dtype = get_data_type_for_reference(et, self.mz_reference)
        self.intensity_reference = get_reference_id(et, "INTENSITY_ARRAY")
        self.intensity_dtype = get_data_type_for_reference(et, self.intensity_reference)

        spectra = (
            Spectrum(s)
            for s in et.iterfind("mz:run/mz:spectrumList/mz:spectrum", MZML_NS)
        )
        self.spectra_map = {(s.x, s.y): s for s in spectra}

    def spectrum_at(self, x: int, y: int) -> Spectrum:
        return self.spectra_map[(x, y)]

    def mass_range(self) -> tuple[float, float]:
        low, high = np.inf, -np.inf
        for spectra in self.spectra_map.values():
            mz_array = spectra.get_binary_data(
                self.mz_reference, self.mz_dtype, external_binary=self.external_binary
            )
            low = min(low, mz_array[0])
            high = max(high, mz_array[-1])
        return low, high

    def extract_tic(self) -> np.ndarray:
        tic = np.zeros(self.image_size, dtype=float)
        for (x, y), spec in self.spectra_map.items():
            tic[x - 1, y - 1] = spec.tic
        return np.rot90(tic, 1)

    def extract_masses(
        self, target_masses: np.ndarray | float, mass_width_ppm: float = 10.0
    ) -> np.ndarray:
        target_masses = np.asanyarray(target_masses)
        target_widths = target_masses * mass_width_ppm / 1e6
        target_windows = np.stack(
            (target_masses - target_widths, target_masses + target_widths), axis=1
        )

        data: np.ndarray = np.full(
            (*self.image_size, len(target_masses)), np.nan, dtype=self.intensity_dtype
        )
        for spectra in self.spectra_map.values():
            mz_array = spectra.get_binary_data(
                self.mz_reference, self.mz_dtype, external_binary=self.external_binary
            )
            intensity_array = spectra.get_binary_data(
                self.intensity_reference,
                self.intensity_dtype,
                external_binary=self.external_binary,
            )
            idx = np.searchsorted(mz_array, target_windows.flat)
            data[spectra.x - 1, spectra.y - 1] = np.add.reduceat(intensity_array, idx)[
                ::2
            ]
        return np.rot90(data, 1)


def load(
    imzml: Path | str | ImzML,
    target_masses: float | np.ndarray,
    external_binary: Path | str | None = None,
    mass_width_ppm: float = 10.0,
) -> tuple[np.ndarray, dict]:
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
        imzml = ImzML(imzml, external_binary)

    return imzml.extract_masses(target_masses, mass_width_ppm), {
        "spotsize": imzml.pixel_size
    }
