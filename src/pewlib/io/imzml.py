from pathlib import Path
from xml.etree import ElementTree

import numpy as np

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
    "LOWEST_OBSERVED_MZ": "MS:1000528",
    "HIGHEST_OBSERVED_MZ": "MS:1000527",
}

MZML_NS = {"mz": "http://psi.hupo.org/ms/mzml"}


class ScanSettings(object):
    def __init__(self, image_size: tuple[int, int], pixel_size: tuple[float, float]):
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
        if x is None or y is None:
            raise ValueError("unable to read image size")
        image_size = int(x.attrib["value"]), int(y.attrib["value"])

        px = element.find(
            f"mz:cvParam[@accession='{CV_SCANSETTINGS['PIXEL_SIZE_X']}']",
            MZML_NS,
        )
        py = element.find(
            f"mz:cvParam[@accession='{CV_SCANSETTINGS['PIXEL_SIZE_Y']}']",
            MZML_NS,
        )
        if px is None or py is None:
            raise ValueError("unable to read image pixel size")
        pixel_size = float(px.attrib["value"]), float(py.attrib["value"])

        return cls(image_size, pixel_size)


class Spectrum(object):
    def __init__(
        self,
        pos: tuple[int, int],
        tic: float,
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

        pos = int(px.get("value", -1)), int(py.get("value", -1))

        tic_element = element.find(
            f"mz:cvParam[@accession='{CV_SPECTRUM['TOTAL_ION_CURRENT']}']", MZML_NS
        )
        if tic_element is None:
            raise ValueError("unable to read TIC from spectrum")
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
                f"mz:cvParam[@accession='{CV_SPECTRUM['EXTERNAL_ARRAY_LENGTH']}']",
                MZML_NS,
            )
            if key is None or offset is None or length is None:
                raise ValueError("invalid binary data array")
            offsets[key.get("ref", "")] = int(offset.get("value", -1))
            lengths[key.get("ref", "")] = int(length.get("value", -1))

        return cls(pos, tic, offsets, lengths)

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


class ParamGroup(object):
    type_names = {
        "BINARY_TYPE_8BIT_INTEGER": np.uint8,
        "BINARY_TYPE_16BIT_INTEGER": np.uint16,
        "BINARY_TYPE_32BIT_INTEGER": np.uint32,
        "BINARY_TYPE_64BIT_INTEGER": np.uint64,
        "BINARY_TYPE_32BIT_FLOAT": np.float32,
        "BINARY_TYPE_64BIT_FLOAT": np.float64,
    }

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
        if id is None:
            raise ValueError(f"invalid param group element {element}")

        dtype = None
        for key, val in CV_BINARYDATA.items():
            if element.find(f"mz:cvParam[@accession='{val}']", MZML_NS) is not None:
                dtype = ParamGroup.type_names[key]
                break

        if dtype is None:
            raise ValueError(f"cannot read dtype for param group {id}")

        compressed, external = False, False

        if (
            element.find(
                f"mz:cvParam[@accession='{CV_PARAMGROUP['NO_COMPRESSION']}']", MZML_NS
            )
            is not None
        ):
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

        if isinstance(spectra, list):
            self.spectra = {(s.x, s.y): s for s in spectra}
        else:
            self.spectra = spectra

        self.external_binary = external_binary

    @classmethod
    def from_etree(
        cls,
        et: ElementTree.ElementTree,
        external_binary: Path | str,
        scan_number: int = 1,
    ) -> "ImzML":
        scans = et.findall("mz:scanSettingsList/mz:scanSettings", MZML_NS)
        if len(scans) < scan_number:
            raise ValueError(f"unable to find scan settings for scan {scan_number}")

        mz_params = et.find(
            "mz:referenceableParamGroupList/mz:referenceableParamGroup"
            f"mz:cvParam[@accession='{CV_PARAMGROUP['MZ_ARRAY']}']/..",
            MZML_NS,
        )
        if mz_params is None:
            raise ValueError("unable to find m/z array parameters")
        intensity_params = et.find(
            "mz:referenceableParamGroupList/mz:referenceableParamGroup"
            f"mz:cvParam[@accession='{CV_PARAMGROUP['INTENSITY_ARRAY']}']/..",
            MZML_NS,
        )
        if intensity_params is None:
            raise ValueError("unable to find intensity array parameters")

        spectra = {}
        for element in et.iterfind("mz:run/mz:spectrumList/mz:spectrum", MZML_NS):
            spectrum = Spectrum.from_xml_element(element)
            spectra[(spectrum.x, spectrum.y)] = spectrum

        return cls(
            ScanSettings.from_xml_element(scans[scan_number - 1]),
            ParamGroup.from_xml_element(mz_params),
            ParamGroup.from_xml_element(intensity_params),
            spectra,
            external_binary=external_binary,
        )

    @classmethod
    def from_file(cls, path: Path | str, external_binary: Path | str) -> "ImzML":
        et = ElementTree.parse(path)
        return ImzML.from_etree(et, external_binary)

    @classmethod
    def from_file_iterative(
        cls,
        path: Path | str,
        external_binary: Path | str,
        scan_number: int = 1,
        callback: None = None,
    ) -> "ImzML":
        iter = ElementTree.iterparse(path, events=("end",))

        spectra = {}

        for event, elem in iter:
            if elem.tag == f"{{{MZML_NS['mz']}}}scanSettingsList":
                scans = elem.findall("mz:scanSettings", MZML_NS)
                scan_settings = ScanSettings.from_xml_element(scans[scan_number - 1])
                elem.clear()
            elif elem.tag == f"{{{MZML_NS['mz']}}}referenceableParamGroup":
                if (
                    elem.find(
                        f"mz:cvParam[@accession='{CV_PARAMGROUP['MZ_ARRAY']}']", MZML_NS
                    )
                    is not None
                ):
                    mz_group = ParamGroup.from_xml_element(elem)
                elif (
                    elem.find(
                        f"mz:cvParam[@accession='{CV_PARAMGROUP['INTENSITY_ARRAY']}']",
                        MZML_NS,
                    )
                    is not None
                ):
                    intensity_group = ParamGroup.from_xml_element(elem)
                elem.clear()
            elif elem.tag == f"{{{MZML_NS['mz']}}}spectrum":
                spectrum = Spectrum.from_xml_element(elem)
                spectra[(spectrum.x, spectrum.y)] = spectrum
                elem.clear()
                if callback is not None:
                    if callback():
                        return
        return cls(
            scan_settings,
            mz_group,
            intensity_group,
            spectra,
            external_binary=external_binary,
        )

    def mass_range(self) -> tuple[float, float]:
        low, high = np.inf, -np.inf
        for spectra in self.spectra.values():
            mz_array = spectra.get_binary_data(
                self.mz_params.id,
                self.mz_params.dtype,
                external_binary=self.external_binary,
            )
            low = min(low, mz_array[0])
            high = max(high, mz_array[-1])
        return low, high

    def extract_tic(self) -> np.ndarray:
        tic = np.zeros(self.scan_settings.image_size, dtype=float)
        for (x, y), spec in self.spectra.items():
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
            (*self.scan_settings.image_size, len(target_masses)),
            np.nan,
            dtype=self.intensity_params.dtype,
        )
        for spectra in self.spectra.values():
            mz_array = spectra.get_binary_data(
                self.mz_params.id,
                self.mz_params.dtype,
                external_binary=self.external_binary,
            )
            intensity_array = spectra.get_binary_data(
                self.intensity_params.id,
                self.intensity_params.dtype,
                external_binary=self.external_binary,
            )
            idx = np.searchsorted(mz_array, target_windows.flat)
            data[spectra.x - 1, spectra.y - 1] = np.add.reduceat(intensity_array, idx)[
                ::2
            ]
        return np.rot90(data, 1)


def load(
    imzml: Path | str | ImzML,
    external_binary: Path | str,
    target_masses: float | np.ndarray,
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
        imzml = ImzML.from_file(imzml, external_binary)

    return imzml.extract_masses(target_masses, mass_width_ppm), {
        "spotsize": imzml.scan_settings.pixel_size
    }
