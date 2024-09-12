import logging
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

logger = logging.getLogger(__name__)

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

#
# class Spectrum(object):
#     def __init__(self, spectrum: ElementTree.Element, scan_number: int = 1):
#         self.xml = spectrum
#         self.scan_number = scan_number
#
#         self._pos: tuple[int, int] | None = None
#
#         self.offsets: dict[str, int] = {}
#         self.lengths: dict[str, int] = {}
#
#     @property
#     def pos(self) -> tuple[int, int]:
#         if self._pos is None:
#             scan = self.xml.find(f"mz:scanList/mz:scan[{self.scan_number}]", MZML_NS)
#             if scan is None:
#                 raise ValueError(f"unable to find scan {self.scan_number}")
#
#             px = scan.find(
#                 f"mz:cvParam[@accession='{CV_SPECTRUM['POSITION_X']}']", MZML_NS
#             )
#             py = scan.find(
#                 f"mz:cvParam[@accession='{CV_SPECTRUM['POSITION_Y']}']", MZML_NS
#             )
#             if px is None or py is None:
#                 raise ValueError("unable to read position from spectrum")
#
#             self._pos = int(px.get("value", -1)), int(py.get("value", -1))
#         return self._pos
#
#     def get_binary_data(self, reference_id: str, dtype: type) -> np.ndarray:
#         offset = self.xml.find(
#             f"mz:cvParam[@accession='{CV_SPECTRUM['EXTERNAL_OFFSET']}']", MZML_NS
#         )
#         length = self.xml.find(
#             f"mz:cvParam[@accession='{CV_SPECTRUM['EXTERNAL_ARRAY_LENGTH']}']", MZML_NS
#         )
#         if offset is None or length is None:
#             raise ValueError("unable to read external offset / length for spectrum")
#
#         return np.fromfile(
#             external_binary,
#             count=int(length.get("value", -1)),
#             offset=int(offset.get("value", -1)),
#             dtype=dtype,
#         )


def get_image_size(
    xml: ElementTree.ElementTree, scan_number: int = 1
) -> tuple[int, int]:
    scan = xml.find(f"mz:scanSettingsList/mz:scanSettings[{scan_number}]", MZML_NS)
    if scan is None:
        raise ValueError(f"unable to read scan settings {scan_number}")
    x = scan.find(
        f"mz:cvParam[@accession='{CV_SCANSETTINGS['MAX_COUNT_OF_PIXEL_X']}']", MZML_NS
    )
    y = scan.find(
        f"mz:cvParam[@accession='{CV_SCANSETTINGS['MAX_COUNT_OF_PIXEL_Y']}']", MZML_NS
    )
    if x is None or y is None:
        raise ValueError("unable to read image size")
    return int(x.get("value", -1)), int(y.get("value", -1))


def get_image_pixel_size(
    xml: ElementTree.ElementTree, scan_number: int = 1
) -> tuple[int, int]:
    scan = xml.find(f"mz:scanSettingsList/mz:scanSettings[{scan_number}]", MZML_NS)
    if scan is None:
        raise ValueError(f"unable to read scan settings {scan_number}")
    x = scan.find(
        f"mz:cvParam[@accession='{CV_SCANSETTINGS['PIXEL_SIZE_X']}']", MZML_NS
    )
    y = scan.find(
        f"mz:cvParam[@accession='{CV_SCANSETTINGS['PIXEL_SIZE_Y']}']", MZML_NS
    )
    if x is None or y is None:
        raise ValueError("unable to read image pixel size")
    return int(x.get("value", -1)), int(y.get("value", -1))


def get_reference_id(xml: ElementTree.ElementTree, cv: str) -> str:
    group = xml.find(
        "mz:referenceableParamGroupList/mz:referenceableParamGroup/"
        f"mz:cvParam[@accession='{CV_PARAMGROUP[cv]}']/..",
        MZML_NS,
    )
    if group is None:
        raise ValueError(f"unable to find '{cv}' reference group")
    id = group.get("id", None)
    if id is None:
        raise ValueError(f"unable to find id for '{cv}' reference group")
    return id


def get_data_type_for_reference(
    xml: ElementTree.ElementTree, reference_id: str
) -> type:
    type_names = {
        "BINARY_TYPE_8BIT_INTEGER": np.uint8,
        "BINARY_TYPE_16BIT_INTEGER": np.uint16,
        "BINARY_TYPE_32BIT_INTEGER": np.uint32,
        "BINARY_TYPE_64BIT_INTEGER": np.uint64,
        "BINARY_TYPE_32BIT_FLOAT": np.float32,
        "BINARY_TYPE_64BIT_FLOAT": np.float64,
    }
    group = xml.find(
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


def get_binary_data_from_spectrum(
    spectrum: ElementTree.Element,
    reference_id: str,
    dtype: type,
    external_binary: Path | str | None = None,
) -> np.ndarray:
    """
    Read binary data from the imzML and optional binary.

    Args:
        xml: the parsed xml
        mz: dict of namespaces
        reference_id: id of reference group, from `get_reference_id`
        dtype: data type, from `get_data_type_for_reference`
        external_binary: path to external binary data file
    """

    if external_binary is None:
        raise NotImplementedError(
            "direct reading of data from imzML not supported, "
            "please pass an external binary"
        )

    array = spectrum.find(
        "mz:binaryDataArrayList/mz:binaryDataArray/"
        f"mz:referenceableParamGroupRef[@ref='{reference_id}']/..",
        MZML_NS,
    )
    if array is None:
        raise ValueError(f"unable to find array with '{reference_id}' in spectrum")

    offset = array.find(
        f"mz:cvParam[@accession='{CV_SPECTRUM['EXTERNAL_OFFSET']}']", MZML_NS
    )
    length = array.find(
        f"mz:cvParam[@accession='{CV_SPECTRUM['EXTERNAL_ARRAY_LENGTH']}']", MZML_NS
    )
    if offset is None or length is None:
        raise ValueError("unable to read external offset / length for spectrum")

    return np.fromfile(
        external_binary,
        count=int(length.get("value", -1)),
        offset=int(offset.get("value", -1)),
        dtype=dtype,
    )


def get_position_from_spectrum(
    spectrum: ElementTree.Element, scan_number: int = 1
) -> tuple[int, int]:
    scan = spectrum.find(f"mz:scanList/mz:scan[{scan_number}]", MZML_NS)
    if scan is None:
        raise ValueError(f"unable to find scan {scan_number}")

    px = scan.find(f"mz:cvParam[@accession='{CV_SPECTRUM['POSITION_X']}']", MZML_NS)
    py = scan.find(f"mz:cvParam[@accession='{CV_SPECTRUM['POSITION_Y']}']", MZML_NS)
    if px is None or py is None:
        raise ValueError("unable to read position from spectrum")

    return int(px.get("value", -1)), int(py.get("value", -1))


def get_mz_range(
    xml: ElementTree.ElementTree,
    external_binary: Path | str | None = None,
    use_binary_data: bool = False,
) -> tuple[float, float]:
    """
    Get the maximum m/z range for the imzML.
    Attempts to read stored lowest and highest m/z,
    but will fallback to reading binary data.

    Args:
        xml: parsed imzML
        mz: namespaces
        external_binary: optional path to binary, required for `use_binary_data`
        use_binary_data: read from m/z array instead of stored lowest / highest m/z
        assume_sorted:

    Returmz:
        tuple of low, high m/z
    """
    spectra = xml.findall("mz:run/mz:spectrumList/mz:spectrum", MZML_NS)

    lowest, highest = np.inf, -np.inf

    if use_binary_data:
        ref_id = get_reference_id(xml, "MZ_ARRAY")
        dtype = get_data_type_for_reference(xml, ref_id)
        for spectrum in spectra:
            mz_array = get_binary_data_from_spectrum(
                spectrum, ref_id, dtype, external_binary=external_binary
            )
            # todo: check data is actually sorted
            lowest = min(lowest, mz_array[0])
            highest = max(highest, mz_array[-1])
    else:
        for spectrum in spectra:
            low_mz = spectrum.find(
                f"mz:cvParam[@accession='{CV_SPECTRUM['LOWEST_OBSERVED_MZ']}']", MZML_NS
            )
            high_mz = spectrum.find(
                f"mz:cvParam[@accession='{CV_SPECTRUM['HIGHEST_OBSERVED_MZ']}']",
                MZML_NS,
            )
            if low_mz is None or high_mz is None:
                logger.warning("falling back to use_binary_data")
                return get_mz_range(
                    xml, external_binary=external_binary, use_binary_data=True
                )
            lowest = min(lowest, float(low_mz.get("value", np.inf)))
            highest = max(highest, float(high_mz.get("value", -np.inf)))

    return lowest, highest


def get_all_mz(
    xml: ElementTree.ElementTree,
    external_binary: Path | str | None = None,
) -> np.ndarray:
    spectra = xml.findall("mz:run/mz:spectrumList/mz:spectrum", MZML_NS)
    ref_id = get_reference_id(xml, "MZ_ARRAY")
    dtype = get_data_type_for_reference(xml, ref_id)

    current_mz: np.ndarray = np.array([], dtype=dtype)

    for spectrum in spectra:
        mz_array = get_binary_data_from_spectrum(
            spectrum, ref_id, dtype, external_binary=external_binary
        )
        current_mz = np.unique_values([current_mz, mz_array])
    return current_mz


def parse_imzml(xml: Path | str) -> ElementTree.ElementTree:
    return ElementTree.parse(xml)


def get_spectrum_at_position(
    # spectra: list[Spectrum],
    xml: ElementTree.ElementTree,
    pos: tuple[int, int],
    scan_number: int = 1,
    ) -> ElementTree.Element:
# ) -> Spectrum:
    """Gets the spectrum element at pos.

    Args:
        xml: parsed xml
        mz: namespaces
        pos: zero-indexed position
        scan_number: which scan set to search pos

    Returmz:
        spectrum xml element
    """
    # for spectrum in spectra:
    #     if spectrum.pos == (pos[0] + 1, pos[1] + 1):
    #         return spectrum
    # raise StopIteration(f"spectrum not found for position {pos}")

    spectra = xml.findall("mz:run/mz:spectrumList/mz:spectrum", MZML_NS)
    # this is way faster than using a weird xPath
    for spectrum in spectra:
        scan = spectrum.find(f"mz:scanList/mz:scan[{scan_number}]", MZML_NS)
        if scan is None:
            raise ValueError("scan not found for spectrum")
        if (
            scan.find(
                f"mz:cvParam[@accession='{CV_SPECTRUM['POSITION_X']}']"
                f"[@value='{pos[0]+1}']",
                MZML_NS,
            )
            is not None
            and scan.find(
                f"mz:cvParam[@accession='{CV_SPECTRUM['POSITION_Y']}']"
                f"[@value='{pos[1]+1}']",
                MZML_NS,
            )
            is not None
        ):
            return spectrum
    raise StopIteration(f"spectrum not found for position {pos}")


def load(
    xml: Path | str | ElementTree.ElementTree,
    external_binary: Path | str | None = None,
    target_masses: float | np.ndarray | None = None,
    mass_width_ppm: float = 10.0,
    scan_number: int = 1,
) -> tuple[np.ndarray, dict]:
    """Load data from an imzML.

    Args:
        xml: path to imzML, or pre-parsed tree
        external_binary: path to binary data, usually .ibd file
        target_masses: masses to import
        mass_width_ppm: width of imported regions
        scan_number: which set of scan / scan settings to import

    Returmz:
        image data, dict of parameters
    """

    if isinstance(xml, (Path, str)):
        xml = ElementTree.parse(xml)

    if isinstance(external_binary, str):
        external_binary = Path(external_binary)

    # calculate target m/z windows
    target_masses = np.asanyarray(target_masses)
    target_widths = target_masses * mass_width_ppm / 1e6
    target_windows = np.stack(
        (target_masses - target_widths, target_masses + target_widths), axis=1
    )

    pixel_size = get_image_pixel_size(xml, scan_number=scan_number)
    size = get_image_size(xml, scan_number=scan_number)

    ref_id_mz = get_reference_id(xml, "MZ_ARRAY")
    dtype_mz = get_data_type_for_reference(xml, ref_id_mz)
    ref_id_signal = get_reference_id(xml, "INTENSITY_ARRAY")
    dtype_signal = get_data_type_for_reference(xml, ref_id_signal)

    data: np.ndarray = np.full((*size, len(target_masses)), np.nan, dtype=dtype_signal)

    spectra = xml.findall("mz:run/mz:spectrumList/mz:spectrum", MZML_NS)
    for spectrum in spectra:
        pos = get_position_from_spectrum(spectrum, scan_number=scan_number)
        mz_array = get_binary_data_from_spectrum(
            spectrum, ref_id_mz, dtype_mz, external_binary
        )
        signal_array = get_binary_data_from_spectrum(
            spectrum, ref_id_signal, dtype_signal, external_binary
        )
        idx = np.searchsorted(mz_array, target_windows.flat)
        data[pos[0] - 1, pos[1] - 1] = np.add.reduceat(signal_array, idx)[::2]

    return data, {"spotsize": pixel_size}
#
#
# import time
#
# import matplotlib.pyplot as plt
#
# targets = np.array([792.555445])
# xml = parse_imzml("/home/tom/Downloads/slide 8 at 19%-total ion count.imzML")
# external_binary = "/home/tom/Downloads/slide 8 at 19%-total ion count.ibd"
#
# data, _ = load(
#     xml,
#     target_masses=targets,
#     external_binary=external_binary,
# )
# data = np.rot90(data, 1)
#
# fig, axes = plt.subplots(2, 1)
#
# axes[0].imshow(data)
#
# ref_id_mz = get_reference_id(xml, "MZ_ARRAY")
# dtype_mz = get_data_type_for_reference(xml, ref_id_mz)
# ref_id_signal = get_reference_id(xml, "INTENSITY_ARRAY")
# dtype_signal = get_data_type_for_reference(xml, ref_id_signal)
#
# t0 = time.time()
# spec = [Spectrum(s) for s in xml.findall("mz:run/mz:spectrumList/mz:spectrum", MZML_NS)]
# t1 = time.time()
# print(t1 - t0)
#
#
# def onclick(event):
#     x = event.xdata
#     y = event.ydata
#
#     t0 = time.time()
#     spectrum = get_spectrum_at_position(spec, (int(x), int(y)))
#     t1 = time.time()
#     mz = get_binary_data_from_spectrum(spectrum, ref_id_mz, dtype_mz, external_binary)
#     t2 = time.time()
#     sig = get_binary_data_from_spectrum(
#         spectrum, ref_id_signal, dtype_signal, external_binary
#     )
#     t3 = time.time()
#     print("time to find spec", t1 - t0)
#     print("time to load mz", t2 - t1)
#     print("time to load sig", t3 - t2)
#     axes[1].clear()
#     axes[1].plot(mz, sig)
#     fig.canvas.draw_idle()
#
#
# fig.canvas.mpl_connect("button_press_event", onclick)
# plt.show()
