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
    "MZ_ARRAY": "MS:1000514",
    "INTENSITY_ARRAY": "MS:1000515",
}
CV_SPECTRUM = {
    "POSITION_X": "IMS:1000050",
    "POSITION_Y": "IMS:1000051",
    "TOTAL_ION_CURRENT": "MS:1000285",
    "EXTERNAL_OFFSET": "IMS:1000102",
    "EXTERNAL_ARRAY_LENGTH": "IMS:1000103",
}
CV = {**CV_SCANSETTINGS, **CV_BINARYDATA, **CV_SPECTRUM}


def load(
    path: Path | str,
    external_binary: Path | str | None = None,
    target_masses: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    xml = ElementTree.parse(path)
    ns = {"ns": xml.getroot().tag.split("}")[0][1:]}

    if isinstance(external_binary, str):
        external_binary = Path(external_binary)

    mz_group = xml.find(
        "ns:referencableParamGroupList/ns:referenceableParamGroup/"
        f"ns:cvParam[@accession='{CV_PARAMGROUP['MZ_ARRAY']}'/..",
        ns,
    )
    if mz_group is None:
        raise ValueError("unable to find mz array")
    mz_id = mz_group.get("id", "")
    signal_group = xml.find(
        "ns:referencableParamGroupList/ns:referenceableParamGroup/"
        f"ns:cvParam[@accession='{CV_PARAMGROUP['INTENSITY_ARRAY']}'/..",
        ns,
    )
    if signal_group is None:
        raise ValueError("unable to find intensity array")
    signal_id = signal_group.get("id", "")

    scan_settings_list = xml.findall("ns:scanSettingsList/ns:scanSettings", ns)
    if len(scan_settings_list) == 0:
        raise ValueError("unable to find scan settings")
    elif len(scan_settings_list) > 1:
        raise ValueError("more than one scan settings found")
    scan_settings = scan_settings_list[0]


class IMZMLImporter(object):
    def __init__(self, imzML: Path | str, external_binary: Path | str | None = None):
        self.xml = ElementTree.parse(imzML)
        self.ns = {"ns": self.xml.getroot().tag.split("}")[0][1:]}

        if isinstance(external_binary, str):
            external_binary = Path(external_binary)
        self.external_binary = external_binary

        param_elements = self.xml.findall(
            "ns:referenceableParamGroupList/ns:referenceableParamGroup", self.ns
        )
        if len(param_elements) == 0:
            raise ValueError(f"invalid imzML '{imzML}'")
        param_groups = [ParamGroup(e, self.ns) for e in param_elements]
        self.param_groups = {g.id: g for g in param_groups}

        scan_settings = self.xml.find("ns:scanSettingsList/ns:scanSettings", self.ns)
        if scan_settings is None:
            raise ValueError(f"invalid imzML '{imzML}'")
        self.scan_settings = ScanSettings(scan_settings, self.ns)

        spectra = self.xml.findall("ns:run/ns:spectrumList/ns:spectrum", self.ns)
        if len(spectra) == 0:
            raise ValueError(f"invalid imzML '{imzML}'")
        self.spectra = [Spectrum(s, self.ns) for s in spectra]

    def extract_mz(
        self,
        mz: float | np.ndarray,
        width_ppm: float = 10.0,
        mz_ref: str = "mzArray",
        signal_ref: str = "intensities",
    ) -> np.ndarray:
        groups = self.param_groups
        if isinstance(mz, float):
            mz = np.array([mz])
        dmz = mz * width_ppm / 1e6

        if self.external_binary is None:
            raise NotImplementedError
        else:
            mz_dtype = groups[mz_ref].dtype
            signal_dtype = groups[signal_ref].dtype
            data: np.ndarray = np.zeros(
                (self.scan_settings.size_x, self.scan_settings.size_y, mz.size),
                dtype=signal_dtype,
            )

            for spectrum in self.spectra:
                mzs = spectrum.binary_data(mz_ref, self.external_binary, dtype=mz_dtype)
                signals = spectrum.binary_data(
                    signal_ref, self.external_binary, dtype=signal_dtype
                ).ravel()
                idx = np.searchsorted(mzs, np.stack((mz - dmz, mz + dmz), axis=1).flat)

                data[spectrum.pos_x - 1, spectrum.pos_y - 1] = np.add.reduceat(
                    signals, idx
                )[::2]
        return data
