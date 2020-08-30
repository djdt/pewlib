import numpy as np
from xml.etree import ElementTree
import os.path

from pew.io import agilent

from typing import Dict, Tuple


scan_record_dtype = np.dtype(
    [
        ("ScanID", np.int32),
        ("ScanMethodID", np.int32),
        ("TimeSegmentID", np.int32),
        ("ScanTime", np.float64),
        ("MSLevel", np.int32),
        ("ScanType", np.int32),
        ("TIC", np.float64),
        ("BasePeakMZ", np.float64),
        ("BasePeakValue", np.float64),
        ("Status", np.int32),
        ("IonMode", np.int32),
        ("IonPolarity", np.int32),
        ("SamplingPeriod", np.float64),
        (
            "SpectrumParamValues",
            np.dtype(
                [
                    ("SpectrumFormatID", np.int32),
                    ("SpectrumOffset", np.int64),
                    ("ByteCount", np.int32),
                    ("PointCount", np.int32),
                    ("MinX", np.float64),
                    ("MaxX", np.float64),
                    ("MinY", np.float64),
                    ("MaxY", np.float64),
                ]
            ),
        ),
        (
            "XSpecificParamType",
            np.dtype([("Offset", np.int64), ("ByteCount", np.int32)]),
        ),
    ]
)

msprofile_record_dtype = np.dtype(
    [
        ("ID", np.float32),
        ("Analog", np.float64),
        ("Analog2", np.float64),
        ("Digital", np.float64),
    ]
)


def get_ms_profile_dtype(n: int):
    # sub_dtype = np.dtype(
    #     {
    #         "names": ["ID", "Analog", "_Analog", "Digital"],
    #         "formats": [np.float32, np.float64, np.float64, np.float64],
    #         "offsets": [0, 4 * n, 12 * n, 20 * n, 28 * n],
    #         # "itemsize": 36 * n,
    #     }
    # )
    # return np.dtype(
    #     {
    #         "names": [str(i) for i in np.arange(n)],
    #         "formats": [sub_dtype for i in np.arange(n)],
    #         "offsets": np.arange(0, 4 * n, 4),
    #         "itemsize": 28 * n * n,
    #     }
    # )
    # return np.dtype({}sub_dtype, n)
    # return sub_dtype
    return np.dtype(
        [
            ("ID", np.float32, n),
            ("Analog", np.float64, n),
            ("Analog2", np.float64, n),
            ("Digital", np.float64, n),
        ]
    )


def read_ms_scan_xspecific(path: str) -> np.ndarray:
    msscan_xspecific_magic_number = 275
    msscan_xspecific_header_size = 68

    with open(path, "rb") as fp:
        if int.from_bytes(fp.read(4), "little") != msscan_xspecific_magic_number:
            raise IOError("Invalid header for MSScan.")
        fp.seek(msscan_xspecific_header_size)
        return np.frombuffer(fp.read(), dtype=[(("_", np.int32), ("MZ", np.float64))])


def read_ms_scan(path: str):
    msscan_magic_number = 257
    msscan_header_size = 68

    with open(path, "rb") as fp:
        if int.from_bytes(fp.read(4), "little") != msscan_magic_number:
            raise IOError("Invalid header for MSScan.")
        fp.seek(msscan_header_size + 20)
        offset = int.from_bytes(fp.read(4), "little")
        fp.seek(offset)
        return np.frombuffer(fp.read(), dtype=scan_record_dtype)


def read_ms_profile(path: str, n: int) -> np.ndarray:
    msprofile_magic_number = 258
    msprofile_header_size = 68
    msprofile_flat_dtype = np.dtype(
        [
            ("ID", np.float32),
            ("Analog", np.float64),
            ("Analog2", np.float64),
            ("Digital", np.float64),
        ]
    )

    with open(path, "rb") as fp:
        if int.from_bytes(fp.read(4), "little") != msprofile_magic_number:
            raise IOError("Invalid header for MSProfile.")
        fp.seek(msprofile_header_size)
        data = np.frombuffer(fp.read(), dtype=get_ms_profile_dtype(n))

    flattened = np.empty(np.prod(data.size * n), dtype=msprofile_flat_dtype)
    for name in data.dtype.names:
        flattened[name] = data[name].flat
    return flattened


def parse_msts_xml(msts_xml: str) -> Tuple[float, float, int]:
    xml = ElementTree.parse(msts_xml)
    time_segment = xml.find("TimeSegment")
    start = float(time_segment.findtext("StartTime") or 0.0)
    end = float(time_segment.findtext("EndTime") or 0.0)
    scans = int(time_segment.findtext("NumOfScans") or 0)
    return start, end, scans


def parse_msts_xspecific_xml(msts_xspecific_xml: str,) -> Dict[int, Tuple[str, float]]:
    xml = ElementTree.parse(msts_xspecific_xml)
    xdict = {}
    for record in xml.iter("IonRecord"):
        for masses in record.iter("Masses"):
            mass = int(masses.findtext("Mass") or 0)
            name = masses.findtext("Name") or ""
            acctime = float(masses.findtext("AccumulationTime") or 0.0)
            xdict[mass] = (name, acctime)
    return xdict


def parse_msts_xaddition_xml(
    msts_xaddition_xml: str,
) -> Tuple[Dict[int, Tuple[int, int]], str]:
    xml = ElementTree.parse(msts_xaddition_xml)
    xdict = {}
    for xaddition in xml.iter("MSTS_XAddition"):
        scan_type = xaddition.findtext("ScanType")
        indexed_masses = xaddition.find("IndexedMasses")
        for msts_index in indexed_masses.iter("MSTS_XAddition_IndexedMasses"):
            index = int(msts_index.findtext("Index") or 0)
            precursor = int(msts_index.findtext("PrecursorIonMZ") or 0)
            product = int(msts_index.findtext("ProductIonMZ") or 0)
            xdict[index] = (precursor, product)
    return xdict, scan_type


class XSpecificMass(object):
    def __init__(self, id: int, name: str, acctime: float, mz: int, mz2: int = None):
        self.id = id
        self.name = name
        self.acctime = acctime
        self.mz = mz
        self.mz2 = mz2

    def __str__(self) -> str:
        if self.mz2 is None:
            return f"{self.name}{self.mz}"
        else:
            print(self.mz2)
            return f"{self.name}{self.mz}->{self.mz2}"


def datafile_msts_mass_info(datafile: str) -> Dict[int, XSpecificMass]:
    msts_xspecific_path = os.path.join(datafile, "AcqData", "MSTS_XSpecific.xml")
    msts_xaddition_path = os.path.join(datafile, "MSTS_XAddition.xml")

    if os.path.exists(msts_xspecific_path):
        xspecific = parse_msts_xspecific_xml(msts_xspecific_path)
    else:
        raise FileNotFoundError("MSTS_XSpecific.xml not found.")

    masses = {k: XSpecificMass(k, v[0], v[1], k) for k, v in xspecific.items()}

    if os.path.exists(msts_xaddition_path):
        xaddition, scan_type = parse_msts_xaddition_xml(msts_xaddition_path)
        if scan_type == "MS_MS":
            for k, v in xaddition.items():
                masses[k].mz = v[0]
                masses[k].mz2 = v[1]

    return masses


def load(batch: str) -> np.ndarray:
    import matplotlib.pyplot as plt

    data_files = agilent.collect_datafiles(batch, ["batch_xml", "batch_csv"])
    masses = datafile_msts_mass_info(data_files[0])
    ids = np.array([k for k in masses.keys()], dtype=int)
    dtype = [(str(masses[k]), np.float64) for k in masses.keys()]

    # msprofile_data = np.stack(
    #     [
    #         read_ms_profile(os.path.join(df, "AcqData", "MSProfile.bin"), len(masses))
    #         for df in data_files
    #     ],
    #     axis=0,
    # )
    # idx = np.argmax(msprofile_data["ID"] == ids[:, None, None], axis=0)

    # data = msprofile_data[""]

    # plt.imshow(data["C1"])
    # plt.show()


# load("/home/tom/Downloads/20200630_agar_test_1.b")
profile = read_ms_profile(
    "/home/tom/Downloads/20200630_agar_test_1.b/001.d/AcqData/MSProfile.bin", 4
)
print(profile[:3], profile.dtype)
# # idx = np.where(profile["ID"][:3] == np.array([4, 2, 3, 1])[:, None])
# # idx = np.argmax(profile["ID"][:3] == np.array([4, 2, 3, 1])[:, None, None], axis=0)
# idx = np.argmax(profile["ID"][:3] == np.array([4, 2, 3, 1])[:, None, None], axis=0)
# print(np.take_along_axis(profile["Digital"][:3], idx, 1), profile.dtype)
