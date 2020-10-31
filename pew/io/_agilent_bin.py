import numpy as np
import numpy.lib.recfunctions as rfn
from xml.etree import ElementTree
import os.path

from pew.io import agilent, error

from typing import Dict, List, Tuple


def read_msscan_xspecific(path: str) -> np.ndarray:
    msscan_xspecific_magic_number = 275
    msscan_xspecific_header_size = 68
    msscan_xspecific_dtype = np.dtype([("_", np.int32), ("MZ", np.float64)])

    with open(path, "rb") as fp:
        if int.from_bytes(fp.read(4), "little") != msscan_xspecific_magic_number:
            raise IOError("Invalid header for MSScan.")
        fp.seek(msscan_xspecific_header_size)
        return np.frombuffer(fp.read(), dtype=msscan_xspecific_dtype)


def read_msscan(path: str):
    msscan_magic_number = 257
    msscan_header_size = 68
    msscan_dtype = np.dtype(
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

    with open(path, "rb") as fp:
        if int.from_bytes(fp.read(4), "little") != msscan_magic_number:
            raise IOError("Invalid header for MSScan.")
        fp.seek(msscan_header_size + 20)
        offset = int.from_bytes(fp.read(4), "little")
        fp.seek(offset)
        return np.frombuffer(fp.read(), dtype=msscan_dtype)


def read_msprofile(path: str, n: int) -> np.ndarray:
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

    def get_msprofile_dtype(n: int):
        return np.dtype(
            [
                ("ID", np.float32, n),
                ("Analog", np.float64, n),
                ("Analog2", np.float64, n),
                ("Digital", np.float64, n),
            ]
        )

    with open(path, "rb") as fp:
        if int.from_bytes(fp.read(4), "little") != msprofile_magic_number:
            raise IOError("Invalid header for MSProfile.")
        fp.seek(msprofile_header_size)
        data = np.frombuffer(fp.read(), dtype=get_msprofile_dtype(n))

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


def parse_msts_xspecific_xml(
    msts_xspecific_xml: str,
) -> Dict[int, Tuple[str, float]]:
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
            return f"{self.name}{self.mz}->{self.mz2}"


def datafile_msts_mass_info(datafile: str) -> List[XSpecificMass]:
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

    return sorted(masses.values(), key=lambda x: x.id)


def read_datafile(datafile: str, masses: List[XSpecificMass]) -> np.ndarray:
    msscan = read_msscan(os.path.join(datafile, "AcqData", "MSScan.bin"))
    msprofile = read_msprofile(
        os.path.join(datafile, "AcqData", "MSProfile.bin"), len(masses)
    )
    offsets = (
        msscan["SpectrumParamValues"]["SpectrumOffset"]
        // msscan["SpectrumParamValues"]["ByteCount"]
    )
    dtype = [(str(mass), np.float64) for mass in masses] + [("Time", np.float64)]
    data = np.empty(offsets.size, dtype=dtype)
    for mass in masses:
        data[str(mass)] = msprofile[(offsets * len(masses)) + (mass.id - 1)]["Analog"]

    data["Time"] = msscan["ScanTime"] * 60.0  # ScanTime in minutes
    return data


def load(
    batch: str,
    collection_methods: List[str] = None,
    counts_per_second: bool = False,
    full: bool = False,
) -> np.ndarray:
    if collection_methods is None:
        collection_methods = ["batch_xml", "batch_csv"]

    datafiles = agilent.collect_datafiles(batch, collection_methods)
    if len(datafiles) == 0:
        raise error.PewException(f"No data files found in {batch}!")  # pragma: no cover

    masses = datafile_msts_mass_info(datafiles[0])
    data = np.stack([read_datafile(df, masses) for df in datafiles], axis=0)

    scantime = np.round(np.mean(np.diff(data["Time"], axis=1)), 4)
    data = rfn.drop_fields(data, "Time")

    if counts_per_second:
        for mass in masses:
            data[str(mass)] /= mass.acctime

    if full:
        return data, dict(scantime=scantime)
    else:
        return data
