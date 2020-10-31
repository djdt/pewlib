import os
import logging
from xml.etree import ElementTree

import numpy as np
import numpy.lib.recfunctions

from pew.io.error import PewException

from typing import Callable, Dict, Generator, List, Tuple

logger = logging.getLogger(__name__)

acq_method_xml_path = os.path.join("Method", "AcqMethod.xml")
batch_csv_path = "BatchLog.csv"
batch_xml_path = os.path.join("Method", "BatchLog.xml")

# msprofile_path = os.path.join("AcqData", "MSProfile.bin")
# msscan_path = os.path.join("AcqData", "MSScan.bin")
# msts_xspecific_path = os.path.join("AcqData", "MSTS_XSpecific.xml")
# msts_xaddition_path = "MSTS_XAddition.xml"


class XSpecificMass(object):
    """Class for MSTS mass information."""

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


# Datafile collection


def acq_method_xml_read_datafiles(batch_root: str, acq_xml: str) -> List[str]:
    xml = ElementTree.parse(acq_xml)
    ns = {"ns": xml.getroot().tag.split("}")[0][1:]}
    samples = xml.findall("ns:SampleParameter", ns)
    samples = sorted(
        samples, key=lambda s: int(s.findtext("ns:SampleID", namespaces=ns) or -1)
    )

    data_files = []
    for sample in samples:
        data_file = sample.findtext("ns:DataFileName", namespaces=ns)
        if data_file is not None:
            data_files.append(os.path.join(batch_root, data_file))
    return data_files


def batch_csv_read_datafiles(batch_root: str, batch_csv: str) -> List[str]:
    batch_log = np.genfromtxt(
        batch_csv,
        delimiter=",",
        comments=None,
        names=True,
        usecols=(0, 5, 6),
        dtype=[np.uint32, "U264", "U4"],
    )
    if batch_log.size == 1:  # Ensure iterable even if one line
        batch_log = batch_log.reshape(1)  # pragma: no cover
    data_files = []
    for _id, data_file, result in batch_log:
        if result == "Pass":
            data_files.append(
                os.path.join(
                    batch_root, data_file[max(map(data_file.rfind, "\\/")) + 1 :]
                )
            )
    return data_files


def batch_xml_read_datafiles(batch_root: str, batch_xml: str) -> List[str]:
    xml = ElementTree.parse(batch_xml)
    ns = {"ns": xml.getroot().tag.split("}")[0][1:]}

    data_files = []
    for log in xml.findall("ns:BatchLogInfo", ns):
        if log.findtext("ns:AcqResult", namespaces=ns) == "Pass":
            data_file = log.findtext("ns:DataFileName", namespaces=ns)
            data_files.append(
                os.path.join(
                    batch_root, data_file[max(map(data_file.rfind, "\\/")) + 1 :]
                )
            )
    return data_files


def collect_datafiles(batch_root: str, methods: List[str]) -> List[str]:
    for method in methods:
        if method == "batch_xml":
            method_path = os.path.join(batch_root, batch_xml_path)
            method_func: Callable[[str, str], List[str]] = batch_xml_read_datafiles
        elif method == "batch_csv":
            method_path = os.path.join(batch_root, batch_csv_path)
            method_func = batch_csv_read_datafiles
        elif method == "acq_method_xml":
            method_path = os.path.join(batch_root, acq_method_xml_path)
            method_func = acq_method_xml_read_datafiles

        if os.path.exists(method_path):
            data_files = method_func(batch_root, method_path)
            missing = len(data_files) - sum([os.path.exists(df) for df in data_files])
            if missing == 0:
                logger.info(f"Datafiles collected using '{method}'.")
                return data_files
            else:  # pragma: no cover
                logger.info(f"Missing {missing} datafiles using '{method}'.")
        else:
            logger.warning(f"Unable to collect datafiles using '{method}'.")

    # Fall back to alphabetical
    logger.info("Falling back to alphabetical order for datafile collection.")
    data_files = find_datafiles_alphabetical(batch_root)
    data_files.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    return data_files


def find_datafiles_alphabetical(path: str) -> List[str]:
    data_files = []
    with os.scandir(path) as it:
        for entry in it:
            if entry.name.lower().endswith(".d") and entry.is_dir():
                data_files.append(os.path.join(path, entry.name))
    return data_files


# Binary Import


def binary_read_datafile(datafile: str, masses: List[XSpecificMass]) -> np.ndarray:
    msscan = binary_read_msscan(os.path.join(datafile, "AcqData", "MSScan.bin"))
    msprofile = binary_read_msprofile(
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


def binary_read_msscan(path: str):
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


def binary_read_msscan_xspecific(path: str) -> np.ndarray:
    msscan_xspecific_magic_number = 275
    msscan_xspecific_header_size = 68
    msscan_xspecific_dtype = np.dtype([("_", np.int32), ("MZ", np.float64)])

    with open(path, "rb") as fp:
        if int.from_bytes(fp.read(4), "little") != msscan_xspecific_magic_number:
            raise IOError("Invalid header for MSScan.")
        fp.seek(msscan_xspecific_header_size)
        return np.frombuffer(fp.read(), dtype=msscan_xspecific_dtype)


def binary_read_msprofile(path: str, n: int) -> np.ndarray:
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


def msts_xspecific_xml_read_info(
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


def msts_xaddition_xml_read_info(
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


def msts_xml_mass_info(datafile: str) -> List[XSpecificMass]:
    msts_xspecific_path = os.path.join(datafile, "AcqData", "MSTS_XSpecific.xml")
    msts_xaddition_path = os.path.join(datafile, "MSTS_XAddition.xml")

    if os.path.exists(msts_xspecific_path):
        xspecific = msts_xspecific_xml_read_info(msts_xspecific_path)
    else:
        raise FileNotFoundError("MSTS_XSpecific.xml not found.")

    masses = {k: XSpecificMass(k, v[0], v[1], k) for k, v in xspecific.items()}

    if os.path.exists(msts_xaddition_path):
        xaddition, scan_type = msts_xaddition_xml_read_info(msts_xaddition_path)
        if scan_type == "MS_MS":
            for k, v in xaddition.items():
                masses[k].mz = v[0]
                masses[k].mz2 = v[1]

    return sorted(masses.values(), key=lambda x: x.id)


def load_binary(
    batch: str,
    collection_methods: List[str] = None,
    counts_per_second: bool = False,
    full: bool = False,
) -> np.ndarray:
    """Imports an Agilent batch (.b) directory, using the MSScan and MSProfile binaries.
    Finds lines using (in order of preference): BatchLog.xml, BatchLog.csv,
     AcqMethod.xml, .d files sorted by name.

    Args:
       path: Path to the .b directory.
       collection_methods: List of data file collection methods to try.
       counts_per_second: Return data in CPS instead of raw counts.
       full: If True then also return a dict of available params.

    Returns:
        The structured numpy array and optionally, params.

    Raises:
        PewException

    """
    if collection_methods is None:
        collection_methods = ["batch_xml", "batch_csv"]

    datafiles = collect_datafiles(batch, collection_methods)
    if len(datafiles) == 0:
        raise PewException(f"No data files found in {batch}!")  # pragma: no cover

    masses = msts_xml_mass_info(datafiles[0])
    data = np.stack([binary_read_datafile(df, masses) for df in datafiles], axis=0)

    scantime = np.round(np.mean(np.diff(data["Time"], axis=1)), 4)
    data = numpy.lib.recfunctions.drop_fields(data, "Time")

    if counts_per_second:
        for mass in masses:
            data[str(mass)] /= mass.acctime

    if full:
        return data, dict(scantime=scantime)
    else:
        return data


# CSV Import


def acq_method_xml_read_elements(acq_xml: str) -> List[str]:
    xml = ElementTree.parse(acq_xml)
    ns = {"ns": xml.getroot().tag.split("}")[0][1:]}

    msms = False
    for tune in xml.findall("ns:TuneStep", namespaces=ns):
        if tune.findtext("ns:ScanType_Acq", namespaces=ns) == "MS_MS":
            msms = True
            break

    elements: List[Tuple[str, int, int]] = []
    for element in xml.findall("ns:IcpmsElement", namespaces=ns):
        name = element.findtext("ns:ElementName", namespaces=ns)
        if name is None:  # pragma: no cover
            continue
        mz = int(element.findtext("ns:MZ", namespaces=ns) or -1)
        mz2 = int(element.findtext("ns:SelectedMZ", namespaces=ns) or -1)
        elements.append((name, mz, mz2))

    elements = sorted(elements, key=lambda e: (e[1], e[2]))
    names = []
    for e in elements:
        names.append(f"{e[0]}{e[2]}->{e[1]}" if msms else f"{e[0]}{e[1]}")
    return names


def csv_read_params(path: str) -> Tuple[List[str], float, int]:
    data = np.genfromtxt(
        csv_valid_lines(path), delimiter=b",", names=True, dtype=np.float64
    )
    total_time = np.max(data["Time_Sec"])
    names = [name for name in data.dtype.names if name != "Time_Sec"]
    return names, np.round(total_time / data.shape[0], 4), data.shape[0]


def csv_valid_lines(csv: str) -> Generator[bytes, None, None]:
    delimiter_count = 0
    past_header = False
    with open(csv, "rb") as fp:
        for line in fp:
            if past_header and line.count(b",") == delimiter_count:
                yield line
            if line.startswith(b"Time"):
                past_header = True
                delimiter_count = line.count(b",")
                yield line


def load_csv(
    path: str,
    collection_methods: List[str] = None,
    use_acq_for_names: bool = True,
    full: bool = False,
) -> np.ndarray:
    """Imports an Agilent batch (.b) directory, returning structured array.
    Finds lines using (in order of preference): BatchLog.xml, BatchLog.csv,
     AcqMethod.xml, .d files sorted by name.

    Args:
       path: Path to the .b directory.
       collection_methods: List of data file collection methods to try.
       full: If True then also return a dict of available params.

    Returns:
        The structured numpy array and optionally, params.

    Raises:
        PewException

    """
    if collection_methods is None:
        collection_methods = ["batch_xml", "batch_csv"]

    # Collect data files
    data_files = collect_datafiles(path, collection_methods)
    if len(data_files) == 0:
        raise PewException(f"No data files found in {path}!")  # pragma: no cover

    # Collect csvs
    csvs: List[str] = []
    for d in data_files:
        csv = os.path.join(d, os.path.splitext(os.path.basename(d))[0] + ".csv")
        logger.debug(f"Looking for csv '{csv}'.")
        if not os.path.exists(csv):
            logger.warning(f"Missing csv '{csv}', line blanked.")
            csvs.append(None)
        else:
            csvs.append(csv)

    names, scan_time, nscans = csv_read_params(next(c for c in csvs if c is not None))
    if use_acq_for_names:
        if os.path.exists(os.path.join(path, acq_method_xml_path)):
            names = acq_method_xml_read_elements(
                os.path.join(path, acq_method_xml_path)
            )
        else:  # pragma: no cover
            logger.warning("AcqMethod.xml not found, cannot read names.")

    data = np.empty(
        (len(data_files), nscans), dtype=[(name, np.float64) for name in names]
    )
    for i, csv in enumerate(csvs):
        if csv is None:
            data[i, :] = np.zeros(data.shape[1], dtype=data.dtype)
        else:
            try:
                data[i, :] = np.genfromtxt(
                    csv_valid_lines(csv),
                    delimiter=b",",
                    names=True,
                    usecols=np.arange(1, len(names) + 1),
                    dtype=np.float64,
                )
            except ValueError:
                logger.warning(f"'{csv}' row {i} missing, line blanked.")
                data[i, :] = np.zeros(data.shape[1], dtype=data.dtype)

    if full:
        return data, dict(scantime=scan_time)
    else:
        return data


def load(
    path: str,
    collection_methods: List[str] = None,
    use_acq_for_names: bool = True,
    counts_per_second: bool = False,
    full: bool = False,
) -> np.ndarray:
    """Attempts to load using the binary method, falling back to csv import.
    See also 'load_binary', 'load_csv'.

    Args:
        path: Path to batch folder.
        collection_methods: Methods to use for datafile collection.
        use_acq_for_names: Read aquistion method to find element names (CSV only).
        counts_per_second: Import as CPS (Binary only).
        full: Return data and a dict of params.

    Returns:
        The structured numpy array and optionally, params.

    Raises:
        PewException
    """
    try:
        result = load_binary(
            path, collection_methods, counts_per_second=counts_per_second, full=full
        )
    except Exception as e:
        logger.info("Unable to import as binary, reverting to CSV import.")
        logger.exception(e)
        result = load_csv(
            path, collection_methods, use_acq_for_names=use_acq_for_names, full=full
        )
    return result
