"""
Import of line-by-line collected Agilent '.b' batches.
Both raw binaries and the '.csv' exports are supported.
Tested with Agilent 7500, 7700 and 8900 ICPs.
"""
import logging
from xml.etree import ElementTree
from pathlib import Path

import numpy as np
import numpy.lib.recfunctions as rfn

from typing import Callable, Dict, Generator, List, Tuple, Union

logger = logging.getLogger(__name__)

acq_method_xml_path = Path("Method", "AcqMethod.xml")
batch_csv_path = Path("BatchLog.csv")
batch_xml_path = Path("Method", "BatchLog.xml")


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


# Datafile collection


def acq_method_xml_read_datafiles(path: Path, acq_xml: Path) -> List[Path]:
    xml = ElementTree.parse(acq_xml)
    ns = {"ns": xml.getroot().tag.split("}")[0][1:]}
    samples = xml.findall("ns:SampleParameter", ns)
    samples = sorted(
        samples, key=lambda s: int(s.findtext("ns:SampleID", namespaces=ns) or -1)
    )

    datafiles = []
    for sample in samples:
        datafile = sample.findtext("ns:DataFileName", namespaces=ns)
        if datafile is not None:
            datafiles.append(path.joinpath(datafile))
    return datafiles


def batch_csv_read_datafiles(path: Path, batch_csv: Path) -> List[Path]:
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
    datafiles = []
    for _id, datafile, result in batch_log:
        if result == "Pass":
            datafiles.append(
                path.joinpath(datafile[max(map(datafile.rfind, "\\/")) + 1 :])
            )
    return datafiles


def batch_xml_read_datafiles(path: Path, batch_xml: Path) -> List[Path]:
    xml = ElementTree.parse(batch_xml)
    ns = {"ns": xml.getroot().tag.split("}")[0][1:]}

    datafiles = []
    for log in xml.findall("ns:BatchLogInfo", ns):
        if log.findtext("ns:AcqResult", namespaces=ns) == "Pass":
            datafile = log.findtext("ns:DataFileName", namespaces=ns)
            if datafile is not None:
                datafiles.append(
                    path.joinpath(datafile[max(map(datafile.rfind, "\\/")) + 1 :])
                )

    return datafiles


def collect_datafiles(path: Union[str, Path], methods: List[str]) -> List[Path]:
    """Finds '.d' datafiles in a directory.

    A list of expected datafiles is created for each method in `methods`.
    Methods are tested in order until ones successfully finds ALL expected datafiles.

    Args:
        path: path to directory
        methods: list of methods to try,
            {'alphabetical', 'acq_method_xml', 'batch_csv', 'batch_xml'}
    Returns:
        A list of datafiles
    """

    if isinstance(path, str):  # pragma: no cover
        path = Path(path)

    for method in methods:
        if method == "batch_xml":
            method_path = path.joinpath(batch_xml_path)
            method_func: Callable[[Path, Path], List[Path]] = batch_xml_read_datafiles
        elif method == "batch_csv":
            method_path = path.joinpath(batch_csv_path)
            method_func = batch_csv_read_datafiles
        elif method == "acq_method_xml":
            method_path = path.joinpath(acq_method_xml_path)
            method_func = acq_method_xml_read_datafiles
        elif method == "alphabetical":
            return find_datafiles_alphabetical(path)

        if method_path.exists():
            datafiles = method_func(path, method_path)
            missing = len(datafiles) - sum([df.exists() for df in datafiles])
            if missing == 0:
                logger.info(f"Datafiles collected using '{method}'.")
                return datafiles
            else:  # pragma: no cover
                logger.info(f"Missing {missing} datafiles using '{method}'.")
        else:  # pragma: no cover
            logger.warning(f"Unable to collect datafiles using '{method}'.")

    logger.error(f"All datafile collection methods '{methods}' failed.")
    raise ValueError(f"All datafile collection methods '{methods}' failed.")


def find_datafiles_alphabetical(path: Union[str, Path]) -> List[Path]:
    if isinstance(path, str):  # pragma: no cover
        path = Path(path)

    datafiles = []
    for entry in path.iterdir():
        if entry.suffix.lower().endswith(".d") and entry.is_dir():
            datafiles.append(entry)

    datafiles.sort(key=lambda f: int("".join(filter(str.isdigit, f.name))))
    return datafiles


# Binary Import


def binary_read_datafile(path: Path, masses: List[XSpecificMass]) -> np.ndarray:
    msscan = binary_read_msscan(path.joinpath("AcqData", "MSScan.bin"))
    msprofile = binary_read_msprofile(
        path.joinpath("AcqData", "MSProfile.bin"), len(masses)
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


def binary_read_msscan(path: Path):
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

    with path.open("rb") as fp:
        if (
            int.from_bytes(fp.read(4), "little") != msscan_magic_number
        ):  # pragma: no cover
            raise IOError("Invalid header for MSScan.")
        fp.seek(msscan_header_size + 20)
        offset = int.from_bytes(fp.read(4), "little")
        fp.seek(offset)
        return np.frombuffer(fp.read(), dtype=msscan_dtype)


def binary_read_msscan_xspecific(path: Path) -> np.ndarray:  # pragma: no cover
    msscan_xspecific_magic_number = 275
    msscan_xspecific_header_size = 68
    msscan_xspecific_dtype = np.dtype([("_", np.int32), ("MZ", np.float64)])

    with path.open("rb") as fp:
        if int.from_bytes(fp.read(4), "little") != msscan_xspecific_magic_number:
            raise IOError("Invalid header for MSScan.")
        fp.seek(msscan_xspecific_header_size)
        return np.frombuffer(fp.read(), dtype=msscan_xspecific_dtype)


def binary_read_msprofile(path: Path, n: int) -> np.ndarray:
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

    with path.open("rb") as fp:
        if (
            int.from_bytes(fp.read(4), "little") != msprofile_magic_number
        ):  # pragma: no cover
            raise IOError("Invalid header for MSProfile.")
        fp.seek(msprofile_header_size)
        data = np.frombuffer(fp.read(), dtype=get_msprofile_dtype(n))

    flattened = np.empty(np.prod(data.size * n), dtype=msprofile_flat_dtype)
    for name in data.dtype.names:
        flattened[name] = data[name].flat
    return flattened


def mass_info_datafile(path: Path) -> List[XSpecificMass]:
    msts_xspecific_path = path.joinpath("AcqData", "MSTS_XSpecific.xml")
    msts_xaddition_path = path.joinpath("MSTS_XAddition.xml")

    if msts_xspecific_path.exists():
        xspecific = msts_xspecific_xml_read_info(msts_xspecific_path)
    else:  # pragma: no cover
        raise FileNotFoundError("MSTS_XSpecific.xml not found.")

    masses = {
        idx: XSpecificMass(idx, name=name, acctime=acctime, mz=mz)
        for idx, (name, mz, acctime) in xspecific.items()
    }

    if msts_xaddition_path.exists():
        xaddition, scan_type = msts_xaddition_xml_read_info(msts_xaddition_path)
        for idx, (mz, mz2) in xaddition.items():
            masses[idx].mz = mz
            if scan_type == "MS_MS":
                masses[idx].mz2 = mz2

    return sorted(masses.values(), key=lambda x: x.id)


def msts_xspecific_xml_read_info(path: Path) -> Dict[int, Tuple[str, int, float]]:
    xml = ElementTree.parse(path)
    idx = 1
    xdict = {}
    for record in xml.iter("IonRecord"):
        for masses in record.iter("Masses"):
            mass = int(masses.findtext("Mass") or 0)
            name = masses.findtext("Name") or ""
            acctime = float(masses.findtext("AccumulationTime") or 0.0)
            xdict[idx] = (name, mass, acctime)
            idx += 1
    return xdict


def msts_xaddition_xml_read_info(path: Path) -> Tuple[Dict[int, Tuple[int, int]], str]:
    xml = ElementTree.parse(path)
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


def load_binary(
    path: Union[str, Path],
    collection_methods: List[str] = None,
    counts_per_second: bool = False,
    drop_names: List[str] = None,
    full: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """Imports an Agilent '.b' batch.

    Import is performed using the 'MSScan.bin', 'MSProfile.bin' binaries and
    'MSTS_XSpecific.xml' document.
    By default `drop_names` drops the 'Time' field.

    Args:
        path: path to batch
        collection_methods: list of datafile collection methods,
            default = ['batch_xml', 'batch_csv']
        counts_per_second: return data in CPS
        drop_names: names to remove from final array
        full: also return dict with scantime

    Returns:
        structured array of data
        dict of params if `full`

    Raises:
        FileNotFoundError: 'MSScan.bin', 'MSProfile.bin' or 'MSTS_XSpecific.xml'
            not found
        IOError: invalid binary format

    See Also:
        :func:`pewlib.io.agilent.collect_datafiles`
    """

    if isinstance(path, str):  # pragma: no cover
        path = Path(path)

    if drop_names is None:
        drop_names = ["Time"]

    if collection_methods is None:
        collection_methods = ["batch_xml", "batch_csv"]

    datafiles = collect_datafiles(path, collection_methods)
    if len(datafiles) == 0:  # pragma: no cover
        logger.info("Falling back to alphabetical order for datafile collection.")
        datafiles = find_datafiles_alphabetical(path)
        if len(datafiles) == 0:  # pragma: no cover
            raise FileNotFoundError(f"No data files found in {path.name}!")

    masses = mass_info_datafile(datafiles[0])
    data = np.stack([binary_read_datafile(df, masses) for df in datafiles], axis=0)

    params = {}
    if full:
        if "Time" in data.dtype.names:
            params["scantime"] = np.round(np.mean(np.diff(data["Time"], axis=1)), 4)
        else:  # pragma: no cover
            logger.warning("'Time' field not found, unable to import scantime.")

        # Read devices.xml

    if counts_per_second:
        for mass in masses:
            data[str(mass)] /= mass.acctime

    data = rfn.drop_fields(data, drop_names)

    if full:
        return data, params
    else:  # pragma: no cover
        return data


# CSV Import


def acq_method_xml_read_elements(path: Path) -> List[str]:
    xml = ElementTree.parse(path)
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


def csv_valid_lines(csv: Path) -> Generator[bytes, None, None]:
    delimiter_count = 0
    past_header = False
    with csv.open("rb") as fp:
        for line in fp:
            if past_header and line.count(b",") == delimiter_count:
                yield line
            elif line.startswith(b"Time"):
                past_header = True
                delimiter_count = line.count(b",")
                yield line


def read_datafile_csvs(datafiles: List[Path]) -> Generator[np.ndarray, None, None]:
    for df in datafiles:
        csv = df.joinpath(df.with_suffix(".csv").name)
        logger.debug(f"Looking for csv '{csv}'.")
        if not csv.exists():
            logger.warning(f"Missing csv '{csv}', line blanked.")
            yield None
        else:
            yield np.genfromtxt(
                csv_valid_lines(csv),
                delimiter=b",",
                names=True,
                dtype=np.float64,
                deletechars="",
            )


def load_csv(
    path: Union[str, Path],
    collection_methods: List[str] = None,
    use_acq_for_names: bool = True,
    drop_names: List[str] = None,
    full: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """Imports an Agilent '.b' batch.

    Import is performed using the '.csv' files found in each '.d' datafile.
    If a '.csv' can not be found then all data in the line is set to 0.
    To load properly formatted element names use `use_acq_for_names`.
    By default `drop_names` drops the 'Time_[Sec]' field.

    Args:
        path: path to batch
        collection_methods: list of datafile collection methods,
            default = ['batch_xml', 'batch_csv']
        use_acq_for_names: read element names from 'AcqMethod.xml'
        drop_names: names to remove from final array
        full: also return dict with scantime

    Returns:
        structured array of data
        dict of params if `full`

    See Also:
        :func:`pewlib.io.agilent.collect_datafiles`
    """

    if isinstance(path, str):  # pragma: no cover
        path = Path(path)

    if drop_names is None:
        drop_names = ["Time_[Sec]"]

    if collection_methods is None:
        collection_methods = ["batch_xml", "batch_csv"]

    # Collect data files
    datafiles = collect_datafiles(path, collection_methods)
    if len(datafiles) == 0:  # pragma: no cover
        logger.info("Falling back to alphabetical order for datafile collection.")
        datafiles = find_datafiles_alphabetical(path)
        if len(datafiles) == 0:  # pragma: no cover
            raise FileNotFoundError(f"No data files found in {path.name}!")

    lines = list(read_datafile_csvs(datafiles))

    data_shape = next(line for line in lines if line is not None).shape
    data_dtype = next(line for line in lines if line is not None).dtype

    data = np.empty((len(datafiles), data_shape[0]), dtype=data_dtype)

    for i, line in enumerate(lines):
        if line is None:
            data[i, :] = np.zeros(data_shape[0], dtype=data_dtype)
        else:
            data[i, :] = line

    if use_acq_for_names:
        if path.joinpath(acq_method_xml_path).exists():
            names = acq_method_xml_read_elements(path.joinpath(acq_method_xml_path))
            data = rfn.rename_fields(
                data, {old: new for old, new in zip(data.dtype.names[1:], names)}
            )
        else:  # pragma: no cover
            logger.warning("AcqMethod.xml not found, cannot read names.")

    params = {}
    if full:
        if "Time_[Sec]" in data.dtype.names:
            params["scantime"] = np.round(
                np.mean(np.diff(data["Time_[Sec]"], axis=1)), 4
            )
        else:  # pragma: no cover
            logger.warning("'Time_[Sec]' field not found, unable to import scantime.")

    data = rfn.drop_fields(data, drop_names)

    if full:
        return data, params
    else:  # pragma: no cover
        return data


def batch_xml_read_info(path: Path) -> Dict[str, str]:
    xml = ElementTree.parse(path)
    ns = {"ns": xml.getroot().tag.split("}")[0][1:]}
    name = xml.getroot().get("BatchName")
    path = xml.getroot().get("BatchDataPath")
    info = xml.find("ns:BatchLogInfo", namespaces=ns)
    date = info.findtext("ns:AcqDateTime", namespaces=ns)
    user = info.findtext("ns:OperatorName", namespaces=ns)

    return {
        "Acquisition Name": name,
        "Acquisition Path": path,
        "Acquisition Date": date,
        "Acquisition User": user,
    }


def device_xml_read_info(path: Path) -> Dict[str, str]:
    xml = ElementTree.parse(path)
    device = xml.find("Device")
    type = device.findtext("Name")
    model = device.findtext("ModelNumber")
    serial = device.findtext("SerialNumber")
    return {
        "Instrument Vendor": "Agilent",
        "Instrument Type": type,
        "Instrument Model": model,
        "Instrument Serial": serial,
    }


def load_info(path: Union[str, Path]) -> Dict[str, str]:
    """Reads information from a batch.

    Instrument info is read from the first Devices.xml found, batch info from
    the BatchLog.xml. An empty dictionary is returned if neither file can be read.

    Possible keys:
        Acquisition {Date,Name,Path,User}
        Instrument {Type,Model,Serial,Vendor}

    Args:
        path: path to batch

    Returns:
        dict
    """

    if isinstance(path, str):  # pragma: no cover
        path = Path(path)

    try:
        device_xml = path.glob("*.d/AcqData/Devices.xml")
        info = device_xml_read_info(next(device_xml))
    except (
        StopIteration,
        FileNotFoundError,
        ElementTree.ParseError,
    ):  # pragma: no cover
        logger.warning("Unable to read info from Devices.xml.")
        info = {}

    try:
        batch_xml = path.joinpath(batch_xml_path)
        info.update(batch_xml_read_info(batch_xml))
    except (FileNotFoundError, ElementTree.ParseError):  # pragma: no cover
        logger.warning("Unable to read info from BatchLog.xml.")

    return {k: v for k, v in sorted(info.items()) if v is not None}


def load(
    path: Union[str, Path],
    collection_methods: List[str] = None,
    use_acq_for_names: bool = True,
    counts_per_second: bool = False,
    drop_names: List[str] = None,
    full: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """Imports an Agilent '.b' batch.

    First attempts a binary import, falling back to importing any '.csv' files.

    Args:
        path: path to batch
        collection_methods: list of datafile collection methods,
            default = ['batch_xml', 'batch_csv']
        use_acq_for_names: read element names from 'AcqMethod.xml', only for csv
        counts_per_second: return data in CPS, only for binary
        drop_names: names to remove from final array
        full: also return dict with scantime

    Returns:
        structured array of data
        dict of params if `full`

    See Also:
        :func:`pewlib.io.agilent.collect_datafiles`
        :func:`pewlib.io.agilent.load_binary`
        :func:`pewlib.io.agilent.load_csv`
    """
    try:
        result = load_binary(
            path,
            collection_methods,
            counts_per_second=counts_per_second,
            drop_names=drop_names,
            full=full,
        )
    except Exception as e:
        logger.info("Unable to import as binary, reverting to CSV import.")
        logger.exception(e)
        result = load_csv(
            path,
            collection_methods,
            use_acq_for_names=use_acq_for_names,
            drop_names=drop_names,
            full=full,
        )
    return result
