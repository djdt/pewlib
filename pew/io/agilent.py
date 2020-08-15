import os
import logging
from xml.etree import ElementTree

import numpy as np
import numpy.lib
import numpy.lib.recfunctions

from pew.io.error import PewException

from typing import Callable, List, Tuple

logger = logging.getLogger(__name__)

# These files are not present in older software, must be able to be ignored safely
# Important files:
#   Method/AcqMethod.xml - Contains the datafile list <SampleParameter>
#   {.d file}/AcqData/MSTS.xml - Contains run time in mins <StartTime>, <EndTime>; number of scans <NumOfScans>
#   {.d file}/AcqData/MSTS_XSpecific.xml - Contains acc time for elements <AccumulationTime>

acq_method_xml_path = os.path.join("Method", "AcqMethod.xml")
batch_csv_path = "BatchLog.csv"
batch_xml_path = os.path.join("Method", "BatchLog.xml")


def clean_lines(csv: str):
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


def csv_read_params(path: str) -> Tuple[List[str], float, int]:
    data = np.genfromtxt(
        clean_lines(path), delimiter=b",", names=True, dtype=np.float64
    )
    total_time = np.max(data["Time_Sec"])
    names = [name for name in data.dtype.names if name != "Time_Sec"]
    return names, np.round(total_time / data.shape[0], 4), data.shape[0]


def find_datafiles_alphabetical(path: str) -> List[str]:
    data_files = []
    with os.scandir(path) as it:
        for entry in it:
            if entry.name.lower().endswith(".d") and entry.is_dir():
                data_files.append(os.path.join(path, entry.name))
    return data_files


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
        if name is None:
            continue
        mz = int(element.findtext("ns:MZ", namespaces=ns) or -1)
        mz2 = int(element.findtext("ns:SelectedMZ", namespaces=ns) or -1)
        elements.append((name, mz, mz2))

    elements = sorted(elements, key=lambda e: (e[1], e[2]))
    names = []
    for e in elements:
        names.append(f"{e[0]}{e[2]}->{e[1]}" if msms else f"{e[0]}{e[1]}")
    return names


def batch_csv_read_datafiles(batch_root: str, batch_csv: str) -> List[str]:
    batch_log = np.genfromtxt(
        batch_csv,
        delimiter=",",
        comments=None,
        names=True,
        usecols=(0, 5, 6),
        dtype=[np.uint32, object, "S4"],
    )
    if batch_log.size == 1:  # Ensure iterable even if one line
        batch_log = batch_log.reshape(1)
    data_files = []
    for _id, data_file, result in batch_log:
        if result.decode() == "Pass":
            data_file = data_file.decode()
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


# def msts_xml_read_params(msts_xml: str) -> Tuple[float, int]:
#     xml = ElementTree.parse(msts_xml)
#     segment = xml.find("TimeSegment")
#     if segment is None:
#         raise PewException("Malformed MSTS.xml")

#     stime = float(segment.findtext("StartTime") or 0)
#     etime = float(segment.findtext("EndTime") or 0)
#     scans = int(segment.findtext("NumOfScans") or 0)

#     return np.round((etime - stime) * 60 / scans, 4), scans


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
            else:
                logger.info(f"Missing {missing} datafiles using '{method}'.")
        else:
            logger.warning(f"Unable to collect datafiles using '{method}'.")

    # Fall back to alphabetical
    logger.info("Falling back to alphabetical order for datafile collection.")
    data_files = find_datafiles_alphabetical(batch_root)
    data_files.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    return data_files


def load(
    path: str,
    collection_methods: List[str] = None,
    use_acq_for_names: bool = True,
    full: bool = False,
) -> np.ndarray:
    """Imports an Agilent batch (.b) directory, returning structured array.
    Finds lines using (in order of preference): BatchLog.xml, BatchLog.csv,
     AcqMethod.xml, .d files sorted by name.

    Args:
       path: Path to the .b directory
       raw: Only use .d and .csv files.
       full: return dict of available params

    Returns:
        The structured numpy array.

    Raises:
        PewException

    """
    if collection_methods is None:
        collection_methods = ["batch_xml", "batch_csv", "acq_method_xml"]

    # Collect data files
    data_files = collect_datafiles(path, collection_methods)
    if len(data_files) == 0:
        raise PewException(f"No data files found in {path}!")

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
        if not os.path.exists(os.path.join(path, acq_method_xml_path)):
            logger.warning("AcqMethod.xml not found, cannot read names.")
        else:
            names = acq_method_xml_read_elements(
                os.path.join(path, acq_method_xml_path)
            )

    data = np.empty(
        (len(data_files), nscans), dtype=[(name, np.float64) for name in names]
    )
    for i, csv in enumerate(csvs):
        if csv is None:
            data[i, :] = np.zeros(data.shape[1], dtype=data.dtype)
        else:
            try:
                data[i, :] = np.genfromtxt(
                    clean_lines(csv),
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
