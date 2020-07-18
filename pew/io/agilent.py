import os
import logging
from xml.etree import ElementTree

import numpy as np
import numpy.lib
import numpy.lib.recfunctions

from pew.io.error import PewException

from typing import Generator, List, Tuple

logger = logging.getLogger(__name__)

# These files are not present in older software, must be able to be ignored safely
# Important files:
#   Method/AcqMethod.xml - Contains the datafile list <SampleParameter>
#   {.d file}/AcqData/MSTS.xml - Contains run time in mins <StartTime>, <EndTime>; number of scans <NumOfScans>
#   {.d file}/AcqData/MSTS_XSpecific.xml - Contains acc time for elements <AccumulationTime>

acq_method_path = os.path.join("Method", "AcqMethod.xml")
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


def find_datafiles(path: str) -> Generator[str, None, None]:
    with os.scandir(path) as it:
        for entry in it:
            if entry.name.lower().endswith(".d") and entry.is_dir():
                yield os.path.join(path, entry.name)


def acq_method_read_datafiles(
    root: str, method_path: str
) -> Generator[str, None, None]:
    xml = ElementTree.parse(os.path.join(root, method_path))
    ns = {"ns": xml.getroot().tag.split("}")[0][1:]}
    samples = xml.findall("ns:SampleParameter", ns)
    samples = sorted(
        samples, key=lambda s: int(s.findtext("ns:SampleID", namespaces=ns) or -1)
    )

    for sample in samples:
        data_file = sample.findtext("ns:DataFileName", namespaces=ns)
        if data_file is not None:
            data_file = os.path.join(root, data_file)
            if os.path.exists(data_file):
                yield data_file


def acq_method_read_elements(root: str, method_path: str) -> List[str]:
    xml = ElementTree.parse(os.path.join(root, method_path))
    ns = {"ns": xml.getroot().tag.split("}")[0][1:]}

    elements: List[Tuple[str, int, int]] = []
    for element in xml.findall("ns:IcpmsElement", ns):
        name = element.findtext("ns:ElementName", namespaces=ns)
        if name is None:
            continue
        mz = int(element.findtext("ns:MZ", namespaces=ns) or -1)
        mzmz = int(element.findtext("ns:SelectedMZ", namespaces=ns) or -1)
        elements.append((name, mz, mzmz))

    elements = sorted(elements, key=lambda e: (e[1], e[2]))
    return [
        f"{e[0]}{e[1]}{'__' if e[2] > 1 else ''}{e[2] if e[2] > -1 else ''}"
        for e in elements
    ]


def batch_csv_read_datafiles(root: str, batch_csv: str) -> Generator[str, None, None]:
    batch_log = np.genfromtxt(
        os.path.join(root, batch_csv),
        delimiter=",",
        comments=None,
        names=True,
        usecols=(0, 5, 6),
        dtype=[np.uint32, object, "S4"],
    )
    for _id, data_file, result in batch_log:
        if result.decode() == "Pass":
            data_file = data_file.decode()
            data_file = os.path.join(
                root, data_file[max(map(data_file.rfind, "\\/")) + 1 :]
            )
            if os.path.exists(data_file):
                yield data_file


def batch_xml_read_datafiles(root: str, batch_xml: str) -> Generator[str, None, None]:
    xml = ElementTree.parse(os.path.join(root, batch_xml))
    ns = {"ns": xml.getroot().tag.split("}")[0][1:]}

    for log in xml.findall("ns:BatchLogInfo", ns):
        if log.findtext("ns:AcqResult", namespaces=ns) == "Pass":
            data_file = log.findtext("ns:DataFileName", namespaces=ns)
            data_file = os.path.join(
                root, data_file[max(map(data_file.rfind, "\\/")) + 1 :]
            )
            logger.debug(f"Looking for datafile '{data_file}'.")
            if os.path.exists(data_file):
                yield data_file


def msts_read_params(msts_path: str) -> Tuple[float, int]:
    xml = ElementTree.parse(msts_path)
    segment = xml.find("TimeSegment")
    if segment is None:
        raise PewException("Malformed MSTS.xml")

    stime = float(segment.findtext("StartTime") or 0)
    etime = float(segment.findtext("EndTime") or 0)
    scans = int(segment.findtext("NumOfScans") or 0)

    return np.round((etime - stime) * 60 / scans, 4), scans


def load(path: str, full: bool = False) -> np.ndarray:
    """Imports an Agilent batch (.b) directory, returning IsotopeData object.

   Scans the given path for .d directories containg a similarly named
   .csv file. These are imported as lines, sorted by their name.

    Args:
       path: Path to the .b directory
       full: return dict of available params

    Returns:
        The structured numpy array.

    Raises:
        PewException

    """
    # Batch files
    # acq_xml = os.path.join(path, "Method", "AcqMethod.xml")
    # batch_csv = os.path.join(path, "BatchLog.csv")
    # batch_xml = os.path.join(path, "Method", "BatchLog.xml")

    # if raw:
    #     acq_xml = batch_xml = batch_csv = ""

    # Collect data files
    ddirs = []
    for file, function in zip(
        [batch_xml_path, batch_csv_path, acq_method_path],
        [
            batch_xml_read_datafiles,
            batch_csv_read_datafiles,
            acq_method_read_datafiles,
        ],
    ):
        if os.path.exists(os.path.join(path, file)):
            ddirs = list(function(path, file))
            if len(ddirs) > 0:
                logger.info(f"Imported files using {os.path.basename(file)}")
                break

    # Fall back to csv name order
    if len(ddirs) == 0:
        logger.warning(
            "Unable to import files from BatchLog or AcqMethod.xml,"
            " falling back to alphabetical order."
        )
        ddirs = list(find_datafiles(path))
        ddirs.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

    msts_xml = os.path.join(path, ddirs[0], "AcqData", "MSTS.xml")

    # Collect csvs
    csvs: List[str] = []
    for d in ddirs:
        csv = os.path.join(d, os.path.splitext(os.path.basename(d))[0] + ".csv")
        logger.debug(f"Looking for csv '{csv}'.")
        if not os.path.exists(csv):
            logger.warning(f"Missing csv '{csv}', line blanked.")
            csvs.append(None)
        else:
            csvs.append(csv)

    # Read elements, the scan time and number fo scans
    if os.path.exists(msts_xml) and os.path.exists(os.path.join(path, acq_method_path)):
        names = acq_method_read_elements(path, acq_method_path)
        scan_time, nscans = msts_read_params(msts_xml)
    else:
        logger.info("AcqMethod.xml and MSTS.xml not found, reading params from csv.")
        names, scan_time, nscans = csv_read_params(
            next(c for c in csvs if c is not None)
        )

    data = np.empty((len(ddirs), nscans), dtype=[(name, np.float64) for name in names])
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
