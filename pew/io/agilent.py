import os
from xml.etree import ElementTree
import warnings
import numpy as np
import numpy.lib
import numpy.lib.recfunctions

from .error import PewException, PewWarning

from typing import Generator, List, Tuple


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


def acq_read_elements(method_path: str) -> List[str]:
    xml = ElementTree.parse(method_path)
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
        f"{e[0]}{e[1]}{'->' if e[2] > 1 else ''}{e[2] if e[2] > -1 else ''}"
        for e in elements
    ]


def acq_read_datafiles(method_path: str) -> Generator[str, None, None]:
    xml = ElementTree.parse(method_path)
    ns = {"ns": xml.getroot().tag.split("}")[0][1:]}
    samples = xml.findall("ns:SampleParameter", ns)
    samples = sorted(
        samples, key=lambda s: int(s.findtext("ns:SampleID", namespaces=ns) or -1)
    )

    for sample in samples:
        data_file = sample.findtext("ns:DataFileName", namespaces=ns)
        if data_file is not None:
            yield data_file


def load(path: str) -> np.ndarray:
    """Imports an Agilent batch (.b) directory, returning IsotopeData object.

   Scans the given path for .d directories containg a similarly named
   .csv file. These are imported as lines, sorted by their name.

    Args:
       path: Path to the .b directory

    Returns:
        The structured numpy array.

    Raises:
        PewException

    """
    names = acq_read_elements(os.path.join(path, "Method", "AcqMethod.xml"))
    ddirs = acq_read_datafiles(os.path.join(path, "Method", "AcqMethod.xml"))
    # ddirs = []
    # with os.scandir(path) as it:
    #     for entry in it:
    #         if entry.name.lower().endswith(".d") and entry.is_dir():
    #             ddirs.append(entry.path)

    csvs = []
    # Sort by name
    ddirs.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    for d in ddirs:
        csv = os.path.splitext(os.path.basename(d))[0] + ".csv"
        csv = os.path.join(d, csv)
        if not os.path.exists(csv):
            warnings.warn(f"{path} missing csv {os.path.basename(csv)}", PewWarning)
            continue
        csvs.append(csv)

    datas = []
    for csv in csvs:
        try:
            datas.append(
                np.genfromtxt(
                    clean_lines(csv),
                    delimiter=b",",
                    names=True,
                    dtype=np.float64,
                )
            )
        except ValueError as e:
            raise PewException(f"{e} Could not parse batch.") from e

    try:
        print([d.shape for d in datas])
        data = np.vstack(datas)
        # We don't care about the time field currently
        data = np.lib.recfunctions.drop_fields(data, "Time_Sec")

    except ValueError as e:
        raise PewException("Mismatched data.") from e

    return data
