"""
Import of line-by-line nu instruments data.
Uses ProcessPoolExecutor for multithreaded import of aquistions.
Untested and under development.
"""
from concurrent.futures import ProcessPoolExecutor
import logging
import numpy as np
import numpy.lib.recfunctions as rfn
from pathlib import Path

from typing import List, Tuple, Union

logger = logging.getLogger(__name__)


def is_valid_directory(path: Union[str, Path]) -> bool:
    """Tests if a directory contains Nu instruments data.

    Ensures the path exists, is a directory and contains a File_Report and at
    least one acquistion '.csv'.
    """
    if isinstance(path, str):
        path = Path(path)

    if not path.exists() or not path.is_dir():
        return False

    if len(list(path.glob("File_Report*.csv"))) == 0:
        return False

    return len(list(path.glob("acq*.csv"))) > 0


def read_report_file(path: Path) -> Tuple[int, int]:
    """Reads parameters from the report.

    The expected number of files and minimum cycles are used while reading
    aquistions.

    Returns:
        number of files, minimum number of cycles
    """
    report = np.genfromtxt(path, delimiter=",", names=True, dtype=np.int32)
    return np.amax(report["File_number"]), np.amin(report["cycles"])


def read_acqusition(path: Path, number_cycles: int) -> np.ndarray:
    """Imports a single acquisition (line).

    The `number_cycles` is used to determine the maximum line length,
    ensuring that lines will stack correctly.

    Args:
        path: path
        number_cycles: maximum line length

    Returns:
        structured array of shape (number_cycles, )
    """
    return np.genfromtxt(
        path, delimiter=",", deletechars="", names=True, dtype=np.float64
    )[:number_cycles]


def load(
    path: Union[str, Path],
    drop_names: List[str] = None,
    full: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """Load a Nu instruments data directory.

    The directory must contain at least one acquistion '.csv' and a File_Report,
    this can be checked using :func:`pewlib.io.nu.is_valid_directory`.
    Names passed to `drop_names` as removed from the final array, the default
    is to drop 'Y_(um)' and 'X_(um)'.
    If `full` then a dict with the spotsize is also returned.

    Args:
        path: directory
        drop_names: names removed from data, default removes positions.
        full: also return parameters

    Returns:
        structured array of data
        dict of params if `full`
    """
    if isinstance(path, str):
        path = Path(path)

    if drop_names is None:
        drop_names = ["X_(um)", "Y_(um)"]

    # Read report
    report_path = list(path.glob("File_Report*.csv"))
    if len(report_path) == 0:
        raise FileNotFoundError("Could not find report file!")
    number_files, min_cycles = read_report_file(report_path[0])

    acq_paths = sorted(
        path.glob("acq*.csv"), key=lambda p: int("".join(filter(str.isdigit, p.stem)))
    )

    # Multithreaded read greatly improves load times
    with ProcessPoolExecutor() as execuctor:
        results = [execuctor.submit(read_acqusition, p, min_cycles) for p in acq_paths]
        data = np.stack([r.result() for r in results], axis=1)

    params = {}
    if full:
        if "Y_(um)" in data.dtype.names:
            params["spotsize"] = np.round(np.mean(np.diff(data["Y_(um)"], axis=1)), 2)
        else:
            logger.warning("'Y_(um)' field not found, unable to import spotsize.")

    data = rfn.drop_fields(data, drop_names)

    if full:
        return data, params
    else:
        return data
