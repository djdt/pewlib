from concurrent.futures import ProcessPoolExecutor
import numpy as np
import numpy.lib.recfunctions as rfn
from pathlib import Path

from typing import Callable, List, Tuple, Union


def read_report_file(path: Path) -> Tuple[int, int]:
    """Reads parameters from the report.

    The expected number of files and minimum cycles are used while reading
    aquistions.

    Returns:
        number of files, minimum number of cycles
    """
    report = np.genfromtxt(path, delimiter=",", names=True, dtype=np.int32)
    return np.amax(report["File_number"]), np.amin(report["cycles"])


def read_acq_file(path: Path, number_cycles: int) -> np.ndarray:
    """Imports a single acquisition (line).

    The `number_cycles` is used to determine the maximum line length.
    This ensures that lines will stack correctly.

    Args:
        path: path
        number_cycles: maximum line length
        callback: callback for progress
    """
    return np.genfromtxt(
        path, delimiter=",", deletechars="", names=True, dtype=np.float64
    )


    def load(path: Union[str, Path], threads: int = None, full: bool = False) -> np.ndarray:
    """Load a Nu instruments batch.

    The value of `threads` is passed to the ProcessPoolExecutor.
    If `full` then a dict with the spotsize is also returned.
    """

    report_path = list(Path().glob("File_Report*.csv"))
    if len(report_path) == 0:
        raise FileNotFoundError("Could no find report file!")
    number_files, min_cycles = read_report_file(report_path[0])
    acq_paths = sorted([p for p in Path().glob("acq*.csv")])

    with ProcessPoolExecutor() as execuctor:
        results = [execuctor.submit(read_acq_file, p, cycle_min) for p in acqpaths]
        data = np.stack([r.result() for r in results], axis=1)
