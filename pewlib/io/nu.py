from concurrent.futures import ProcessPoolExecutor
import numpy as np
import numpy.lib.recfunctions as rfn
from pathlib import Path

from typing import List, Tuple, Union


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
    )[:number_cycles]


def load(
    path: Union[str, Path],
    threads: int = None,
    drop_names: List[str] = None,
    full: bool = False,
) -> np.ndarray:
    """Load a Nu instruments batch.

    The value of `threads` is passed to the ProcessPoolExecutor.
    If `full` then a dict with the spotsize is also returned.
    """
    if isinstance(path, str):
        path = Path(path)

    if drop_names is None:
        drop_names = ["X (um)", "Y (um)"]

    # Read report
    report_path = list(path.glob("File_Report*.csv"))
    if len(report_path) == 0:
        raise FileNotFoundError("Could not find report file!")
    number_files, min_cycles = read_report_file(report_path[0])

    acq_paths = sorted(
        path.glob("acq*.csv"), key=lambda p: int("".join(filter(str.isdigit, p.stem)))
    )

    with ProcessPoolExecutor() as execuctor:
        results = [execuctor.submit(read_acq_file, p, min_cycles) for p in acq_paths]
        data = np.stack([r.result() for r in results], axis=1)

    data = rfn.drop_fields(data, drop_names)

    return data
