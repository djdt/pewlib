import numpy as np

from pew.io.error import PewException

from typing import Generator, List, Set, Tuple


def clean_lines(csv: str) -> Generator[str, None, None]:
    with open(csv, "r") as fp:
        for line in fp:
            yield line.replace(",", ";").replace("\t", ";")


def get_name_data(path: str, name: str) -> Generator[str, None, None]:
    for line in clean_lines(path):
        run_type, _, _name, data_type, data = line.split(";", 4)
        if run_type != "MainRuns":
            continue
        if _name == name and data_type == "Counter":
            yield data


def preprocess_file(path: str) -> Tuple[List[str], float, Tuple[int, int]]:
    names: Set[str] = set()
    time = 0.0
    nscans = 0

    lines = clean_lines(path)
    line1 = next(lines)
    if "Sample" not in line1:
        raise PewException("Unknown iCap CSV formatting.")
    nlines = line1.count(";") - 4

    for line in lines:
        run_type, n, name, data_type, data = line.split(";", 4)
        if name:
            names.add(name)
        nscans = max(nscans, int(n or -1) + 1)
        if run_type != "MainRuns":
            continue
        if data_type == "Time":
            time = max(time, float(next(s for s in data.split(";") if s)))

    return (
        sorted(names, key=lambda f: int("".join(filter(str.isdigit, f)))),
        np.round(time / nscans, 4),
        (nlines, nscans),
    )


def load(path: str, full: bool = False) -> np.ndarray:
    """Imports iCap data exported using the CSV export function.

    Data is read from the "Counts" column.
    If full and a "Time" column is available then the scan time is also returned.

    Args:
        path: Path to CSV

    Returns:
        Structured numpy array.

    Raises:
        PewException

    """
    names, scan_time, shape = preprocess_file(path)
    data = np.empty(shape, dtype=[(name, np.float64) for name in names])
    cols = np.arange(0, shape[0])

    for name in names:
        try:
            data[name] = np.genfromtxt(
                get_name_data(path, name),
                delimiter=";",
                usecols=cols,
                max_rows=shape[1],
                filling_values=0.0,
            ).T
        except ValueError as e:
            raise PewException("Could not parse file.") from e

    if full:
        return data, dict(scantime=scan_time)
    else:
        return data
