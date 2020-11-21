"""
Import of line-by-line data exported from Qtegra using the '.csv' export function.
Both column and row formats are supported.
Tested with Thermo iCAP RQ ICP-MS.
"""
import numpy as np
from pathlib import Path

from typing import Generator, TextIO, Tuple, Union


def _icap_csv_columns_read(
    path: Path, line_type: str, delimiter: str = None, comma_decimal: bool = False
) -> np.ndarray:
    def _read_lines(
        fp: TextIO, line_type: str, replace_decimal: bool = False
    ) -> Generator[str, None, None]:
        for line in fp:
            if line.startswith("MainRuns") and line_type in line:
                yield line.replace(",", ".") if replace_decimal else line

    with path.open("r", encoding="utf-8-sig") as fp:
        line = fp.readline()
        if delimiter is None:
            delimiter = line[0]  # Should be delimiter
        nlines = np.count_nonzero(
            np.genfromtxt([line], dtype="S1", delimiter=delimiter)
        )
        if nlines == 0:
            raise ValueError(
                "Invalid iCap export, expected samples in columns."
            )  # pragma: no cover

        dtype = [
            ("run", "S8"),
            ("scan", int),
            ("isotope", "S12"),
            ("type", "S7"),
            ("data", np.float64, nlines),
        ]
        return np.genfromtxt(
            _read_lines(fp, line_type, replace_decimal=comma_decimal),
            dtype=dtype,
            delimiter=delimiter,
        )


def icap_csv_columns_read_data(
    path: Union[str, Path],
    delimiter: str = None,
    comma_decimal: bool = False,
    use_analog: bool = False,
) -> np.ndarray:
    if isinstance(path, str):  # pragma: no cover
        path = Path(path)

    line_type = "Analog" if use_analog else "Counter"
    data = _icap_csv_columns_read(
        path, line_type=line_type, delimiter=delimiter, comma_decimal=comma_decimal
    )
    nscans = np.amax(data["scan"]) + 1

    names, idx = np.unique(data["isotope"], return_index=True)
    names = names[np.argsort(idx)]

    structured = np.empty(
        (data["data"].shape[1], nscans),
        dtype=[(name.decode(), np.float64) for name in names],
    )
    for name in names:
        structured[name.decode()] = data[data["isotope"] == name]["data"].T

    return structured


def icap_csv_columns_read_params(
    path: Union[str, Path],
    delimiter: str = None,
    comma_decimal: bool = False,
) -> dict:
    if isinstance(path, str):  # pragma: no cover
        path = Path(path)

    data = _icap_csv_columns_read(
        path, line_type="Time", delimiter=delimiter, comma_decimal=comma_decimal
    )
    data = data[data["isotope"] == data["isotope"][0]]["data"]
    scantime = np.round(np.nanmean(np.diff(data, axis=0)), 4)

    return dict(scantime=scantime)


def icap_csv_rows_read_data(
    path: Union[str, Path],
    delimiter: str = None,
    comma_decimal: bool = False,
    use_analog: bool = False,
) -> np.ndarray:
    def _read_lines(
        fp: TextIO, replace_decimal: bool = False
    ) -> Generator[str, None, None]:
        for line in fp:
            yield line.replace(",", ".") if replace_decimal else line

    if isinstance(path, str):  # pragma: no cover
        path = Path(path)

    with path.open("r", encoding="utf-8-sig") as fp:
        line = fp.readline()
        if delimiter is None:
            delimiter = line[0]  # Should be delimiter

        run_mask = np.genfromtxt([line], dtype="S8", delimiter=delimiter) == b"MainRuns"
        if np.count_nonzero(run_mask) == 0:  # pragma: no cover
            raise ValueError("Invalid iCap export, expected samples in rows.")

        scans = np.genfromtxt([fp.readline()], dtype=int, delimiter=delimiter)
        nscans = np.amax(scans) + 1
        isotopes = np.genfromtxt([fp.readline()], dtype="S12", delimiter=delimiter)
        dtype = np.genfromtxt([fp.readline()], dtype="S7", delimiter=delimiter)

        cols = np.nonzero(
            np.logical_and(dtype == (b"Analog" if use_analog else b"Counter"), run_mask)
        )[0]
        names, idx = np.unique(isotopes[cols], return_index=True)
        names = names[np.argsort(idx)]

        data = np.genfromtxt(
            _read_lines(fp, replace_decimal=comma_decimal),
            dtype=np.float64,
            delimiter=delimiter,
            usecols=cols,
        )

    structured = np.empty(
        (data.shape[0], nscans),
        dtype=[(name.decode(), np.float64) for name in names],
    )
    for name in names:
        structured[name.decode()] = data[:, isotopes[cols] == name]

    return structured


def icap_csv_rows_read_params(
    path: Union[str, Path],
    delimiter: str = None,
    comma_decimal: bool = False,
) -> dict:
    def _read_lines(
        fp: TextIO, replace_decimal: bool = False
    ) -> Generator[str, None, None]:
        for line in fp:
            yield line.replace(",", ".") if replace_decimal else line

    if isinstance(path, str):  # pragma: no cover
        path = Path(path)

    with path.open("r", encoding="utf-8-sig") as fp:
        line = fp.readline()
        if delimiter is None:
            delimiter = line[0]  # Should be delimiter

        run_mask = np.genfromtxt([line], dtype="S8", delimiter=delimiter) == b"MainRuns"
        if np.count_nonzero(run_mask) == 0:  # pragma: no cover
            raise ValueError("Invalid iCap export, expected samples in rows.")

        fp.readline()  # scans
        isotopes = np.genfromtxt([fp.readline()], dtype="S12", delimiter=delimiter)
        dtype = np.genfromtxt([fp.readline()], dtype="S4", delimiter=delimiter)

        cols = np.nonzero(
            np.logical_and.reduce((isotopes == isotopes[2], dtype == b"Time", run_mask))
        )[0]
        names, idx = np.unique(isotopes[cols], return_index=True)
        names = names[np.argsort(idx)]

        data = np.genfromtxt(
            _read_lines(fp, replace_decimal=comma_decimal),
            dtype=np.float64,
            delimiter=delimiter,
            usecols=cols,
            max_rows=1,
        )

    scantime = np.round(np.nanmean(np.diff(data, axis=0)), 4)

    return dict(scantime=scantime)


def icap_csv_sample_format(path: Union[str, Path]) -> str:
    """Determines CSVsample format.

    Valid formats are 'columns' and 'rows' depending on Qtegra export option.

    Args:
        Path: path to CSV

        Returns:
            'rows', 'columns' or 'unknown' if invalid
    """
    if isinstance(path, str):  # pragma: no cover
        path = Path(path)

    with path.open("r", encoding="utf-8-sig") as fp:
        lines = [next(fp) for i in range(3)]
    if "MainRuns" in lines[0]:
        return "rows"
    elif "MainRuns" in lines[2]:
        return "columns"
    else:
        return "unknown"


def load(
    path: Union[str, Path], use_analog: bool = False, full: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:  # pragma: no cover
    """Imports iCap CSV export.

    Data must be exported from Qtegra using the CSV export option.
    At a mininmum the 'Counts' column must exported for each element.
    If `use_analog` the 'Analog' channel must be exported.
    If `full` and the 'Time' column is exported then the scantime can be determined.
    Samples in columns and rows are both supported.

    Args:
        path: path to CSV
        use_analog: ues 'Analog' instead of 'Counts'
        full: also export a dict of params

    Returns:
        structured array of data
        dict of params if `full`

    Raises:
        ValueError: unknown or invalid CSV
    """
    if isinstance(path, str):  # pragma: no cover
        path = Path(path)

    sample_format = icap_csv_sample_format(path)
    if sample_format == "rows":
        data = icap_csv_rows_read_data(path)
        if full:
            params = icap_csv_rows_read_params(path)
    elif sample_format == "columns":
        data = icap_csv_columns_read_data(path)
        if full:
            params = icap_csv_columns_read_params(path)
    else:  # pragma: no cover
        raise ValueError("Unknown iCap CSV format.")

    if full:
        return data, dict(scantime=params["scantime"])
    else:  # pragma: no cover
        return data
