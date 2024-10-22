"""
Import of line-by-line data exported from Qtegra using the '.csv' export function.
Both column and row formats are supported.
Tested with Thermo iCAP RQ ICP-MS.
"""

import logging
from collections.abc import Generator
from io import TextIOBase
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _icap_csv_columns_read(
    path: Path,
    line_type: str,
    delimiter: str | None = None,
    comma_decimal: bool = False,
) -> np.ndarray:
    def _read_lines(
        fp: TextIOBase, line_type: str, replace_decimal: bool = False
    ) -> Generator[str, None, None]:
        for line in fp:
            if line.startswith("MainRuns") and line_type in line:
                yield line.replace(",", ".") if replace_decimal else line

    with path.open("r", encoding="utf-8-sig") as fp:
        line = fp.readline()
        if delimiter is None:
            delimiter = line[0]  # Should be delimiter
        nlines = np.count_nonzero(
            np.genfromtxt([line], dtype="U1", delimiter=delimiter)
        )
        if nlines == 0:
            raise ValueError(
                "Invalid iCap export, expected samples in columns."
            )  # pragma: no cover

        dtype = [
            ("run", "U8"),
            ("scan", int),
            ("name", "U32"),
            ("type", "U7"),
            ("data", np.float64, nlines),
        ]
        data = np.genfromtxt(
            _read_lines(fp, line_type, replace_decimal=comma_decimal),
            dtype=dtype,
            delimiter=delimiter,
        )

        names, idx = np.unique(data["name"], return_index=True)
        names = names[np.argsort(idx)]

        structured = np.empty(
            (data["data"].shape[1], np.amax(data["scan"]) + 1),
            dtype=[(name, np.float64) for name in names],
        )
        for name in names:
            structured[name] = data[data["name"] == name]["data"].T

        return structured


def icap_csv_columns_read_data(
    path: str | Path,
    delimiter: str | None = None,
    comma_decimal: bool = False,
    use_analog: bool = False,
) -> np.ndarray:
    if isinstance(path, str):  # pragma: no cover
        path = Path(path)

    return _icap_csv_columns_read(
        path,
        line_type="Analog" if use_analog else "Counter",
        delimiter=delimiter,
        comma_decimal=comma_decimal,
    )


def icap_csv_columns_read_params(
    path: str | Path, delimiter: str | None = None, comma_decimal: bool = False
) -> dict:
    if isinstance(path, str):  # pragma: no cover
        path = Path(path)

    data = _icap_csv_columns_read(
        path, line_type="Time", delimiter=delimiter, comma_decimal=comma_decimal
    )
    data = data[data.dtype.names[0]]
    scantime = np.round(np.nanmean(np.diff(data, axis=1)), 4)

    return dict(times=data, scantime=scantime)


def _icap_csv_rows_read(
    path: Path,
    col_type: str,
    delimiter: str | None = None,
    comma_decimal: bool = False,
) -> np.ndarray:
    def _read_lines(
        fp: TextIOBase, replace_decimal: bool = False
    ) -> Generator[str, None, None]:
        for line in fp:
            yield line.replace(",", ".") if replace_decimal else line

    with path.open("r", encoding="utf-8-sig") as fp:
        line = fp.readline()
        if delimiter is None:
            delimiter = line[0]  # Should be delimiter

        run_mask = np.array(line.split(delimiter), dtype="U8") == "MainRuns"
        if np.count_nonzero(run_mask) == 0:  # pragma: no cover
            raise ValueError("Invalid iCap export, expected samples in rows.")

        scans = np.array(
            fp.readline().split(delimiter), dtype="U4"
        )  # nscans = np.amax(scans) + 1
        names = np.array(fp.readline().split(delimiter), dtype="U32")
        type_mask = np.array(fp.readline().split(delimiter), dtype="U7") == col_type

        col_mask = np.logical_and(run_mask, type_mask)
        scans = scans[col_mask].astype(int)
        names = names[col_mask]

        data = np.genfromtxt(
            _read_lines(fp, replace_decimal=comma_decimal),
            dtype=np.float64,
            usecols=np.flatnonzero(col_mask),
            delimiter=delimiter,
        )
        unames, idx = np.unique(names, return_index=True)
        unames = unames[np.argsort(idx)]

        structured = np.empty(
            (data.shape[0], np.amax(scans) + 1), dtype=[(n, float) for n in unames]
        )
        for name in structured.dtype.names:
            structured[name] = data[:, names == name]

        return structured


def icap_csv_rows_read_data(
    path: str | Path,
    delimiter: str | None = None,
    comma_decimal: bool = False,
    use_analog: bool = False,
) -> np.ndarray:
    if isinstance(path, str):  # pragma: no cover
        path = Path(path)

    return _icap_csv_rows_read(
        path,
        "Analog" if use_analog else "Counter",
        delimiter=delimiter,
        comma_decimal=comma_decimal,
    )


def icap_csv_rows_read_params(
    path: str | Path,
    delimiter: str | None = None,
    comma_decimal: bool = False,
) -> dict:
    if isinstance(path, str):  # pragma: no cover
        path = Path(path)

    data = _icap_csv_rows_read(
        path,
        "Time",
        delimiter=delimiter,
        comma_decimal=comma_decimal,
    )
    data = data[data.dtype.names[0]]

    scantime = np.round(np.nanmean(np.diff(data, axis=1)), 4)

    return dict(times=data, scantime=scantime)


def icap_csv_sample_format(path: str | Path) -> str:
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
    path: str | Path, use_analog: bool = False, full: bool = False
) -> np.ndarray | tuple[np.ndarray, dict]:  # pragma: no cover
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
    elif sample_format == "columns":
        data = icap_csv_columns_read_data(path)
    else:  # pragma: no cover
        raise ValueError("Unknown iCap CSV format.")

    if full:
        try:
            if sample_format == "rows":
                params = icap_csv_rows_read_params(path)
            elif sample_format == "columns":
                params = icap_csv_columns_read_params(path)
            return data, params
        except (IndexError, ValueError):
            logger.warning(f"Unabled to read params from {path.name}")
            return data, {}
    else:  # pragma: no cover
        return data
