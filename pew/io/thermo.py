import numpy as np

from pew.io.error import PewException

from typing import Generator, TextIO


def _icap_csv_columns_read(
    path: str, line_type: str, delimiter: str = None, comma_decimal: bool = False
) -> np.ndarray:
    def _read_lines(
        fp: TextIO, line_type: str, replace_decimal: bool = False
    ) -> Generator[str, None, None]:
        for line in fp:
            if line.startswith("MainRuns") and line_type in line:
                yield line.replace(",", ".") if replace_decimal else line

    with open(path, "r", encoding="utf-8-sig") as fp:
        line = fp.readline()
        if delimiter is None:
            delimiter = line[0]  # Should be delimiter
        nlines = np.count_nonzero(
            np.genfromtxt([line], dtype="S1", delimiter=delimiter)
        )
        if nlines == 0:
            raise PewException("Invalid iCap export, expected samples in columns.")

        dtype = [
            ("run", "S8"),
            ("scan", int),
            ("isotope", "S12"),
            ("type", "S7"),
            ("data", np.float64, nlines),
        ]
        try:
            return np.genfromtxt(
                _read_lines(fp, line_type, replace_decimal=comma_decimal),
                dtype=dtype,
                delimiter=delimiter,
            )
        except ValueError as e:
            raise PewException("Could not read iCap CSV (samples in columns).") from e


def icap_csv_columns_read_data(
    path: str,
    delimiter: str = None,
    comma_decimal: bool = False,
    use_analog: bool = False,
) -> np.ndarray:
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
    path: str, delimiter: str = None, comma_decimal: bool = False,
) -> dict:
    data = _icap_csv_columns_read(
        path, line_type="Time", delimiter=delimiter, comma_decimal=comma_decimal
    )
    data = data[data["isotope"] == data["isotope"][0]]["data"]
    scantime = np.round(np.nanmean(np.diff(data, axis=0)), 4)

    return dict(scantime=scantime)


def icap_csv_rows_read_data(
    path: str,
    delimiter: str = None,
    comma_decimal: bool = False,
    use_analog: bool = False,
) -> np.ndarray:
    def _read_lines(
        fp: TextIO, replace_decimal: bool = False
    ) -> Generator[str, None, None]:
        for line in fp:
            yield line.replace(",", ".") if replace_decimal else line

    with open(path, "r", encoding="utf-8-sig") as fp:
        line = fp.readline()
        if delimiter is None:
            delimiter = line[0]  # Should be delimiter

        run_mask = np.genfromtxt([line], dtype="S8", delimiter=delimiter) == b"MainRuns"
        if np.count_nonzero(run_mask) == 0:
            raise PewException("Invalid iCap export, expected samples in rows.")

        scans = np.genfromtxt([fp.readline()], dtype=int, delimiter=delimiter)
        nscans = np.amax(scans) + 1
        isotopes = np.genfromtxt([fp.readline()], dtype="S12", delimiter=delimiter)
        dtype = np.genfromtxt([fp.readline()], dtype="S7", delimiter=delimiter)

        cols = np.nonzero(
            np.logical_and(dtype == (b"Analog" if use_analog else b"Counter"), run_mask)
        )[0]
        names, idx = np.unique(isotopes[cols], return_index=True)
        names = names[np.argsort(idx)]

        try:
            data = np.genfromtxt(
                _read_lines(fp, replace_decimal=comma_decimal),
                dtype=np.float64,
                delimiter=delimiter,
                usecols=cols,
            )
        except ValueError as e:
            raise PewException("Could not read iCap CSV (samples in rows).") from e

    structured = np.empty(
        (data.shape[0], nscans), dtype=[(name.decode(), np.float64) for name in names],
    )
    for name in names:
        structured[name.decode()] = data[:, isotopes[cols] == name]

    return structured


def icap_csv_rows_read_params(
    path: str, delimiter: str = None, comma_decimal: bool = False,
) -> dict:
    def _read_lines(
        fp: TextIO, replace_decimal: bool = False
    ) -> Generator[str, None, None]:
        for line in fp:
            yield line.replace(",", ".") if replace_decimal else line

    with open(path, "r", encoding="utf-8-sig") as fp:
        line = fp.readline()
        if delimiter is None:
            delimiter = line[0]  # Should be delimiter

        run_mask = np.genfromtxt([line], dtype="S8", delimiter=delimiter) == b"MainRuns"
        if np.count_nonzero(run_mask) == 0:
            raise PewException("Invalid iCap export, expected samples in rows.")

        fp.readline()  # scans
        isotopes = np.genfromtxt([fp.readline()], dtype="S12", delimiter=delimiter)
        dtype = np.genfromtxt([fp.readline()], dtype="S4", delimiter=delimiter)

        cols = np.nonzero(
            np.logical_and.reduce((isotopes == isotopes[2], dtype == b"Time", run_mask))
        )[0]
        names, idx = np.unique(isotopes[cols], return_index=True)
        names = names[np.argsort(idx)]

        try:
            data = np.genfromtxt(
                _read_lines(fp, replace_decimal=comma_decimal),
                dtype=np.float64,
                delimiter=delimiter,
                usecols=cols,
                max_rows=1,
            )
        except ValueError as e:
            raise PewException("Could not read iCap CSV (samples in rows).") from e

    scantime = np.round(np.nanmean(np.diff(data, axis=0)), 4)

    return dict(scantime=scantime)


def load(path: str, samples_in_rows: bool = None, full: bool = False) -> np.ndarray:
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
    if samples_in_rows is None:
        with open(path, "r") as fp:
            lines = [next(fp) for i in range(3)]
        if "MainRuns" in lines[0]:
            samples_in_rows = True
        elif "MainRuns" in lines[2]:
            samples_in_rows = False
        else:
            raise PewException("Unknown iCap CSV format.")
    if samples_in_rows:
        data = icap_csv_rows_read_data(path)
        if full:
            params = icap_csv_rows_read_params(path)
    else:
        data = icap_csv_columns_read_data(path)
        if full:
            params = icap_csv_columns_read_params(path)

    if full:
        return data, dict(scantime=params["scantime"])
    else:
        return data
