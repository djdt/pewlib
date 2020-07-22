import numpy as np

from pew.io.error import PewException

from typing import Generator, List, Set, Tuple, TextIO


def icap_csv_columns_read_data(
    path: str,
    delimiter: str = None,
    comma_decimal: bool = False,
    use_analog: bool = False,
) -> np.ndarray:
    # Greatly speeds up reading
    def _read_lines(
        fp: TextIO, analog: bool = False, replace_decimal: bool = False
    ) -> Generator[str, None, None]:
        for line in fp:
            if (
                line.startswith("MainRuns")
                and ("Analog" if analog else "Counter") in line
            ):
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
            data = np.genfromtxt(
                _read_lines(fp, use_analog, replace_decimal=comma_decimal),
                dtype=dtype,
                delimiter=delimiter,
            )
        except ValueError as e:
            raise PewException("Could not parse iCap CSV as columns.") from e

        nscans = np.amax(data["scan"]) + 1

        names, idx = np.unique(data["isotope"], return_index=True)
        names = names[np.argsort(idx)]

    structured = np.empty(
        (nlines, nscans), dtype=[(name.decode(), np.float64) for name in names],
    )
    for name in names:
        structured[name.decode()] = data[data["isotope"] == name]["data"].T

    return structured


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
            raise PewException("Could not parse iCap CSV as rows.") from e

    structured = np.empty(
        (data.shape[0], nscans), dtype=[(name.decode(), np.float64) for name in names],
    )
    for name in names:
        structured[name.decode()] = data[:, isotopes[cols] == name]

    return structured


# def preprocess_file(path: str) -> Tuple[List[str], float, Tuple[int, int]]:
#     names: Set[str] = set()
#     time = 0.0
#     nscans = 0

#     lines = clean_lines(path)
#     line1 = next(lines)
#     if "Sample" not in line1:
#         raise PewException("Unknown iCap CSV formatting.")
#     nlines = line1.count(";") - 4

#     for line in lines:
#         run_type, n, name, data_type, data = line.split(";", 4)
#         if name:
#             names.add(name)
#         nscans = max(nscans, int(n or -1) + 1)
#         if run_type != "MainRuns":
#             continue
#         if data_type == "Time":
#             time = max(time, float(next(s for s in data.split(";") if s)))

#     return (
#         sorted(names, key=lambda f: int("".join(filter(str.isdigit, f)))),
#         np.round(time / nscans, 4),
#         (nlines, nscans),
#     )


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
        lines = [next(open(path, "r")) for i in range(3)]
        if "MainRuns" in lines[0]:
            samples_in_rows = True
        elif "MainRuns" in lines[2]:
            samples_in_rows = False
        else:
            raise PewException("Unknown iCap CSV format.")
    if samples_in_rows:
        data = icap_csv_rows_read_data(path)
    else:
        data = icap_csv_columns_read_data(path)

    if full:
        return data, dict(scantime=0.250)
        # return data, dict(scantime=scan_time)
    else:
        return data


if __name__ == "__main__":
    import time

    t0 = time.time()
    d = icap_csv_rows_read_data(
        "/home/tom/Downloads/thermo/20200721_GelatineStdsLaserJacob.csv",
    )
    t1 = time.time()
    d = icap_csv_columns_read_data(
        "/home/tom/Downloads/thermo/20200721_GelatineStdsLaserJacobTest.csv",
        use_analog=True,
    )
    t2 = time.time()
    print(t1 - t0, t2 - t1)
    print(d.shape)
    np.savetxt("/home/tom/Downloads/out.csv", d["172Yb"], delimiter=",")
