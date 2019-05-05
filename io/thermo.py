import numpy as np

from .error import LaserLibException

from typing import Dict, List


def load(path: str) -> np.ndarray:
    """Imports iCap data exported using the CSV export function.

    Data is read from the "Counts" column.

    Args:
        path: Path to CSV
        config: Config to apply
        calibration: Calibration to apply

    Returns:
        The LaserData object.

    Raises:
        PewPewFileError: Unreadable file.
        PewPewDataError: Invalid data.

    """
    data: Dict[str, List[np.ndarray]] = {}
    with open(path, "r") as fp:
        cleaned = (
            line.replace(",", ";").replace("\t", ";").strip().rstrip(";")
            for line in fp
            if "Counter" in line
        )

        for cline in cleaned:
            try:
                _, _, isotope, _, line_data = cline.split(";", 4)
                data.setdefault(isotope, []).append(
                    np.genfromtxt(
                        [line_data], delimiter=";", dtype=float, filling_values=0.0
                    )
                )
            except ValueError as e:
                raise LaserLibException("Could not parse file.") from e

    keys = list(data.keys())
    # Stack lines to form 2d
    dtype = [(k, float) for k in keys]
    structured = np.empty((data[keys[0]][0].shape[0], len(data[keys[0]])), dtype)
    for k in keys:
        stack = np.vstack(data[k][:]).transpose()
        if stack.ndim != 2:
            raise LaserLibException(f"Invalid data dimensions '{stack.ndim}'.")
        structured[k] = stack

    return structured


# def load_ldr(path: str, config: dict, calibration: dict = None) -> Laser:
#     """Imports data exported using \"Laser Data Reduction\".
#     CSVs in the given directory are imported as
#     lines in the image and are sorted by name.

#     path -> path to directory containing CSVs
#     config -> config to apply
#     calibration -> calibration to apply

#     returns LaserData"""
#     data_files = []
#     with os.scandir(path) as it:
#         for entry in it:
#             if entry.name.lower().endswith(".csv") and entry.is_file():
#                 data_files.append(entry.path)
#     # Sort by name
#     data_files.sort()

#     with open(data_files[0], "r") as fp:
#         line = fp.readline()
#         skip_header = 0
#         while line and not line.startswith("Time"):
#             line = fp.readline()
#             skip_header += 1

#         delimiter = line[-1]

#     cols = np.arange(1, line.count(delimiter))

#     try:
#         lines = [
#             np.genfromtxt(
#                 f,
#                 delimiter=delimiter,
#                 names=True,
#                 usecols=cols,
#                 skip_header=skip_header,
#                 dtype=np.float64,
#             )
#             for f in data_files
#         ]
#     except ValueError as e:
#         raise PewPewFileError("Could not parse batch.") from e
#     # We need to skip the first row as it contains junk
#     try:
#         data = np.vstack(lines)[1:]
#     except ValueError as e:
#         raise PewPewDataError("Mismatched data.") from e

#     return LaserData(data, config=config, calibration=calibration, source=path)
