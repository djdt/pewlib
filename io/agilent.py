import os
import numpy as np

from .error import LaserLibException


def load(path: str) -> np.ndarray:
    """Imports an Agilent batch (.b) directory, returning LaserData object.

   Scans the given path for .d directories containg a similarly named
   .csv file. These are imported as lines, sorted by their name.

    Args:
       path: Path to the .b directory
       config: Config to be applied
       calibration: Calibration to be applied

    Returns:
        The LaserData object.

    Raises:
        PewPewFileError: Missing or malformed files.
        PewPewDataError: Invalid data.

    """
    data_files = []
    with os.scandir(path) as it:
        for entry in it:
            if entry.name.lower().endswith(".d") and entry.is_dir():
                csv = entry.name[: entry.name.rfind(".")] + ".csv"
                csv = os.path.join(entry.path, csv)
                if not os.path.exists(csv):
                    raise LaserLibException(f"Missing csv '{csv}'.")
                data_files.append(csv)
    # Sort by name
    data_files.sort()

    with open(data_files[0], "r") as fp:
        line = fp.readline()
        skip_header = 0
        while line and not line.startswith("Time [Sec]"):
            line = fp.readline()
            skip_header += 1

        skip_footer = 0
        if "Print" in fp.read().splitlines()[-1]:
            skip_footer = 1

    cols = np.arange(1, line.count(",") + 1)

    try:
        lines = [
            np.genfromtxt(
                f,
                delimiter=",",
                names=True,
                usecols=cols,
                skip_header=skip_header,
                skip_footer=skip_footer,
                dtype=np.float64,
            )
            for f in data_files
        ]
    except ValueError as e:
        raise LaserLibException("Could not parse batch.") from e

    try:
        data = np.vstack(lines)

    except ValueError as e:
        raise LaserLibException("Mismatched data.") from e

    return data
