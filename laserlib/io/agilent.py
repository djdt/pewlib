import os
import numpy as np
import numpy.lib
import numpy.lib.recfunctions

from .error import LaserLibException


def load(path: str) -> np.ndarray:
    """Imports an Agilent batch (.b) directory, returning LaserData object.

   Scans the given path for .d directories containg a similarly named
   .csv file. These are imported as lines, sorted by their name.

    Args:
       path: Path to the .b directory

    Returns:
        The structured numpy array.

    Raises:
        LaserLibException

    """
    ddirs = []
    with os.scandir(path) as it:
        for entry in it:
            if entry.name.lower().endswith(".d") and entry.is_dir():
                ddirs.append(entry.path)

    csvs = []
    # Sort by name
    ddirs.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    for d in ddirs:
        csv = os.path.splitext(os.path.basename(d))[0] + ".csv"
        csv = os.path.join(d, csv)
        if not os.path.exists(csv):
            raise LaserLibException(f"Missing csv '{csv}'.")
        csvs.append(csv)

    def get_clean_lines(csv: str):
        past_header = False
        with open(csv, "rb") as fp:
            for line in fp:
                if past_header and b"," in line:
                    yield line
                if line.startswith(b"Time"):
                    past_header = True
                    yield line

    datas = []
    for csv in csvs:
        try:
            datas.append(np.genfromtxt(
                get_clean_lines(csv),
                delimiter=b",",
                names=True,
                dtype=np.float64
            ))
        except ValueError as e:
            raise LaserLibException("Could not parse batch.") from e

    try:
        data = np.vstack(datas)
        # We don't care about the time field currently
        data = np.lib.recfunctions.drop_fields(data, "Time_Sec")

    except ValueError as e:
        raise LaserLibException("Mismatched data.") from e

    return data
