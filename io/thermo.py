import numpy as np

from .error import LaserLibException

from typing import Dict, List


def load(path: str) -> np.ndarray:
    """Imports iCap data exported using the CSV export function.

    Data is read from the "Counts" column.

    Args:
        path: Path to CSV

    Returns:
        Structured numpy array.

    Raises:
        LaserLibException

    """
    data: Dict[str, List[np.ndarray]] = {}
    with open(path, "r") as fp:
        cleaned = (
            line.replace(",", ";").replace("\t", ";").strip()
            for line in fp
            if "Counter" in line
        )

        for cline in cleaned:
            try:
                # Trim off last delimiter
                if cline.endswith(";"):
                    cline = cline[:-1]
                _, _, isotope, _, line_data = cline.split(";", 4)
                data.setdefault(isotope, []).append(
                    np.genfromtxt(
                        [line_data], delimiter=";", dtype=float, filling_values=0.0
                    )
                )
            except ValueError as e:
                raise LaserLibException("Could not parse file.") from e

    if not data:  # Data is empty
        raise LaserLibException("Empty data, nothing to import.")

    # Stack lines to form 2d
    keys = list(data.keys())
    dtype = [(k, float) for k in keys]
    structured = np.empty((data[keys[0]][0].shape[0], len(data[keys[0]])), dtype=dtype)
    for k in keys:
        stack = np.vstack(data[k][:]).transpose()
        if stack.ndim != 2:
            raise LaserLibException(f"Invalid data dimensions '{stack.ndim}'.")
        structured[k] = stack

    return structured
