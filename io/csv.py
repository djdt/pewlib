import numpy as np

from .error import LaserLibException


def load(path: str) -> np.ndarray:
    with open(path, "rb") as fp:
        try:
            cleaned = (line.replace(b";", b",").replace(b"\t", b",") for line in fp)
            data = np.genfromtxt(cleaned, delimiter=b",", comments=b"#", dtype=float)
        except ValueError as e:
            raise LaserLibException("Could not parse file.") from e

        if data.ndim != 2:
            raise LaserLibException(f"Invalid data dimensions '{data.ndim}'.")

    data.dtype = [("Unknown", data.dtype)]
    return data


def save(path: str, data: np.ndarray) -> None:
    np.savetxt(path, data, fmt="%g", delimiter=",")
