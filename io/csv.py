import numpy as np

from .error import LaserLibException


def load(path: str, isotope: str = "CSV") -> np.ndarray:
    with open(path, "rb") as fp:
        cleaned = (line.replace(b";", b",").replace(b"\t", b",") for line in fp)
        try:
            data = np.genfromtxt(cleaned, delimiter=b",", comments=b"#", dtype=float, loose=False)
        except ValueError as e:
            raise LaserLibException("Could not parse file.") from e
    if data.ndim != 2:
        raise LaserLibException(f"Invalid data dimensions '{data.ndim}'.")

    data.dtype = [(isotope, data.dtype)]
    return data


def save(path: str, data: np.ndarray, header: str = "") -> None:
    np.savetxt(path, data, delimiter=",", comments="#", header=header)
