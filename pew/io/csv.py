import numpy as np

from .error import PewException


def load(path: str, isotope: str = None) -> np.ndarray:
    with open(path, "rb") as fp:
        cleaned = (line.replace(b";", b",").replace(b"\t", b",") for line in fp)
        try:
            data = np.genfromtxt(
                cleaned, delimiter=b",", comments=b"#", dtype=float, loose=False
            )
        except ValueError as e:
            raise PewException("Could not parse file.") from e
    if data.ndim != 2:
        raise PewException(f"Invalid data dimensions '{data.ndim}'.")

    if isotope is not None:
        data.dtype = [(isotope, data.dtype)]
    return data


def save(path: str, data: np.ndarray, header: str = "") -> None:
    np.savetxt(path, data, delimiter=",", comments="#", header=header)
