import numpy as np

from pew.io.error import PewException


def load(
    path: str, delimiter: str = None, comments: str = "#", name: str = None
) -> np.ndarray:
    try:
        if delimiter is None:
            with open(path, "r") as fp:
                gen = [line.replace(";", ",").replace("\t", ",") for line in fp]
                data = np.genfromtxt(
                    gen, delimiter=",", comments=comments, dtype=np.float64
                )
        else:
            data = np.genfromtxt(
                path, delimiter=delimiter, comments=comments, dtype=np.float64
            )
    except ValueError as e:
        raise PewException("Could not parse file.") from e
    if data.ndim != 2:
        raise PewException(f"Invalid data dimensions '{data.ndim}'.")

    if name is not None:
        data.dtype = [(name, data.dtype)]
    return data


def save(path: str, data: np.ndarray, header: str = "") -> None:
    np.savetxt(path, data, delimiter=",", comments="#", header=header)
