import numpy as np
from pathlib import Path

from pew.io.error import PewException

from typing import Union


def load(
    path: Union[str, Path], delimiter: str = None, comments: str = "#", name: str = None
) -> np.ndarray:
    if isinstance(path, str):  # pragma: no cover
        path = Path(path)

    try:
        if delimiter is None:
            with path.open("r") as fp:
                gen = (line.replace(";", ",").replace("\t", ",") for line in fp)
                data = np.genfromtxt(
                    gen, delimiter=",", comments=comments, dtype=np.float64
                )
        else:
            data = np.genfromtxt(
                path, delimiter=delimiter, comments=comments, dtype=np.float64
            )
    except ValueError as e:  # pragma: no cover
        raise PewException("Could not parse file.") from e
    if data.ndim != 2:  # pragma: no cover
        raise PewException(f"Invalid data dimensions '{data.ndim}'.")

    if name is not None:
        data.dtype = [(name, data.dtype)]
    return data


def save(path: Union[str, Path], data: np.ndarray, header: str = "") -> None:
    np.savetxt(path, data, delimiter=",", comments="#", header=header)
