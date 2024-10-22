"""
Import and export of text-images, files where data is stored as delimited text values.
Data is read in order from the first line.
"""

from pathlib import Path

import numpy as np


def load(
    path: str | Path,
    delimiter: str | None = None,
    comments: str = "#",
    name: str | None = None,
) -> np.ndarray:
    """Load text-image.

    Loads 2d data from file. If `delimiter` is specified then all tab and ';'
    are converted to ',' before import. If `name` is specified then a single
    field structured array is returned.

    Args:
        path: path to file
        delimiter: file delimiter
        comments: file comment character
        name: return single `name` field structured array
    """
    if isinstance(path, str):  # pragma: no cover
        path = Path(path)

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

    data = np.atleast_2d(data)

    if name is not None:
        data.dtype = [(name, data.dtype)]
    return data


def save(path: str | Path, data: np.ndarray, header: str = "") -> None:
    """Save data to csv.

    See :func:`numpy.savetxt`

    Args:
        path: path to file
        data: unstructured array
        header: file header
    """
    if isinstance(path, str):  # pragma: no cover
        path = Path(path)
    np.savetxt(path, data, delimiter=",", comments="#", header=header, fmt="%.18g")
