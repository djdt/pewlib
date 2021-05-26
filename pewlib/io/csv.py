"""
Import of line-by-line data stored as a series of .csv files.
"""

from concurrent.futures import ProcessPoolExecutor
import logging
from pathlib import Path
import time
import re

import numpy as np
import numpy.lib.recfunctions as rfn

from typing import Any, List, Tuple, Union


logger = logging.getLogger(__name__)


class GenericOption(object):
    """Options for instrument specific csv imports.

    Options are used by :func:`pewlib.io.csv.load` to filter and sort paths,
    generate data and read parameters from csvs.

    Args:
        drop_names: columns dropped from imports
        kw_genfromtxt: kwargs for numpy.genfromtxt
        regex: regex string for matching filenames
    """

    def __init__(
        self,
        drop_names: List[str] = [],
        kw_genfromtxt: dict = None,
        regex: str = r".*\.csv",
        drop_nan_rows: bool = False,
        drop_nan_columns: bool = False,
        transposed: bool = False,
    ):
        self.drop_names = drop_names
        self.kw_genfromtxt = kw_genfromtxt
        self.regex = re.compile(regex, re.IGNORECASE)

        self.drop_nan_rows = drop_nan_rows
        self.drop_nan_columns = drop_nan_columns
        self.transposed = transposed

    def filter(self, paths: List[Path]) -> List[Path]:
        """Filter non matching paths."""
        return [path for path in paths if self.regex.match(path.name) is not None]

    def validForPath(self, path: Path) -> bool:
        """Checks if option is valid for a file or directory."""
        if path.is_dir():
            paths = (p for p in path.glob("*") if p.is_file())
            return any(self.regex.match(p.name) is not None for p in paths)
        else:
            return self.regex.match(path.name) is not None

    def readParams(self, data: np.ndarray) -> dict:
        """Read parameters from data."""
        return {}

    def sort(self, paths: List[Path]) -> List[Path]:
        """Sort paths using 'sortkey'."""
        return sorted(paths, key=self.sortkey)  # type: ignore

    def sortkey(self, path: Path) -> Any:
        return path


class NuOption(GenericOption):
    """Option for Nu Instruments data."""

    def __init__(self):
        super().__init__(drop_names=["X_(um)", "Y_(um)"], regex=r"acq.*\.csv")

    def readParams(self, data: np.ndarray) -> dict:
        if "Y_(um)" in data.dtype.names:
            return {"spotsize": np.round(np.mean(np.diff(data["Y_(um)"], axis=0)), 2)}
        else:  # pragma: no cover
            logger.warning("Y_(um) not found, unable to read spotsize.")
            return super().readParams(data)

    def sortkey(self, path: Path) -> int:
        """Sorts files numerically."""
        return int("".join(filter(str.isdigit, path.stem)) or -1)


class ThermoLDROption(GenericOption):
    """Option for Thermo iCAP LDR data."""

    def __init__(self):
        super().__init__(
            drop_names=["Time"],
            kw_genfromtxt={"skip_header": 13},
            regex=r"\w*_ldr_\d+\.csv",
            drop_nan_rows=True,
            drop_nan_columns=True,
        )

    def readParams(self, data: np.ndarray) -> dict:
        if "Time" in data.dtype.names:
            return {"scantime": np.round(np.mean(np.diff(data["Time"], axis=1)), 4)}
        else:  # pragma: no cover
            logger.warning("Time not found, unable to read scantime.")
            return super().readParams(data)

    def sortkey(self, path: Path) -> int:
        """Sorts files numerically."""
        return int("".join(filter(str.isdigit, path.stem)) or -1)


class TofwerkOption(GenericOption):
    """Option for TOFWERK data."""

    def __init__(self):
        super().__init__(
            drop_names=["t_elapsed_Buf"],
            kw_genfromtxt={"deletechars": "'"},
            regex=r"\w+?([0-9.]+-\d\dh\d\dm\d\ds).*\.csv",
        )

    def readParams(self, data: np.ndarray) -> dict:
        if "t_elapsed_Buf" in data.dtype.names:
            return {
                "scantime": np.round(np.mean(np.diff(data["t_elapsed_Buf"], axis=1)), 4)
            }
        else:  # pragma: no cover
            logger.warning("Y_(um) not found, unable to read spotsize.")
            return super().readParams(data)

    def sortkey(self, path: Path) -> float:
        """Sorts files using the timestamp in name."""
        match = self.regex.match(path.name)
        return time.mktime(time.strptime(match.group(1), "%Y.%m.%d-%Hh%Mm%Ss"))


def is_valid_directory(path: Union[str, Path]) -> bool:
    """Tests if a directory contains at least one csv."""
    if isinstance(path, str):  # pragma: no cover
        path = Path(path)

    if not path.exists() or not path.is_dir():
        return False

    return len(list(path.glob("*.csv"))) > 0


def option_for_path(path: Union[str, Path]) -> GenericOption:
    """Attempts to find the correct type hint for the directory.
    If no specific type hint is found then a GenericOption."""
    options = [NuOption(), ThermoLDROption(), TofwerkOption()]

    if isinstance(path, str):  # pragma: no cover
        path = Path(path)

    return next(
        (op for op in options if op.validForPath(path)),
        GenericOption(),
    )


def load(
    path: Union[str, Path],
    option: GenericOption = None,
    full: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """Load a directory where lines are stored in separate .csv files.

    Paths are filtered and sorted according to the `option` used, defaulting
    to the value of :func:`pewlib.io.csv.option_for_path`.

    Args:
        path: directory
        hint: type hint (NuHint, TofwerkHint)
        genfromtxtkws: kwargs for numpy.genfromtxt
        full: also return parameters

    Returns:
        structured array of data
        dict of params if `full`

    See Also:
        :class:`pewlib.io.csv.GenericOption`
        :func:`numpy.genfromtxt`
    """

    if isinstance(path, str):  # pragma: no cover
        path = Path(path)

    if option is None:
        option = option_for_path(path)

    kwargs = dict(delimiter=",", deletechars="", names=True, dtype=np.float64)
    if option.kw_genfromtxt is not None:
        kwargs.update(option.kw_genfromtxt)

    paths = list(
        p for p in path.glob("*") if p.is_file() and not p.name.startswith(".")
    )

    if len(paths) == 0:  # pragma: no cover
        raise ValueError(f"No csv files found with '{option.regex}' in {path.name}!")

    paths = option.filter(paths)
    paths = option.sort(paths)

    with ProcessPoolExecutor() as execuctor:
        futures = [execuctor.submit(np.genfromtxt, path, **kwargs) for path in paths]
    lines = [future.result() for future in futures]

    length = min(line.size for line in lines)
    data = np.stack([line[:length] for line in lines], axis=0)

    if option.transposed:  # pragma: no cover
        data = data.T

    if option.drop_nan_rows:
        mask = np.all(np.isnan(rfn.structured_to_unstructured(data)), axis=(0, 2))
        rows = np.flatnonzero(mask)
        data = np.delete(data, rows, 1)

    if option.drop_nan_columns:
        column_drop_names = []
        for name in data.dtype.names:
            if np.all(np.isnan(data[name])):
                column_drop_names.append(name)
        data = rfn.drop_fields(data, column_drop_names)

    if full:
        params = option.readParams(data)

    data = rfn.drop_fields(data, option.drop_names)

    if full:
        return data, params
    else:  # pragma: no cover
        return data
