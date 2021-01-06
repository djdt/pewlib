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


class CsvTypeHint(object):
    """Contains hints for instrument specific imports.

    These hints are used to filter and sort the .csv files in an imported directory,
    provide kwargs to numpy.genfromtxt and read parameters from imported data.

    `drop_names` determine columns to be dropped from the final import.
    `kw_genfromtxt` are kwargs that should be passed to numpy.genfromtxt.
    `regex` is a regex str that should match file names.
    """

    def __init__(
        self,
        drop_names: List[str] = [],
        kw_genfromtxt: dict = None,
        regex: str = r".*\.csv",
    ):
        self.drop_names = drop_names
        self.kw_genfromtxt = kw_genfromtxt or {}
        self.regex = re.compile(regex, re.IGNORECASE)

    def filter(self, paths: List[Path]) -> List[Path]:
        return [path for path in paths if self.regex.match(path.name) is not None]

    def directoryIsType(self, path: Path) -> bool:
        return any(self.regex.match(p.name) is not None for p in path.glob("*.csv"))

    def readParams(self, data: np.ndarray) -> dict:
        return {}

    def sort(self, paths: List[Path]) -> List[Path]:
        return sorted(paths, key=self.sortkey)  # type: ignore

    def sortkey(self, path: Path) -> Any:
        return path


class NuHint(CsvTypeHint):
    def __init__(self):
        super().__init__(drop_names=["X_(um)", "Y_(um)"], regex=r"acq.*\.csv")

    def readParams(self, data: np.ndarray) -> dict:
        if "Y_(um)" in data.dtype.names:
            return {"spotsize": np.round(np.mean(np.diff(data["Y_(um)"], axis=0)), 2)}
        logger.warning("Y_(um) not found, unable to read spotsize.")
        return super().readParams(data)

    def sortkey(self, path: Path) -> int:
        return int("".join(filter(str.isdigit, path.stem)))


class TofwerkHint(CsvTypeHint):
    def __init__(self):
        super().__init__(
            drop_names=["t_elapsed_Buf"],
            kw_genfromtxt={"deletechars": "'"},
            regex=r"(\w+?)([0-9.]+-\d\dh\d\dm\d\ds).*\.csv",
        )

    def readParams(self, data: np.ndarray) -> dict:
        if "t_elapsed_Buf" in data.dtype.names:
            return {
                "scantime": np.round(np.mean(np.diff(data["t_elapsed_Buf"], axis=0)), 4)
            }
        logger.warning("t_elapsed_Buf not found, unable to read scantime.")
        return super().readParams(data)

    def sortkey(self, path: Path) -> float:
        match = self.regex.match(path.name)
        return time.mktime(time.strptime(match.group(2), "%Y.%m.%d-%Hh%Mm%Ss"))


def is_valid_directory(path: Union[str, Path]) -> bool:
    """Tests if a directory contains at least one csv."""
    if isinstance(path, str):
        path = Path(path)

    if not path.exists() or not path.is_dir():
        return False

    return len(list(path.glob("*.csv"))) > 0


def load(
    path: Union[str, Path],
    hint: CsvTypeHint = None,
    kw_genfromtxt: dict = None,
    full: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """Load a directory where lines are stored in separate .csv files.

    Paths are filtered and sorted according to the `hint` used. The default hint
    is generated from the file names within the directory.
    Kwargs in `kw_genfromtxt` are passed directly to :func:`numpy.genfromtxt`
    and override any previous default or hint kwargs.

    Args:
        path: directory
        hint: type hint (NuHint, TofwerkHint)
        genfromtxtkws: kwargs for numpy.genfromtxt
        full: also return parameters

    Returns:
        structured array of data
        dict of params if `full`

    See Also:
        :class:`pewlib.io.csv.CsvTypeHint`
        :func:`numpy.genfromtxt`
    """
    if isinstance(path, str):
        path = Path(path)

    if hint is None:
        hint = next(
            (t for t in [NuHint(), TofwerkHint()] if t.directoryIsType(path)),
            CsvTypeHint(),
        )

    kwargs = dict(delimiter=",", deletechars="", names=True, dtype=np.float64)
    kwargs.update(hint.kw_genfromtxt)
    if kw_genfromtxt is not None:
        kwargs.update(kw_genfromtxt)

    paths = list(path.glob("*.csv"))

    if len(paths) == 0:
        raise ValueError(f"No csv files found with '{hint.regex}' in {path.name}!")

    paths = hint.filter(paths)
    paths = hint.sort(paths)

    with ProcessPoolExecutor() as execuctor:
        futures = [execuctor.submit(np.genfromtxt, path, **kwargs) for path in paths]

    lines = [future.result() for future in futures]

    length = min(line.size for line in lines)
    data = np.stack((line[:length] for line in lines), axis=1)

    if full:
        params = hint.readParams(data)

    drop_names = hint.drop_names

    data = rfn.drop_fields(data, drop_names)

    if full:
        return data, params
    else:
        return data