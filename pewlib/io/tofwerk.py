from concurrent.futures import ProcessPoolExecutor
import logging
import numpy as np
import numpy.lib.recfunctions as rfn
from pathlib import Path
import re
import time

from typing import List, Tuple, Union

logger = logging.getLogger(__name__)


acquistion_regex = re.compile(r"(\w+?)([0-9.]+-\d\dh\d\dm\d\ds).*\.csv")


def is_valid_directory(path: Union[str, Path]) -> bool:
    """Tests if a directory contains TOFWERK data.

    Ensures the path exists, is a directory and contains a File_Report and at
    least one acquistion '.csv'.
    """
    if isinstance(path, str):
        path = Path(path)

    if not path.exists() or not path.is_dir():
        return False

    paths = list(path.glob("*.csv"))
    return any(acquistion_regex.match(p.name) for p in paths)


def acquistion_name(path: Path) -> str:
    m = acquistion_regex.match(path.name)
    if m is None:
        raise ValueError(f"Invalid acquistion name '{path.name}'")

    return m.group(1)


def acquistion_time_seconds(path: Path) -> float:
    m = acquistion_regex.match(path.name)
    if m is None:
        raise ValueError(f"Invalid acquistion time '{path.name}'")

    return time.mktime(time.strptime(m.group(2), "%Y.%m.%d-%Hh%Mm%Ss"))


def read_acqusition(path: Path) -> np.ndarray:
    return np.genfromtxt(
        path, delimiter=",", deletechars="", names=True, dtype=np.float64
    )


def load(
    path: Union[str, Path],
    drop_names: List[str] = None,
    full: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """Load a Nu instruments data directory.

    The directory must contain at least one acquistion CSV and a File_Report,
    this can be checked using :func:`pewlib.io.nu.is_valid_directory`.
    Names passed to `drop_names` as removed from the final array, the default
    is to drop 'Y_(um)' and 'X_(um)'.
    If `full` then a dict with the spotsize is also returned.

    Args:
        path: directory
        drop_names: names removed from data, default removes positions.
        full: also return parameters

    Returns:
        structured array of data
        dict of params if `full`
    """
    if isinstance(path, str):
        path = Path(path)

    if drop_names is None:
        drop_names = ["t_elapsed_Buf"]

    # Read report
    acq_paths = sorted(path.glob("*.csv"), key=lambda p: acquistion_time_seconds(p))

    acq_name = acquistion_name(acq_paths[0])
    for name in [acquistion_name(p) for p in acq_paths]:
        if name != acq_name:
            raise ValueError(
                f"Directory contains mixed acquistions! {name}, {acq_name}"
            )

    # Multithreaded read greatly improves load times
    with ProcessPoolExecutor() as execuctor:
        results = [execuctor.submit(read_acqusition, p) for p in acq_paths]
        data = np.stack([r.result() for r in results], axis=1)

    params = {}
    if full:
        if "t_elapsed_Buf" in data.dtype.names:
            params["scantime"] = np.round(
                np.mean(np.diff(data["t_elapsed_Buf"], axis=1)), 4
            )
        else:
            logger.warning(
                "'t_elapsed_Buf' field not found, unable to import scantime."
            )

    data = rfn.drop_fields(data, drop_names)

    if full:
        return data, params
    else:
        return data
