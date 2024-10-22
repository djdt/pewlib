"""
Import of line-by-line PerkinElmer ELAN 'XL' directories.
"""
import logging
from pathlib import Path

import numpy as np
import numpy.lib.recfunctions

logger = logging.getLogger(__name__)


def is_valid_directory(path: str | Path) -> bool:
    """Tests if a directory contains PerkinElmer data.

    Ensures the path exists, is a directory and contains at least one '.xl' file.
    """
    if isinstance(path, str):  # pragma: no cover
        path = Path(path)

    if not path.exists() or not path.is_dir():
        return False

    return len(list(path.glob("*.xl"))) > 0


def load(
    path: str | Path, import_parameters: bool = True, full: bool = False
) -> np.ndarray| tuple[np.ndarray, dict]:
    """Loads PerkinElmer directory.

    Searches the directory `path` for '.xl' files and used them to reconstruct data.
    If `import_parameters` and a 'parameters.conf' is used then the scantime,
    speed and spotsize can be imported.

    Args:
        path: path to directory
        import_parameters: import params from 'parameters.conf'
        full: also return dict with params

    Returns:
        structured array of data
        dict of params if `full`

    See Also:
        :func:`pewlib.io.perkinelmer.collect_datafiles`
    """
    param_conversion = {
        "ablation.speed": ("speed", 1e3),
        "acquisition.time": ("scantime", 1.0),
        "space.interval": ("spotsize", 1e3),
    }
    if not isinstance(path, Path):  # pragma: no cover
        path = Path(path)

    datafiles = sorted(
        path.glob("*.xl"), key=lambda p: int("".join(filter(str.isdigit, p.stem)))
    )

    data = np.stack(
        [
            np.genfromtxt(df, skip_header=1, delimiter=",", names=True, deletechars="")
            for df in datafiles
        ],
        axis=1,
    )
    params: dict = {"origin": (0.0, 0.0), "times": data["Time_in_Seconds"]}
    data = numpy.lib.recfunctions.drop_fields(data, "Time_in_Seconds")

    if import_parameters:
        parameters = path.joinpath("parameters.conf")
        if parameters.exists():
            try:
                with parameters.open() as fp:
                    for line in fp:
                        if "=" in line:
                            k, v = line.split("=")
                            params[k.strip()] = v.strip()
            except ValueError:  # pragma: no cover
                logger.warning("Parameters could not be read from parameters.conf.")

        for old, (new, mult) in param_conversion.items():
            if old in params:
                params[new] = float(params.pop(old)) * mult

    # positions = path.joinpath("positions.txt")
    # if positions.exists():
    #     np.genfromtxt(
    #         (line for line in positions.open() if "," in line),
    #         delimiter=",",
    #         dtype=float,
    #     )

    if full:
        return data, params
    else:  # pragma: no cover
        return data
