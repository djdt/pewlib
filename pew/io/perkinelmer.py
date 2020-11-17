import logging

import numpy as np
import numpy.lib.recfunctions

from pathlib import Path

from typing import List, Union

logger = logging.getLogger(__name__)


def collect_datafiles(path: Path) -> List[Path]:
    datafiles = []

    for child in path.iterdir():
        if child.suffix == ".xl":
            datafiles.append(child)

    # Sort by any numerical order
    datafiles.sort(key=lambda f: int("".join(filter(str.isdigit, f.name))))
    return datafiles


def load(
    path: Union[str, Path], import_parameters: bool = True, full: bool = False
) -> np.ndarray:
    param_conversion = {
        "ablation.speed": ("speed", 1e3),
        "acquisition.time": ("scantime", 1e3),
        "space.interval": ("spotsize", 1.0),
    }
    if not isinstance(path, Path):  # pragma: no cover
        path = Path(path)

    datafiles = collect_datafiles(path)

    data = np.vstack(
        [
            np.genfromtxt(df, skip_header=1, delimiter=",", names=True)
            for df in datafiles
        ]
    )
    data = numpy.lib.recfunctions.drop_fields(data, "Time_in_Seconds")
    params: dict = {"origin": (0.0, 0.0)}

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
                params[new] = params.pop(old) * mult

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
