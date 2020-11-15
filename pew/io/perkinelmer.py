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


def load(path: Union[str, Path], full: bool = False) -> np.ndarray:
    key_exchange = {
        "ablation.speed": "speed",
        "acquisition.time": "scantime",
        "space.interval": "spotsize",
    }
    if not isinstance(path, Path):
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

    parameters = path.joinpath("parameters.conf")
    if parameters.exists():
        try:
            with parameters.open() as fp:
                for line in fp:
                    if "=" in line:
                        k, v = line.split("=")
                        if k in key_exchange:
                            params[key_exchange[k.strip()]] = float(v.strip())
                        else:
                            params[k.strip()] = v.strip()
        except ValueError:  # pragma: no cover
            logger.warning("Parameters could not be read from parameters.conf.")

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
