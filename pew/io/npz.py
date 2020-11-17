import time
from pathlib import Path

import numpy as np

from pew import __version__
from pew import Laser, Calibration, Config

from pew.laser import _Laser
from pew.io.error import PewException
from pew.srr import SRRLaser, SRRConfig

from typing import Union


def load(path: Union[str, Path]) -> _Laser:
    if isinstance(path, str):  # pragma: no cover
        path = Path(path)

    npz = np.load(path)

    if "_version" not in npz.files or npz["_version"] < "0.6.0":  # pragma: no cover
        raise PewException("NPZ Version mismatch, only versions >=0.6.0 are supported.")
    data = npz["data"]

    calibration = {}
    for name in data.dtype.names:
        calibration[name] = Calibration.from_array(npz[f"calibration_{name}"])

    if npz["_class"] == "Laser":
        laser = Laser
        config = Config.from_array(npz["config"])
    elif npz["_class"] == "SRRLaser":
        laser = SRRLaser  # type: ignore
        config = SRRConfig.from_array(npz["config"])
    else:  # pragma: no cover
        raise PewException("NPZ unable to import laser class {npz['_class']}.")

    return laser(
        data=data,
        calibration=calibration,
        config=config,
        name=str(npz["name"]),
        path=str(path.resolve()),
    )


def save(path: Union[str, Path], laser: Laser) -> None:
    savedict: dict = {"_version": __version__, "_time": time.time(), "_multiple": False}
    savedict["_class"] = laser.__class__.__name__
    savedict["data"] = laser.data
    savedict["name"] = laser.name
    savedict["config"] = laser.config.to_array()
    for name in laser.calibration:
        savedict[f"calibration_{name}"] = laser.calibration[name].to_array()
    np.savez_compressed(path, **savedict)
