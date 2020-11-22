"""
Import and export in pew's custom file format, based on numpy's compressed '.npz'.
This format svaes image data, laser parameters and calibrations in one file.
"""
import time
from pathlib import Path

import numpy as np

from pewlib import __version__
from pewlib import Laser, Calibration, Config

from pewlib.laser import _Laser
from pewlib.srr import SRRLaser, SRRConfig

from typing import Union


def load(path: Union[str, Path]) -> _Laser:
    """Loads data from '.npz' file.

    Loads files created using :func:`pewlib.io.npz.save`.
    On load the a :class:`Laser` or :class:`SRRLaser` is reformed from the saved data.

    Args:
        path: path to '.npz'

    Returns:
        :class:`Laser` or :class:`SRRLaser`

    Raises:
        ValueError: incomatible version

    See Also:
        :func:`numpy.load`
    """
    if isinstance(path, str):  # pragma: no cover
        path = Path(path)

    npz = np.load(path)

    if "_version" not in npz.files or npz["_version"] < "0.6.0":  # pragma: no cover
        raise ValueError("NPZ Version mismatch, only versions >=0.6.0 are supported.")
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
        raise ValueError("NPZ unable to import laser class {npz['_class']}.")

    return laser(
        data=data,
        calibration=calibration,
        config=config,
        name=str(npz["name"]),
        path=path,
    )


def save(path: Union[str, Path], laser: _Laser) -> None:
    """Saves data to '.npz' file.

    Converts a :class:`Laser` or :class:`SRRLaser` to a series of `np.ndarray`
    which are then saved to a compressed '.npz' archive. The time and current
    version are also saved. If `path` does not end in '.npz' it is
    appended.

    Args:
        path: path to save to
        laser: :class:`Laser` or :class:`SRRLaser`

    See Also:
        :func:`numpy.savez_compressed`
    """
    savedict: dict = {"_version": __version__, "_time": time.time(), "_multiple": False}
    savedict["_class"] = laser.__class__.__name__
    savedict["data"] = laser.data
    savedict["name"] = laser.name
    savedict["config"] = laser.config.to_array()
    for name in laser.calibration:
        savedict[f"calibration_{name}"] = laser.calibration[name].to_array()
    np.savez_compressed(path, **savedict)
