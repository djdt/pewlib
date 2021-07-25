"""
Import and export in pew's custom file format, based on numpy's compressed '.npz'.
This format svaes image data, laser parameters and calibrations in one file.
"""
import time
from pathlib import Path

import numpy as np

from pewlib import __version__
from pewlib import Laser, Calibration, Config
from pewlib.config import SpotConfig

from pewlib.laser import Laser
from pewlib.srr import SRRLaser, SRRConfig

from typing import Dict, Union


def pack_info(info: Dict[str, str], sep: str = "\t") -> np.ndarray:
    string = sep.join(
        f"{key.replace(sep, ' ')}{sep}{val.replace(sep, ' ')}"
        for key, val in info.items()
        if key not in ["File Path"]
    )  # Makes no sense to store file path so drop it
    return np.array(string)


def unpack_info(info: np.ndarray, sep: str = "\t") -> Dict[str, str]:
    tokens = str(info).split(sep)
    return {key: val for key, val in zip(tokens[::2], tokens[1::2])}


def load(path: Union[str, Path]) -> Laser:
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

    if npz["_class"] in ["Laser", "Raster"]:
        laser = Laser
        config = Config.from_array(npz["config"])
    elif npz["_class"] in ["Spot"]:
        laser = Laser
        config = SpotConfig.from_array(npz["config"])
    elif npz["_class"] in ["SRRLaser", "SRR"]:
        laser = SRRLaser  # type: ignore
        config = SRRConfig.from_array(npz["config"])
    else:  # pragma: no cover
        raise ValueError("NPZ unable to import laser class {npz['_class']}.")

    if npz["_version"] < "0.7.0":  # Prior to use of info dict
        info = {"Name": str(npz["name"])}
    else:
        info = unpack_info(npz["info"])

    # Update the path
    info["Name"] = info.get("Name", path.stem)  # Ensure name
    info["File Path"] = str(path.resolve())

    return laser(
        data=data,
        calibration=calibration,
        config=config,
        info=info,
    )


def save(path: Union[str, Path], laser: Union[Laser, SRRLaser]) -> None:
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
    calibrations = {}
    for name in laser.calibration:
        calibrations[f"calibration_{name}"] = laser.calibration[name].to_array()
    np.savez_compressed(
        path,
        _version=__version__,
        _time=time.time(),
        _class=laser.config._class,
        _multiple=False,
        data=laser.data,
        info=pack_info(laser.info),
        config=laser.config.to_array(),
        **calibrations,
    )
