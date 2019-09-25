import numpy as np

from .. import __version__

from .error import PewException

from typing import Any, Dict, List
from .. import Laser, Calibration, Config, IsotopeData
from ..srr import SRR, KrissKrossConfig, KrissKrossData


def load(path: str) -> List[Laser]:
    """Imports the given numpy archive given, returning a list of data.

    Both the config and calibration read from the archive may be overriden.

    Args:
        path: Path to numpy archive
        config_override: If not None will be applied to all imports
        calibration_override: If not None will be applied to all imports

    Returns:
        List of IsotopeData and SRRData

    Raises:
        PewPewFileError: Version of archive missing or incompatable.

    """
    lasers: List[Laser] = []
    npz = np.load(path, allow_pickle=True)

    if "version" not in npz.files:
        raise PewException("Archive version mismatch.")
    elif npz["version"] < "0.1.1":
        raise PewException(f"Archive version mismatch: {npz['version']}.")

    for f in npz.files:
        if f == "version":
            continue
        data: Dict[str, IsotopeData] = {}
        laserdict: Dict[str, Any] = npz[f].item()
        # Config
        if laserdict["type"] == "SRR":
            config = SRRConfig()
        else:
            config = Config()
        for k, v in laserdict["config"].items():
            setattr(config, k, v)

        # Calibration and data
        for isotope in laserdict["data"].keys():
            calibration = Calibration()
            for k, v in laserdict["calibration"][isotope].items():
                if npz["version"] < "0.1.5" and k == "points":
                    k = "_points"
                setattr(calibration, k, v)

            if laserdict["type"] == "SRR":
                data[isotope] = SRRData(laserdict["data"][isotope], calibration)
            else:
                data[isotope] = IsotopeData(laserdict["data"][isotope], calibration)

        if laserdict["type"] == "SRR":
            laser = SRR(
                data=data, config=config, name=laserdict["name"], path=path
            )
        else:
            laser = Laser(
                data=data, config=config, name=laserdict["name"], path=path
            )

        lasers.append(laser)

    return lasers


def save(path: str, laser_list: List[Laser]) -> None:
    savedict: Dict[str, Any] = {"version": __version__}
    for laser in laser_list:
        laserdict = {
            "type": laser.__class__.__name__,
            "name": laser.name,
            "config": laser.config.__dict__,
            "data": {k: v.data for k, v in laser.data.items()},
            "calibration": {k: v.calibration.__dict__ for k, v in laser.data.items()},
        }
        name = laser.name
        if name in savedict:
            i = 0
            while f"{name}{i}" in savedict:
                i += 1
            name += f"{name}{i}"
        savedict[name] = laserdict
    np.savez_compressed(path, **savedict)
