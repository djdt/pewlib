import numpy as np

from pew import __version__

from pew.io.error import PewException

from typing import Any, Dict, List
from pew.laser import _Laser
from pew import Laser, Calibration, Config
from pew.srr import SRRLaser, SRRConfig


def load(path: str) -> List[_Laser]:
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
    lasers: List[_Laser] = []
    npz = np.load(path, allow_pickle=True)

    if "version" not in npz.files:
        raise PewException("Archive version mismatch.")
    elif npz["version"] < "0.2.5":
        raise PewException(f"Archive version mismatch: {npz['version']}.")

    for f in npz.files:
        if f == "version":
            continue
        laserdict: Dict[str, Any] = npz[f].item()
        # Config
        if laserdict["type"] == "SRRLaser":
            config: Config = SRRConfig()
        else:
            config = Config()
        for k, v in laserdict["config"].items():
            setattr(config, k, v)

        # Calibration
        calibration = {}
        for name, cal in laserdict["calibration"].items():
            calibration[name] = Calibration()
            for k, v in cal.items():
                setattr(calibration[name], k, v)

        if laserdict["type"] == "Laser":
            lasers.append(
                Laser(
                    data=laserdict["data"],
                    calibration=calibration,
                    config=config,
                    name=laserdict["name"],
                    path=path,
                )
            )
        elif laserdict["type"] == "SRRLaser":
            assert isinstance(config, SRRConfig)
            lasers.append(
                SRRLaser(
                    data=laserdict["data"],
                    calibration=calibration,
                    config=config,
                    name=laserdict["name"],
                    path=path,
                )
            )
        else:
            raise PewException("Unknown Laser type.")

    return lasers


def save(path: str, laser_list: List[Laser]) -> None:
    savedict: Dict[str, Any] = {"version": __version__}
    for laser in laser_list:
        laserdict = {
            "type": laser.__class__.__name__,
            "name": laser.name,
            "config": laser.config.__dict__,
            "calibration": {k: v.__dict__ for k, v in laser.calibration.items()},
            "data": laser.data,
        }
        name = laser.name
        if name in savedict:
            i = 0
            while f"{name}{i}" in savedict:
                i += 1
            name += f"{name}{i}"
        savedict[name] = laserdict
    np.savez_compressed(path, **savedict)
