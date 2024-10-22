"""
Import and export in pew's custom file format, based on numpy's compressed '.npz'.
This format svaes image data, laser parameters and calibrations in one file.
"""

import logging
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from pewlib import Calibration, Config, Laser
from pewlib.config import SpotConfig
from pewlib.srr import SRRConfig, SRRLaser

logger = logging.getLogger(__name__)


def compare_version(va: str, vb: str) -> int:

    for a, b in zip(va.split("."), vb.split(".")):
        if int(a) > int(b):
            return 1
        elif int(a) < int(b):
            return -1
    return 0


def pack_info(
    info: dict[str, str], sep: str = "\t", remove_keys: list[str] | None = None
) -> np.ndarray:
    if remove_keys is None:
        remove_keys = ["File Path"]
    string = sep.join(
        f"{key.replace(sep, ' ')}{sep}{val.replace(sep, ' ')}"
        for key, val in info.items()
        if key not in remove_keys
    )  # Makes no sense to store file path so drop it
    return np.array(string)


def unpack_info(x: np.ndarray, sep: str = "\t") -> dict[str, str]:
    tokens = str(x).split(sep)
    return {key: val for key, val in zip(tokens[::2], tokens[1::2])}


def pack_calibration(dict: dict[str, Calibration]) -> np.ndarray:
    size = max(v.x.size for v in dict.values())
    data = np.stack([v.to_array(size=size) for v in dict.values()])
    elements = np.array([k for k in dict.keys()])
    # recfunctions.append_fields does not like 0 length,
    # i.e. when no calibration with points
    packed = np.empty(
        data.size, dtype=[("element", elements.dtype), ("calibration", data.dtype)]
    )
    packed["element"] = elements
    packed["calibration"] = data
    return packed


def unpack_calibration(x: np.ndarray) -> dict[str, Calibration]:
    calibration = {i["element"]: Calibration.from_array(i["calibration"]) for i in x}
    return calibration


def load(path: str | Path) -> Laser:
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

    if "header" not in npz.files:
        if (
            "_version" not in npz.files
            or compare_version(str(npz["_version"]), "0.6.0") == -1
        ):  # pragma: no cover
            raise ValueError(
                "NPZ Version mismatch, only versions >=0.6.0 are supported."
            )
        else:  # < 0.8.0 Prior to use of header
            header = {"version": npz["_version"], "class": npz["_class"]}
    else:
        header = unpack_info(npz["header"])

    data = npz["data"]

    # Compatibility with old file verions
    if (
        compare_version(str(header["version"]), "0.7.0") == -1
    ):  # Prior to use of info dict
        info = {"Name": str(npz["name"])}
    else:
        info = unpack_info(npz["info"])

    if (
        compare_version(str(header["version"]), "0.8.0") == -1
    ):  # Prior to use of packed calibrations
        calibration = {}
        for name in data.dtype.names:
            calibration[name] = Calibration.from_array(npz[f"calibration_{name}"])
    else:
        calibration = unpack_calibration(npz["calibration"])

    # if header["version"] < version("pewlib"):
    #     logger.info(
    #         f"NPZ version of {path} is out of date. {header['version']} < 0.8.0."
    #     )

    if header["class"] in ["Laser", "Raster"]:
        laser = Laser
        config = Config.from_array(npz["config"])
    elif header["class"] in ["Spot"]:
        laser = Laser
        config = SpotConfig.from_array(npz["config"])
    elif header["class"] in ["SRRLaser", "SRR"]:
        laser = SRRLaser  # type: ignore
        config = SRRConfig.from_array(npz["config"])
    else:  # pragma: no cover
        raise ValueError("NPZ unable to import laser class {npz['_class']}.")

    # Update the path
    info["Name"] = info.get("Name", path.stem)  # Ensure name
    info["File Path"] = str(path.resolve())
    info["File Version"] = str(header["version"])

    return laser(
        data=data,
        calibration=calibration,
        config=config,  # type: ignore
        info=info,
    )


def save(path: str | Path, laser: Laser | SRRLaser) -> None:
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
    np.savez_compressed(
        path,
        header=pack_info(
            {
                "version": version("pewlib"),
                "class": str(laser.config._class),
                "time": str(time.time()),
            }
        ),
        data=laser.data,
        calibration=pack_calibration(laser.calibration),
        info=pack_info(laser.info),
        config=laser.config.to_array(),
    )
