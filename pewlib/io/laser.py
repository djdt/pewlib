"""
Synchronisation of laser parameters (ablation times and locations) with signal data.
Data should be imported using the other pewlib.io modules then passed with the laser
parameters file to these functions.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def read_nwi_laser_log(log_path: Path | str) -> np.ndarray:
    log = np.genfromtxt(
        log_path,
        usecols=(0, 1, 2, 4, 5, 6, 10, 11, 13),
        delimiter=",",
        skip_header=1,
        dtype=[
            ("time", "datetime64[ms]"),
            ("sequence", int),
            ("subpoint", int),
            ("comment", "U64"),
            ("x", float),
            ("y", float),
            ("state", "U3"),
            ("rate", int),
            ("spotsize", "U16"),
        ],
    )
    return log


def sync_data_nwi_laser_log(
    data: np.ndarray,
    times: np.ndarray | float,
    log: np.ndarray | Path | str,
    sequence: np.ndarray | int | None = None,
    squeeze: bool = False,
) -> np.ndarray:
    """
    Args:
        squeeze: remove any rows and columns of all NaNs
    """

    if isinstance(log, (Path, str)):
        log = read_nwi_laser_log(log)

    if isinstance(times, float):
        times = np.arange(data.size) * times

    # remove patterns that were not selected
    if sequence is not None:
        log = log[np.in1d(log["sequence"], sequence)]

    # Get laser start and end events only
    start_idx = np.flatnonzero(log["state"] == "On")
    log = log[np.stack((start_idx, start_idx + 1), axis=1).flat]

    first_line = log[0]
    # check for inconsistencies and warn
    if not np.all(log["spotsize"] == first_line["spotsize"]):
        logger.warning("importing multiple spot sizes")

    # get the spotsize, for square or circular spots
    if "x" in first_line["spotsize"]:
        spot_size = np.array([float(x) for x in first_line["spotsize"].split(" x ")])
    else:
        spot_size = np.array(
            [float(first_line["spotsize"]), float(first_line["spotsize"])]
        )

    # reshape into (start, end)
    log = np.reshape(log, (-1, 2))

    if not np.all(log["rate"] == log["rate"][0]):
        logger.warning("importing multiple laser firing rates")

    # find the shape of the data
    origin = np.amin(log["x"]), np.amin(log["y"])
    px = ((log["x"] - origin[0]) / spot_size[0]).astype(int)
    py = ((log["y"] - origin[1]) / spot_size[1]).astype(int)
    sync = np.full((py.max() + 1, px.max() + 1), np.nan, dtype=data.dtype)

    # calculate the indicies for start and end times of lines
    laser_times = log["time"] - log["time"][0][0]
    laser_idx = np.searchsorted(times, laser_times.astype(float) / 1000.0)

    # read and fill in data
    for line, (t0, t1), (x0, x1), (y0, y1) in zip(log, laser_idx, px, py):
        x = data.flat[t0:t1]
        if y0 == y1:
            if x0 > x1:  # flip right-to-left
                x = x[::-1]
                x0, x1 = x1, x0

            size = min(x.size, x1 - x0)
            sync[y0, x0:x1][:size] = x[:size]
        elif x0 == x1:
            if y0 > y1:  # flip bottom-to-top
                x = x[::-1]
                y0, y1 = y1, y0

            size = min(x.size, y1 - y0)
            sync[y0:y1, x0][:size] = x[:size]
        else:
            raise ValueError("unable to import non-vertical or non-horizontal lines.")

    if squeeze:
        nan_rows = np.all(np.isnan(sync), axis=0)
        sync = sync[nan_rows, :]
        nan_cols = np.all(np.isnan(sync), axis=1)
        sync = sync[:, nan_cols]

    return sync
