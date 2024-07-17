"""
Synchronisation of laser parameters (ablation times and locations) with signal data.
Data should be imported using the other pewlib.io modules then passed with the laser
parameters file to these functions.
"""

import logging
from pathlib import Path

import numpy as np
import numpy.lib.recfunctions as rfn

logger = logging.getLogger(__name__)


def is_nwi_laser_log(log_path: Path | str) -> bool:
    log_path = Path(log_path)

    if log_path.suffix.lower() != ".csv":
        return False
    with log_path.open("r") as fp:
        header = fp.readline()
        if not (
            header.replace(", ", ",").startswith(
                "Timestamp,Sequence Number,SubPoint Number,Vertex Number,Comment"
            )
        ):
            return False
    return True


def read_nwi_laser_log(log_path: Path | str) -> np.ndarray:
    def fill_ints(x: np.ndarray) -> None:
        max = np.maximum.accumulate(x)
        x[x == -1] = max[x == -1]

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
    fill_ints(log["sequence"])
    fill_ints(log["subpoint"])
    return log


def sync_data_nwi_laser_log(
    data: np.ndarray,
    times: np.ndarray | float,
    log_file: np.ndarray | Path | str,
    sequence: np.ndarray | int | None = None,
    delay: float | None = None,
    squeeze: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Syncs ICP-MS data collected as a single line per raster with the laser log file.

    Args:
        data: 1d ICP-MS data
        times: array of times (s) the same size as ``data``, or pixel acquistion time
        log: log data or path to LaserLog csv
        sequence: select raster(s) to import, defaults to all
        delay: delay in s between laser and ICP-MS, default calculates from the TIC
        squeeze: remove any rows and columns of all NaNs
    """

    if isinstance(log_file, (Path, str)):
        log = read_nwi_laser_log(log_file)
    else:
        log = log_file

    if isinstance(times, float):
        logger.info(f"generating times with interval {times:.2f}")
        times = np.arange(data.size) * times
    elif times.ndim > 1:
        logger.warning("times has more than one dimension, flattening")

    times = (times - times.min() + delay).ravel()

    if delay is None:
        tic = rfn.structured_to_unstructured(data[: np.argmax(times > 1.0)])
        tic = np.sum(tic, axis=-1)
        tic = np.diff(tic)
        delay = times[np.argmax((tic / tic.mean()) > 0.1)]

    # remove patterns that were not selected
    if sequence is not None:
        log = log[np.isin(log["sequence"], sequence)]

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
    laser_times = (log["time"] - log["time"][0][0]).astype(float) / 1000.0
    laser_idx = np.searchsorted(times, laser_times)

    # read and fill in data
    for line, (t0, t1), (x0, x1), (y0, y1) in zip(log, laser_idx, px, py):
        x = data.flat[t0:t1]
        if y0 == y1:  # horizontal
            s0, s1 = -min(x.size, abs(x1 - x0)), None
            if x0 > x1:  # flip right-to-left
                x = x[::-1]
                x0, x1 = x1, x0
                s0, s1 = None, -s0

            sync[y0, x0:x1][s0:s1] = x[s0:s1]
        elif x0 == x1:  # vertical
            s0, s1 = -min(x.size, abs(y1 - y0)), None
            if y0 > y1:  # flip bottom-to-top
                x = x[::-1]
                y0, y1 = y1, y0
                s0, s1 = None, -s0
            sync[y0:y1, x0][s0:s1] = x[s0:s1]
        else:
            raise ValueError("unable to import non-vertical or non-horizontal lines.")

    if squeeze:
        nan_rows = np.all(np.isnan(sync), axis=0)
        sync = sync[nan_rows, :]
        nan_cols = np.all(np.isnan(sync), axis=1)
        sync = sync[:, nan_cols]

    return sync, {"delay": delay, "origin": origin, "spotsize": spot_size}
