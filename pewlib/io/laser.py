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
) -> np.ndarray:
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

    # check for inconsistencies and warn
    if not np.all(log["spotsize"] == log["spotsize"][0]):
        logger.warning("importing multiple spot sizes")

    # get the spotsize, for square or circular spots
    if "x" in log["spotsize"][0]:
        spot_size = np.array([float(x) for x in log["spotsize"][0].split(" x ")])
    else:
        spot_size = np.array([float(log["spotsize"][0]), float(log["spotsize"][0])])

    # reshape into (start, end)
    log = np.reshape(log, (-1, 2))

    if not np.all(log["rate"] == log["rate"][0]):
        logger.warning("importing multiple laser firing rates")

    # find the shape of the data
    px = ((log["x"] - np.amin(log["x"])) / spot_size[0]).astype(int)
    py = ((log["y"] - np.amin(log["y"])) / spot_size[1]).astype(int)
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

    return sync


from pewlib.io import agilent, textimage

data, params = agilent.load_binary(
    "/home/tom/Downloads/lasso.b/",
    full=True,
)
# data = np.genfromtxt(
#     "/home/tom/Downloads/lasso.b/",
#     delimiter=",",
#     skip_footer=3,
#     skip_header=1,
#     usecols=(0, 1),
#     names=["times", "Ho165"],
# )
# data = textimage.load("/home/tom/Downloads/001.csv", name="Ho165")

if "times" in params:
    times = params["times"]
else:
    times = params["scantime"]
# times = data["times"]
# times -= times[0]
# print(times)

import matplotlib.pyplot as plt

a = sync_data_nwi_laser_log(
    data,
    times,
    "/home/tom/Downloads/LaserLog_24-05-02_13-49-34.csv",
)
plt.imshow(a["P31"], vmax=np.nanpercentile(a["P31"], 95))
plt.show()
