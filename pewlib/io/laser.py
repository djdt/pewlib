"""
Synchronisation of laser parameters (ablation times and locations) with signal data.
Data should be imported using the other pewlib.io modules then passed with the laser
parameters file to these functions.
"""

from pathlib import Path

import numpy as np


def sync_data_nwi_laser_log(
    data: np.ndarray, times: np.ndarray | float, log_path: Path | str
) -> np.ndarray:
    def date_parse(date: bytes) -> np.datetime64:
        return np.datetime64(date.decode())

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

    # need a way to select sequence / subpoint

    if isinstance(times, float):
        times = np.arange(data.size) * times

    if "x" in log["spotsize"][0]:
        spot_size = np.array([float(x) for x in log["spotsize"][0].split(" x ")])
    else:
        spot_size = np.array([float(log["spotsize"][0]), float(log["spotsize"][0])])

    start_idx = np.flatnonzero(log["state"] == "On")
    log = log[np.stack((start_idx, start_idx + 1), axis=1).flat]  # remove movements

    # min_x, max_x = np.amin(log["x"]), np.amax(log["x"])
    # min_y, may_y = np.amin(log["y"]), np.amax(log["y"])
    xspots = int(np.ptp(log["x"]) / spot_size[0])
    yspots = int(np.ptp(log["y"]) / spot_size[1])
    sync = np.empty((yspots, xspots), dtype=data.dtype)
    print(sync.shape)

    laser_times = (log["time"] - log["time"][0]).astype(float) / 1000.0
    line_idx = np.searchsorted(times, laser_times[::2], side="right") - 1
    for i, idx in enumerate(line_idx):
        if i >= sync.shape[0]:
            break
        sync[i] = data.flat[idx:idx+sync.shape[1]]

    import matplotlib.pyplot as plt
    plt.imshow(sync["Ho165"])
    plt.show()


from pewlib.io import agilent

data, params = agilent.load(
    "/home/tom/Downloads/20240430 laser multi layer raster test/4 layers.b",
    full=True,
)
if "times" in params:
    times = params["times"]
else:
    times = params["scantime"]

sync_data_nwi_laser_log(
    data,
    times,
    "/home/tom/Downloads/20240430 laser multi layer raster test/LaserLog_24-04-30_14-17-12.csv",
)
