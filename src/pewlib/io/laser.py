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


def is_iolite_laser_log(log_path: Path | str) -> bool:
    log_path = Path(log_path)

    if log_path.suffix.lower() != ".csv":
        return False
    with log_path.open("r") as fp:
        header = fp.readline()
        if not (
            header.replace(", ", ",").startswith(
                "Timestamp,Sequence Number,SubPoint Number,Vert"  # Vertex or Vertix?
            )
        ):
            return False
    return True


def read_iolite_laser_log(log_path: Path | str, log_style: str = "raw") -> np.ndarray:
    """Reads an Iolite style log.
    Different vendors will have slighly different styles of log, so passing 'log_style'
    is reccommended to reduce the log to only laser start and end events.
    Currently NWL ActiveView2 and Teledyne Chromium2 are supported.
    Passing 'raw' as a style will prevent processing.

    Args:
        log_path: path to iolilte
        log_style: style of log ('activeview2', 'chromium2', 'raw')

    Returns:
        log as a numpy array, trimmed to useful lines
    """

    def fill_ints(x: np.ndarray) -> None:
        max = np.maximum.accumulate(x)
        x[x == -1] = max[x == -1]

    def fill_strings(x: np.ndarray) -> None:
        idx = np.cumsum(x != "") - 1
        strings = x[np.flatnonzero(x != "")]
        x[:] = strings[idx]

    log = np.genfromtxt(
        log_path,
        usecols=(0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 13),
        delimiter=",",
        skip_header=1,
        converters={10: lambda x: 1 if x == "On" else 0},
        dtype=[
            ("time", "datetime64[ms]"),
            ("sequence", int),
            ("subpoint", int),
            ("vertix", int),
            ("comment", "U64"),
            ("x", float),
            ("y", float),
            ("velocity", float),
            ("state", int),
            ("rate", int),
            ("spotsize", "U16"),
        ],
    )
    fill_ints(log["sequence"])
    fill_ints(log["subpoint"])
    fill_strings(log["comment"])

    if log_style == "chromium2":
        start_idx = np.flatnonzero(np.logical_and(log["vertix"] > 0, log["state"] == 1))
        log = log[np.stack((start_idx, start_idx + 2), axis=1).flat]
    elif log_style == "activeview2":
        log = log[np.argsort(log["time"])]
        start_idx = (
            np.flatnonzero(
                np.logical_and(log["state"][1:] == 1, log["state"][:-1] == 0)
            )
            + 1
        )
        # Get laser end event, next event after start to be 'Off'
        log = log[np.stack((start_idx, start_idx + 1), axis=1).flat]
    elif log_style != "raw":  # pragma: no cover
        raise ValueError(f"invalid log style {log_style}")

    return log


def guess_delay_from_data(data: np.ndarray, times: np.ndarray) -> float:
    """Guess delay from laser firing to ICP-MS measurement.

    Looks for a change of > 10% in the TIC, up to 1 second into data.

    Args:
        data: structured array of signals, flatttend
        times: array of times, same length as data

    Returns:
        delay in ms"""
    tic = rfn.structured_to_unstructured(data.flat[: np.searchsorted(times, 1.0)])
    tic = np.sum(tic, axis=-1)
    tic = np.diff(tic)
    return times.flat[np.argmax((tic / tic.mean()) > 0.1)]


def sync_data_with_laser_log(
    data: np.ndarray,
    times: np.ndarray | float,
    log: np.ndarray,
    sequence: np.ndarray | int | None = None,
    delay: float | None = None,
    squeeze: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Syncs ICP-MS data collected as a single line per raster with the laser log file.
    Times in the log are modified to start at 0.

    Args:
        data: 1d ICP-MS data
        times: array of times (s) the same size as ``data``, or pixel acquistion time
        log: log data or path to LaserLog csv
        sequence: select raster(s) to import, defaults to all
        delay: delay in s between laser and ICP-MS, default calculates from the TIC
        squeeze: remove any rows and columns of all NaNs
    """

    if isinstance(times, float):  # pragma: no cover, warning
        logger.info(f"generating times with interval {times:.2f}")
        times = np.arange(data.size) * times
    elif times.ndim > 1:  # pragma: no cover, warning
        logger.warning("times has more than one dimension, flattening")
    times = times.ravel()

    if delay is None:  # pragma: no cover, tested elsewhere
        delay = guess_delay_from_data(data, times)

    times += delay

    # remove patterns that were not selected
    if sequence is not None:
        log = log[np.isin(log["sequence"], sequence)]

    first_line = log[0]
    # check for inconsistencies and warn
    if not np.all(
        log["spotsize"] == first_line["spotsize"]
    ):  # pragma: no cover, warning
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

    if not np.all(log["rate"] == log["rate"][0]):  # pragma: no cover, warning
        logger.warning("importing multiple laser firing rates")

    # find the shape of the data
    origin = np.amin(log["x"]), np.amin(log["y"])
    px = ((log["x"] - origin[0]) / spot_size[0]).astype(int)
    py = ((log["y"] - origin[1]) / spot_size[1]).astype(int)
    sync = np.full((py.max() + 1, px.max() + 1), np.nan, dtype=data.dtype)
    assert sync.dtype.names is not None

    # calculate the indicies for start and end times of lines
    laser_times = (log["time"] - log["time"][0][0]).astype(float) / 1000.0
    laser_idx = np.searchsorted(times, laser_times)

    # read and fill in data
    for line, (i0, i1), (x0, x1), (y0, y1) in zip(log, laser_idx, px, py):
        x = data.flat[i0:i1]
        if x.size == 0:
            continue
        if y0 == y1:  # horizontal
            if x0 > x1:  # flip right-to-left
                x = x[::-1]
                x0, x1 = x1, x0
            for name in sync.dtype.names:
                sync[y0, x0:x1][name] = np.add.reduceat(
                    x[name], np.linspace(0, x.size, x1 - x0, endpoint=False).astype(int)
                )
        elif x0 == x1:  # vertical
            if y0 > y1:  # flip bottom-to-top
                x = x[::-1]
                y0, y1 = y1, y0
            for name in sync.dtype.names:
                sync[y0:y1, x0][name] = np.add.reduceat(
                    x[name], np.linspace(0, x.size, y1 - y0, endpoint=False).astype(int)
                )
        else:  # pragma: no cover
            raise ValueError("unable to import non-vertical or non-horizontal lines.")

    if squeeze:
        if sync.dtype.names is not None:
            nans = np.all([np.isnan(sync[n]) for n in sync.dtype.names], axis=0)
        else:
            nans = np.isnan(sync)  # pragma: no cover
        nan_rows = np.all(nans, axis=1)
        sync = sync[~nan_rows, :]
        if sync.dtype.names is not None:
            nans = np.all([np.isnan(sync[n]) for n in sync.dtype.names], axis=0)
        else:
            nans = np.isnan(sync)  # pragma: no cover
        nan_cols = np.all(nans, axis=0)
        sync = sync[:, ~nan_cols]

    return sync, {"delay": delay, "origin": origin, "spotsize": spot_size}
