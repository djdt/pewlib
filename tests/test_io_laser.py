from pathlib import Path

import numpy as np
import pytest

from pewlib.io.laser import (
    guess_delay_from_data,
    is_nwi_laser_log,
    read_nwi_laser_log,
    sync_data_nwi_laser_log,
)


@pytest.fixture(scope="module")
def laserlog() -> np.ndarray:
    path = Path(__file__).parent.joinpath(
        "data", "laser_nwi", "LaserLog_test_rasters.csv"
    )
    return read_nwi_laser_log(path)


@pytest.fixture(scope="module")
def laserlog_data() -> tuple[np.ndarray, np.ndarray]:
    path = Path(__file__).parent.joinpath("data", "laser_nwi", "laserlog_test_data.npz")
    npz = np.load(path)
    return npz["data"], npz["times"]


def test_is_nwi_laser_log():
    assert not is_nwi_laser_log("fake.bad")
    path = Path(__file__).parent.joinpath(
        "data", "laser_nwi", "LaserLog_test_rasters.csv"
    )
    assert is_nwi_laser_log(path)
    path = Path(__file__).parent.joinpath("data", "csv", "nu", "dummy.csv")
    assert not is_nwi_laser_log(path)


def test_read_nwi_laser_log():
    path = Path(__file__).parent.joinpath(
        "data", "laser_nwi", "LaserLog_test_rasters.csv"
    )
    laserlog = read_nwi_laser_log(path)
    assert np.unique(laserlog["comment"]).size == 5 + 1
    assert len(laserlog) == 110
    assert np.count_nonzero(laserlog["state"] == "On") == 25

    assert laserlog[0]["comment"] == "Image Raster1"
    assert laserlog[0]["sequence"] == 1
    assert laserlog[0]["spotsize"] == "40 x 40"


def test_guess_delay_from_data():
    data = np.zeros(100, dtype=[("A", float)])
    data["A"][10:] = 100.0
    times = np.linspace(0.0, 2.0, 100)
    delay = guess_delay_from_data(data, times)
    assert delay == times[9]


def test_sync_data_nwi_laser_log_left_to_right(laserlog, laserlog_data):
    data, times = laserlog_data
    sync, params = sync_data_nwi_laser_log(
        data[0], times[0], laserlog, delay=0.25, sequence=1, squeeze=True
    )
    assert np.all(sync["Ho165"][:, :5] < 1e1, where=~np.isnan(sync["Ho165"][:, :5]))
    assert np.all(sync["Ho165"][:, -5:] > 1e3, where=~np.isnan(sync["Ho165"][:, -5:]))


def test_sync_data_nwi_laser_log_right_to_left(laserlog, laserlog_data):
    data, times = laserlog_data
    sync, params = sync_data_nwi_laser_log(
        data[1], times[1], laserlog, delay=0.25, sequence=2, squeeze=True
    )
    assert np.all(sync["Ho165"][:, :5] < 1e1, where=~np.isnan(sync["Ho165"][:, :5]))
    assert np.all(sync["Ho165"][:, -5:] > 1e3, where=~np.isnan(sync["Ho165"][:, -5:]))


def test_sync_data_nwi_laser_log_horz_raster(laserlog, laserlog_data):
    data, times = laserlog_data
    sync, params = sync_data_nwi_laser_log(
        data[4], times[4], laserlog, delay=0.25, sequence=5, squeeze=True
    )
    assert np.all(sync["Ho165"][:, :5] < 1e1, where=~np.isnan(sync["Ho165"][:, :5]))
    assert np.all(sync["Ho165"][:, -5:] > 1e3, where=~np.isnan(sync["Ho165"][:, -5:]))


def test_sync_data_nwi_laser_log_top_to_bottom(laserlog, laserlog_data):
    data, times = laserlog_data
    sync, params = sync_data_nwi_laser_log(
        data[2], times[2], laserlog, delay=0.25, sequence=3, squeeze=True
    )
    assert np.all(sync["Ho165"][:5] < 1e1, where=~np.isnan(sync["Ho165"][:5]))
    assert np.all(sync["Ho165"][-5:] > 1e3, where=~np.isnan(sync["Ho165"][-5:]))


def test_sync_data_nwi_laser_log_bottom_to_top(laserlog, laserlog_data):
    data, times = laserlog_data
    laserlog["spotsize"] = "40"  # test for IVA style
    sync, params = sync_data_nwi_laser_log(
        data[3], times[3], laserlog, delay=0.25, sequence=4, squeeze=True
    )
    assert np.all(sync["Ho165"][:5] < 1e1, where=~np.isnan(sync["Ho165"][:5]))
    assert np.all(sync["Ho165"][-5:] > 1e3, where=~np.isnan(sync["Ho165"][-5:]))


def test_zero_size_line_in_vert():
    log = read_nwi_laser_log(
        Path(__file__).parent.joinpath(
            "data", "laser_nwi", "LaserLog_zero_size_vert.csv"
        )
    )
    npz = np.load(
        Path(__file__).parent.joinpath(
            "data", "laser_nwi", "laserlog_zero_size_vert.npz"
        )
    )
    sync_data_nwi_laser_log(npz["data"], npz["times"], log, 1)
