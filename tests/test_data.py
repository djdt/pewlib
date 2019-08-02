import numpy as np

from laserlib.calibration import LaserCalibration
from laserlib.config import LaserConfig
from laserlib.data import LaserData


def test_data_calibrate():
    calibration = LaserCalibration(2.0, 1.0)
    data = LaserData(np.random.random([10, 10]), calibration)
    assert np.all(calibration.calibrate(data.data) == data.get(None, calibrate=True))


def test_data_extent():
    config = LaserConfig(10.0, 10.0, 0.5)
    data = LaserData(np.random.random([10, 10]), None)
    assert np.all(
        data.data[8:10, 0:4] == data.get(config, extent=(0.0, 20.0, 0.0, 20.0))
    )
