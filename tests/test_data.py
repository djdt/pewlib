import numpy as np

from pew.calibration import Calibration
from pew.config import Config
from pew.data import IsotopeData


def test_data_calibrate():
    calibration = Calibration(2.0, 1.0)
    data = IsotopeData(np.random.random([10, 10]), calibration)
    assert np.all(calibration.calibrate(data.data) == data.get(None, calibrate=True))


def test_data_extent():
    config = Config(10.0, 10.0, 0.5)
    data = IsotopeData(np.random.random([10, 10]), None)
    assert np.all(
        data.data[8:10, 0:4] == data.get(config, extent=(0.0, 20.0, 0.0, 20.0))
    )
