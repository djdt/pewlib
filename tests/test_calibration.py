import numpy as np
import pytest

from laserlib.calibration import LaserCalibration


def test_default_calibration():
    calibration = LaserCalibration()
    assert calibration.concentrations().size == 0
    assert calibration.counts().size == 0
    calibration.update_linreg()


def test_calibration_from_points():
    calibration = LaserCalibration.from_points([[0, 1], [1, 2], [1, 3], [2, 4]])
    assert calibration.gradient == pytest.approx(1.5)
    assert calibration.intercept == pytest.approx(1.0)
    assert calibration.rsq == pytest.approx(0.9000)
    # With nans
    calibration = LaserCalibration.from_points(
        [[0, 1], [1, np.nan], [np.nan, 3], [2, 4]]
    )
    assert calibration.gradient == pytest.approx(1.5)
    assert calibration.intercept == pytest.approx(1.0)
    assert calibration.rsq == pytest.approx(1.0)
    with pytest.raises(ValueError):
        calibration = LaserCalibration.from_points([[0, 1]])
    with pytest.raises(ValueError):
        calibration = LaserCalibration.from_points([[0, 1, 1]])
    with pytest.raises(ValueError):
        calibration = LaserCalibration.from_points([[[0, 1]]])
