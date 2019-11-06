import numpy as np
import pytest

from pew.calibration import Calibration


def test_default_calibration():
    calibration = Calibration()
    assert calibration.concentrations().size == 0
    assert calibration.counts().size == 0
    calibration.update_linreg()
    # Default should just return data
    data = np.random.random([10, 10])
    assert np.all(calibration.calibrate(data) == data)


def test_calibration_calibrate():
    calibration = Calibration(2.0, 2.0, unit="ppm")
    # Test data
    data = np.random.random([10, 10])
    assert np.all(calibration.calibrate(data) == ((data - 2.0) / 2.0))


def test_calibration_from_points():
    # Test shape[0] = 1
    with pytest.raises(ValueError):
        calibration = Calibration.from_points([[0, 1]])
    # Test shape[1] = 3
    with pytest.raises(ValueError):
        calibration = Calibration.from_points([[0, 1, 1]])
    # Test one dimnesion
    with pytest.raises(ValueError):
        calibration = Calibration.from_points([0, 1])

    calibration = Calibration.from_points([[0, 1], [1, 2], [1, 3], [2, 4]])
    assert calibration.gradient == pytest.approx(1.5)
    assert calibration.intercept == pytest.approx(1.0)
    assert calibration.rsq == pytest.approx(0.9000)
    # Test returns
    assert np.all(calibration.concentrations() == np.array([0.0, 1.0, 1.0, 2.0]))
    assert np.all(calibration.counts() == np.array([1.0, 2.0, 3.0, 4.0]))
    # With nans
    calibration = Calibration.from_points([[0, 1], [1, np.nan], [1, 3], [2, 4]])


def test_calibration_from_points_weights():
    points = np.vstack([[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 1.0, 2.0, 4.0, 8.0]]).T
    calibration = Calibration.from_points(points=points, weights="x")
    assert pytest.approx(calibration.gradient, 2.085814)
    assert pytest.approx(calibration.intercept, -3.314286)
    assert pytest.approx(calibration.rsq, 0.865097)

    calibration = Calibration.from_points(
        points=points, weights=[1.0, 2.0, 3.0, 4.0, 5.0]
    )
    assert pytest.approx(calibration.gradient, 2.085814)
    assert pytest.approx(calibration.intercept, -3.314286)
    assert pytest.approx(calibration.rsq, 0.865097)
