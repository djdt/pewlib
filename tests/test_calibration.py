import numpy as np
import pytest

from pew.calibration import weighting, weighted_rsq, weighted_linreg
from pew.calibration import Calibration


def test_weighting():
    # Test all the weights, safe return
    x = np.random.normal(loc=2, size=10)
    assert weighting(x, None) is None
    assert np.all(weighting(x, "x") == x)
    assert np.all(weighting(x, "1/x") == 1 / x)
    assert np.all(weighting(x, "1/(x^2)") == 1 / (x * x))

    # Test safe return
    x[0] = 0.0
    assert weighting(x, "x", True)[0] == np.amin(x[1:])
    assert np.all(weighting(x, "x", False) == x)
    with pytest.raises(ValueError):
        weighting(x, "invalid")

    # Nan ignored when looking for min value in safe
    assert np.all(
        weighting(np.array([0.0, 1.0, np.nan, 2.0]), "1/x", True)[[0, 1, 3]]
        == [1.0, 1.0, 0.5]
    )


def test_weighted_rsq():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 1.0, 2.0, 4.0, 8.0])
    # From http://www.xuru.org/rt/WLR.asp
    assert weighted_rsq(x, y, None) == pytest.approx(8.30459e-1)
    assert weighted_rsq(x, y, x) == pytest.approx(8.65097e-1)
    assert weighted_rsq(x, y, 1 / x) == pytest.approx(7.88696e-1)
    assert weighted_rsq(x, y, 1 / (x ** 2)) == pytest.approx(7.22560e-1)


def test_weighted_linreg():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 1.0, 2.0, 4.0, 8.0])
    # Normal
    assert weighted_linreg(x, y, None) == pytest.approx((1.7, -1.9, 0.830459, 1.402379))
    # Weighted
    assert weighted_linreg(x, y, x) == pytest.approx(
        (2.085714, -3.314286, 0.865097, 2.296996)
    )


def test_default_calibration():
    calibration = Calibration()
    assert calibration.concentrations().size == 0
    assert calibration.counts().size == 0
    calibration.update_linreg()

    # Default should just return data
    data = np.random.random([10, 10])
    assert np.all(calibration.calibrate(data) == data)
    assert calibration.rsq is None


def test_calibration_calibrate():
    calibration = Calibration(2.0, 2.0, unit="ppm")
    # Test data
    data = np.random.random([10, 10])
    assert np.all(calibration.calibrate(data) == ((data - 2.0) / 2.0))

    assert calibration.weights is None


def test_calibration_from_points():
    calibration = Calibration.from_points([[0, 1], [1, 2], [1, 3], [2, 4]])
    assert calibration.gradient == pytest.approx(1.5)
    assert calibration.intercept == pytest.approx(1.0)
    assert calibration.rsq == pytest.approx(0.9000)

    # Test returns
    assert np.all(calibration.concentrations() == np.array([0.0, 1.0, 1.0, 2.0]))
    assert np.all(calibration.counts() == np.array([1.0, 2.0, 3.0, 4.0]))

    # With nans
    calibration = Calibration.from_points([[0, 1], [1, np.nan], [1, 3], [2, 4]])
    assert calibration.rsq is not None
    calibration = Calibration.from_points([[0, 1], [np.nan, 2], [1, 3], [2, 4]])
    assert calibration.rsq is not None
    # If all nan should return to default
    calibration = Calibration.from_points([[1.0, np.nan], [1.0, np.nan]])
    assert calibration.gradient == 1.0


def test_calibration_from_points_invalid():
    # Test shape[0] = 1
    with pytest.raises(ValueError):
        Calibration.from_points([[0, 1]])
    # Test shape[1] = 3
    with pytest.raises(ValueError):
        Calibration.from_points([[0, 1, 1]])
    # Test one dimnesion
    with pytest.raises(ValueError):
        Calibration.from_points([0, 1])


def test_calibration_from_points_weights():
    points = np.vstack([[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 1.0, 2.0, 4.0, 8.0]]).T
    calibration = Calibration.from_points(points=points, weights="x")
    assert np.all(calibration.weights == [1.0, 2.0, 3.0, 4.0, 5.0])
    assert pytest.approx(calibration.gradient, 2.085814)
    assert pytest.approx(calibration.intercept, -3.314286)
    assert pytest.approx(calibration.rsq, 0.865097)

    calibration = Calibration.from_points(
        points=points, weights=[1.0, 2.0, 3.0, 4.0, 5.0]
    )
    assert pytest.approx(calibration.gradient, 2.085814)
    assert pytest.approx(calibration.intercept, -3.314286)
    assert pytest.approx(calibration.rsq, 0.865097)

    with pytest.raises(ValueError):
        calibration = Calibration.from_points(points=points, weights=[1.0])


def test_calibration_update_linreg():
    points = np.vstack([[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]]).T
    calibration = Calibration.from_points(points, weighting="1/x")
    calibration.points[0, 0] = 0
    calibration.update_linreg()
    calibration.points[1, 0] = np.nan
    calibration.update_linreg()
    calibration.points[2, 1] = np.nan
    calibration.update_linreg()


def test_calibration_str():
    calibration = Calibration(1.0, 2.0, rsq=0.999)
    assert str(calibration) == "y = 2 · x - 1\nr² = 0.9990"
