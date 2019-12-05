import pytest
import numpy as np

from pew.calc import get_weights, weighted_rsq, weighted_linreg
from pew.srr.calc import subpixel_offset


def test_get_weights():
    # Test all the weights, safe return
    x = np.array([1.0, 0.4, 0.2, 0.5, 0.8])
    assert get_weights(x, None) is None
    assert np.all(get_weights(x, "x") == x)
    assert np.all(get_weights(x, "1/x") == 1 / x)
    assert np.all(get_weights(x, "1/(x^2)") == 1 / (x * x))
    # Test safe return
    x = np.array([0.0, 0.1, 0.2])
    assert np.all(get_weights(x, "x", True) == np.array([0.1, 0.1, 0.2]))
    assert np.all(get_weights(x, "x", False) == x)


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
    assert weighted_linreg(x, y, x) == pytest.approx((2.085714, -3.314286, 0.865097, 2.296996))


def test_subpixel_offset():
    x = np.ones((10, 10, 3))

    y = subpixel_offset(x, [(0, 0), (1, 1), (2, 3)], (2, 3))
    assert y.shape == (22, 33, 3)
    assert np.all(y[0:20, 0:30, 0] == 1)
    assert np.all(y[1:21, 1:31, 1] == 1)
    assert np.all(y[2:22, 3:33, 2] == 1)
