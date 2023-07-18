import numpy as np

from pewlib.process import filters


def test_mean_filter_1d():
    # test specific, 1.6 filtered, 1.5 not
    y = np.array([1.0, 1.1, 1.5, 1.3, 1.2, 1.1, 1.0, 1.6, 1.2, 1.3])
    f = filters.rolling_mean(y, 5, threshold=3.0)
    assert np.all(f == [1.0, 1.1, 1.5, 1.3, 1.2, 1.1, 1.0, 1.15, 1.2, 1.3])


def test_mean_filter_2d():
    # Test zeros
    d = np.zeros((10, 10))
    d[5, 5] = 100.0
    f = filters.rolling_mean(d, (3, 3), threshold=1.0)
    assert np.all(f[5, 5] == 0.0)


def test_median_filter_1d():
    # test specific, 1.5 filtered, 1.6 not
    y = np.array([1.0, 1.1, 1.5, 1.3, 1.2, 1.1, 1.0, 1.6, 1.2, 1.3])
    f = filters.rolling_median(y, 5, threshold=3.0)
    assert np.all(f == [1.0, 1.1, 1.2, 1.3, 1.2, 1.1, 1.0, 1.6, 1.2, 1.3])


def test_median_filter():
    # Test zeros
    d = np.zeros((10, 10))
    d[5, 5] = 100.0
    f = filters.rolling_median(d, (3, 3), threshold=1.0)
    assert np.all(f == 0.0)
