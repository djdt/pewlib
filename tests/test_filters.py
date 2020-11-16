import numpy as np

from pew.lib import filters

def test_mean_filter_1d():
    np.random.seed(93546376)
    x = np.random.random(30)
    x[15] = 3.0
    f = filters.rolling_mean(x, 5, threshold=3.0)
    assert np.all(f < 0.3)


def test_mean_filter_2d():
    x = np.random.random((50, 50))
    x[10, 10] = 2.0
    x[10, 20] = -1.0

    # Nothing filtered when threshold is high
    f = filters.rolling_mean(x, (5, 5), threshold=100.0)
    assert np.all(x == f)

    f = filters.rolling_mean(x, (5, 5), threshold=3.0)
    assert np.allclose(f[10, 10], np.nanmean(np.where(x <= 1.0, x, np.nan)[8:13, 8:13]))
    assert np.allclose(f[10, 20], np.nanmean(np.where(x >= 0.0, x, np.nan)[8:13, 18:23]))


def test_median_filter():
    x = np.random.random((50, 50))
    x[10, 10] = 2.0
    x[10, 20] = -1.0

    # Nothing filtered when threshold is high
    f = filters.rolling_median(x, (5, 5), threshold=100.0)
    assert np.all(x == f)

    f = filters.rolling_median(x, (5, 5), threshold=3.0)
    assert np.allclose(f[10, 10], np.median(x[8:13, 8:13]))
    assert np.allclose(f[10, 20], np.median(x[8:13, 18:23]))
