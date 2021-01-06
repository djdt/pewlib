import numpy as np

from pewlib.process import calc


def test_local_maxima():
    x = np.linspace(0.0, 1.0, 100)
    i = np.arange(5, 95, 5)
    x[i] = 10.0
    assert np.all(calc.local_maxima(x)[:-1] == i)


def test_normalise():
    x = np.random.random(100)
    x = calc.normalise(x, -1.0, 2.33)
    assert x.min() == -1.0
    assert x.max() == 2.33


def test_reset_cumsum():
    x = np.array([1, 2, 3, 0, 4, 5, 0, 6, 7, 0, 8, 0, 0, 9, 0])
    assert np.all(
        calc.reset_cumsum(x) == [1, 3, 6, 0, 4, 9, 0, 6, 13, 0, 8, 0, 0, 9, 0]
    )


def test_shuffle_blocks():
    x = np.random.random((100, 100))
    m = np.zeros((100, 100))
    m[:52] = 1.0

    y = calc.shuffle_blocks(x, (5, 20), mask=m, shuffle_partial=False)

    assert np.allclose(y[50:], x[50:])
    assert not np.allclose(y[:50], x[:50])
    assert np.allclose(y.sum(), x.sum())


def test_subpixel_offset():
    x = np.ones((10, 10, 3))

    y = calc.subpixel_offset(x, [(0, 0), (1, 1), (2, 3)], (2, 3))
    assert y.shape == (22, 33, 3)
    assert np.all(y[0:20, 0:30, 0] == 1)
    assert np.all(y[1:21, 1:31, 1] == 1)
    assert np.all(y[2:22, 3:33, 2] == 1)

    assert np.all(
        calc.subpixel_offset(x, [(0, 0), (1, 1)], (2, 2))
        == calc.subpixel_offset_equal(x, [0, 1], 2)
    )


def test_subpixel_offset_means():
    x = np.stack((np.random.random((10, 10)), np.random.random((10, 10))), axis=2)

    z = calc.subpixel_offset(x, [(0, 0), (1, 1)], (2, 2))

    x = np.repeat(x, 2, axis=0)
    x = np.repeat(x, 2, axis=1)

    assert np.isclose(
        z[1:-1, 1:-1].mean(), (x[1:, 1:, 0].mean() + x[:-1, :-1, 1].mean()) / 2.0
    )
