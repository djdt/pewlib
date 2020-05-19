import numpy as np

from pew.lib import calc


def test_local_maxima():
    x = np.linspace(0.0, 1.0, 100)
    i = np.arange(5, 95, 5)
    x[i] = 10.0
    assert np.all(calc.local_maxima(x)[:-1] == i)


def test_cwt():
    # ???
    x = np.random.random(50)
    calc.cwt(x, np.arange(2, 5), calc.ricker_wavelet)


def test_sliding_window():
    x = np.arange(10)
    w = calc.sliding_window(x, 3)
    assert np.all(np.mean(w, axis=1) == [1, 2, 3, 4, 5, 6, 7, 8])
    w = calc.sliding_window_centered(x, 3)
    assert np.all(np.mean(w, axis=1) == [1 / 3.0, 1, 2, 3, 4, 5, 6, 7, 8, 26 / 3.0])


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
