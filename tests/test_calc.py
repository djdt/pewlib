import numpy as np

from pew.lib.calc import subpixel_offset


def test_subpixel_offset():
    x = np.ones((10, 10, 3))

    y = subpixel_offset(x, [(0, 0), (1, 1), (2, 3)], (2, 3))
    assert y.shape == (22, 33, 3)
    assert np.all(y[0:20, 0:30, 0] == 1)
    assert np.all(y[1:21, 1:31, 1] == 1)
    assert np.all(y[2:22, 3:33, 2] == 1)
