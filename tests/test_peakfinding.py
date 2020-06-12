import numpy as np

from pew.lib.peakfinding import cwt, ricker_wavelet


def test_cwt():
    # ???
    x = np.random.random(50)
    cwt(x, np.arange(2, 5), ricker_wavelet)
