import numpy as np

from pew.lib.threshold import kmeans_threshold, otsu


def test_kmeans_threshold():
    x = np.zeros(100)
    x[:25] += 1.0
    x[:50] += 1.0
    x[:75] += 1.0

    t = kmeans_threshold(x, 4)

    assert np.allclose(t, [1.0, 2.0, 3.0])


def test_otsu():
    x = np.hstack(
        (np.random.normal(1.0, 1.0, size=500), np.random.normal(4.0, 2.0, size=500))
    )

    assert np.allclose(otsu(x), 3.0, atol=2e-1)
