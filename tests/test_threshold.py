import numpy as np

from pewlib.process.threshold import otsu


def test_otsu():
    np.random.seed(61945331)
    x = np.hstack(
        (np.random.normal(1.0, 1.0, size=500), np.random.normal(4.0, 2.0, size=500))
    )

    assert np.allclose(otsu(x), 3.0, atol=2e-1)

    x[::10] = np.nan

    assert np.allclose(otsu(x, remove_nan=True), 3.0, atol=2e-1)
