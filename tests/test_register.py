import numpy as np

from pewlib.laser import Laser
from pewlib.process import register


def test_fft_regsiter_images():
    np.random.seed(940926)

    # random no offset
    a = np.random.random((100, 100))
    b = np.random.random((100, 100))

    offset = register.fft_register_images(a, b)
    assert np.all(offset == (0, 0))

    # perfect overlap circles
    xx, yy = np.mgrid[:100, :100]
    a[(xx - 50) ** 2 + (yy - 75) ** 2 < 50] += 5.0
    b[(xx - 25) ** 2 + (yy - 25) ** 2 < 50] += 5.0

    offset = register.fft_register_images(a, b)
    assert np.all(offset == (25, 50))
    offset = register.fft_register_images(b, a)
    assert np.all(offset == (-25, -50))

    # bars
    a = np.random.random((100, 100))
    b = np.random.random((100, 100))

    a[10:20] += 5.0
    b[60:70] += 5.0

    offset = register.fft_register_images(a, b)
    assert np.all(offset == (-50, 0))
    offset = register.fft_register_images(b, a)
    assert np.all(offset == (50, 0))


def test_overlap_arrays():
    a = np.arange(9.0).reshape(3, 3)
    b = np.arange(9.0).reshape(3, 3)

    # Test overlap methods
    c = register.overlap_arrays(a, b, offset=(1, 1), mode="replace")
    assert c.shape == (4, 4)
    assert np.isnan(c[0, 3])
    assert np.isnan(c[3, 0])
    assert np.all(c[~np.isnan(c)] == [0, 1, 2, 3, 0, 1, 2, 6, 3, 4, 5, 6, 7, 8])
    c = register.overlap_arrays(a, b, offset=(1, 1), mode="mean")
    assert np.all(c[~np.isnan(c)] == [0, 1, 2, 3, 2, 3, 2, 6, 5, 6, 5, 6, 7, 8])
    c = register.overlap_arrays(a, b, offset=(1, 1), mode="sum")
    assert np.all(c[~np.isnan(c)] == [0, 1, 2, 3, 4, 6, 2, 6, 10, 12, 5, 6, 7, 8])

    # Test various offsets
    for i in range(-5, 5):
        for j in range(-5, 5):
            c = register.overlap_arrays(a, b, offset=(i, j), mode="sum")
            assert np.nansum(c) == 72.0

    # Test non nan fill
    c = register.overlap_arrays(a, b, offset=(-3, 0), fill=0.0, mode="replace")
    assert np.all(c[:3] == b)
    c = register.overlap_arrays(a, b, offset=(-3, 0), fill=0.0, mode="mean")
    assert np.all(c[:3] == b)
    c = register.overlap_arrays(a, b, offset=(-3, 0), fill=0.0, mode="sum")
    assert np.all(c[:3] == b)


def test_overlap_structured_arrays():
    a = np.empty((4, 4), dtype=[("a", float), ("c", float)])
    a["a"].flat = np.arange(16.0)
    a["c"].flat = np.arange(16.0)
    b = np.empty((4, 4), dtype=[("b", float), ("c", float)])
    b["b"].flat = np.arange(16.0)
    b["c"] = 1.0

    d = register.overlap_structured_arrays(a, b, offset=(2, 2), mode="replace")

    # Test overlap methods
    assert d.shape == (6, 6)
    assert np.all(d["a"][:4, :4] == a["a"])
    assert np.all(d["b"][2:, 2:] == b["b"])
    assert np.all(d["c"][2:, 2:] == 1.0)
    assert np.all(d["c"][:2, :2] == a["a"][:2, :2])

    d = register.overlap_structured_arrays(a, b, offset=(2, 2), mode="mean")

    # Test overlap methods
    assert np.all(d["c"][2:4, 2:4] == np.mean([a["c"][:2, :2], b["c"][:2, :2]], axis=0))

    d = register.overlap_structured_arrays(a, b, offset=(2, 2), mode="sum")
    assert np.all(d["c"][2:4, 2:4] == np.sum([a["c"][:2, :2], b["c"][:2, :2]], axis=0))


def test_overlap_laser():
    a = np.empty((3, 3), dtype=[("a", float), ("c", float)])
    a["a"].flat = np.arange(9.0)
    a["c"].flat = 1.0
    b = np.empty((5, 5), dtype=[("b", float), ("c", float)])
    b["b"].flat = np.arange(25.0)
    b["c"] = 1.0

    la = Laser(data=a)
    lb = Laser(data=b)

    lc = register.overlap_lasers(la, lb, anchor="top left")
    assert np.all(
        lc.data["c"][~np.isnan(lc.data["c"])]
        == [
            [2, 2, 2, 1, 1],
            [2, 2, 2, 1, 1],
            [2, 2, 2, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    lc = register.overlap_lasers(la, lb, anchor="top right")
    assert np.all(
        lc.data["c"][~np.isnan(lc.data["c"])]
        == [
            [1, 1, 2, 2, 2],
            [1, 1, 2, 2, 2],
            [1, 1, 2, 2, 2],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    lc = register.overlap_lasers(la, lb, anchor="bottom left")
    lc = register.overlap_lasers(la, lb, anchor="bottom right")
    lc = register.overlap_lasers(la, lb, anchor="center")

test_overlap_laser()
