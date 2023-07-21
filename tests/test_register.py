import numpy as np

from pewlib.laser import Laser
from pewlib.process import register


def test_anchor_offset():
    a = np.empty((9, 9))
    b = np.empty((5, 5))

    assert register.anchor_offset(a, b, "top left") == (0, 0)
    assert register.anchor_offset(a, b, "top right") == (0, 4)
    assert register.anchor_offset(a, b, "bottom left") == (4, 0)
    assert register.anchor_offset(a, b, "bottom right") == (4, 4)
    assert register.anchor_offset(a, b, "center") == (2, 2)


def test_fft_regsiter_offset():
    np.random.seed(940926)

    # random no offset
    a = np.random.random((100, 100))
    b = np.random.random((100, 100))

    offset = register.fft_register_offset(a, b)
    assert np.all(offset == (0, 0))

    # perfect overlap circles
    xx, yy = np.mgrid[:100, :100]
    a[(xx - 50) ** 2 + (yy - 75) ** 2 < 50] += 5.0
    b[(xx - 25) ** 2 + (yy - 25) ** 2 < 50] += 5.0

    offset = register.fft_register_offset(a, b)
    assert np.all(offset == (25, 50))
    offset = register.fft_register_offset(b, a)
    assert np.all(offset == (-25, -50))

    # bars
    a = np.random.random((100, 100))
    b = np.random.random((100, 100))

    a[10:20] += 5.0
    b[60:70] += 5.0

    offset = register.fft_register_offset(a, b)
    assert np.all(offset == (-50, 0))
    offset = register.fft_register_offset(b, a)
    assert np.all(offset == (50, 0))


def test_overlap_arrays():
    a = np.ones((2, 2))
    b = np.ones((2, 2)) * 2
    c = np.ones((2, 2)) * 3

    offsets = [(0, 0), (-1, 1), (0, 1)]
    # _   b   b
    # a   abc bc
    # a   ac  c

    # Test overlap methods
    d = register.overlap_arrays([a, b, c], offsets=offsets, mode="replace")
    assert d.shape == (3, 3)
    assert np.isnan(d[0][0])
    assert np.all(d[~np.isnan(d)] == [2, 2, 1, 3, 3, 1, 3, 3])

    d = register.overlap_arrays([a, b, c], offsets=offsets, mode="mean")
    assert np.all(d[~np.isnan(d)] == [2, 2, 1, 2, 2.5, 1, 2, 3])

    d = register.overlap_arrays([a, b, c], offsets=offsets, mode="sum")
    assert np.all(d[~np.isnan(d)] == [2, 2, 1, 6, 5, 1, 4, 3])

    d = register.overlap_arrays([a, b, c], offsets=[(-1, -1), (2, 2), (-2, 2)], fill=10)
    assert np.all(
        d
        == [
            [10, 10, 10, 3, 3],
            [1, 1, 10, 3, 3],
            [1, 1, 10, 10, 10],
            [10, 10, 10, 10, 10],
            [10, 10, 10, 2, 2],
            [10, 10, 10, 2, 2],
        ]
    )


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
