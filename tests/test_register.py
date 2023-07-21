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


def test_overlap_arrays_nan():
    a = np.ones((3, 3))
    b = np.full_like(a, np.nan)

    c = register.overlap_arrays([a, b], offsets=[(0, 0), (0, 0)])
    assert np.all(c == 1.0)
    c = register.overlap_arrays([a, b], offsets=[(0, 0), (0, 0)], mode="mean")
    assert np.all(c == 1.0)
    c = register.overlap_arrays([a, b], offsets=[(0, 0), (0, 0)], mode="sum")
    assert np.all(c == 1.0)


def test_overlap_structured_arrays():
    a = np.empty((2, 2), dtype=[("a", float), ("c", float)])
    b = np.empty((2, 2), dtype=[("b", float), ("c", float)])
    a["a"] = 1
    a["c"] = 2
    b["b"] = 3
    b["c"] = 4

    c = register.overlap_structured_arrays([a, b], offsets=[(0, 0), (1, 1)])
    assert c.shape == (3, 3)
    assert c.dtype.names == ("a", "c", "b")

    assert np.all(np.nan_to_num(c["a"]) == [[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    assert np.all(np.nan_to_num(c["b"]) == [[0, 0, 0], [0, 3, 3], [0, 3, 3]])
    assert np.all(np.nan_to_num(c["c"]) == [[2, 2, 0], [2, 4, 4], [0, 4, 4]])

    c = register.overlap_structured_arrays(
        [a, b], offsets=[(0, 0), (1, 1)], mode="mean"
    )

    assert np.all(np.nan_to_num(c["a"]) == [[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    assert np.all(np.nan_to_num(c["b"]) == [[0, 0, 0], [0, 3, 3], [0, 3, 3]])
    assert np.all(np.nan_to_num(c["c"]) == [[2, 2, 0], [2, 3, 4], [0, 4, 4]])

    c = register.overlap_structured_arrays(
        [a, b], offsets=[(0, 0), (1, 1)], mode="sum"
    )

    assert np.all(np.nan_to_num(c["a"]) == [[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    assert np.all(np.nan_to_num(c["b"]) == [[0, 0, 0], [0, 3, 3], [0, 3, 3]])
    assert np.all(np.nan_to_num(c["c"]) == [[2, 2, 0], [2, 6, 4], [0, 4, 4]])
