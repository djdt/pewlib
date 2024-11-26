import numpy as np

from pewlib.process import colocal

a = np.tile([[0.0, 1.0], [0.0, 1.0]], (10, 10))
b = np.tile([[0.0, 1.0], [1.0, 0.0]], (10, 10))
c = np.tile([[1.0, 0.0], [1.0, 0.0]], (10, 10))
d = np.tile([[1.0, 2.0], [3.0, 4.0]], (10, 10))
e = np.tile([[1.0, 2.0], [4.0, 3.0]], (10, 10))


def test_li_icq():
    assert colocal.li_icq(a, a) == 0.5
    assert colocal.li_icq(a, b) == 0.0
    assert colocal.li_icq(a, c) == -0.5


def test_pearson_r():
    assert colocal.pearsonr(a, a) == 1.0
    assert colocal.pearsonr(a, b) == 0.0
    assert colocal.pearsonr(a, c) == -1.0


def test_pearson_r_probability():
    np.random.seed(872634)
    r, p = colocal.pearsonr_probablity(a, b, block=3, n=500, shuffle_partial=False)
    assert r == 0.0
    assert 0.66 > p > 0.33

    r, p = colocal.pearsonr_probablity(a, a, block=3, n=500, shuffle_partial=False)
    assert r == 1.0

    r, p = colocal.pearsonr_probablity(
        a, np.random.random(a.shape), block=3, n=500, shuffle_partial=False
    )
    assert p >= 0.9


def test_manders():
    assert colocal.manders(a, b) == (0.5, 0.5)  # Tx, Ty as min
    assert colocal.manders(a, a, 0) == (1.0, 1.0)
    assert colocal.manders(a, b, 0, 0) == (0.5, 0.5)
    assert colocal.manders(a, c, 0, 0) == (0.0, 0.0)
    assert colocal.manders(a, d, 0, 0) == (1.0, 0.6)
    assert colocal.manders(a, e, 0, 0) == (1.0, 0.5)


def test_costes_threshold():
    assert np.allclose(colocal.costes_threshold(a, a), (0.0, 1.0, 0.0))
    assert np.allclose(colocal.costes_threshold(a, b), (1.0, 0.0, 0.5))
