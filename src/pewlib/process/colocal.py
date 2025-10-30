"""Colcalisation can be used to quantify the spacial relationship between
elements. A few ofthe many available algorithms are implemented in this file.

"""

import numpy as np

from pewlib.process.calc import normalise, shuffle_blocks


def li_icq(x: np.ndarray, y: np.ndarray) -> float:
    """Calculates Li's ICQ.

    The intenisty correlation quotient calculates the number of pixels where both
    x and y are above or below their means. A value of 0 indicates no correlation,
    below 0 segregation and above 0 colocalisation.

    Args:
        x: array
        y: array, same shape as `x`

    Returns:
        value between -0.5 and 0.5

    References:
        Li, Q. A Syntaxin 1, G o, and N-Type Calcium Channel Complex at
            a Presynaptic Nerve Terminal: Analysis by Quantitative Immunocolocalization
            Journal of Neuroscience, Society for Neuroscience, 2004, 24, 4070-4081
    """
    ux, uy = np.mean(x), np.mean(y)
    return np.sum((x - ux) * (y - uy) >= 0.0) / x.size - 0.5


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson's colocalisation coefficient.

    A value of 0 indicates no correlation, below 0 segregation and above 0
    colocalisation.

    Args:
        x: array
        y: array, same shape as `x`

    Returns:
        value between -1 and 1
    """
    return (np.mean(x * y) - (np.mean(x) * np.mean(y))) / (np.std(x) * np.std(y))


def pearsonr_probablity(
    x: np.ndarray,
    y: np.ndarray,
    block: int = 3,
    mask: np.ndarray | None = None,
    shuffle_partial: bool = False,
    n: int = 500,
) -> tuple[float, float]:
    """Evalulates Probability of Pearson's coefficient.

    Calculates Pearson's R of `x` and `y` then shuffles `y` `n` times, retesting
    Pearson's R. The probability is defined as the ratio of R's produced by the
    shuffling that are lower than the original R. Args `block`, `mask` and
    `shuffle_partial` are passed to 'shuffle_blocks'. Implemented as per
    Costes [1].

    Args:
        x: array
        y: array, same shape as `x`
        block: block size for shuffle
        mask: mask for shuffle
        shuffle_partial: shuffle partially masked blocks
        n: number of shuffles to perform

    Returns:
        Pearsons's r
        probability, p, of the r

    See Also:
        :func:`pewlib.process.colocal.pearsonr`
        :func:`pewlib.process.calc.shuffle_blocks`

    References:
        .. [1] Costes, S. V.; Daelemans, D.; Cho, E. H.; Dobbin, Z.; Pavlakis, G.
            & Lockett, S. Automatic and Quantitative Measurement of Protein-Protein
            Colocalization in Live Cells Biophysical Journal, Elsevier BV,
            2004, 86, 3993-4003
    """
    if mask is None:
        mask = np.ones(x.shape, dtype=bool)

    r = pearsonr(x[mask], y[mask])
    rs = np.empty(n, dtype=float)
    shuffled = y.copy()
    for i in range(n):
        shuffled = shuffle_blocks(
            shuffled,
            (block, block),
            mask,
            mode="inplace",
            shuffle_partial=shuffle_partial,
        )
        rs[i] = pearsonr(x[mask], shuffled[mask])

    return r, (rs > r).sum() / n


def manders(
    x: np.ndarray, y: np.ndarray, tx: float | None = None, ty: float | None = None
) -> tuple[float, float]:
    """Manders' correlation coefficients.

    Args:
        x: array
        y: array, same shape as `x`
        tx: threshold for `x`, defaults to x.min()
        ty: threshold for `y`, defaults to y.min()

    Returns:
        M1, factional overlap of `x` to `y`
        M2, factional overlap of `y` to `x`

    References:
        MANDERS, E. M. M.; VERBEEK, F. J. & J. A., ATEN
            Measurement of co-localization of objects in dual-colour confocal images
            Journal of Microscopy, Wiley, 1993, 169, 375-382
    """
    if tx is None:
        tx = np.amin(x)
    if ty is None:
        ty = np.amin(y)

    return np.sum(x, where=y > ty) / x.sum(), np.sum(y, where=x > tx) / y.sum()


def costes_threshold(
    x: np.ndarray, y: np.ndarray, target_r: float = 0.0
) -> tuple[float, float, float]:
    """Calculates Costes thresholds.

    Pearson's R is calculated for values of `x` and `y` that are above an increasing
    threshold. Once the calculated R value is above `target_r` the thresholds are
    returned. The threshold for `y` equals 'tx' * 'a' + 'b'.

    Args:
        x: array
        y: array, same shape as `x`
        target_r: value of R at which stop incrementing

    Returns:
        threshold for x, tx
        slope, a
        intercept, b

    See Also:
        :func:`pewlib.process.colocal.pearsonr`

    References:
        Costes, S. V.; Daelemans, D.; Cho, E. H.; Dobbin, Z.; Pavlakis, G.
            & Lockett, S. Automatic and Quantitative Measurement of Protein-Protein
            Colocalization in Live Cells Biophysical Journal, Elsevier BV,
            2004, 86, 3993-4003
    """
    b, a = np.polynomial.polynomial.polyfit(x.flat, y.flat, 1)

    thresholds = np.linspace(x.min(), x.max(), 256)

    for threshold in thresholds:
        idx = np.logical_or(x <= threshold, y <= (a * threshold + b))
        if np.all(x[idx] == x[idx][0]) or np.all(y[idx] == y[idx][0]):
            return thresholds[0], a, b
        if pearsonr(x[idx], y[idx]) > target_r:  # pragma: no cover
            break

    return threshold, a, b


def costes(
    x: np.ndarray, y: np.ndarray, n_scrambles: int = 200
) -> tuple[float, float, float, float]:  # pragma: no cover, covered in other funcs
    """Performs Costes colocalisation.

    The threshold at which no colocalisation appears (R < 0) is first calculated
    and then used to find Manders M1 and M2.

    Args:
        x: array
        y: array, same shape as `x`
        n_scrambles: scrambles for Pearson probability

    Returns:
        Pearson's R
        probability of Pearson's R
        Mander's M1
        Mander's M2

    See Also:
        :func:`pewlib.process.colocal.costes_threshold`
        :func:`pewlib.process.colocal.manders`
        :func:`pewlib.process.colocal.pearsonr_probablity`

    References:
        Costes, S. V.; Daelemans, D.; Cho, E. H.; Dobbin, Z.; Pavlakis, G.
            & Lockett, S. Automatic and Quantitative Measurement of Protein-Protein
            Colocalization in Live Cells Biophysical Journal, Elsevier BV,
            2004, 86, 3993-4003
    """
    x, y = normalise(x), normalise(y)
    t, a, b = costes_threshold(x, y)
    tx, ty = t, t * a + b
    pearson_r, r_prob = pearsonr_probablity(
        x, y, mask=np.logical_and(x > tx, y > ty), n=n_scrambles
    )
    m1, m2 = np.sum(x, where=x > tx) / x.sum(), np.sum(y, where=y > ty) / y.sum()

    return pearson_r, r_prob, m1, m2
