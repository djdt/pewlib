import numpy as np


def otsu(x: np.ndarray) -> float:
    """Calculates the otsu threshold of the input array.
    https://github.com/scikit-image/scikit-image/blob/master/skimage/filters/thresholding.py
"""
    hist, bin_edges = np.histogram(x, bins=256)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

    w1 = np.cumsum(hist)
    w2 = np.cumsum(hist[::-1])[::-1]

    u1 = np.cumsum(hist * bin_centers) / w1
    u2 = (np.cumsum((hist * bin_centers)[::-1]) / w2[::-1])[::-1]

    i = np.argmax(w1[:-1] * w2[1:] * (u1[:-1] - u2[1:]) ** 2)
    return bin_centers[i]
