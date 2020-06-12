import numpy as np

from pew.lib.calc import kmeans


def kmeans_threshold(x: np.ndarray, k: int) -> np.ndarray:
    """Uses k-means clustering to group array into k clusters and produces k - 1
     thresholds using the minimum value of each cluster.
"""
    assert k > 1

    idx = kmeans(x.ravel(), k, max_iterations=k * 100).reshape(x.shape)
    return np.array([np.amin(x[idx == i]) for i in range(1, k)])


# def multiotsu(x: np.ndarray, levels: int, nbins: int = 256) -> np.ndarray:
#     assert levels == 2 or levels == 3
#     return _multiotsu.multiotsu(x, levels, nbins)


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
