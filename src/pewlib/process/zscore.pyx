#cython: language_level=3

import numpy as np

from libc.math cimport sqrt

DTYPE = np.byte

def zscore_peaks(
    x: np.ndarray, lag: Py_ssize_t, threshold: float = 3.3, influence: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    cdef double[:] x_view = x
    cdef char[:] signal = np.zeros(x.size, dtype=DTYPE)
    cdef double[:] filtered = x.copy()

    cdef Py_ssize_t i
    cdef double mean
    cdef double std
    for i in range(lag, x.shape[0]):
        mean = calcmean(filtered, i - lag, i)
        std = calcstd(filtered, i - lag, i , mean)

        if abs(x_view[i] - mean) > std * threshold:
            signal[i] = 1 if x_view[i] > mean else -1
            filtered[i] = influence * x_view[i] + (1.0 - influence) * filtered[i - 1]


    return np.array(signal), np.array(filtered)


cdef double calcmean(double[:] x, Py_ssize_t start, Py_ssize_t end):
    cdef double sum = 0.0
    cdef Py_ssize_t i
    for i in range(start, end):
        sum += x[i]
    return sum / (end - start)


cdef double calcstd(double[:] x, Py_ssize_t start, Py_ssize_t end, double mean):
    cdef double std = 0.0
    cdef Py_ssize_t i
    for i in range(start, end):
        std += (x[i] - mean) ** 2
    return sqrt(std / (end - start))
