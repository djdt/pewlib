import numpy as np

from pewlib.process import peakfinding


def test_peakfinding_cwt():
    x = np.concatenate((np.zeros(10), np.sin(np.linspace(0.0, 25.0, 100))))
    x[x < 0.0] = 0.0
    x[np.array([35, 55])] += 0.5

    peaks = peakfinding.find_peaks_cwt(x, 8, 12, ridge_min_snr=3.3)

    assert peaks.size == 4
    assert np.all(peaks["width"] == 20)


def test_peakfinding_windowed():
    x = np.concatenate((np.zeros(10), np.sin(np.linspace(0.0, 25.0, 100))))
    x[x < 0.0] = 0.0
    x[np.array([35, 55])] += 0.5

    def constant_threshold(x, axis):
        return 0.1

    peaks = peakfinding.find_peaks_windowed(x, 29, np.median, constant_threshold)
    assert peaks.size == 4


def test_peakfinding_peaks_from_edges():
    np.random.seed(83717)
    x = np.random.random(20)
    peaks = peakfinding.peaks_from_edges(x, np.array([5]), [15])

    # Test the baseline and height methods

    assert peaks.size == 1
    assert peaks[0]["width"] == 10
    assert peaks[0]["height"] + peaks[0]["base"] == np.amax(x[5:15])

    peaks = peakfinding.peaks_from_edges(
        x, np.array([5]), [15], baseline=np.ones_like(x), height_method="center"
    )
    assert peaks[0]["base"] == 1.0
    assert peaks[0]["height"] == x[10] - 1.0

    peaks = peakfinding.peaks_from_edges(x, np.array([5]), [15], base_method="edge")
    assert peaks[0]["base"] == np.amin((x[5], x[15]))

    peaks = peakfinding.peaks_from_edges(x, np.array([5]), [15], base_method="minima")
    assert peaks[0]["base"] == np.amin(x[5:15])

    peaks = peakfinding.peaks_from_edges(
        x, np.array([5]), [15], base_method="prominence"
    )
    assert peaks[0]["base"] == np.amax((x[5], x[15]))

    peaks = peakfinding.peaks_from_edges(x, np.array([5]), [15], base_method="zero")
    assert peaks[0]["base"] == 0.0


def test_peakfinding_insert_missing():
    peaks = np.array(
        [
            (1.0, 2, 10.0, 0.0, 10, 10, 10, 12),
            (1.0, 2, 10.0, 0.0, 15, 15, 15, 17),
            (1.0, 2, 10.0, 0.0, 20, 20, 20, 22),
            # (1.0, 2, 10.0, 0.0, 25, 25, 25, 27),
            (1.0, 2, 10.0, 0.0, 30, 30, 30, 32),
            (1.0, 2, 10.0, 0.0, 35, 35, 35, 37),
            (1.0, 2, 10.0, 0.0, 40, 40, 40, 42),
            (1.0, 2, 10.0, 0.0, 45, 45, 45, 47),
            # (1.0, 2, 10.0, 0.0, 50, 50, 50, 52),
            (1.0, 2, 10.0, 0.0, 55, 55, 55, 57),
        ],
        dtype=peakfinding.PEAK_DTYPE,
    )

    peaks = peakfinding.insert_missing_peaks(peaks)
    assert peaks.size == 10


def test_peakfinding_peak_filtering():
    peaks = np.array(
        [
            (10.0, 10, 100.0, 0.0, 10, 10, 5, 15),
            (10.0, 10, 10.0, 0.0, 10, 10, 15, 25),
            (10.0, 2, 100.0, 0.0, 10, 10, 25, 30),
            (1.0, 10, 100.0, 0.0, 10, 10, 35, 45),
        ],
        dtype=peakfinding.PEAK_DTYPE,
    )
    assert peakfinding.filter_peaks(peaks, min_area=50.0).size == 3
    assert peakfinding.filter_peaks(peaks, min_height=5.0).size == 3
    assert peakfinding.filter_peaks(peaks, min_width=5).size == 3
