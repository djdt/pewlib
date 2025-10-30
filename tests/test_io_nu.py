import json
from pathlib import Path

import numpy as np

from pewlib.io import nu


def test_is_nu_acquisition_directory():
    path = Path(__file__).parent.joinpath("data", "nu", "Image001", "00001")
    assert nu.is_nu_acquisition_directory(path)
    assert not nu.is_nu_acquisition_directory(path.parent)


def test_is_nu_image_directory():
    path = Path(__file__).parent.joinpath("data", "nu", "Image001")
    assert nu.is_nu_image_directory(path)
    assert not nu.is_nu_image_directory(path.parent)


def test_apply_corrections():
    path = Path(__file__).parent.joinpath(
        "data", "nu", "Image001", "TriggerCorrections.dat"
    )

    with path.open() as fp:
        corrections = json.load(fp)

    times = np.arange(10)
    corr = nu.apply_trigger_correction(times, corrections)
    assert np.all(corr == times + 80.0e-3)


def test_read_acuistion():
    path = Path(__file__).parent.joinpath("data", "nu", "Image001", "00001")

    signals, masses, times, pulses, info = nu.read_laser_acquisition(path)

    assert masses.size == 166
    assert np.isclose(masses[0], 50.9371, atol=0.001)
    assert np.isclose(masses[-1], 239.0213, atol=0.001)

    assert signals.shape == (734, 166)
    assert times.shape == (734,)
    assert pulses.shape == (1767,)

    assert info["SampleName"] == "area"
    assert info["FirstLaserLineNumber"] == 1
    assert info["Username"] == "engineer"


def test_read_laser_image():
    path = Path(__file__).parent.joinpath("data", "nu", "Image001")

    signals, masses, times = nu.read_laser_image(path)

    assert masses.size == 166
    assert np.isclose(masses[0], 50.9371, atol=0.001)
    assert np.isclose(masses[-1], 239.0213, atol=0.001)

    assert signals.shape == (734, 166)
    assert times.shape == (734,)
    assert times[0] < 0.0  # from subtracting first pulse
