import json
import zipfile
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


def test_read_acquistion(tmp_path: Path):
    path = Path(__file__).parent.joinpath("data", "nu", "Image001.zip")
    zp = zipfile.ZipFile(path)
    zp.extractall(tmp_path)

    signals, masses, times, pulses, info = nu.read_laser_acquisition(
        tmp_path.joinpath("Image001", "00001"), cycle=1, segment=1
    )

    assert masses.size == 166
    assert np.isclose(masses[0], 50.9371, atol=0.001)
    assert np.isclose(masses[-1], 239.0213, atol=0.001)

    assert signals.shape == (734, 166)
    assert times.shape == (734,)
    assert pulses.shape == (1767,)

    assert info["SampleName"] == "area"
    assert info["FirstLaserLineNumber"] == 1
    assert info["Username"] == "engineer"

    assert np.isclose(nu.eventtime_from_info(info), 0.05126)


def test_read_acquistion_blanking(tmp_path: Path):
    path = Path(__file__).parent.joinpath("data", "nu", "autob.zip")
    zp = zipfile.ZipFile(path)
    zp.extractall(tmp_path)

    signals, masses, times, pulses, info = nu.read_laser_acquisition(
        tmp_path.joinpath("Image001", "00001"), autoblank=False
    )
    assert np.all(~np.isnan(signals[8191:9999, 0:14]))
    signals, masses, times, pulses, info = nu.read_laser_acquisition(
        tmp_path.joinpath("Image001", "00001"), autoblank=True
    )
    assert np.all(np.isnan(signals[8191:9999, 0:14]))


def test_read_laser_image(tmp_path: Path):
    path = Path(__file__).parent.joinpath("data", "nu", "Image001.zip")
    zp = zipfile.ZipFile(path)
    zp.extractall(tmp_path)

    signals, masses, times = nu.read_laser_image(tmp_path.joinpath("Image001"))

    assert masses.size == 166
    assert np.isclose(masses[0], 50.9371, atol=0.001)
    assert np.isclose(masses[-1], 239.0213, atol=0.001)

    assert signals.shape == (734, 166)
    assert times.shape == (734,)
    assert times[0] < 0.0  # from subtracting first pulse
