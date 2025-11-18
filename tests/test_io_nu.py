import json
import zipfile
from pathlib import Path
import pytest

import numpy as np

from pewlib.io import nu

@pytest.fixture(scope="module")
def image_path(tmp_path_factory) -> Path:
    path = Path(__file__).parent.joinpath("data", "nu", "Image001.zip")
    zp = zipfile.ZipFile(path)
    tmp_path = tmp_path_factory.mktemp("Image001")
    zp.extractall(tmp_path)
    return tmp_path.joinpath("Image001")

@pytest.fixture(scope="module")
def autob_path(tmp_path_factory) -> Path:
    path = Path(__file__).parent.joinpath("data", "nu", "autob.zip")
    zp = zipfile.ZipFile(path)
    tmp_path = tmp_path_factory.mktemp("Image001")
    zp.extractall(tmp_path)
    return tmp_path.joinpath("Image001")


def test_is_nu_acquisition_directory(image_path: Path):
    path = image_path.joinpath("00001")
    assert nu.is_nu_acquisition_directory(path)
    assert not nu.is_nu_acquisition_directory(path.parent)


def test_is_nu_image_directory(image_path: Path):
    assert nu.is_nu_image_directory(image_path)
    assert not nu.is_nu_image_directory(image_path.parent)


def test_apply_corrections(image_path: Path):
    with image_path.joinpath("TriggerCorrections.dat").open() as fp:
        corrections = json.load(fp)

    times = np.arange(10)
    corr = nu.apply_trigger_correction(times, corrections)
    assert np.all(corr == times + 80.0e-3)


def test_read_acquistion(image_path: Path):
    signals, masses, times, pulses, info = nu.read_laser_acquisition(
        image_path.joinpath("00001"), cycle=1, segment=1
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


def test_read_acquistion_blanking(autob_path: Path):
    signals, masses, times, pulses, info = nu.read_laser_acquisition(
        autob_path.joinpath("00001"), autoblank=False
    )
    assert np.all(~np.isnan(signals[8191:9999, 0:14]))
    signals, masses, times, pulses, info = nu.read_laser_acquisition(
        autob_path.joinpath("00001"), autoblank=True
    )
    assert np.all(np.isnan(signals[8191:9999, 0:14]))


def test_read_laser_image(image_path: Path):
    signals, masses, times, pulses, info = nu.read_laser_image(image_path)

    assert masses.size == 166
    assert np.isclose(masses[0], 50.9371, atol=0.001)
    assert np.isclose(masses[-1], 239.0213, atol=0.001)

    assert signals.shape == (734, 166)
    assert times.shape == (734,)
