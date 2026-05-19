import json
import zipfile
from pathlib import Path
import pytest

import numpy as np

from pewlib.io import nu



@pytest.fixture(scope="module")
def image_path(tmp_path_factory, request) -> Path:
    path = Path(__file__).parent.joinpath("data", "nu", request.param)
    zp = zipfile.ZipFile(path.with_suffix(".zip"))
    tmp_path = tmp_path_factory.mktemp("TestImage")
    zp.extractall(tmp_path)
    return tmp_path.joinpath(request.param)


@pytest.mark.parametrize("image_path", ["TestImage"], indirect=True)
def test_is_nu_acquisition_directory(image_path: Path):
    path = image_path.joinpath("00001")
    assert nu.is_nu_acquisition_directory(path)
    assert not nu.is_nu_acquisition_directory(path.parent)


@pytest.mark.parametrize("image_path", ["TestImage"], indirect=True)
def test_is_nu_image_directory(image_path: Path):
    assert nu.is_nu_image_directory(image_path)
    assert not nu.is_nu_image_directory(image_path.parent)

@pytest.mark.parametrize("image_path", ["TestImage"], indirect=True)
def test_contains_nu_image_directory(image_path: Path):
    assert not nu.contains_nu_image_directory(image_path)
    assert nu.contains_nu_image_directory(image_path.parent)


@pytest.mark.parametrize("image_path", ["TestImage"], indirect=True)
def test_apply_corrections(image_path: Path):
    with image_path.joinpath("TriggerCorrections.dat").open() as fp:
        corrections = json.load(fp)

    times = np.arange(10)
    corr = nu.apply_trigger_correction(times, corrections)
    assert np.all(corr == times + 24.0e-3)


@pytest.mark.parametrize("image_path", ["TestImage"], indirect=True)
def test_read_acquistion(image_path: Path):
    signals, masses, times, pulses, info = nu.read_laser_acquisition(
        image_path.joinpath("00001"), cycle=1, segment=1
    )

    assert masses.size == 186
    assert np.isclose(masses[0], 30.9552, atol=0.001)
    assert np.isclose(masses[-1], 240.0225, atol=0.001)

    assert signals.shape == (4273, 186)
    assert times.shape == (4273,)
    assert pulses.shape == (805,)

    assert info["SampleName"] == "Y sample cut 8"
    assert info["FirstLaserLineNumber"] == 1
    assert info["Username"] == "engineer"

    assert np.isclose(nu.eventtime_from_info(info), 0.005178)


@pytest.mark.parametrize("image_path", ["TestImage"], indirect=True)
def test_read_laser_image(image_path: Path):
    signals, masses, times, pulses, info = nu.read_laser_image(image_path)

    assert masses.size == 186
    assert np.isclose(masses[0], 30.9552, atol=0.001)
    assert np.isclose(masses[-1], 240.0225, atol=0.001)

    assert len(signals) == 2
    assert signals[0].shape == (4273, 186)
    assert len(times) == 2
    assert times[0].shape == (4273,)

    image, pos = nu.sync_data_with_laser_info(signals, times, pulses, info)

    assert image.shape == (10, 80, 186)

    image, pos = nu.sync_data_with_laser_info(
        signals, times, pulses, info, apply_overlap=False
    )

    assert image.shape == (10, 161, 186)


@pytest.mark.parametrize("image_path", ["ImageRTL"], indirect=True)
def test_read_laser_image_right_to_left(image_path: Path):
    signals, masses, times, pulses, info = nu.read_laser_image(image_path)
    image, pos = nu.sync_data_with_laser_info(signals, times, pulses, info)

    assert image.shape == (25, 121, 195)


@pytest.mark.parametrize("image_path", ["ImageTTB"], indirect=True)
def test_read_laser_image_top_to_bottom(image_path: Path):
    signals, masses, times, pulses, info = nu.read_laser_image(image_path)
    image, pos = nu.sync_data_with_laser_info(signals, times, pulses, info)

    assert image.shape == (103, 25, 195)


@pytest.mark.parametrize("image_path", ["ImageLASSO"], indirect=True)
def test_read_laser_image_lasso(image_path: Path):
    signals, masses, times, pulses, info = nu.read_laser_image(image_path)
    image, pos = nu.sync_data_with_laser_info(signals, times, pulses, info)

    assert image.shape == (25, 86, 195)
