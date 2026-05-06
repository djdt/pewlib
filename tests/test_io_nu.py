import json
import zipfile
from pathlib import Path
import pytest

import numpy as np

from pewlib.io import nu


@pytest.fixture(scope="module")
def image_path(tmp_path_factory) -> Path:
    path = Path(__file__).parent.joinpath("data", "nu", "TestImage.zip")
    zp = zipfile.ZipFile(path)
    tmp_path = tmp_path_factory.mktemp("TestImage")
    zp.extractall(tmp_path)
    return tmp_path.joinpath("TestImage")


# @pytest.fixture(scope="module")
# def autob_path(tmp_path_factory) -> Path:
#     path = Path(__file__).parent.joinpath("data", "nu", "autob.zip")
#     zp = zipfile.ZipFile(path)
#     tmp_path = tmp_path_factory.mktemp("Image001")
#     zp.extractall(tmp_path)
#     return tmp_path.joinpath("Image001")


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
    assert np.all(corr == times + 24.0e-3)


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


# def test_read_acquistion_blanking(autob_path: Path):
#     raise NotImplementedError
    # signals, masses, times, pulses, info = nu.read_laser_acquisition(
    #     autob_path.joinpath("00001"), autoblank=False
    # )
    # assert np.all(~np.isnan(signals[8191:9999, 0:14]))
    # signals, masses, times, pulses, info = nu.read_laser_acquisition(
    #     autob_path.joinpath("00001"), autoblank=True
    # )
    # assert np.all(np.isnan(signals[8191:9999, 0:13]))


def test_read_laser_image(image_path: Path):
    signals, masses, times, pulses, info = nu.read_laser_image(image_path)

    assert masses.size == 186
    assert np.isclose(masses[0], 30.9552, atol=0.001)
    assert np.isclose(masses[-1], 240.0225, atol=0.001)

    assert len(signals) == 2
    assert signals[0].shape == (4273, 186)
    assert len(times) == 2
    assert times[0].shape == (4273,)

    image = nu.sync_data_with_laser_info(signals, times, pulses, info)

    assert image.shape == (10, 80, 186)

    image = nu.sync_data_with_laser_info(
        signals, times, pulses, info, sum_overlaps=False
    )

    assert image.shape == (10, 160, 186)
