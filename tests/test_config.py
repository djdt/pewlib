import numpy as np
import pytest

from pewlib.config import Config, SpotConfig
from pewlib.srr import SRRConfig


def test_config():
    config = Config(10.0, 10.0, 0.5)
    # Standard stuff
    assert config.get_pixel_width() == 5.0
    assert config.get_pixel_height() == 10.0
    assert config.data_extent((10, 10)) == (0.0, 50.0, 0.0, 100.0)


def test_config_to_from_array():
    config = Config(1.0, 2.0, 3.0)
    array = config.to_array()
    assert array["spotsize"] == 1.0
    assert array["speed"] == 2.0
    assert array["scantime"] == 3.0
    config = Config.from_array(array)
    assert config.spotsize == 1.0
    assert config.speed == 2.0
    assert config.scantime == 3.0


def test_config_spot():
    config = SpotConfig(20.0, 50.0)
    # Standard stuff
    assert config.get_pixel_width() == 20.0
    assert config.get_pixel_height() == 50.0
    assert config.data_extent((10, 10)) == (0.0, 200.0, 0.0, 500.0)


def test_config_spot_to_from_array():
    config = SpotConfig(10.0)
    array = config.to_array()
    assert np.all(array["spotsize"] == [10.0, 10.0])
    config = SpotConfig.from_array(array)
    assert config.spotsize == 10.0
    assert config.spotsize_y == 10.0


def test_config_srr():
    config = SRRConfig(10.0, 10.0, 0.5, warmup=10.0, subpixel_offsets=[[0, 2], [1, 2]])
    # Standard stuff
    assert config._warmup == 20
    assert config.magnification == 2
    assert config.get_pixel_width() == 5.0
    assert config.get_pixel_height() == 5.0
    assert config.data_extent((10, 20)) == (100.0, 200.0, 100.0, 150.0)
    assert np.all(config.subpixel_offsets == [[0, 2], [1, 2]])

    # Test layers
    assert config.get_pixel_width(0) == 5.0
    assert config.get_pixel_height(0) == 10.0
    assert config.data_extent((10, 20), layer=0) == (0.0, 100.0, 0.0, 100.0)
    assert config.get_pixel_width(1) == 10.0
    assert config.get_pixel_height(1) == 5.0
    assert config.data_extent((10, 20), layer=1) == (0.0, 200.0, 0.0, 50.0)

    # Tests for subpixel offsets
    with pytest.raises(ValueError):
        config.subpixel_offsets = [0, 1]
    config.set_equal_subpixel_offsets(4)
    assert np.all(config._subpixel_offsets == [0, 1, 2, 3])
    assert config._subpixel_size == 4
    assert config.subpixels_per_pixel == 2


def test_config_srr_to_from_array():
    config = SRRConfig(1.0, 2.0, 1.0, warmup=4.0, subpixel_offsets=[[0, 2], [1, 2]])
    array = config.to_array()
    assert array["spotsize"] == 1.0
    assert array["speed"] == 2.0
    assert array["scantime"] == 1.0
    assert array["warmup"] == 4.0
    assert np.all(array["subpixel_offsets"] == [[0, 2], [1, 2]])
    config = SRRConfig.from_array(array)
    assert config.spotsize == 1.0
    assert config.speed == 2.0
    assert config.scantime == 1.0
    assert config.warmup == 4.0
    assert np.all(config.subpixel_offsets == [[0, 2], [1, 2]])
