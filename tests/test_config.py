import pytest
import numpy as np


from laserlib.config import LaserConfig
from laserlib.krisskross.config import KrissKrossConfig


def test_config():
    config = LaserConfig(10.0, 10.0, 0.5)
    # Standard stuff
    assert config.get_pixel_width() == 5.0
    assert config.get_pixel_height() == 10.0
    data = np.random.random([10, 10])
    assert config.data_extent(data) == (0.0, 50.0, 0.0, 100.0)


def test_config_krisskross():
    config = KrissKrossConfig(10.0, 10.0, 0.5, warmup=10.0)
    # Standard stuff
    assert config._warmup == 20
    assert config.magnification == 2
    assert config.get_pixel_width() == 5.0
    assert config.get_pixel_height() == 5.0
    data = np.random.random([10, 10])
    assert config.data_extent(data) == (100.0, 150.0, 100.0, 150.0)
    # Test layers
    assert config.get_pixel_width(0) == 5.0
    assert config.get_pixel_height(0) == 10.0
    assert config.data_extent(data, 0) == (0.0, 50.0, 0.0, 100.0)
    assert config.get_pixel_width(1) == 10.0
    assert config.get_pixel_height(1) == 5.0
    assert config.data_extent(data, 1) == (0.0, 100.0, 0.0, 50.0)
    # Tests for subpixel offsets
    with pytest.raises(ValueError):
        config.subpixel_offsets = [[0, 1]]
    config.set_equal_subpixel_offsets(4)
    assert np.all(config._subpixel_offsets == [0, 1, 2, 3])
    assert config._subpixel_size == 4
