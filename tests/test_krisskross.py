import pytest
import numpy as np

from laserlib.krisskross.krisskross import KrissKross
from laserlib.krisskross.config import KrissKrossConfig
from laserlib.krisskross.data import KrissKrossData


def test_config_krisskross():
    config = KrissKrossConfig(
        10.0, 10.0, 0.5, warmup=10.0, subpixel_offsets=[[0, 1], [0, 2]]
    )
    # Standard stuff
    assert config._warmup == 20
    assert config.magnification == 2
    assert config.get_pixel_width() == 5.0
    assert config.get_pixel_height() == 5.0
    data = np.zeros([10, 20])
    assert config.data_extent(data) == (100.0, 200.0, 100.0, 150.0)
    # Test layers
    assert config.get_pixel_width(0) == 5.0
    assert config.get_pixel_height(0) == 10.0
    assert config.data_extent(data, 0) == (0.0, 100.0, 0.0, 100.0)
    assert config.get_pixel_width(1) == 10.0
    assert config.get_pixel_height(1) == 5.0
    assert config.data_extent(data, 1) == (0.0, 200.0, 0.0, 50.0)
    # Tests for subpixel offsets
    with pytest.raises(ValueError):
        config.subpixel_offsets = [0, 1]
    config.set_equal_subpixel_offsets(4)
    assert np.all(config._subpixel_offsets == [0, 1, 2, 3])
    assert config._subpixel_size == 4
    assert config.subpixels_per_pixel == 2


def test_data_krisskross():
    config = KrissKrossConfig(5, 10, 0.5, warmup=0)
    data = KrissKrossData([np.random.random((10, 20)), np.random.random((10, 20))])

    assert data.get(config).shape == (21, 21, 2)
    assert data.get(config, flat=True).shape == (21, 21)
    assert data.get(config, layer=1).shape == (20, 10)
    assert data.get(config, extent=(0, 10, 0, 10)).shape == (4, 4, 2)
    assert data.get(config, layer=1).shape == (20, 10)


def test_krisskross():
    kk = KrissKross(
        config=KrissKrossConfig(1, 1, 1, warmup=0),
        data={
            "A": KrissKrossData(
                [np.random.random((10, 15)), np.random.random((10, 20))]
            )
        },
    )
    assert kk.layers == 2
    # Check config params
    assert kk.check_config_valid(KrissKrossConfig(1, 1, 1, warmup=0))
    assert not kk.check_config_valid(KrissKrossConfig(2, 1, 1, warmup=0))
    assert not kk.check_config_valid(KrissKrossConfig(3, 1, 1, warmup=0))
    assert not kk.check_config_valid(KrissKrossConfig(warmup=100))
    assert not kk.check_config_valid(KrissKrossConfig(warmup=-1))


def test_krisskross_from_structured():
    structured = np.empty((10, 20), dtype=[("A", float), ("B", float)])
    structured["A"] = np.random.random([10, 20])
    structured["B"] = np.random.random([10, 20])
    kk = KrissKross.from_structured(
        [structured, structured], config=KrissKrossConfig()
    )
    assert kk.layers == 2
