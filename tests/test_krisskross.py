import pytest
import numpy as np

from pew.srr.krisskross import SRR
from pew.srr.config import SRRConfig
from pew.srr.data import SRRData


def test_config_srr():
    config = SRRConfig(
        10.0, 10.0, 0.5, warmup=10.0, subpixel_offsets=[[0, 1], [0, 2]]
    )
    # Standard stuff
    assert config._warmup == 20
    assert config.magnification == 2
    assert config.get_pixel_width() == 5.0
    assert config.get_pixel_height() == 5.0
    assert config.data_extent((10, 20)) == (100.0, 200.0, 100.0, 150.0)
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


def test_data_srr():
    config = SRRConfig(5, 10, 0.5, warmup=0)
    data = SRRData([np.random.random((10, 20)), np.random.random((10, 20))])

    assert data.get(config).shape == (21, 21, 2)
    assert data.get(config, flat=True).shape == (21, 21)
    assert data.get(config, layer=1).shape == (20, 10)
    assert data.get(config, extent=(0, 10, 0, 10)).shape == (4, 4, 2)
    assert data.get(config, layer=1).shape == (20, 10)


def test_srr():
    kk = SRR(
        config=SRRConfig(1, 1, 1, warmup=0),
        data={
            "A": SRRData(
                [np.random.random((10, 15)), np.random.random((10, 20))]
            )
        },
    )
    assert kk.layers == 2
    # Check config params
    assert kk.check_config_valid(SRRConfig(1, 1, 1, warmup=0))
    assert not kk.check_config_valid(SRRConfig(2, 1, 1, warmup=0))
    assert not kk.check_config_valid(SRRConfig(3, 1, 1, warmup=0))
    assert not kk.check_config_valid(SRRConfig(warmup=100))
    assert not kk.check_config_valid(SRRConfig(warmup=-1))


def test_srr_from_structured():
    structured = np.empty((10, 20), dtype=[("A", float), ("B", float)])
    structured["A"] = np.random.random([10, 20])
    structured["B"] = np.random.random([10, 20])
    kk = SRR.from_structured(
        [structured, structured], config=SRRConfig()
    )
    assert kk.layers == 2
