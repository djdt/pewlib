import numpy as np


from laserlib.config import LaserConfig


def test_config():
    config = LaserConfig(10.0, 10.0, 0.5)

    assert config.get_pixel_width() == 5.0
    assert config.get_pixel_height() == 10.0
    data = np.random.random([10, 10])
    assert config.data_extent(data) == (0.0, 50.0, 0.0, 100.0)
