import numpy as np


from pew.config import Config


def test_config():
    config = Config(10.0, 10.0, 0.5)
    # Standard stuff
    assert config.get_pixel_width() == 5.0
    assert config.get_pixel_height() == 10.0
    assert config.data_extent((10, 10)) == (0.0, 50.0, 0.0, 100.0)
