import pytest
import numpy as np

from pew import Laser
from pew.srr import SRRLaser, SRRConfig

from typing import List, Tuple


def rand_data(names: List[str], shape: Tuple[int, int]) -> np.ndarray:
    dtype = [(name, float) for name in names]
    data = np.empty(shape, dtype=dtype)
    for name in names:
        data[name] = np.random.random(shape)
    return data


def test_srr():
    laser = SRRLaser(
        config=SRRConfig(1, 1, 1, warmup=0),
        data=[rand_data(["A", "B"], (10, 20)), rand_data(["A", "B"], (10, 20))],
    )
    assert laser.layers == 2
    assert laser.shape == (10, 10, 2)
    # Check config params
    assert laser.check_config_valid(SRRConfig(1, 1, 1, warmup=0))
    assert not laser.check_config_valid(SRRConfig(2, 1, 1, warmup=0))
    assert not laser.check_config_valid(SRRConfig(3, 1, 1, warmup=0))
    assert not laser.check_config_valid(SRRConfig(warmup=100))
    assert not laser.check_config_valid(SRRConfig(warmup=-1))


def test_srr_from_list():
    names = ["A", "B"]
    layers = [[np.random.random((10, 20)) for _ in range(2)] for _ in range(3)]
    with pytest.raises(AssertionError):
        SRRLaser.from_list(["A"], [[1, 2], [1, 2]])
    laser = SRRLaser.from_list(names, layers)
    assert laser.layers == 3
    assert laser.shape == (10, 10, 3)


def test_srr_from_lasers():
    lasers = [Laser(rand_data(["A", "B"], (10, 20))) for _ in range(2)]
    laser = SRRLaser.from_lasers(lasers)
    assert laser.layers == 2
    assert laser.shape == (10, 10, 2)
