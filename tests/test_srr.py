import pytest
import numpy as np

from pewlib import Laser
from pewlib.srr import SRRLaser, SRRConfig


def rand_data(names: list[str], shape: tuple[int, int]) -> np.ndarray:
    dtype = [(name, float) for name in names]
    data = np.empty(shape, dtype=dtype)
    for name in names:
        data[name] = np.random.random(shape)
    return data


def test_srr():
    data = [rand_data(["A", "B"], (20, 20)), rand_data(["A", "B"], (20, 20))]
    laser = SRRLaser(config=SRRConfig(1, 1, 1, warmup=0), data=data)
    assert laser.layers == 2
    assert laser.shape == (20, 20, 2)
    assert laser.extent == (0, 20.5, 0, 20.5)
    # Check config params
    assert laser.check_config_valid(SRRConfig(1, 1, 1, warmup=0))
    assert not laser.check_config_valid(SRRConfig(1, 3, 1, warmup=0))
    assert not laser.check_config_valid(SRRConfig(3, 1, 1, warmup=0))
    assert not laser.check_config_valid(SRRConfig(warmup=100))
    assert not laser.check_config_valid(SRRConfig(warmup=-1))

    # Get layer
    assert np.all(laser.get("A", layer=1) == data[1]["A"].T)

    # Get flat
    assert laser.get("A", flat=True).ndim == 2

    # Test adding, removing, renaming datas
    new_data = [np.random.random((20, 20)), np.random.random((20, 20))]
    laser.add("C", new_data)
    assert laser.elements == ("A", "B", "C")
    laser.remove("B")
    assert laser.elements == ("A", "C")
    laser.rename({"C": "B"})
    assert laser.elements == ("A", "B")


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


def test_srr_krisskross():
    data = np.empty((3, 3), dtype=[("a", float)])
    data["a"] = [[3, 2, 1], [2, 1, 0], [1, 0, 0]]

    laser = SRRLaser(config=SRRConfig(1, 1, 1, warmup=0), data=[data, data.copy()])

    kk = laser.get(flat=True)
    assert np.all(kk["a"][1] == np.array([1.5, 3, 2.5, 2, 1.5, 1, 0.5]))
    kk = laser.get(extent=(0, 1, 2, 3.5), flat=True)
    assert np.all(kk["a"][1] == np.array([1.5, 3]))


def test_srr_krisskross_mag_factors():
    a = np.tile([[0, 1], [1, 0]], (5, 5))
    b = np.rot90(a, 2)

    config = SRRConfig(15, 30, 0.5, 0, [(1, 2)])  # Equal
    laser = SRRLaser(
        [np.array(a, dtype=[("a", float)]), np.array(b, dtype=[("a", float)])],
        config=config,
    )
    assert laser.get("a", flat=True).shape == (21, 21)

    a = np.tile([[0, 0, 1, 1], [1, 1, 0, 0]], (5, 5))
    b = np.rot90(a, 2)

    config = SRRConfig(15, 30, 0.25, 0, [(1, 2)])  # Positive mag
    laser = SRRLaser(
        [np.array(a, dtype=[("a", float)]), np.array(b, dtype=[("a", float)])],
        config=config,
    )
    assert laser.get("a", flat=True).shape == (21, 21)

    a = np.tile([[0, 1], [0, 1], [1, 0], [1, 0]], (5, 5))
    b = np.rot90(a, 2)

    config = SRRConfig(15, 30, 1, 0, [(1, 2)])  # Negative mag
    laser = SRRLaser(
        [np.array(a, dtype=[("a", float)]), np.array(b, dtype=[("a", float)])],
        config=config,
    )
    assert laser.get("a", flat=True).shape == (21, 21)
