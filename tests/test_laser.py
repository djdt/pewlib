import numpy as np
import pytest

from laserlib.laser import Laser
from laserlib.data import LaserData


def test_default_laser():
    laser = Laser()
    assert len(laser.isotopes) == 0
    assert laser.layers == 1
    assert laser.shape is None
    assert laser.extent == (0, 0, 0, 0)
    with pytest.raises(KeyError):
        assert laser.get("NONE")


def test_laser_bad_shape():
    with pytest.raises(AssertionError):
        Laser(
            data={
                "A": LaserData(np.zeros([5, 5])),
                "B": LaserData(np.zeros([5, 5])),
                "C": LaserData(np.zeros([4, 5])),
            }
        )


def test_laser():
    laser = Laser(
        data={
            "A": LaserData(np.random.random([10, 10])),
            "B": LaserData(np.random.random([10, 10])),
        }
    )
    assert laser.isotopes == ["A", "B"]
    assert laser.shape == (10, 10)
    assert laser.extent == (0, 350, 0, 350)

    assert np.all(laser.get("A") == laser.data["A"].data)

    structured = laser.get_structured()
    assert np.all(structured["A"] == laser.get("A"))
    assert np.all(structured["B"] == laser.get("B"))


def test_laser_from_structured():
    structured = np.empty((10, 10), dtype=[("A", float), ("B", float)])
    structured["A"] = np.random.random([10, 10])
    structured["B"] = np.random.random([10, 10])
    laser = Laser.from_structured(structured)
    assert np.all(structured["A"] == laser.get("A"))
    assert np.all(structured["B"] == laser.get("B"))
