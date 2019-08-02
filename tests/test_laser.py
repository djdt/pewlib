import numpy as np

from laserlib.laser import Laser
from laserlib.data import LaserData


def test_default_laser():
    laser = Laser()
    assert len(laser.isotopes) == 0
    assert laser.layers == 1
    assert laser.get("NONE") == np.zeros((1, 1))


def test_laser():
    laser = Laser(
        data={
            "A": LaserData(np.random.random([10, 10])),
            "B": LaserData(np.random.random([10, 10])),
        }
    )
    assert laser.isotopes == ["A", "B"]
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
