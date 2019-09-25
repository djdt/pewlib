import numpy as np
import pytest

from pew.laser import Laser, Calibration, Config

from typing import List


def rand_data(names: List[str]) -> np.ndarray:
    dtype = [(name, float) for name in names]
    data = np.empty((10, 10), dtype=dtype)
    for name in names:
        data[name] = np.random.random((10, 10))
    return data


def test_laser():
    laser = Laser(rand_data(["A", "B"]))
    assert laser.extent == (0, 350, 0, 350)
    assert laser.isotopes == ("A", "B")
    assert laser.shape == (10, 10)
    assert laser.layers == 1

    with pytest.raises(AssertionError):
        laser.add("C", np.random.random((1, 1)))

    laser.add("C", np.random.random((10, 10)))
    assert laser.isotopes == ("A", "B", "C")

    laser.remove("A")
    assert laser.isotopes == ("B", "C")


def test_laser_get():
    data = rand_data(["A", "B"])
    laser = Laser(
        data, calibration=dict(A=Calibration(1.0, 2.0)), config=Config(10, 10, 0.5)
    )

    assert np.all(laser.get("A") == data["A"])
    assert np.all(laser.get("A", calibrate=True) == (data["A"] - 1.0) / 2.0)
    assert np.all(laser.get("A", extent=(0.0, 20.0, 10.0, 30)) == data["A"][7:9, 0:4])


def test_laser_from_list():
    names = ["A", "B", "C"]
    datas = [np.random.random((10, 10)) for i in range(3)]
    laser = Laser.from_list(names, datas)
    assert np.all(datas[0] == laser.get("A"))
    assert np.all(datas[1] == laser.get("B"))
    assert np.all(datas[2] == laser.get("C"))
