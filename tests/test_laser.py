import numpy as np
import pytest

from pewlib.laser import Laser
from pewlib import Calibration, Config


def rand_data(names: list[str]) -> np.ndarray:
    dtype = [(name, float) for name in names]
    data = np.empty((10, 10), dtype=dtype)
    for name in names:
        data[name] = np.random.random((10, 10))
    return data


# def test_laser_base():
#     laser = Laser()
#     assert laser.extent == (0.0, 0.0, 0.0, 0.0)
#     assert laser.elements == ()
#     assert laser.layers == 1
#     assert laser.shape == (0, 0)


def test_laser():
    laser = Laser(rand_data(["A", "B"]))
    assert laser.extent == (0, 350, 0, 350)
    assert laser.elements == ("A", "B")
    assert laser.shape == (10, 10)
    assert laser.layers == 1

    with pytest.raises(AssertionError):
        laser.add("C", np.random.random((1, 1)))

    laser.add("C", np.random.random((10, 10)))
    assert laser.elements == ("A", "B", "C")
    assert "C" in laser.calibration

    laser.remove("A")
    assert laser.elements == ("B", "C")
    assert "A" not in laser.calibration

    laser.rename({"B": "D"})
    assert laser.elements == ("D", "C")
    assert "D" in laser.calibration


def test_laser_get():
    data = rand_data(["A", "B"])
    laser = Laser(
        data, calibration=dict(A=Calibration(1.0, 2.0)), config=Config(10, 10, 0.5)
    )

    assert np.all(laser.get() == data)
    assert np.all(laser.get(calibrate=True)["A"] == (data["A"] - 1.0) / 2.0)

    assert np.all(laser.get("A") == data["A"])
    assert np.all(laser.get("A", calibrate=True) == (data["A"] - 1.0) / 2.0)
    assert np.all(laser.get("A", extent=(0.0, 20.0, 10.0, 30)) == data["A"][1:3, 0:4])


def test_laser_from_list():
    names = ["A", "B", "C"]
    datas = [np.random.random((10, 10)) for i in range(3)]
    laser = Laser.from_list(names, datas)
    assert np.all(datas[0] == laser.get("A"))
    assert np.all(datas[1] == laser.get("B"))
    assert np.all(datas[2] == laser.get("C"))
