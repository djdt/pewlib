import filecmp
import numpy as np
from pathlib import Path
import pytest
import tempfile

from pewlib import io
from pewlib.config import SpotConfig
from pewlib.srr import SRRLaser, SRRConfig


def test_io_perkinelmer():
    path = Path(__file__).parent.joinpath("data", "perkinelmer")

    assert io.perkinelmer.is_valid_directory(path.joinpath("perkinelmer"))
    assert not io.perkinelmer.is_valid_directory(path.joinpath("fake"))
    assert not io.perkinelmer.is_valid_directory(path)

    data, params = io.perkinelmer.load(path.joinpath("perkinelmer"), full=True)
    assert np.isclose(np.sum(data["A1"]), 12.0)
    assert np.isclose(np.sum(data["B2"]), 15.0)

    assert params["speed"] == 100.0
    assert params["scantime"] == 0.200
    assert params["spotsize"] == 300.0


def test_io_textimage():
    path = Path(__file__).parent.joinpath("data", "textimage")
    # Test loading
    data = io.textimage.load(path.joinpath("csv.csv"), delimiter=",", name="CSV")
    assert data.dtype.names == ("CSV",)
    assert data.shape == (5, 5)
    assert np.sum(data["CSV"]) == 85.0

    # Delimiter loading
    data = io.textimage.load(path.joinpath("delimiter.csv"))
    assert data.shape == (5, 5)
    assert np.sum(data) == 85.0

    # Test saving
    temp = tempfile.NamedTemporaryFile()
    data = np.random.random([10, 10])
    io.textimage.save(temp.name, data)
    assert np.all(data == io.textimage.load(temp.name))
    temp.close()


def test_io_vtk():
    path = Path(__file__).parent.joinpath("data", "vtk")

    np.random.seed(12718736)
    data = np.random.random((10, 10, 3))
    data.dtype = [("A1", float)]
    temp = tempfile.NamedTemporaryFile(suffix=".vti")
    io.vtk.save(temp.name, data, (1, 1, 1))
    assert filecmp.cmp(temp.name, path.joinpath("test.vti"))
    temp.close()


def test_io_npz():
    path = Path(__file__).parent.joinpath("data", "npz")

    laser = io.npz.load(path.joinpath("test.npz"))
    # Info
    assert laser.info["Name"] == "Test"
    assert Path(laser.info["File Path"]).samefile(path.joinpath("test.npz"))
    assert laser.info["1"] == "1"
    assert laser.info["tab"] == "tab "  # Replace value tab with space
    assert laser.info["tab name"] == "tabname"  # Replace key tab with space
    # Config
    assert laser.config.spotsize == 1
    assert laser.config.speed == 2
    assert laser.config.scantime == 3
    # Data
    assert laser.elements == ("A1", "B2")
    assert laser.get("A1").shape == (10, 10)
    assert laser.get("A1").sum() == pytest.approx(100)
    # Calibration
    calibration = laser.calibration["A1"]
    assert calibration.gradient == pytest.approx(1.0)
    assert calibration.intercept == 2.0
    assert calibration.rsq is None
    assert calibration.weighting == "x"
    assert np.all(calibration.points == np.array([[1, 1], [2, 2], [3, 3]]))
    assert calibration.unit == "test"
    # Saving
    temp = tempfile.NamedTemporaryFile(suffix=".npz")
    io.npz.save(temp.name, laser)
    loaded = io.npz.load(temp.name)
    temp.close()

    assert np.all(loaded.data == laser.data)
    assert loaded.config.spotsize == laser.config.spotsize
    assert loaded.config.speed == laser.config.speed
    assert loaded.config.scantime == laser.config.scantime

    assert loaded.calibration["A1"].gradient == laser.calibration["A1"].gradient
    assert loaded.calibration["A1"].intercept == laser.calibration["A1"].intercept
    assert loaded.calibration["A1"].rsq == laser.calibration["A1"].rsq
    assert loaded.calibration["A1"].weighting == laser.calibration["A1"].weighting
    assert loaded.calibration["A1"].unit == laser.calibration["A1"].unit
    assert np.all(loaded.calibration["A1"].points == laser.calibration["A1"].points)

    assert loaded.calibration["B2"].gradient == laser.calibration["B2"].gradient
    assert loaded.calibration["B2"].intercept == laser.calibration["B2"].intercept
    assert loaded.calibration["B2"].rsq == laser.calibration["B2"].rsq
    assert loaded.calibration["B2"].weighting == laser.calibration["B2"].weighting
    assert loaded.calibration["B2"].unit == laser.calibration["B2"].unit
    assert np.all(loaded.calibration["B2"].points == laser.calibration["B2"].points)


def test_io_npz_spot():
    path = Path(__file__).parent.joinpath("data", "npz")
    laser = io.npz.load(path.joinpath("spot.npz"))

    assert isinstance(laser.config, SpotConfig)
    assert laser.config.spotsize == 10.0
    assert laser.config.spotsize_y == 20.0


def test_io_npz_srr():
    path = Path(__file__).parent.joinpath("data", "npz")

    laser = io.npz.load(path.joinpath("test_srr.npz"))
    assert isinstance(laser, SRRLaser)
    assert isinstance(laser.config, SRRConfig)
