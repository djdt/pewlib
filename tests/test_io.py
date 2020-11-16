import filecmp
import numpy as np
import os.path
import pytest
import tempfile

from pew import io
from pew.srr import SRRLaser, SRRConfig


def test_io_perkinelmer():
    data_path = os.path.join(os.path.dirname(__file__), "data", "perkinelmer")
    data, params = io.perkinelmer.load(
        os.path.join(data_path, "perkinelmer"), full=True
    )
    assert np.isclose(np.sum(data["A1"]), 12.0)
    assert np.isclose(np.sum(data["B2"]), 15.0)

    assert params["speed"] == 0.100
    assert params["scantime"] == 0.200
    assert params["spotsize"] == 0.300


def test_io_textimage():
    data_path = os.path.join(os.path.dirname(__file__), "data", "textimage")
    # Test loading
    data = io.textimage.load(
        os.path.join(data_path, "csv.csv"), delimiter=",", name="CSV"
    )
    assert data.dtype.names == ("CSV",)
    assert data.shape == (5, 5)
    assert np.sum(data["CSV"]) == 85.0

    # Delimiter loading
    data = io.textimage.load(os.path.join(data_path, "delimiter.csv"))
    assert data.shape == (5, 5)
    assert np.sum(data) == 85.0

    # Test saving
    temp = tempfile.NamedTemporaryFile()
    data = np.random.random([10, 10])
    io.textimage.save(temp.name, data)
    assert np.all(data == io.textimage.load(temp.name))
    temp.close()

def test_io_thermo_load():
    data_path = os.path.join(os.path.dirname(__file__), "data", "thermo")
    data, params = io.thermo.load(
        os.path.join(data_path, "icap_columns.csv"), full=True
    )
    assert data.shape == (3, 3)
    assert data.dtype.names == ("1A", "2B")
    assert np.sum(data["1A"]) == pytest.approx(45.0)
    assert np.sum(data["2B"]) == pytest.approx(450.0)

    assert params["scantime"] == 0.1


def test_io_thermo_format():
    data_path = os.path.join(os.path.dirname(__file__), "data", "thermo")
    data_path_csv = os.path.join(os.path.dirname(__file__), "data", "csv")
    # Make sure csv data raises an error
    assert (
        io.thermo.icap_csv_sample_format(os.path.join(data_path, "icap_columns.csv"))
        == "columns"
    )
    assert (
        io.thermo.icap_csv_sample_format(os.path.join(data_path, "icap_rows.csv"))
        == "rows"
    )
    assert (
        io.thermo.icap_csv_sample_format(os.path.join(data_path_csv, "csv.csv"))
        == "unknown"
    )


def test_io_thermo_columns():
    data_path = os.path.join(os.path.dirname(__file__), "data", "thermo")
    data = io.thermo.icap_csv_columns_read_data(
        os.path.join(data_path, "icap_columns.csv")
    )
    assert data.shape == (3, 3)
    assert data.dtype.names == ("1A", "2B")
    assert np.sum(data["1A"]) == pytest.approx(45.0)
    assert np.sum(data["2B"]) == pytest.approx(450.0)

    # Test analog
    data = io.thermo.icap_csv_columns_read_data(
        os.path.join(data_path, "icap_columns.csv"), use_analog=True
    )
    assert data.shape == (3, 3)
    assert data.dtype.names == ("1A", "2B")
    assert np.sum(data["1A"]) == pytest.approx(4.5)
    assert np.sum(data["2B"]) == pytest.approx(0.0)

    # Params
    params = io.thermo.icap_csv_columns_read_params(
        os.path.join(data_path, "icap_columns.csv")
    )
    assert params["scantime"] == 0.1


def test_io_thermo_rows():
    data_path = os.path.join(os.path.dirname(__file__), "data", "thermo")
    data = io.thermo.icap_csv_rows_read_data(os.path.join(data_path, "icap_rows.csv"))
    assert data.shape == (3, 3)
    assert data.dtype.names == ("1A", "2B")
    assert np.sum(data["1A"]) == pytest.approx(45.0)
    assert np.sum(data["2B"]) == pytest.approx(450.0)

    # Test analog
    data = io.thermo.icap_csv_rows_read_data(
        os.path.join(data_path, "icap_rows.csv"), use_analog=True
    )
    assert data.shape == (3, 3)
    assert data.dtype.names == ("1A", "2B")
    assert np.sum(data["1A"]) == pytest.approx(4.5)
    assert np.sum(data["2B"]) == pytest.approx(0.0)

    # Params
    params = io.thermo.icap_csv_rows_read_params(
        os.path.join(data_path, "icap_rows.csv")
    )
    assert params["scantime"] == 0.1


def test_io_vtk():
    data_path = os.path.join(os.path.dirname(__file__), "data", "vtk")

    np.random.seed(12718736)
    data = np.random.random((10, 10, 3))
    data.dtype = [("A1", float)]
    temp = tempfile.NamedTemporaryFile(suffix=".vti")
    io.vtk.save(temp.name, data, (1, 1, 1))
    assert filecmp.cmp(temp.name, os.path.join(data_path, "test.vti"))
    temp.close()


def test_io_npz():
    data_path = os.path.join(os.path.dirname(__file__), "data", "npz")
    laser = io.npz.load(os.path.join(data_path, "test.npz"))
    assert laser.name == "Test"
    assert laser.path == os.path.join(data_path, "test.npz")
    # Config
    assert laser.config.spotsize == 1
    assert laser.config.speed == 2
    assert laser.config.scantime == 3
    # Data
    assert laser.isotopes == ("A1", "B2")
    assert laser.get("A1").shape == (10, 10)
    pytest.approx(laser.get("A1").sum(), 100)
    # Calibration
    calibration = laser.calibration["A1"]
    pytest.approx(calibration.gradient, 1.0)
    assert calibration.intercept == 2.0
    assert calibration.rsq is None
    assert calibration.weighting == "x"
    assert np.all(calibration.points == np.array([[1, 1], [2, 2], [3, 3]]))
    assert calibration.unit == "test"
    # Saving
    temp = tempfile.NamedTemporaryFile(suffix=".npz")
    io.npz.save(temp.name, laser)  # type: ignore
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


def test_io_npz_srr():
    data_path = os.path.join(os.path.dirname(__file__), "data", "npz")
    laser = io.npz.load(os.path.join(data_path, "test_srr.npz"))
    assert isinstance(laser, SRRLaser)
    assert isinstance(laser.config, SRRConfig)
