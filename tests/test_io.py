import filecmp
import numpy as np
import os.path
import pytest
import tempfile

from pew import io
from pew.srr import SRRLaser, SRRConfig


def test_io_agilent_load():
    data_path = os.path.join(os.path.dirname(__file__), "data", "agilent")
    # Test loading 7700
    data = io.agilent.load(os.path.join(data_path, "7700.b"))
    assert data.shape == (3, 3)
    assert data.dtype.names == ("A1", "B2")
    assert np.sum(data["A1"]) == pytest.approx(9.0)
    assert np.sum(data["B2"]) == pytest.approx(0.9)
    # Test loading from 7500
    data = io.agilent.load(os.path.join(data_path, "7500.b"))
    assert data.shape == (3, 3)
    assert data.dtype.names == ("A1", "B2")
    assert np.sum(data["A1"]) == pytest.approx(9.0)
    assert np.sum(data["B2"]) == pytest.approx(0.9)
    # Make sure no error raised on missing data
    data = io.agilent.load(os.path.join(data_path, "missing_line.b"))
    assert np.all(data["A1"][1, :] == [0.0, 0.0, 0.0, 0.0, 0.0])


def test_io_agilent_params():
    data_path = os.path.join(os.path.dirname(__file__), "data", "agilent")
    _, params = io.agilent.load(os.path.join(data_path, "7700.b"), full=True)
    assert params["scantime"] == 0.1


def test_io_agilent_data_files_collection():
    acq_method_dfiles = ["c.d", "b.d", "a.d"]
    batch_csv_dfiles = ["c.d", "a.d", "b.d"]
    batch_xml_dfiles = ["a.d", "c.d", "b.d"]

    data_path = os.path.join(
        os.path.dirname(__file__), "data", "agilent", "acq_batch_files.b"
    )
    dfiles = io.agilent.collect_datafiles(data_path, ["acq_method_xml"])
    assert dfiles == [os.path.join(data_path, df) for df in acq_method_dfiles]
    dfiles = io.agilent.batch_csv_read_datafiles(
        data_path, os.path.join(data_path, io.agilent.batch_csv_path)
    )
    assert dfiles == [os.path.join(data_path, df) for df in batch_csv_dfiles]
    dfiles = io.agilent.batch_xml_read_datafiles(
        data_path, os.path.join(data_path, io.agilent.batch_xml_path)
    )
    assert dfiles == [os.path.join(data_path, df) for df in batch_xml_dfiles]


def test_io_agilent_acq_method_elements():
    data_path = os.path.join(
        os.path.dirname(__file__), "data", "agilent", "acq_batch_files.b"
    )
    elements = io.agilent.acq_method_xml_read_elements(
        os.path.join(data_path, io.agilent.acq_method_xml_path)
    )
    assert elements == ["A1", "B2"]


def test_io_agilent_acq_method_elements_msms():
    data_path = os.path.join(
        os.path.dirname(__file__), "data", "agilent", "acq_method_msms.xml"
    )
    elements = io.agilent.acq_method_xml_read_elements(data_path)
    assert elements == ["A1->2", "B3->4"]


def test_io_csv():
    data_path = os.path.join(os.path.dirname(__file__), "data", "csv")
    data_path_thermo = os.path.join(os.path.dirname(__file__), "data", "thermo")
    # Make sure error raised on thermo data
    with pytest.raises(io.error.PewException):
        io.csv.load(os.path.join(data_path_thermo, "icap_columns.csv"))
    # Test loading
    data = io.csv.load(os.path.join(data_path, "csv.csv"), "CSV")
    assert data.dtype.names == ("CSV",)
    assert data.shape == (20, 5)
    assert np.sum(data["CSV"]) == pytest.approx(100.0)
    # Test saving
    temp = tempfile.NamedTemporaryFile()
    data = np.random.random([10, 10])
    io.csv.save(temp.name, data)
    assert np.all(data == io.csv.load(temp.name))
    temp.close()
    # Delimiter loading
    data = io.csv.load(os.path.join(data_path, "delimiters.csv"))
    assert data.shape == (10, 5)
    assert np.sum(data) == pytest.approx(100.0)


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
    laser = io.npz.load(os.path.join(data_path, "test.npz"))[0]
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
    assert calibration.gradient == 1.0
    assert calibration.intercept == 2.0
    assert calibration.rsq is None
    assert calibration.weighting == "x"
    assert np.all(calibration.points == np.array([[1, 1], [2, 2], [3, 3]]))
    assert calibration.unit == "test"
    # Saving
    temp = tempfile.NamedTemporaryFile(suffix=".npz")
    io.npz.save(temp.name, [laser])  # type: ignore
    loaded = io.npz.load(temp.name)[0]
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
    laser = io.npz.load(os.path.join(data_path, "test_srr.npz"))[0]
    assert isinstance(laser, SRRLaser)
    assert isinstance(laser.config, SRRConfig)
