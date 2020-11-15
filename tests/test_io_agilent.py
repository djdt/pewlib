import numpy as np
import os.path

from pew.io import agilent


def test_io_agilent_load():
    data_path = os.path.join(os.path.dirname(__file__), "data", "agilent")
    # Test loading 7700
    data = agilent.load(os.path.join(data_path, "7700.b"))
    assert data.shape == (3, 3)
    assert data.dtype.names == ("A1", "B2")
    assert np.isclose(np.sum(data["A1"]), 9.0)
    assert np.isclose(np.sum(data["B2"]), 0.9)
    # Test loading from 7500
    data = agilent.load(os.path.join(data_path, "7500.b"))
    assert data.shape == (3, 3)
    assert data.dtype.names == ("A1", "B2")
    assert np.isclose(np.sum(data["A1"]), 9.0)
    assert np.isclose(np.sum(data["B2"]), 0.9)
    # Make sure no error raised on missing data
    data = agilent.load(os.path.join(data_path, "missing_line.b"))
    assert np.all(data["A1"][1, :] == [0.0, 0.0, 0.0, 0.0, 0.0])


def test_io_agilent_params():
    data_path = os.path.join(os.path.dirname(__file__), "data", "agilent")
    _, params = agilent.load(os.path.join(data_path, "7700.b"), full=True)
    assert params["scantime"] == 0.1


def test_io_agilent_data_files_collection():
    acq_method_dfiles = ["c.d", "b.d", "a.d"]
    batch_csv_dfiles = ["c.d", "a.d", "b.d"]
    batch_xml_dfiles = ["a.d", "c.d", "b.d"]

    data_path = os.path.join(
        os.path.dirname(__file__), "data", "agilent", "acq_batch_files.b"
    )
    dfiles = agilent.collect_datafiles(data_path, ["acq_method_xml"])
    assert dfiles == [os.path.join(data_path, df) for df in acq_method_dfiles]
    dfiles = agilent.batch_csv_read_datafiles(
        data_path, os.path.join(data_path, agilent.batch_csv_path)
    )
    assert dfiles == [os.path.join(data_path, df) for df in batch_csv_dfiles]
    dfiles = agilent.batch_xml_read_datafiles(
        data_path, os.path.join(data_path, agilent.batch_xml_path)
    )
    assert dfiles == [os.path.join(data_path, df) for df in batch_xml_dfiles]


def test_io_agilent_acq_method_elements():
    data_path = os.path.join(
        os.path.dirname(__file__), "data", "agilent", "acq_batch_files.b"
    )
    elements = agilent.acq_method_xml_read_elements(
        os.path.join(data_path, agilent.acq_method_xml_path)
    )
    assert elements == ["A1", "B2"]


def test_io_agilent_acq_method_elements_msms():
    data_path = os.path.join(
        os.path.dirname(__file__), "data", "agilent", "acq_method_msms.xml"
    )
    elements = agilent.acq_method_xml_read_elements(data_path)
    assert elements == ["A1->2", "B3->4"]


