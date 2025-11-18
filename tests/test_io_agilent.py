import numpy as np
import pytest
from pathlib import Path

from pewlib.io import agilent


sums_7700 = {
    "P31": 7.4188098658e3,
    "Eu153": 2.5000009499e1,
    "W182": 0.0000000000e0,
}

sums_8900 = {
    "P31": 9.5571921387e5,
    "Eu153": 1.6875011255e2,
    "W182": 7.9762025047e2,
    "P31->47": 1.8750004889e1,
    "Eu153->153": 9.3750033388e1,
    "W182->182": 4.4047663919e2,
}


def test_io_agilent_collection():
    path = Path(__file__).parent.joinpath("data", "agilent", "8900", "test_ms.b")

    truefiles = ["001.d", "002.d", "003.d", "004.d", "005.d"]

    for method in ["batch_xml", "batch_csv", "acq_method_xml", "alphabetical"]:
        datafiles = agilent.collect_datafiles(path, [method])
        assert [df.name for df in datafiles] == truefiles


def test_io_agilent_mass_info():
    path = Path(__file__).parent.joinpath("data", "agilent", "8900")

    # Single quad data
    masses = agilent.mass_info_datafile(path.joinpath("test_ms.b", "001.d"))
    assert np.all([m.id for m in masses] == [1, 2, 3])
    assert np.all(np.isclose([m.acctime for m in masses], [0.16, 0.16, 0.168]))
    assert np.all([str(m) for m in masses] == ["P31", "Eu153", "W182"])

    # Triple quad
    masses = agilent.mass_info_datafile(path.joinpath("test_ms_ms.b", "001.d"))
    assert np.all([m.id for m in masses] == [1, 2, 3])
    assert np.all(np.isclose([m.acctime for m in masses], [0.16, 0.16, 0.168]))
    assert np.all([str(m) for m in masses] == ["P31->47", "Eu153->153", "W182->182"])


def test_io_agilent_load_7700_binary():
    path = Path(__file__).parent.joinpath("data", "agilent", "7700")

    data, params = agilent.load_binary(path.joinpath("test.b"), counts_per_second=True)
    assert data.shape == (5, 5)
    assert data.dtype.names == ("P31", "Eu153", "W182")
    for name in data.dtype.names:
        assert np.isclose(data[name].sum(), sums_7700[name])

    assert np.isclose(params["scantime"], 0.5, rtol=1e-2)


def test_io_agilent_load_7700_csv():
    path = Path(__file__).parent.joinpath("data", "agilent", "7700")

    data, params = agilent.load_csv(
        path.joinpath("test.b"),
    )
    assert data.shape == (5, 5)
    assert data.dtype.names == ("P31", "Eu153", "W182")
    for name in data.dtype.names:
        assert np.isclose(data[name].sum(), sums_7700[name], rtol=1e-4)  # Lowered tol

    assert np.isclose(params["scantime"], 0.5, rtol=1e-2)


def test_io_agilent_load_8900_binary():
    path = Path(__file__).parent.joinpath("data", "agilent", "8900")

    data, params = agilent.load_binary(
        path.joinpath("test_ms.b"), counts_per_second=True
    )
    assert data.shape == (5, 5)
    assert data.dtype.names == ("P31", "Eu153", "W182")
    for name in data.dtype.names:
        assert np.isclose(data[name].sum(), sums_8900[name])

    assert np.isclose(params["scantime"], 0.5, rtol=1e-2)

    data, params = agilent.load_binary(
        path.joinpath("test_ms_ms.b"), counts_per_second=True
    )
    assert data.shape == (5, 5)
    assert data.dtype.names == ("P31->47", "Eu153->153", "W182->182")
    for name in data.dtype.names:
        assert np.isclose(data[name].sum(), sums_8900[name])

    assert np.isclose(params["scantime"], 0.5, rtol=1e-2)


def test_io_agilent_load_8900_csv():
    path = Path(__file__).parent.joinpath("data", "agilent", "8900")

    data, params = agilent.load_csv(
        path.joinpath("test_ms.b"),
    )
    assert data.shape == (5, 5)
    assert data.dtype.names == ("P31", "Eu153", "W182")
    for name in data.dtype.names:
        assert np.isclose(data[name].sum(), sums_8900[name], rtol=1e-4)  # Lowered tol

    assert np.isclose(params["scantime"], 0.5, rtol=1e-2)

    data, params = agilent.load_csv(
        path.joinpath("test_ms_ms.b"),
    )
    assert data.shape == (5, 5)
    assert data.dtype.names == ("P31->47", "Eu153->153", "W182->182")
    for name in data.dtype.names:
        assert np.isclose(data[name].sum(), sums_8900[name], rtol=1e-4)  # Lowered tol

    assert np.isclose(params["scantime"], 0.5, rtol=1e-2)


def test_io_agilent_load_flatten():
    path = Path(__file__).parent.joinpath("data", "agilent", "8900")

    data, params = agilent.load_csv(path.joinpath("test_ms.b"), flatten=True)
    assert data.shape == (25,)


def test_io_agilent_load_missing():
    path = Path(__file__).parent.joinpath("data", "agilent", "8900")

    with pytest.raises(ValueError):
        agilent.load(
            path.joinpath("test_ms_missing.b"),
            use_acq_for_names=False,
            collection_methods=["batch_csv"],  # Missing batch csv
        )
    data, _ = agilent.load(
        path.joinpath("test_ms_missing.b"),
        use_acq_for_names=False,
        collection_methods=["alphabetical"],
    )
    assert isinstance(data, np.ndarray)
    assert data.shape == (3, 5)


def test_io_agilent_load_info_7700():
    path = Path(__file__).parent.joinpath("data", "agilent", "7700")
    info = agilent.load_info(path.joinpath("test.b"))

    assert info["Acquisition Date"] == "2020-11-18T14:24:29+11:00"
    assert info["Acquisition Name"] == "test_7700.b"
    assert info["Acquisition Path"] == "D:\\DATA\\OM\\test_7700.b"
    assert info["Acquisition User"] == "Student"

    assert info["Instrument Type"] == "ICPMS"
    assert info["Instrument Model"] == "G3282A"
    assert info["Instrument Serial"] == "JP11161014"


def test_io_agilent_load_info_8900():
    path = Path(__file__).parent.joinpath("data", "agilent", "8900")
    info = agilent.load_info(path.joinpath("test_ms_ms.b"))

    assert info["Acquisition Date"] == "2020-11-16T13:05:17+11:00"
    assert info["Acquisition Name"] == "test_ms_ms.b"
    assert (
        info["Acquisition Path"]
        == "D:\\Agilent\\ICPMH\\1\\DATA\\Tom\\Tests\\test_ms_ms.b"
    )
    assert info["Acquisition User"] == "ICPMS8900"

    assert info["Instrument Type"] == "ICPQQQ"
    assert info["Instrument Model"] == "G3665A"
    assert info["Instrument Serial"] == "SG19441319"


def test_io_agilent_load_info_8900_missing():
    path = Path(__file__).parent.joinpath("data", "agilent", "8900")
    info = agilent.load_info(path.joinpath("test_ms_missing.b"))

    assert not any("Acquisition " + x in info for x in ["Date", "Name", "Path", "User"])
    assert all("Instrument " + x in info for x in ["Type", "Model", "Serial"])
