import numpy as np
from pathlib import Path

from pew.io import agilent


sums = {
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


def test_io_agilent_load_binary():
    path = Path(__file__).parent.joinpath("data", "agilent", "8900")

    data, params = agilent.load_binary(
        path.joinpath("test_ms.b"), counts_per_second=True, full=True
    )
    assert data.shape == (5, 5)
    assert data.dtype.names == ("P31", "Eu153", "W182")
    for name in data.dtype.names:
        assert np.isclose(data[name].sum(), sums[name])

    assert params["scantime"] == 0.5

    data, params = agilent.load_binary(
        path.joinpath("test_ms_ms.b"), counts_per_second=True, full=True
    )
    assert data.shape == (5, 5)
    assert data.dtype.names == ("P31->47", "Eu153->153", "W182->182")
    for name in data.dtype.names:
        assert np.isclose(data[name].sum(), sums[name])

    assert params["scantime"] == 0.5


def test_io_agilent_load_csv():
    path = Path(__file__).parent.joinpath("data", "agilent", "8900")

    data, params = agilent.load_csv(path.joinpath("test_ms.b"), full=True)
    assert data.shape == (5, 5)
    assert data.dtype.names == ("P31", "Eu153", "W182")
    for name in data.dtype.names:
        assert np.isclose(data[name].sum(), sums[name], rtol=1e-4)  # Lowered tol

    assert params["scantime"] == 0.50

    data, params = agilent.load_csv(path.joinpath("test_ms_ms.b"), full=True)
    assert data.shape == (5, 5)
    assert data.dtype.names == ("P31->47", "Eu153->153", "W182->182")
    for name in data.dtype.names:
        assert np.isclose(data[name].sum(), sums[name], rtol=1e-4)  # Lowered tol

    assert params["scantime"] == 0.5


def test_io_agilent_load_missing():
    path = Path(__file__).parent.joinpath("data", "agilent", "8900")

    data = agilent.load(  # Will fall back to csv
        path.joinpath("test_ms_missing.b"),
        use_acq_for_names=False,
        collection_methods=["batch_csv"],
    )
    assert isinstance(data, np.ndarray)
    assert data.shape == (4, 5)
    assert np.all(data["P31"][3] == 0.0)
