import numpy as np
from pathlib import Path

from pewlib import io


def test_io_csv_generic():
    path = Path(__file__).parent.joinpath("data", "csv", "generic")

    assert io.csv.is_valid_directory(path)
    assert not io.csv.is_valid_directory(path.joinpath("non-exist"))
    assert isinstance(io.csv.option_for_path(path), io.csv.GenericOption)
    assert isinstance(
        io.csv.option_for_path(path.joinpath("1,csv")), io.csv.GenericOption
    )

    data, params = io.csv.load(path, full=True)

    assert data.shape == (3, 5)
    assert data.dtype.names == ("A", "B")

    assert np.all(data["A"].ravel() == np.arange(1, 16))
    assert np.isclose(np.sum(data["A"]), 120.0)
    assert np.isclose(np.sum(data["B"]), 12.0)


def test_io_csv_nu_instruments():
    path = Path(__file__).parent.joinpath("data", "csv", "nu")

    assert io.csv.is_valid_directory(path)
    assert isinstance(io.csv.option_for_path(path), io.csv.NuOption)

    data, params = io.csv.load(path, full=True)

    assert data.shape == (3, 5)
    assert data.dtype.names == ("A", "B")

    assert np.all(data["A"].ravel() == np.arange(1, 16))
    assert np.isclose(np.sum(data["A"]), 120.0)
    assert np.isclose(np.sum(data["B"]), 12.0)

    assert params["spotsize"] == 10.0


def test_io_csv_tofwerk():
    path = Path(__file__).parent.joinpath("data", "csv", "tofwerk")

    assert io.csv.is_valid_directory(path)
    assert isinstance(io.csv.option_for_path(path), io.csv.TofwerkOption)

    data, params = io.csv.load(path, full=True)

    assert data.shape == (3, 5)
    assert data.dtype.names == ("A", "B")

    assert np.all(data["A"].ravel() == np.arange(1, 16))
    assert np.isclose(np.sum(data["A"]), 120.0)
    assert np.isclose(np.sum(data["B"]), 12.0)

    assert params["scantime"] == 0.1
