import numpy as np
from pathlib import Path

from pewlib import io


def test_io_csv_generic():
    path = Path(__file__).parent.joinpath("data", "csv", "generic")

    assert io.csv.is_valid_directory(path)
    assert isinstance(io.csv.directory_type(path), io.csv.CsvTypeHint)

    data, params = io.csv.load(path, full=True)

    assert data.dtype.names == ()
    assert np.isclose(np.sum(data["_"]), 0.0)


def test_io_csv_nu_instruments():
    path = Path(__file__).parent.joinpath("data", "csv", "nu")

    assert io.csv.is_valid_directory(path)
    assert isinstance(io.csv.directory_type(path), io.csv.NuHint)

    data, params = io.csv.load(path, full=True)

    assert data.dtype.names == ()
    assert np.isclose(np.sum(data["_"]), 0.0)

    assert params["spotsize"] == 0.0


def test_io_csv_tofwerk():
    path = Path(__file__).parent.joinpath("data", "csv", "tofwerk")

    assert io.csv.is_valid_directory(path)
    assert isinstance(io.csv.directory_type(path), io.csv.TofwerkHint)

    data, params = io.csv.load(path, full=True)

    assert data.dtype.names == ()
    assert np.isclose(np.sum(data["_"]), 0.0)

    assert params["scantime"] == 0.0
