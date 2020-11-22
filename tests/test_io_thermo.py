import numpy as np
from pathlib import Path

from pewlib.io import thermo


sums = {"31P": 4.8708803607e4, "153Eu": 5.5000012000e1, "182W": 2.5000003000e1}
asums = {"31P": -2.7566666667e-1, "153Eu": -1.1089166667e0, "182W": -1.0259166667e0}


def test_io_thermo_format():
    path = Path(__file__).parent.joinpath("data", "thermo")
    path_csv = Path(__file__).parent.joinpath("data", "textimage")

    assert thermo.icap_csv_sample_format(path.joinpath("icap_columns.csv")) == "columns"
    assert thermo.icap_csv_sample_format(path.joinpath("icap_rows.csv")) == "rows"
    assert thermo.icap_csv_sample_format(path_csv.joinpath("csv.csv")) == "unknown"


def test_io_thermo_columns():
    path = Path(__file__).parent.joinpath("data", "thermo")

    data = thermo.icap_csv_columns_read_data(path.joinpath("icap_columns.csv"))
    assert data.shape == (5, 5)
    assert data.dtype.names == ("31P", "153Eu", "182W")

    for name in data.dtype.names:
        assert np.isclose(np.sum(data[name]), sums[name])

    # Test analog
    data = thermo.icap_csv_columns_read_data(
        path.joinpath("icap_columns.csv"), use_analog=True
    )

    for name in data.dtype.names:
        assert np.isclose(np.sum(data[name]), asums[name])

    # Params
    params = thermo.icap_csv_columns_read_params(path.joinpath("icap_columns.csv"))
    assert params["scantime"] == 1.0049


def test_io_thermo_rows():
    path = Path(__file__).parent.joinpath("data", "thermo")

    data = thermo.icap_csv_rows_read_data(path.joinpath("icap_rows.csv"))
    assert data.shape == (5, 5)
    assert data.dtype.names == ("31P", "153Eu", "182W")

    for name in data.dtype.names:
        assert np.isclose(np.sum(data[name]), sums[name])

    # Test analog
    data = thermo.icap_csv_rows_read_data(
        path.joinpath("icap_rows.csv"), use_analog=True
    )

    for name in data.dtype.names:
        assert np.isclose(np.sum(data[name]), asums[name])

    # Params
    params = thermo.icap_csv_rows_read_params(path.joinpath("icap_rows.csv"))
    assert params["scantime"] == 1.0049
