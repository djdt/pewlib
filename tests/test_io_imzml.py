from pathlib import Path

import numpy as np
import pytest

from pewlib.io import imzml


def test_is_imzml():
    path = Path(__file__).parent.joinpath("data", "imzml")
    assert imzml.is_imzml(path.joinpath("test.imzML"))
    assert not imzml.is_imzml(path.joinpath("test.ibd"))
    assert not imzml.is_imzml_binary_data(path.joinpath("test.imzML"))
    assert imzml.is_imzml_binary_data(path.joinpath("test.ibd"))


def test_io_imzml():
    path = Path(__file__).parent.joinpath("data", "imzml")

    # auto find .ibd
    imz = imzml.ImzML.from_file(path.joinpath("test.imzML"))

    assert imz.scan_settings.image_size == (2, 2)
    assert imz.scan_settings.pixel_size == (30.0, 30.0)

    assert imz.mz_params.id == "mzArray"
    assert imz.mz_params.dtype == np.float64

    assert imz.intensity_params.id == "intensities"
    assert imz.intensity_params.dtype == np.float32

    tic = imz.extract_tic()
    assert tic.shape == (2, 2)
    assert np.allclose(tic, [[80436.0, 72098.0], [52676.0, 75608.0]])

    targets = imz.extract_masses([300.0, 600.0, 900.0], mass_width_ppm=10.0)
    assert targets.shape == (2, 2, 3)

    targets = imz.extract_masses([300.0, 600.0, 900.0], mass_width_mz=1.0)
    assert targets.shape == (2, 2, 3)

    with pytest.raises(ValueError):
        imz.extract_masses([500.0], mass_width_mz=1.0, mass_width_ppm=1.0)

    mz00 = imz.spectra[(1, 1)].get_binary_data(
        imz.mz_params.id, imz.mz_params.dtype, imz.external_binary
    )
    assert mz00.shape == (852,)
    assert np.isclose(np.amin(mz00), 200.273186)
    assert np.isclose(np.amax(mz00), 992.453628)

    mz_range = imz.mass_range()
    assert np.allclose(mz_range, (200.081016, 992.925155))

    bins, data = imz.binned_masses(mass_width_mz=1.0)
    assert bins.size == int((mz_range[1] + 1.0) - mz_range[0]) + 1

    assert data.shape == (2, 2, 794)
    assert data[1][1][0] == 16.0

    mzs, signals = imz.untargeted_extraction(
        100, min_pixel_count=3, min_height_absolute=10.0
    )
    assert signals.shape[:2] == (2, 2)
    assert len(mzs) == signals.shape[-1]


def test_io_imzml_orbi():
    path = Path(__file__).parent.joinpath("data", "imzml")

    # auto find .ibd
    imz = imzml.ImzML.from_file(path.joinpath("test_orbi.imzml"))

    assert imz.mz_params.dtype == np.float32
    assert imz.intensity_params.dtype == np.float32

    assert imz.image_size == (1, 32)

    tic = imz.extract_tic()
    assert tic[30, 0] == 5.3958265e6  # pixel 1, 31
    assert tic.shape == (32, 1)

    data = imz.extract_masses([420.1603], mass_width_ppm=10.0)
    assert np.isclose(data[31, 0], 3.126e6, rtol=1e-4)

    # test when size is missing
    imz.scan_settings.image_size = None
    assert imz.image_size == (1, 32)


def test_io_imzml_fast_parse():
    path = Path(__file__).parent.joinpath("data", "imzml")

    # auto find .ibd
    imz = imzml.ImzML.from_file(path.joinpath("test_orbi.imzml"), use_fast_parse=True)

    assert imz.mz_params.dtype == np.float32
    assert imz.intensity_params.dtype == np.float32

    assert imz.image_size == (1, 32)

    tic = imz.extract_tic()
    assert tic[30, 0] == 5.3958265e6  # pixel 1, 31
    assert tic.shape == (32, 1)

    data = imz.extract_masses([420.1603], mass_width_ppm=10.0)
    assert np.isclose(data[31, 0], 3.126e6, rtol=1e-4)

    # test when size is missing
    imz.scan_settings.image_size = None
    assert imz.image_size == (1, 32)
