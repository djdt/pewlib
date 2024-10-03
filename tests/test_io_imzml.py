from pathlib import Path

import numpy as np

from pewlib.io import imzml


def test_is_imzml():
    path = Path(__file__).parent.joinpath("data", "imzml")
    assert imzml.is_imzml(path.joinpath("test.imzML"))
    assert not imzml.is_imzml(path.joinpath("test.ibd"))
    assert not imzml.is_imzml_binary_data(path.joinpath("test.imzML"))
    assert imzml.is_imzml_binary_data(path.joinpath("test.ibd"))


def test_io_imzml():
    path = Path(__file__).parent.joinpath("data", "imzml")

    imz = imzml.ImzML.from_file(path.joinpath("test.imzML"), path.joinpath("test.ibd"))

    assert imz.scan_settings.image_size == (2, 2)
    assert imz.scan_settings.pixel_size == (30.0, 30.0)

    assert imz.mz_params.id == "mzArray"
    assert imz.mz_params.dtype == np.float64

    assert imz.intensity_params.id == "intensities"
    assert imz.intensity_params.dtype == np.float32

    tic = imz.extract_tic()
    assert tic.shape == (2, 2)
    print(tic)
    assert np.allclose(tic, [[52676.0,  75608.0], [80436.0, 72098.0]])

    targets = imz.extract_masses([300.0, 600.0, 900.0])
    assert targets.shape == (2, 2, 3)

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