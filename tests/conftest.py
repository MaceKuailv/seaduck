import os

import pooch
from pooch import Untar
import pytest
import xarray as xr
import numpy as np

from seaduck import OceData


# # Download data if necessary
# def pytest_configure():
#     fnames = pooch.retrieve(
#         url="https://zenodo.org/record/7916559/files/Data.tar.gz?download=1",
#         processor=Untar(),
#         known_hash="c27a566ed280f3a579dd2bf6846"
#         "2a427d7d0d1286cdd8db2a4a035495f40f7e4",
#     )
#     symlink_args = dict(
#         src=f"{os.path.commonpath(fnames)}",
#         dst="./tests/Data",
#         target_is_directory=True,
#     )
#     try:
#         print(f"Linking {symlink_args['src']!r} to {symlink_args['dst']!r}")
#         os.symlink(**symlink_args)
#     except FileExistsError:
#         os.unlink("./tests/Data")
#         os.symlink(**symlink_args)
path = "tests/testData/"


@pytest.fixture(scope="module")
def xr_ecco():
    fnames = pooch.retrieve(
        url="https://zenodo.org/record/7916559/files/Data.tar.gz?download=1",
        processor=Untar(),
        known_hash="ab5ef7bbd3c0ba05c858e136656508"
        "6d9873bc8271d37f7912143512b892d2d9",
        )
    ds = xr.open_zarr(fnames)
    rand1 = np.random.random((50, 13, 90, 90))
    rand2 = np.random.random((50, 13, 90, 90))
    rand3 = np.random.random((50, 13, 90, 90))
    ds["UVELMASS"] = xr.DataArray(
        np.stack([rand1, rand2, rand3], axis=0), dims=("time", "Z", "face", "Y", "Xp1")
    )
    ds["UVELMASS"][0] = ds.UVELMASS1
    ds["UVELMASS"][1] = ds.UVELMASS1 * rand1
    ds["UVELMASS"][2] = ds.UVELMASS1 * rand2

    ds["WVELMASS"] = xr.DataArray(
        np.stack([rand1, rand2, rand3], axis=0), dims=("time", "Zl", "face", "Y", "X")
    )
    ds["WVELMASS"][0] = ds.WVELMASS1
    ds["WVELMASS"][1] = ds.WVELMASS1 * rand1
    ds["WVELMASS"][2] = ds.WVELMASS1 * rand2

    ds["VVELMASS"] = xr.DataArray(
        np.stack([rand1, rand2, rand3], axis=0), dims=("time", "Z", "face", "Yp1", "X")
    )
    ds["VVELMASS"][0] = ds.VVELMASS1
    ds["VVELMASS"][1] = ds.VVELMASS1 * rand1
    ds["VVELMASS"][2] = ds.VVELMASS1 * rand2

    ds["SALT"] = xr.DataArray(
        np.stack([rand1, rand2, rand3], axis=0), dims=("time", "Z", "face", "Y", "X")
    )
    ds["SALT_snap"] = xr.DataArray(
        np.stack([rand3, rand1], axis=0), dims=("time_midp", "Z", "face", "Y", "X")
    )
    ds["ETAN"] = xr.DataArray(rand1[:3], dims=("time", "face", "Y", "X"))
    ds["ETAN_snap"] = xr.DataArray(rand3[:2], dims=("time_midp", "face", "Y", "X"))
    return ds


@pytest.fixture(scope="module")
def xr_aviso():
    return xr.open_zarr(path + "aviso.zarr")


@pytest.fixture(scope="module")
def xr_curv():
    return xr.open_zarr(path + "curv.zarr")


@pytest.fixture(scope="module")
def xr_rect():
    return xr.open_zarr(path + "rect.zarr")


@pytest.fixture(scope="module")
def ecco(xr_ecco):
    return OceData(xr_ecco)


@pytest.fixture(scope="module")
def avis(xr_aviso):
    return OceData(xr_aviso)


@pytest.fixture(scope="module")
def curv(xr_curv):
    return OceData(xr_curv)


@pytest.fixture(scope="module")
def rect(xr_rect):
    return OceData(xr_rect)
