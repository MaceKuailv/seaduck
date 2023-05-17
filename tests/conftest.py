import os

import pooch
from pooch import Untar
import pytest
import xarray as xr

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
path = "tests/Data/"


@pytest.fixture(scope="module")
def xr_ecco():
    return xr.open_zarr(path + "small_ecco")


@pytest.fixture(scope="module")
def xr_aviso():
    return xr.open_dataset(path + "aviso_example.nc")


@pytest.fixture(scope="module")
def xr_curv():
    return xr.open_dataset(path + "MITgcm_curv_nc.nc")


@pytest.fixture(scope="module")
def xr_rect():
    return xr.open_dataset(path + "MITgcm_rect_nc.nc")


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
