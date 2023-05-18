import os

import pooch
from pooch import Untar
import pytest
import xarray as xr
import numpy as np

from seaduck import OceData

path = "tests/testData/"


@pytest.fixture(scope="module")
def xr_ecco():
    fnames = pooch.retrieve(
        url="https://zenodo.org/record/7949168/files/ecco.tar.gz?download=1",
        processor=Untar(),
        known_hash="ab5ef7bbd3c0ba05c858e1366565086d9873bc8271d37f7912143512b892d2d9",
    )
    ds = xr.open_zarr(os.path.commonpath(fnames))
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
    fnames = pooch.retrieve(
        url="https://zenodo.org/record/7949168/files/aviso.tar.gz?download=1",
        processor=Untar(),
        known_hash="e2377044ee4ea27aee582ede5b753fa7371d7557c11b2fa3bf90bd1dad24c287",
    )
    ds = xr.open_zarr(os.path.commonpath(fnames))
    return ds


@pytest.fixture(scope="module")
def xr_curv():
    fnames = pooch.retrieve(
        url="https://zenodo.org/record/7949168/files/curv.tar.gz?download=1",
        processor=Untar(),
        known_hash="8773c8640be1f2770782a1780afbf6564f34670e6c0052274e3cc61be2e1a055",
    )
    ds = xr.open_zarr(os.path.commonpath(fnames))
    return ds


@pytest.fixture(scope="module")
def xr_rect():
    fnames = pooch.retrieve(
        url="https://zenodo.org/record/7949168/files/rect.tar.gz?download=1",
        processor=Untar(),
        known_hash="34d0a7c22b79f0bf2926f90e99a9da7bb13a148027155cec116d4b544ace4f3b",
    )
    ds = xr.open_zarr(os.path.commonpath(fnames))
    return ds


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
