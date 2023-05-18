import xarray as xr
from seaduck.smart_read import smart_read as srd
import pytest
import numpy as np


@pytest.fixture
def ind():
    N = 3
    it = np.arange(N).astype(int)
    iz = np.arange(N).astype(int)
    fc = np.arange(N).astype(int)
    ix = np.arange(N).astype(int)
    iy = np.arange(N, 2 * N).astype(int)
    return (it, iz, fc, iy, ix)


@pytest.mark.parametrize("chunk", [{"time": 1}, {"time": 1, "Z": 1}, {}])
def test_just_read(ind, xr_ecco, chunk):
    xr_ecco["SALT"] = xr_ecco["SALT"].chunk(chunk)
    srd(xr_ecco["SALT"], ind)


def test_read_xarray(ind, xr_ecco):
    xr_ecco["SALT"] = xr_ecco["SALT"].chunk({"time": 1})
    srd(xr_ecco["SALT"], ind, xarray_more_efficient=1)


def test_mismatch_read(ind, xr_ecco):
    with pytest.raises(ValueError):
        srd(xr_ecco["XC"], ind)
