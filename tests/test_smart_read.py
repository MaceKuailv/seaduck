import numpy as np
import pytest

from seaduck.smart_read import smart_read


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
@pytest.mark.parametrize("ds", ["ecco"], indirect=True)
def test_just_read(ind, ds, chunk):
    ds["SALT"] = ds["SALT"].chunk(chunk)
    smart_read(ds["SALT"], ind)


@pytest.mark.parametrize("ds", ["ecco"], indirect=True)
def test_read_xarray(ind, ds):
    ds["SALT"] = ds["SALT"].chunk({"time": 1})
    smart_read(ds["SALT"], ind, dask_more_efficient=1)


@pytest.mark.parametrize("ds", ["ecco"], indirect=True)
def test_mismatch_read(ind, ds):
    with pytest.raises(ValueError):
        smart_read(ds["XC"], ind)
