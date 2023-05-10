import numpy as np
import pytest
import xarray as xr

from seaduck.topology import topology

Datadir = "tests/Data/"
ecco = xr.open_zarr(Datadir + "small_ecco")
tp = topology(ecco)

mundane = np.array([[[1.0, 1.0]], [[0.0, 0.0]], [[0.0, -0.0]], [[1.0, 1.0]]])


@pytest.mark.parametrize(
    "fface,cis",
    [
        (np.array([[1, 1]]), True),
        (np.array([[1, 2]]), True),
        (np.array([[10, 2]]), False),
        (np.array([[6, 10]]), False),
    ],
)
def test_4_matrix(fface, cis):
    ans = np.array(tp.four_matrix_for_uv(fface))
    if cis:
        assert np.allclose(ans, mundane)
    else:
        assert not np.allclose(ans, mundane)
