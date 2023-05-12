import pytest
import xarray as xr

import seaduck.OceData as ocedata
import seaduck.utils as _u

Datadir = "tests/Data/"
ecco = xr.open_zarr(Datadir + "small_ecco")

curv = xr.open_dataset("{}MITgcm_curv_nc.nc" "".format(Datadir))

rect = xr.open_dataset("{}MITgcm_rect_nc.nc" "".format(Datadir))


def test_create_tree_cartesian():
    _u.create_tree(curv["XC"], curv["YC"], R=None)


@pytest.mark.parametrize("od", [ecco, curv, rect])
def test_convert(od):
    ocedata(od)
