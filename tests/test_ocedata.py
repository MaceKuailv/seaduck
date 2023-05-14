import pytest
import xarray as xr

import seaduck.OceData as ocedata
import seaduck.utils as _u

Datadir = "tests/Data/"
ecco = xr.open_zarr(Datadir + "small_ecco")
curv = xr.open_dataset("{}MITgcm_curv_nc.nc" "".format(Datadir))
rect = xr.open_dataset("{}MITgcm_rect_nc.nc" "".format(Datadir))

curv_prime = curv.drop_vars(["YG"])
curv_prprm = curv.drop_vars(["dyG"])
curv_prprp = curv.drop_vars(
    ["time_midp"] + [i for i in curv.data_vars if "time_midp" in curv[i].dims]
)

oce = ocedata(curv)

oce._add_missing_cs_sn()
curv_prime["CS"] = xr.DataArray(oce["CS"], dims=curv_prime["XC"].dims)
curv_prime["SN"] = xr.DataArray(oce["SN"], dims=curv_prime["XC"].dims)


def test_create_tree_cartesian():
    _u.create_tree(curv["XC"], curv["YC"], R=None)


@pytest.mark.parametrize("od", [ecco, rect])
def test_convert(od):
    oo = ocedata(od)


@pytest.mark.parametrize("data", [curv_prime, curv_prprm, curv_prprp])
def test_incomplete_data(data):
    oo = ocedata(data)
    oo.find_rel_h(np.array([-14]), np.array([70.5]))
