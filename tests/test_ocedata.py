import pytest
import numpy as np
import xarray as xr

import seaduck.OceData as ocedata
import seaduck.utils as _u


# Datadir = "tests/Data/"
# ecco = xr.open_zarr(Datadir + "small_ecco")
# curv = xr.open_dataset("{}MITgcm_curv_nc.nc" "".format(Datadir))
# rect = xr.open_dataset("{}MITgcm_rect_nc.nc" "".format(Datadir))
@pytest.fixture
def curv_prime(xr_curv, curv):
    od = xr_curv.drop_vars(["YG"])
    curv._add_missing_cs_sn()
    od["CS"] = xr.DataArray(curv["CS"], dims=xr_curv["XC"].dims)
    od["SN"] = xr.DataArray(curv["SN"], dims=xr_curv["XC"].dims)
    return od


@pytest.fixture
def curv_prprm(xr_curv):
    return xr_curv.drop_vars(["dyG"])


@pytest.fixture
def curv_prprp(xr_curv):
    return xr_curv.drop_vars(
        ["time_midp"] + [i for i in xr_curv.data_vars if "time_midp" in xr_curv[i].dims]
    )


@pytest.fixture
def test_create_tree_cartesian():
    _u.create_tree(xr_curv["XC"], xr_curv["YC"], R=None)


@pytest.mark.parametrize("data", ["curv_prime", "curv_prprm", "curv_prprp"])
def test_incomplete_data(data, curv_prime, curv_prprm, curv_prprp):
    data = eval(data)
    oo = ocedata(data)
    oo.find_rel_h(np.array([-14]), np.array([70.5]))
