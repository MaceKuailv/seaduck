import numpy as np
import pytest
import xarray as xr

import seaduck as sd
from seaduck import utils


@pytest.fixture
def curv_prime():
    ds = utils.get_dataset("curv")
    od = sd.OceData(ds)
    ds_out = ds.drop_vars(["YG"])
    od._add_missing_cs_sn()
    ds_out["CS"] = xr.DataArray(od["CS"], dims=ds["XC"].dims)
    ds_out["SN"] = xr.DataArray(od["SN"], dims=ds["XC"].dims)
    return ds_out


@pytest.fixture
def curv_prprm():
    ds = utils.get_dataset("curv")
    return ds.drop_vars(["dyG"])


@pytest.fixture
def curv_prprp():
    ds = utils.get_dataset("curv")
    return ds.drop_vars(
        ["time_midp"] + [i for i in ds.data_vars if "time_midp" in ds[i].dims]
    )


@pytest.fixture
def test_create_tree_cartesian(xr_curv):
    sd.utils.create_tree(xr_curv["XC"], xr_curv["YC"], R=None)


@pytest.mark.parametrize("data", ["curv_prime", "curv_prprm", "curv_prprp"])
def test_incomplete_data(data, curv_prime, curv_prprm, curv_prprp):
    data = eval(data)
    oo = sd.OceData(data)
    oo.find_rel_h(np.array([-14]), np.array([70.5]))


@pytest.mark.parametrize("ds", ["curv"], indirect=True)
def test_auto_alias(ds):
    with pytest.raises(NotImplementedError):
        sd.OceData(ds, alias="auto")


@pytest.mark.parametrize("ds", ["curv"], indirect=True)
def test_manual_alias(ds):
    an_alias = {
        "dXC": "dxC",
        "dYC": "dyC",
        "dZ": "drC",
        "dXG": "dxG",
        "dYG": "dyG",
        "dZl": "drF",
        "SALT": "S",
    }
    od = sd.OceData(ds, an_alias)
    od["SALT"] = xr.ones_like(od._ds["XC"])
    try:
        od.show_alias()
    except ImportError:
        with pytest.raises(NameError):
            od.show_alias()


@pytest.mark.parametrize("od", ["ecco"], indirect=True)
def test_add_missing_grid(od):
    # TODO
    od._add_missing_grid()


@pytest.mark.parametrize("ds", ["curv"], indirect=True)
def test_not_h_ready(ds):
    temp = ds.drop_vars(["XG", "YG"])
    with pytest.raises(ValueError):
        sd.OceData(temp)
