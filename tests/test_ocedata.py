import pytest
import numpy as np
import xarray as xr

import seaduck.OceData as ocedata
import seaduck.utils as _u


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


def test_auto_alias(xr_curv):
    with pytest.raises(NotImplementedError):
        ocedata(xr_curv, alias="auto")


def test_manual_alias(xr_curv):
    an_alias = {
        "dXC": "dxC",
        "dYC": "dyC",
        "dZ": "drC",
        "dXG": "dxG",
        "dYG": "dyG",
        "dZl": "drF",
        "SALT": "S",
    }
    od = ocedata(xr_curv, an_alias)
    od["SALT"] = xr.ones_like(od._ds["XC"])
    try:
        import pandas

        od.show_alias()
    except ImportError:
        with pytest.raises(NameError):
            od.show_alias()


def test_add_missing_grid(ecco):
    # TODO
    ecco._add_missing_grid()


def test_not_h_ready(xr_curv):
    temp = xr_curv.drop_vars(["XG", "YG"])
    with pytest.raises(ValueError):
        ocedata(temp)
