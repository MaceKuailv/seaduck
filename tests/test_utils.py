import math

import numpy as np
import pytest
import xarray as xr

import seaduck.utils as _u


def test_rel_lon():
    assert _u.rel_lon(-5, 125) == 230


@pytest.mark.parametrize("item,length", [([1, 2, 3], 3), (0, 1)])
def test_general_len(item, length):
    assert _u._general_len(item) == length


def test_to_180():
    assert _u.to_180(365) == 5


def test_key_by_values():
    dic = {1: "a", "hello": "b", "alpha": "a"}
    assert _u.get_key_by_value(dic, "b") == "hello"
    assert _u.get_key_by_value(dic, "a") in [1, "alpha"]
    assert _u.get_key_by_value(dic, "c") is None


@pytest.mark.parametrize(
    "lat,lon",
    [
        (0, 0),
        (45, 45),
        (50, 180),
    ],
)
def test_spherical2cartesian(lat, lon):
    X, Y, Z = _u.spherical2cartesian(lon, lat)
    assert np.isclose(X**2 + Y**2 + Z**2, 6371.0**2)


def test_local_to_latlon():
    u = 1
    v = 0.618
    cs = np.random.random()
    sn = np.sqrt(1 - cs**2)
    uu, vv = _u.local_to_latlon(u, v, cs, sn)
    assert np.isclose(np.hypot(uu, vv), np.hypot(u, v))


@pytest.mark.parametrize("lst", [[1, 2, 3], np.arange(3)])
@pytest.mark.parametrize("select", [1, 3])
def test_combination(lst, select):
    the_ls = _u.get_combination(lst, select)
    assert len(the_ls) == math.factorial(len(lst)) / (math.factorial(select)) / (
        math.factorial(len(lst) - select)
    )


def test_none_in():
    assert _u.NoneIn([1, 2, None])
    assert not (_u.NoneIn([1, 2, 3]))


def test_cs_sn(xr_curv):
    _u.missing_cs_sn(xr_curv)
    assert isinstance(xr_curv["CS"], xr.DataArray)
