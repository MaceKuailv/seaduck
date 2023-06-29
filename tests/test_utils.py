import numpy as np
import pytest
import xarray as xr
from scipy import spatial

import seaduck as sd


def test_chg_ref_lon():
    assert sd.utils.chg_ref_lon(-5, 125) == 230


@pytest.mark.parametrize("item,length", [([1, 2, 3], 3), (0, 1)])
def test_general_len(item, length):
    assert sd.utils._general_len(item) == length


def test_to_180():
    assert sd.utils.to_180(365) == 5


def test_to_180_underflow():
    assert abs(sd.utils.to_180(-1e-20)) < 1e-10


def test_key_by_values():
    dic = {1: "a", "hello": "b", "alpha": "a"}
    assert sd.utils.get_key_by_value(dic, "b") == "hello"
    assert sd.utils.get_key_by_value(dic, "a") in [1, "alpha"]
    assert sd.utils.get_key_by_value(dic, "c") is None


@pytest.mark.parametrize(
    "lat,lon",
    [
        (0, 0),
        (45, 45),
        (50, 180),
    ],
)
def test_spherical2cartesian(lat, lon):
    X, Y, Z = sd.utils.spherical2cartesian(lon, lat)
    assert np.isclose(X**2 + Y**2 + Z**2, 6371.0**2)


def test_local_to_latlon():
    u = 1
    v = 0.618
    cs = np.random.random()
    sn = np.sqrt(1 - cs**2)
    uu, vv = sd.utils.local_to_latlon(u, v, cs, sn)
    assert np.isclose(np.hypot(uu, vv), np.hypot(u, v))


def test_none_in():
    assert sd.utils.NoneIn([1, 2, None])
    assert not (sd.utils.NoneIn([1, 2, 3]))


@pytest.mark.parametrize("ds", ["curv"], indirect=True)
def test_cs_sn(ds):
    sd.utils.missing_cs_sn(ds, return_xr=True)
    assert isinstance(ds["CS"], xr.DataArray)


def test_covert_time():
    rand = round(np.random.random() * 1e7)
    time = np.datetime64("1970-01-01") + np.timedelta64(rand, "s")
    ress = sd.utils.convert_time(time)
    assert np.isclose(rand, ress, atol=1)


def test_easy_cube():
    east = -80.0
    west = 0.0
    south = 40.0
    north = 75.0
    shallow = -10.0
    deep = -10.0
    time_string = "1977-01-01"

    Nlon = 10
    Nlat = 10
    Ndep = 1

    x, y, z, t = sd.utils.easy_3d_cube(
        (east, west, Nlon),
        (south, north, Nlat),
        (shallow, deep, Ndep),
        time_string,
        print_total_number=True,
    )
    assert len(x) == Nlon * Nlat * Ndep


ascend = np.array([1, 11, 111, 1111])
dscend = -ascend
darray = np.array([1, 10, 100, 1000])


@pytest.mark.parametrize(
    "value,array,ascending,above,peri,ans",
    [
        (10, ascend, 1, True, None, 0),
        (10, ascend, 1, False, None, 1),
        (0, ascend, 1, False, None, 0),
        (-2, dscend, -1, True, None, 1),
        (-2, dscend, -1, False, None, 0),
        (0, ascend, 1, False, 110.5, 2),
        (0, ascend, 1, False, 1111.5, 3),
    ],
)
def test_find_ind(value, array, ascending, above, peri, ans):
    ix, bx = sd.utils.find_ind(
        array, value, peri=peri, ascending=ascending, above=above
    )
    assert ix == ans
    assert isinstance(ix, int)


@pytest.mark.parametrize(
    "value, array, ascending",
    [(0.5, ascend, 1), (-2000, dscend, -1)],
)
def test_find_ind_out_of_bound(value, array, ascending):
    with pytest.raises(ValueError):
        ix, bx = sd.utils.find_ind(array, value, ascending=ascending, above=True)


@pytest.mark.parametrize(
    "value,array,darray,ascending,above,peri, dx_right,ans",
    [
        (10, ascend, darray, 1, True, None, False, 0.9),
        (10, ascend, 10 * darray, 1, True, None, True, 0.9),
        (10, ascend, None, 1, True, None, True, 0.9),
        (10, ascend, darray, 1, False, None, False, -0.1),
        (0, ascend, darray, 1, False, None, False, -1),
        (-2, dscend, darray, -1, True, None, False, 0.9),
        (-2, dscend, darray, -1, False, None, False, -0.1),
        (-2, dscend, 10 * darray, -1, False, None, True, -0.1),
        (0, dscend, darray, -1, True, None, False, 1),
        (0, ascend, darray, 1, False, 110.5, False, -0.5 / 100),
        (0, ascend, 10 * darray, 1, True, 1111.5, True, 0.5 / 10000),
    ],
)
def test_find_rel(value, array, darray, ascending, above, peri, dx_right, ans):
    value = np.array([value])
    ixs, rxs, dxs, bxs = sd.utils.find_rel(
        value, array, darray, ascending, above, peri, dx_right
    )
    assert rxs[0] == ans
    assert np.issubdtype(ixs.dtype, int)


@pytest.mark.parametrize("ds", ["curv", "ecco"], indirect=True)
def test_create_tree_cartesian(ds):
    tree = sd.utils.create_tree(ds["XC"], ds["YC"], R=None)
    assert isinstance(tree, spatial.cKDTree)
