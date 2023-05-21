import math

import numpy as np
import pytest
import xarray as xr

import seaduck as sd


def test_rel_lon():
    assert sd.utils.rel_lon(-5, 125) == 230


@pytest.mark.parametrize("item,length", [([1, 2, 3], 3), (0, 1)])
def test_general_len(item, length):
    assert sd.utils._general_len(item) == length


def test_to_180():
    assert sd.utils.to_180(365) == 5


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


@pytest.mark.parametrize("lst", [[1, 2, 3], np.arange(3)])
@pytest.mark.parametrize("select", [1, 3])
def test_combination(lst, select):
    the_ls = sd.utils.get_combination(lst, select)
    assert len(the_ls) == math.factorial(len(lst)) / (math.factorial(select)) / (
        math.factorial(len(lst) - select)
    )


def test_none_in():
    assert sd.utils.NoneIn([1, 2, None])
    assert not (sd.utils.NoneIn([1, 2, 3]))


@pytest.mark.parametrize("ds", ["curv"], indirect=True)
def test_cs_sn(ds):
    sd.utils.missing_cs_sn(ds)
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
    shallow = -10
    deep = -10
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
