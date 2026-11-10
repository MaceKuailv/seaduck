import numpy as np
import pytest
import xarray as xr
from scipy import spatial

try:
    import numba

    HAS_NUMBA = True
    _ = numba
except ImportError:
    HAS_NUMBA = False

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
    cs, sn = sd.utils.missing_cs_sn(ds, return_xr=True)
    assert isinstance(cs, xr.DataArray)


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


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_ray_trace():
    poly = np.array([[0, 0], [0.5, 1], [0.5, 2], [1, 1], [0.5, 0]])
    xs = np.array([0.1, 0.49, 10, 0.6])
    ys = np.array([0.1, 1, 10, 0.6])
    ans = np.array([True, False, False, True])
    assert np.allclose(sd.utils.parallelpointinpolygon(xs, ys, poly), ans)


# ---------------------------------------------------------------------
# 3D Cartesian conversion between lon/lat and the position in a cell
# ---------------------------------------------------------------------
def _polar_cell(lat=89.5):
    # A cell with the North Pole in its interior, corners ordered
    # SW, SE, NE, NW (counter-clockwise seen from above).
    px = np.array([225.0, 315.0, 45.0, 135.0])
    py = np.full(4, lat)
    return px, py


def _tile(corner, n):
    return np.tile(np.asarray(corner, float)[:, None], n)


@pytest.mark.parametrize("lon_ref", [None, 700.0])
def test_xyz_latlon_round_trip(lon_ref):
    lat = np.array([-89.0, -30.0, 0.0, 45.0, 89.9])
    lon = np.array([-179.0, -20.0, 0.0, 170.0, 359.0])
    new_lat, new_lon = sd.utils.xyz2latlon(sd.utils.latlon2xyz(lat, lon), lon_ref)
    assert np.allclose(new_lat, lat)
    assert np.allclose(sd.utils.to_180(new_lon - lon), 0)
    if lon_ref is not None:
        assert np.all(np.abs(new_lon - lon_ref) <= 180)


def test_oceanparcel_cartesian_round_trip_pole_cell():
    np.random.seed(0)
    n = 50
    rx, ry = np.random.uniform(-0.5, 0.5, (2, n))
    px, py = (_tile(c, n) for c in _polar_cell())
    lon, lat = sd.utils.rel2latlon_oceanparcel(rx, ry, px, py)
    new_rx, new_ry = sd.utils.find_rx_ry_oceanparcel_cartesian(lon, lat, px, py)
    assert np.allclose(new_rx, rx, atol=1e-10)
    assert np.allclose(new_ry, ry, atol=1e-10)
    # the center of this symmetric cell is the pole
    _, lat_c = sd.utils.rel2latlon_oceanparcel(
        np.zeros(1), np.zeros(1), px[:, :1], py[:, :1]
    )
    assert np.isclose(lat_c[0], 90.0)


def test_oceanparcel_cartesian_shared_face():
    # cell A is the pole cell, cell B is its northern neighbor (across the
    # NE-NW face, which is on the other side of the pole).
    s = np.linspace(-0.45, 0.45, 7)
    n = len(s)
    pxa, pya = _polar_cell()
    pxb = np.array([135.0, 45.0, 45.0, 135.0])
    pyb = np.array([89.5, 89.5, 88.5, 88.5])
    lon, lat = sd.utils.rel2latlon_oceanparcel(
        s, np.full(n, 0.5), _tile(pxa, n), _tile(pya, n)
    )
    rx, ry = sd.utils.find_rx_ry_oceanparcel_cartesian(
        lon, lat, _tile(pxb, n), _tile(pyb, n)
    )
    assert np.allclose(ry, -0.5, atol=1e-10)
    assert np.allclose(rx, s, atol=1e-10)


def test_oceanparcel_cartesian_close_to_lonlat_for_small_cells():
    np.random.seed(1)
    n = 20
    rx, ry = np.random.uniform(-0.5, 0.5, (2, n))
    px = _tile([10.0, 10.1, 10.12, 10.02], n)
    py = _tile([30.0, 30.01, 30.11, 30.1], n)
    lon, lat = sd.utils.rel2latlon_oceanparcel(rx, ry, px, py, cartesian=False)
    rx3, ry3 = sd.utils.find_rx_ry_oceanparcel_cartesian(lon, lat, px, py)
    # the two schemes only differ by the curvature within a cell
    assert np.allclose(rx3, rx, atol=1e-3)
    assert np.allclose(ry3, ry, atol=1e-3)


@pytest.mark.parametrize("by", [0.0, 60.0, 89.95])
def test_naive_cartesian_round_trip(by):
    np.random.seed(2)
    n = 20
    rx, ry = np.random.uniform(-0.5, 0.5, (2, n))
    bx = np.full(n, 100.0)
    by = np.full(n, by)
    cs = np.full(n, np.cos(0.3))
    sn = np.full(n, np.sin(0.3))
    dx = np.full(n, 20e3)
    dy = np.full(n, 15e3)
    lon, lat = sd.utils.rel2latlon_cartesian(rx, ry, cs, sn, dx, dy, bx, by)
    new_rx, new_ry = sd.utils.find_rx_ry_naive_cartesian(
        lon, lat, bx, by, cs, sn, dx, dy
    )
    assert np.allclose(new_rx, rx, atol=1e-10)
    assert np.allclose(new_ry, ry, atol=1e-10)


def test_naive_cartesian_close_to_lonlat():
    # away from the pole, the 3D and lon/lat versions agree up to the
    # earth radius used by the lon/lat version (utils.deg2m)
    x, y = np.array([10.01]), np.array([20.02])
    args = tuple(np.array([v]) for v in [10.0, 20.0, 1.0, 0.0, 1e4, 1e4])
    rx3, ry3 = sd.utils.find_rx_ry_naive_cartesian(x, y, *args)
    rx2, ry2 = sd.utils.find_rx_ry_naive(x, y, *args)
    ratio = sd.utils.R_EARTH * np.pi / 180 / sd.utils.deg2m
    assert np.allclose(rx3, rx2 * ratio, rtol=1e-3)
    assert np.allclose(ry3, ry2 * ratio, rtol=1e-3)
