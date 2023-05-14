import numpy as np
import pytest
import xarray as xr
import seaduck.OceData as ocedata
import seaduck.utils as _u
'''
This test does not run. it is for confidence.
'''

Datadir = "tests/Data/"
ecco = xr.open_zarr(Datadir + "small_ecco")

curv = xr.open_dataset("{}MITgcm_curv_nc.nc" "".format(Datadir))

rect = xr.open_dataset("{}MITgcm_rect_nc.nc" "".format(Datadir))

_u.missing_cs_sn(curv)
_u.missing_cs_sn(rect)


# @pytest.mark.parametrize("od", [ecco, curv, rect])
# def test_grid2array(od):
#     _u.grid2array(od)
#     assert _u.XC.any()


@pytest.mark.parametrize(
    "lat,lon,od",
    [
        (0, 0, ecco),
        (45, 45, ecco),
        (50, 180, ecco),
        (45, 260, ecco),
        (85, 10, ecco),
        (-85, 20, ecco),
        (70.5, -14, curv),  # the domain is too small
        (70.5, -17.7, rect),
    ],
)
def test_find_ind_h(lat, lon, od):
    #     _u.grid2array(od)
    tree = od.create_tree("C")
    h_shape = od._ds.XC.shape
    face, iy, ix = _u.find_ind_h(np.array([lon]), np.array([lat]), tree, h_shape)
    if face is not None:
        near_x = od._ds.XC[face, iy, ix][0, 0, 0].values
        near_y = od._ds.YC[face, iy, ix][0, 0, 0].values
    else:
        near_x = od._ds.XC[iy, ix][0, 0].values
        near_y = od._ds.YC[iy, ix][0, 0].values
    print(near_x, near_y)
    X, Y, Z = _u.spherical2cartesian(lon, lat)
    NX, NY, NZ = _u.spherical2cartesian(near_x, near_y)

    thres = 1.5 * float(od._ds.dxC.max().values)
    print(thres)
    dist = np.sqrt((X - NX) ** 2 + (Y - NY) ** 2 + (Z - NZ) ** 2) * 1e3
    print(dist)
    assert thres > dist


@pytest.mark.parametrize("z", [-5, -10, -4000, -8000, 0])
@pytest.mark.parametrize("which", ["Z", "Zl"])
@pytest.mark.parametrize("od", [ecco, curv, rect])
def test_find_ind_z(z, which, od):
    somez = np.array(od._ds[which])
    iz = _u.find_ind_z(somez, z)
    try:
        lower = somez[iz]
    except KeyError:
        lower = -np.inf
    higher = max([somez[max([0, iz - 1])], 0])
    assert lower <= z
    assert z <= higher


@pytest.mark.parametrize("t", [0, 1e7, 3.5e8, 1e10, 1e13])
def test_find_ind_t(t):
    somet = np.arange(0, 1e10, 1e8)
    it = _u.find_ind_t(somet, t)
    try:
        later = somet[it + 1]
    except:
        later = np.inf
    earlier = somet[it]
    assert earlier <= t
    assert t <= later


@pytest.mark.parametrize(
    "x", [np.linspace(-179, 180, 100), np.linspace(150, 220, 100), np.ones(100)]
)
@pytest.mark.parametrize(
    "y", [np.linspace(-89, 90, 100), np.ones(100), np.ones(100) * 89.9]
)
def test_rel_2d(x, y):
    _u.grid2array(ecco)
    faces, iys, ixs, rx, ry, cs, sn, dx, dy = _u.find_rel_2d(x, y)
    assert faces.dtype == "int"
    assert abs(rx).max() < 5  # actually for all the grid of interest 1 is enough.
    assert abs(ry).max() < 5


@pytest.mark.parametrize(
    "x,y,od",
    [
        (np.linspace(-17.95, -17.55, 10), np.linspace(70.3, 70.5, 10), rect),
        (np.linspace(-16, -13, 10), np.linspace(70, 70.7, 10), curv),
    ],
)
def test_rel_2d_on_small(x, y, od):
    _u.grid2array(od)
    faces, iys, ixs, rx, ry, cs, sn, dx, dy = _u.find_rel_2d(x, y)
    assert faces is None
    assert abs(rx).max() < 5  # actually for all the grid of interest 1 is enough.
    assert abs(ry).max() < 5


@pytest.mark.parametrize(
    "x", [np.linspace(-179, 180, 100), np.linspace(150, 220, 100), np.ones(100)]
)
@pytest.mark.parametrize(
    "y", [np.linspace(-89, 90, 100), np.ones(100), np.ones(100) * 89.9]
)
@pytest.mark.parametrize(
    "z,od",
    [
        (-10 * np.ones(100), ecco),
        (-4000 * np.ones(100), ecco),
        (np.linspace(0, -1000, 100), ecco),
        (-10 * np.ones(100), rect),
        (-10 * np.ones(100), curv),
    ],
)
def test_rel_3d(x, y, z, od):
    _u.grid2array(od)
    iz, faces, iys, ixs, rx, ry, rz, cs, sn, dx, dy, dz = _u.find_rel_3d(x, y, z)
    print(iz.dtype)
    assert iz.dtype == "int"
    assert abs(rz).max() <= 1


@pytest.mark.parametrize(
    "x",
    [
        np.linspace(-179, 180, 100),
    ],
)
@pytest.mark.parametrize(
    "y",
    [
        np.linspace(-89, 90, 100),
    ],
)
@pytest.mark.parametrize("z", [np.linspace(0, -95, 100)])
@pytest.mark.parametrize("t", [np.zeros(100)])
@pytest.mark.parametrize(
    "od", [ecco, curv, rect]
)
def test_rel_4d(x, y, z, t, od):
    _u.grid2array(od)
    it, iz, faces, iys, ixs, rx, ry, rz, rt, cs, sn, dx, dy, dz, dt = _u.find_rel_4d(
        x, y, z, t
    )
    assert it.dtype == "int"
    assert abs(rt).max() <= 1


# if __name__ =='__main__':
#     x = np.linspace(-179,180,100)
#     y = np.linspace(-89,90,100)
#     z = -10*np.ones(100)
#     test_rel_3d(x,y,z)
#     test_find_ind_h(50,180,ecco)
