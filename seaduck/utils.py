import functools
import os

import numpy as np

# Required dependencies (private)
import xarray as xr
from scipy import spatial

from seaduck.runtime_conf import compileable

try:
    import pooch
except ImportError:
    pass


@functools.cache
def pooch_prepare():
    """Prepare for loading datasets using pooch."""
    pooch_testdata = pooch.create(
        path=pooch.os_cache("seaduck"),
        base_url="doi:10.5281/zenodo.7949168",
        registry=None,
    )
    pooch_testdata.load_registry_from_doi()  # Automatically populate the registry
    pooch_fetch_kwargs = {"progressbar": True}
    return pooch_testdata, pooch_fetch_kwargs


def process_ecco(ds):
    """Add more meat to ECCO dataset after the skeleton is downloaded."""
    rand1 = np.random.random((50, 13, 90, 90))
    rand2 = np.random.random((50, 13, 90, 90))
    rand3 = np.random.random((50, 13, 90, 90))
    ds["UVELMASS"] = xr.DataArray(
        np.stack([rand1, rand2, rand3], axis=0), dims=("time", "Z", "face", "Y", "Xp1")
    )
    ds["UVELMASS"][0] = ds.UVELMASS1
    ds["UVELMASS"][1] = ds.UVELMASS1 * rand1
    ds["UVELMASS"][2] = ds.UVELMASS1 * rand2

    ds["WVELMASS"] = xr.DataArray(
        np.stack([rand1, rand2, rand3], axis=0), dims=("time", "Zl", "face", "Y", "X")
    )
    ds["WVELMASS"][0] = ds.WVELMASS1
    ds["WVELMASS"][1] = ds.WVELMASS1 * rand1
    ds["WVELMASS"][2] = ds.WVELMASS1 * rand2

    ds["VVELMASS"] = xr.DataArray(
        np.stack([rand1, rand2, rand3], axis=0), dims=("time", "Z", "face", "Yp1", "X")
    )
    ds["VVELMASS"][0] = ds.VVELMASS1
    ds["VVELMASS"][1] = ds.VVELMASS1 * rand1
    ds["VVELMASS"][2] = ds.VVELMASS1 * rand2

    ds["SALT"] = xr.DataArray(
        np.stack([rand1, rand2, rand3], axis=0), dims=("time", "Z", "face", "Y", "X")
    )
    ds["SALT_snap"] = xr.DataArray(
        np.stack([rand3, rand1], axis=0), dims=("time_midp", "Z", "face", "Y", "X")
    )
    ds["ETAN"] = xr.DataArray(rand1[:3], dims=("time", "face", "Y", "X"))
    ds["ETAN_snap"] = xr.DataArray(rand3[:2], dims=("time_midp", "face", "Y", "X"))
    return ds


@functools.cache
def get_dataset(name):
    """Use pooch to download datasets from cloud.

        This is just for testing purposes.

    Parameters
    ----------
        + name: string
            The name of dataset, now support "ecco", "aviso", "curv", "rect"
    """
    pooch_testdata, pooch_fetch_kwargs = pooch_prepare()
    fnames = pooch_testdata.fetch(f"{name}.tar.gz", pooch.Untar(), **pooch_fetch_kwargs)
    ds = xr.open_zarr(os.path.commonpath(fnames))
    if name == "ecco":
        return process_ecco(ds)
    return ds


def chg_ref_lon(x, ref_lon):
    """Change the definition of 0 longitude.

    Return how much east one need to go from ref_lon to x
    This function aims to address
    the confusion caused by the discontinuity in longitude.
    """
    return (x - ref_lon) % 360


def _general_len(thing):
    """Change the definition of len such that the length of a float is 1."""
    try:
        return len(thing)
    except TypeError:
        return 1


def get_key_by_value(d, value):
    """Find one of the keys in a dictionary.

    the key that correspond to the given value.

    Parameters
    ----------
    + d: dictionaty
        dictionary to lookup key from
    + value: object
        A object that has __eq__ method.
    """
    for k, v in d.items():
        if v == value:
            return k
    return None


@compileable
def spherical2cartesian(lat, lon, R=6371.0):
    """Convert spherical coordinates to cartesian.

    Parameters
    ----------
    + lat: np.array
        Spherical Y coordinate (latitude)
    + lon: np.array
        Spherical X coordinate (longitude)
    + R: scalar
        Earth radius in km
        If None, use geopy default

    Returns
    -------
    + x: np.array
        Cartesian x coordinate
    + y: np.array
        Cartesian y coordinate
    + z: np.array
        Cartesian z coordinate
    """
    # Convert
    y_rad = np.deg2rad(lat)
    x_rad = np.deg2rad(lon)
    x = R * np.cos(y_rad) * np.cos(x_rad)
    y = R * np.cos(y_rad) * np.sin(x_rad)
    z = R * np.sin(y_rad)

    return x, y, z


@compileable
def to_180(x, peri=360):
    """Convert any longitude scale to [-180,180)."""
    x = x % peri
    return x + (-1) * (x // (peri / 2)) * peri


def local_to_latlon(u, v, cs, sn):
    """Convert local vector to north-east."""
    uu = u * cs - v * sn
    vv = u * sn + v * cs
    return uu, vv


@compileable
def rel2latlon(rx, ry, rzl, cs, sn, dx, dy, dzl, bx, by, bzl):
    """Translate the spatial rel-coords into lat-lon-dep coords."""
    temp_x = rx * dx / deg2m
    temp_y = ry * dy / deg2m
    dlon = (temp_x * cs - temp_y * sn) / np.cos(by * np.pi / 180)
    dlat = temp_x * sn + temp_y * cs
    lon = dlon + bx
    lat = dlat + by
    dep = bzl + dzl * rzl
    return lon, lat, dep


def create_tree(x, y, R=6371.0, leafsize=16):
    """Create a cKD tree object.

    Parameters
    ----------
    + x,y: np.ndarray
        longitude and latitude of the grid location
    + R: float
        The radius in kilometers of the planet.
    + leafsize: int
        When to switch to brute force search.
    """
    if R:
        x, y, z = spherical2cartesian(lat=y, lon=x, R=R)
    else:
        z = xr.zeros_like(y)

    # Stack
    rid_value = 777777
    if isinstance(x, xr.DataArray):
        x = x.stack(points=x.dims).fillna(rid_value).data
        y = y.stack(points=y.dims).fillna(rid_value).data
        z = z.stack(points=z.dims).fillna(rid_value).data
    elif isinstance(x, np.ndarray):
        x = x.ravel()
        np.nan_to_num(x.ravel(), nan=rid_value, copy=False)
        y = y.ravel()
        np.nan_to_num(y.ravel(), nan=rid_value, copy=False)
        z = z.ravel()
        np.nan_to_num(z.ravel(), nan=rid_value, copy=False)

    # Construct KD-tree
    tree = spatial.cKDTree(np.column_stack((x, y, z)), leafsize=leafsize)

    return tree


def NoneIn(lst):
    """See if there is a None in the iterable object. Return a Boolean."""
    ans = False
    for i in lst:
        if i is None:
            ans = True
            break
    return ans


@compileable
def find_ind(array, value, peri=None, ascending=1, above=True):
    """Find the index of the nearest value to the given value.

    Parameters
    ----------
    + array: numpy.ndarray
        1D numpy array to search index from
    + value: number
        The value to find nearest neighbor with
    + peri: number
        The periodicity of the array.
        For example, 360 for longitude.
    + ascending: int
        Whether the array is in ascending order.
        1 for ascending order, -1 for descending order.
    + above: boolean
        If True, return the index of the largest item in
        array smaller than value.
        Otherwise, return the closest value.
    """
    array = np.asarray(array)
    if peri is None:
        idx = np.argmin(np.abs(array - value))
    else:
        idx = np.argmin(np.abs(to_180((array - value), peri=peri)))
    if above and array[idx] > value and peri is None:
        idx -= ascending * 1
    if idx < 0:
        raise ValueError("Value out of bound.")
    idx = int(idx)
    return idx, array[idx]


deg2m = 6271e3 * np.pi / 180


def find_ind_h(lons, lats, tree, h_shape):
    """Use ckd tree to find the horizontal indexes,."""
    x, y, z = spherical2cartesian(lats, lons)
    _, index1d = tree.query(np.array([x, y, z]).T)
    if len(h_shape) == 3:
        faces, iys, ixs = np.unravel_index((index1d), h_shape)
    elif len(h_shape) == 2:
        faces = None
        iys, ixs = np.unravel_index((index1d), h_shape)
    return faces, iys, ixs


@compileable
def find_rel(
    value, array, darray=None, ascending=1, above=True, peri=None, dx_right=True
):
    """Find the rel-coords of the 1D coords.

    The backend for all find_rel functions

    Parameters
    ----------
    + value: numpy.ndarray
        1D array for the value to find rel-coords.
    + array: numpy.ndarray
        The array of potential reference levels.
    + darray: numpy.ndarray, optional
        The distances between reference levels.
    + peri: number
        The periodicity of the array.
        For example, 360 for longitude.
    + ascending: int
        Whether the array is in ascending order.
        1 for ascending order, -1 for descending order.
    + above: boolean
        If True, return the index of the largest item in
        array smaller than value.
        Otherwise, return the closest value.
    + dx_right: boolean
        If True, darray[i] = abs(array[i+1] - array[i])

    Returns
    -------
    + ix: numpy.ndarray
        Indexes of the reference level
    + rx: numpy.ndarray
        Non-dimensional distance to the reference level
    + dx: numpy.ndarray
        distance between the reference t level and the next one.
    + bx: numpy.ndarray
        Value of the reference level
    """
    if darray is None:
        darray = np.abs(array[1:] - array[:-1])
        darray = np.append(darray, darray[-1])
        dx_right = True
    ixs = np.zeros_like(value)
    rxs = np.ones_like(value) * 0.0
    dxs = np.ones_like(value) * 0.0
    bxs = np.ones_like(value) * 0.0

    dx_offset = int(not dx_right) + int(ascending > 0) - 1
    for i, x in enumerate(value):
        ix, bx = find_ind(array, x, ascending=ascending, above=above, peri=peri)
        if peri is None:
            dx = x - bx
        else:
            dx = to_180(x - bx, peri=peri)

        if not above or peri is not None:
            # peri not None will effectively overwrite above
            dx_offset = int(not dx_right) + int(ascending * dx > 0) - 1
        idx = ix + dx_offset

        ixs[i] = ix
        rxs[i] = dx / darray[idx]
        dxs[i] = darray[idx]
        bxs[i] = bx
    return ixs, rxs, dxs, bxs


# Here are a few partial functions for find_rel
@compileable
def find_rel_nearest(value, ts):
    """Find the rel-coords based on the find_ind_nearest method."""
    return find_rel(value, ts, above=False)


@compileable
def find_rel_periodic(value, ts, peri):
    """Find the rel-coords based on the find_ind_periodic method."""
    return find_rel(value, ts, peri=peri)


@compileable
def find_rel_z(depth, some_z, some_dz=None, dz_above_z=True):
    """Find the rel-coords of the vertical coords.

    Parameters
    ----------
    + depth: numpy.ndarray
        1D array for the depth of interest in meters.
        More negative means deeper.
    + some_z: numpy.ndarray
        The depth of reference depth.
    + some_dz: numpy.ndarray or None
        dz_i = abs(z_{i+1}- z_i)
    + dz_above_z: Boolean
        Whether the dz as the distance between the depth level and
        a shallower one(True) or a deeper one(False)

    Returns
    -------
    + iz: numpy.ndarray
        Indexes of the reference z level
    + rz: numpy.ndarray
        Non-dimensional distance to the reference z level
    + dz: numpy.ndarray
        distance between the reference z level and the next one.
    """
    return find_rel(
        depth, some_z, darray=some_dz, ascending=-1, dx_right=(not dz_above_z)
    )


@compileable
def find_rel_time(time, ts):
    """Find the rel-coords of the temporal coords.

    Parameters
    ----------
    + time: numpy.ndarray
        1D array for the time since 1970-01-01 in seconds.
    + ts: numpy.ndarray
        The time of model time steps also in seconds.

    Returns
    -------
    + it: numpy.ndarray
        Indexes of the reference t level
    + rt: numpy.ndarray
        Non-dimensional distance to the reference t level
    + dt: numpy.ndarray
        distance between the reference t level and the next one.
    """
    return find_rel(time, ts)


def _read_h(some_x, some_y, some_dx, some_dy, CS, SN, ind):
    """Read the grid coords at given index.

    Parameters
    ----------
    + some_x: numpy.ndarray
        array of longitude, could be XC or XG
    + some_y: numpy.ndarray
        array of latitude, could be YC or YG
    + some_dx: numpy.ndarray or None
        array of distances between grid in the longitudinal direction.
    + some_dy: numpy.ndarray or None
        array of distances between grid in the latitudinal direction.
    + CS: numpy.ndarray or None
        array of the cosine of the angle between grid and meridian.
    + SN: numpy.ndarray or None
        array of the sine of the angle between grid and meridian.
    + ind: tuple
        indexes to read the grid data from.
    """
    bx = some_x[ind]
    by = some_y[ind]
    if some_dx is not None and some_dy is not None:
        dx = some_dx[ind]
        dy = some_dy[ind]
    else:
        dx = None
        dy = None

    if CS is not None and SN is not None:
        cs = CS[ind]
        sn = SN[ind]
    else:
        cs = None
        sn = None
    return cs, sn, dx, dy, bx, by


def find_px_py(XG, YG, tp, ind, cuvwg="G"):
    """Find the nearest 4 corner points.

    This is used in oceanparcel interpolation scheme.
    """
    N = len(ind[0])
    ind1 = tuple(i for i in tp.ind_tend_vec(ind, np.ones(N) * 3, cuvwg=cuvwg))
    ind2 = tuple(i for i in tp.ind_tend_vec(ind1, np.zeros(N), cuvwg=cuvwg))
    ind3 = tuple(i for i in tp.ind_tend_vec(ind, np.zeros(N), cuvwg=cuvwg))

    x0 = XG[ind]
    x1 = XG[ind1]
    x2 = XG[ind2]
    x3 = XG[ind3]

    y0 = YG[ind]
    y1 = YG[ind1]
    y2 = YG[ind2]
    y3 = YG[ind3]

    px = np.vstack([x0, x1, x2, x3]).astype("float64")
    py = np.vstack([y0, y1, y2, y3]).astype("float64")

    return px, py


@compileable
def find_rx_ry_naive(x, y, bx, by, cs, sn, dx, dy):
    """Find the non-dimensional coords using the local cartesian scheme."""
    dlon = to_180(x - bx)
    dlat = to_180(y - by)
    rx = (dlon * np.cos(by * np.pi / 180) * cs + dlat * sn) * deg2m / dx
    ry = (dlat * cs - dlon * sn * np.cos(by * np.pi / 180)) * deg2m / dy
    return rx, ry


@compileable
def find_rx_ry_oceanparcel(x, y, px, py):
    """Find the non-dimensional horizontal distance.

    This is done using the oceanparcel scheme.
    """
    rx = np.ones_like(x) * 0.0
    ry = np.ones_like(y) * 0.0
    x0 = px[0]

    x = to_180(x - x0)
    px = to_180(px - x0)

    invA = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
        ]
    )
    a = np.dot(invA, px)
    b = np.dot(invA, py)

    aa = a[3] * b[2] - a[2] * b[3]
    bb = a[3] * b[0] - a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + x * b[3] - y * a[3]
    cc = a[1] * b[0] - a[0] * b[1] + x * b[1] - y * a[1]

    det2 = bb * bb - 4 * aa * cc

    order1 = np.abs(aa) < 1e-12
    order2 = np.logical_and(~order1, det2 >= 0)
    ry = -(cc / bb)  # if it is supposed to be nan, just try linear solve.
    ry[order2] = ((-bb + np.sqrt(det2)) / (2 * aa))[order2]

    rot_rectilinear = np.abs(a[1] + a[3] * ry) < 1e-12
    rx[rot_rectilinear] = (
        (y - py[0]) / (py[1] - py[0]) + (y - py[3]) / (py[2] - py[3])
    )[rot_rectilinear] * 0.5
    rx[~rot_rectilinear] = ((x - a[0] - a[2] * ry) / (a[1] + a[3] * ry))[
        ~rot_rectilinear
    ]

    return rx - 1 / 2, ry - 1 / 2


def find_rel_h_naive(lon, lat, some_x, some_y, some_dx, some_dy, CS, SN, tree):
    """Find the rel-coords in the horizontal.

    very similar to find_rel_time/v
    rx,ry,dx,dy are defined the same way
    for example
    rx = "how much to the right of the node"/"size of the cell in left-right direction"
    dx = "size of the cell in left-right direction".

    cs,sn is just the cos and sin of the grid orientation.
    It will come in handy when we transfer vectors.
    """
    if NoneIn([lon, lat, some_x, some_y, some_dx, some_dy, CS, SN, tree]):
        raise ValueError("Some of the required variables are missing")
    h_shape = some_x.shape
    faces, iys, ixs = find_ind_h(lon, lat, tree, h_shape)
    if faces is not None:
        ind = (faces, iys, ixs)
    else:
        ind = (iys, ixs)
    cs, sn, dx, dy, bx, by = _read_h(some_x, some_y, some_dx, some_dy, CS, SN, ind)
    rx, ry = find_rx_ry_naive(lon, lat, bx, by, cs, sn, dx, dy)
    return faces, iys, ixs, rx, ry, cs, sn, dx, dy, bx, by


def find_rel_h_rectilinear(x, y, lon, lat):
    """Find the rel-coords using the rectilinear scheme."""
    ratio = 6371e3 * np.pi / 180
    ix, rx, dx, bx = find_rel_periodic(x, lon, 360.0)
    iy, ry, dy, by = find_rel_periodic(y, lat, 360.0)
    dx = np.cos(y * np.pi / 180) * ratio * dx
    dy = ratio * dy
    face = None
    cs = np.ones_like(x)
    sn = np.zeros_like(x)
    return face, iy, ix, rx, ry, cs, sn, dx, dy, bx, by


def find_rel_h_oceanparcel(
    x, y, some_x, some_y, some_dx, some_dy, CS, SN, XG, YG, tree, tp
):
    """Find the rel-coords using the rectilinear scheme."""
    if NoneIn([x, y, some_x, some_y, XG, YG, tree]):
        raise ValueError("Some of the required variables are missing")
    h_shape = some_x.shape
    faces, iys, ixs = find_ind_h(x, y, tree, h_shape)
    if faces is not None:
        ind = (faces, iys, ixs)
    else:
        ind = (iys, ixs)
    cs, sn, dx, dy, bx, by = _read_h(some_x, some_y, some_dx, some_dy, CS, SN, ind)
    px, py = find_px_py(XG, YG, tp, ind)
    rx, ry = find_rx_ry_oceanparcel(x, y, px, py)
    return faces, iys, ixs, rx, ry, cs, sn, dx, dy, bx, by


def weight_f_node(rx, ry):
    """Assign weights to four corners.

    assign weight based on the non-dimensional
    coords to the four corner points.
    """
    return np.vstack(
        [
            (0.5 - rx) * (0.5 - ry),
            (0.5 + rx) * (0.5 - ry),
            (0.5 + rx) * (0.5 + ry),
            (0.5 - rx) * (0.5 + ry),
        ]
    ).T


def find_cs_sn(thetaA, phiA, thetaB, phiB):
    """Find a spherical angle OAB.

    theta is the angle
    between the meridian crossing point A
    and the geodesic connecting A and B.

    this function return cos and sin of theta
    """
    # O being north pole
    AO = np.pi / 2 - thetaA
    BO = np.pi / 2 - thetaB
    dphi = phiB - phiA
    # Spherical law of cosine on AOB
    cos_AB = np.cos(BO) * np.cos(AO) + np.sin(BO) * np.sin(AO) * np.cos(dphi)
    sin_AB = np.sqrt(1 - cos_AB**2)
    # spherical law of sine on triangle AOB
    SN = np.sin(BO) * np.sin(dphi) / sin_AB
    CS = np.sign(thetaB - thetaA) * np.sqrt(1 - SN**2)
    return CS, SN


def missing_cs_sn(ds):
    """Fill in the CS,SN of a dataset."""
    xc = np.deg2rad(np.array(ds.XC))
    yc = np.deg2rad(np.array(ds.YC))
    cs = np.zeros_like(xc)
    sn = np.zeros_like(xc)
    cs[0], sn[0] = find_cs_sn(yc[0], xc[0], yc[1], xc[1])
    cs[-1], sn[-1] = find_cs_sn(yc[-2], xc[-2], yc[-1], xc[-1])
    cs[1:-1], sn[1:-1] = find_cs_sn(yc[:-2], xc[:-2], yc[2:], xc[2:])
    ds["CS"] = ds["XC"]
    ds["CS"].values = cs

    ds["SN"] = ds["XC"]
    ds["SN"].values = sn


def convert_time(time):
    """Convert time into seconds after 1970-01-01.

    time needs to be a string or a np.datetime64 object.
    """
    t0 = np.datetime64("1970-01-01")
    one_sec = np.timedelta64(1, "s")
    if isinstance(time, str):
        dt = np.datetime64(time) - t0
        return dt / one_sec
    elif isinstance(time, np.datetime64):
        return (time - t0) / one_sec


def easy_3d_cube(lon, lat, dep, tim, print_total_number=False):
    """Create 4D coords for initializing Position/Particle."""
    east, west, Nlon = lon
    south, north, Nlat = lat
    shallow, deep, Ndep = dep
    t_in_sec = convert_time(tim)

    x = np.linspace(east, west, Nlon)
    y = np.linspace(south, north, Nlat)
    x, y = np.meshgrid(x, y)
    x = x.ravel()
    y = y.ravel()
    levels = np.linspace(shallow, deep, Ndep)
    x, z = np.meshgrid(x, levels)
    y, z = np.meshgrid(y, levels)
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()

    t = np.ones_like(x) * t_in_sec
    if print_total_number:
        print(f"A total {len(x)} positions defined.")
    return x, y, z, t
