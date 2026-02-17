import numpy as np
from packaging.version import parse

from seaduck.utils import _left90, _right90

try:  # pragma: no cover
    import xgcm
except ImportError:  # pragma: no cover
    pass


def _raise_if_no_xgcm():
    try:
        import xgcm

        xgcm
    except ImportError:  # pragma: no cover
        raise ImportError(
            "The python package xgcm is needed."
            "You can install it with:"
            "conda install -c conda-forge xgcm"
        )


def create_ecco_grid(ds, for_outer=False):
    """Create xgcm object for an ECCO dataset."""
    _raise_if_no_xgcm()
    face_connections = {
        "face": {
            0: {"X": ((12, "Y", False), (3, "X", False)), "Y": (None, (1, "Y", False))},
            1: {
                "X": ((11, "Y", False), (4, "X", False)),
                "Y": ((0, "Y", False), (2, "Y", False)),
            },
            2: {
                "X": ((10, "Y", False), (5, "X", False)),
                "Y": ((1, "Y", False), (6, "X", False)),
            },
            3: {"X": ((0, "X", False), (9, "Y", False)), "Y": (None, (4, "Y", False))},
            4: {
                "X": ((1, "X", False), (8, "Y", False)),
                "Y": ((3, "Y", False), (5, "Y", False)),
            },
            5: {
                "X": ((2, "X", False), (7, "Y", False)),
                "Y": ((4, "Y", False), (6, "Y", False)),
            },
            6: {
                "X": ((2, "Y", False), (7, "X", False)),
                "Y": ((5, "Y", False), (10, "X", False)),
            },
            7: {
                "X": ((6, "X", False), (8, "X", False)),
                "Y": ((5, "X", False), (10, "Y", False)),
            },
            8: {
                "X": ((7, "X", False), (9, "X", False)),
                "Y": ((4, "X", False), (11, "Y", False)),
            },
            9: {"X": ((8, "X", False), None), "Y": ((3, "X", False), (12, "Y", False))},
            10: {
                "X": ((6, "Y", False), (11, "X", False)),
                "Y": ((7, "Y", False), (2, "X", False)),
            },
            11: {
                "X": ((10, "X", False), (12, "X", False)),
                "Y": ((8, "Y", False), (1, "X", False)),
            },
            12: {
                "X": ((11, "X", False), None),
                "Y": ((9, "Y", False), (0, "X", False)),
            },
        }
    }
    coords = {"X": {"center": "X", "left": "Xp1"}, "Y": {"center": "Y", "left": "Yp1"}}
    if "Z" in ds.dims:
        coords["Z"] = {"center": "Z", "left": "Zl"}
        if for_outer:
            coords["Z"] = {"center": "Z", "outer": "Zl"}
    if "time" in ds.dims:
        coords["time"] = {"center": "time", "inner": "time_midp"}
    if parse(xgcm.__version__) >= parse("0.9.0"):
        # xgcm trying to be smart.
        xgcmgrd = xgcm.Grid(
            ds,
            periodic=False,
            face_connections=face_connections,
            coords=coords,
            autoparse_metadata=False,
        )
    else:
        xgcmgrd = xgcm.Grid(
            ds, periodic=False, face_connections=face_connections, coords=coords
        )
    return xgcmgrd


def create_periodic_grid(ds):  # pragma: no cover
    _raise_if_no_xgcm()
    xgcmgrd = xgcm.Grid(
        ds,
        periodic=["X"],
        coords={
            "X": {"center": "X", "outer": "Xp1"},
            "Y": {"center": "Y", "outer": "Yp1"},
            "Z": {"center": "Z", "left": "Zl"},
            "time": {"center": "time", "outer": "time_outer"},
        },
    )
    return xgcmgrd


def hor_div(tub, xgcmgrd, xfluxname, yfluxname):
    """Calculate horizontal divergence using xgcm.

    Parameters
    ----------
    tub: sd.OceData or xr.Dataset
        The dataset to calculate data from
    xgcmgrd: xgcm.Grid
        The Grid of the dataset
    xfluxname, yfluxname: string
        The name of the variables corresponding to the horizontal fluxes
        in concentration m^3/s
    """
    try:
        tub["Vol"]
    except KeyError:
        tub._add_missing_vol()
    xy_diff = xgcmgrd.diff_2d_vector(
        {"X": tub[xfluxname].fillna(0), "Y": tub[yfluxname].fillna(0)},
        boundary="fill",
        fill_value=0.0,
    )
    x_diff = xy_diff["X"]
    y_diff = xy_diff["Y"]
    hConv = (x_diff + y_diff) / tub["Vol"]
    return hConv


def ver_div(tub, xgcmgrd, zfluxname):
    """Calculate horizontal divergence using xgcm.

    Parameters
    ----------
    tub: sd.OceData or xr.Dataset
        The dataset to calculate data from
    xgcmgrd: xgcm.Grid
        The Grid of the dataset
    xfluxname, yfluxname, zfluxname: string
        The name of the variables corresponding to the fluxes
        in concentration m^3/s
    """
    try:
        tub["Vol"]
    except KeyError:
        tub._add_missing_vol()
    vConv = (
        xgcmgrd.diff(tub[zfluxname].fillna(0), "Z", boundary="fill", fill_value=0.0)
        / tub["Vol"]
    )
    return -vConv


def total_div(tub, xgcmgrd, xfluxname, yfluxname, zfluxname):
    """Calculate 3D divergence using xgcm.

    Parameters
    ----------
    tub: sd.OceData or xr.Dataset
        The dataset to calculate data from
    xgcmgrd: xgcm.Grid
        The Grid of the dataset
    zfluxname: string
        The name of the variables corresponding to the vertical flux
        in concentration m^3/s
    """
    hDiv = hor_div(tub, xgcmgrd, xfluxname, yfluxname)
    vDiv = ver_div(tub, xgcmgrd, zfluxname)
    return hDiv + vDiv


def bolus_vel_from_psi(tub, xgcmgrd, psixname="GM_PsiX", psiyname="GM_PsiY"):
    """Calculate bolus velocity based on its streamfunction."""
    strmx = tub[psixname].fillna(0)
    strmy = tub[psiyname].fillna(0)

    u = xgcmgrd.diff(strmx, "Z", boundary="fill", fill_value=0.0) / tub["drF"]
    v = xgcmgrd.diff(strmy, "Z", boundary="fill", fill_value=0.0) / tub["drF"]

    vstrmx = strmx * np.array(tub["dyG"])  # there is some fucking problem with xgcm
    vstrmy = strmy * np.array(tub["dxG"])
    print(vstrmy.dims, vstrmx.dims)

    xy_diff = xgcmgrd.diff_2d_vector(
        {"X": vstrmx, "Y": vstrmy}, boundary="fill", fill_value=0.0
    )
    x_diff = xy_diff["X"]
    y_diff = xy_diff["Y"]
    hDiv = x_diff + y_diff

    w = hDiv / tub["rA"]
    return u, v, w


def _slice_corner(array, fc, iy1, iy2, ix1, ix2):
    left = np.minimum(ix1, ix2)
    righ = np.maximum(ix1, ix2)
    down = np.minimum(iy1, iy2)
    uppp = np.maximum(iy1, iy2)
    return array[..., fc, down : uppp + 1, left : righ + 1]


def buffer_x_withface(s, face, lm, rm, tp):
    """Create buffer zone in x direction for one face of an array.

    Parameters
    ----------
    s: numpy.ndarray
        the center field, the last dimension being X,
        the third last dimension being face.
    face: int
        which face to create buffer for.
    lm: int
        the size of the margin to the left.
    rm: int
        the size of the margin to the right.
    tp: seaduck.Topology
        the topology object of the
    """
    shape = list(s.shape)
    shape.pop(-3)
    shape[-1] += lm + rm
    xbuffer = np.zeros(shape)
    xbuffer[..., lm:-rm] = s[..., face, :, :]
    try:
        fc1, iy1, ix1 = tp.ind_moves((face, tp.iymax, 0), [2 for i in range(lm)])
        fc2, iy2, ix2 = tp.ind_moves((face, 0, 0), [2])
        left = _slice_corner(s, fc1, iy1, iy2, ix1, ix2)
    except IndexError:
        left = np.zeros_like(xbuffer[..., :lm])

    try:
        fc3, iy3, ix3 = tp.ind_moves((face, tp.iymax, tp.ixmax), [3 for i in range(rm)])
        fc4, iy4, ix4 = tp.ind_moves((face, 0, tp.ixmax), [3])
        righ = _slice_corner(s, fc3, iy3, iy4, ix3, ix4)
    except IndexError:
        righ = np.zeros_like(xbuffer[..., -rm:])
    try:
        xbuffer[..., :lm] = left
    except ValueError:
        xbuffer[..., :lm] = _right90(left)

    try:
        xbuffer[..., -rm:] = righ
    except ValueError:
        xbuffer[..., -rm:] = _right90(righ)

    return xbuffer


def buffer_y_withface(s, face, lm, rm, tp):
    """Create buffer zone in x direction for one face of an array.

    Parameters
    ----------
    s: numpy.ndarray
        the center field, the last dimension being X,
        the third last dimension being face.
    face: int
        which face to create buffer for.
    lm: int
        the size of the margin to the bottom.
    rm: int
        the size of the margin to the top.
    tp: seaduck.Topology
        the topology object of the
    """
    shape = list(s.shape)
    shape.pop(-3)
    shape[-2] += lm + rm
    ybuffer = np.zeros(shape)
    ybuffer[..., lm:-rm, :] = s[..., face, :, :]
    try:
        fc1, iy1, ix1 = tp.ind_moves((face, 0, tp.ixmax), [1 for i in range(lm)])
        fc2, iy2, ix2 = tp.ind_moves((face, 0, 0), [1])
        left = _slice_corner(s, fc1, iy1, iy2, ix1, ix2)
    except IndexError:
        left = np.zeros_like(ybuffer[..., :lm, :])

    try:
        fc3, iy3, ix3 = tp.ind_moves((face, tp.iymax, tp.ixmax), [0 for i in range(rm)])
        fc4, iy4, ix4 = tp.ind_moves((face, tp.iymax, 0), [0])
        righ = _slice_corner(s, fc3, iy3, iy4, ix3, ix4)
    except IndexError:
        righ = np.zeros_like(ybuffer[..., -rm:, :])
    try:
        ybuffer[..., :lm, :] = left
    except ValueError:
        ybuffer[..., :lm, :] = _left90(left)

    try:
        ybuffer[..., -rm:, :] = righ
    except ValueError:
        ybuffer[..., -rm:, :] = _left90(righ)
    return ybuffer


def buffer_x_periodic(s, lm, rm):
    shape = list(s.shape)
    shape[-1] += lm + rm
    xbuffer = np.zeros(shape)
    xbuffer[..., lm : shape[-1] - rm] = s
    if lm > 0:
        xbuffer[..., :lm] = s[..., -lm:]
    if rm > 0:
        xbuffer[..., -rm:] = s[..., :rm]
    return xbuffer


def buffer_y_periodic(s, lm, rm):
    shape = list(s.shape)
    shape[-2] += lm + rm
    ybuffer = np.zeros(shape)
    ybuffer[..., lm : shape[-2] - rm, :] = s
    if lm > 0:
        ybuffer[..., :lm, :] = s[..., -lm:, :]
    if rm > 0:
        ybuffer[..., -rm:, :] = s[..., :rm, :]
    return ybuffer


def buffer_z_nearest(s, lm, rm):
    shape = list(s.shape)
    shape[-3] += lm + rm
    zbuffer = np.zeros(shape)
    zbuffer[..., lm : shape[-3] - rm, :, :] = s
    if lm > 0:
        zbuffer[..., :lm, :, :] = s[..., :1, :, :]
    if rm > 0:
        zbuffer[..., -rm:, :, :] = s[..., -1:, :, :]
    return zbuffer


def _slope_ratio(Rjm, Rj, Rjp, u_sign, not_z=1):
    """Calculate slope ratio for flux limiter."""
    cr_max = 1e6  # doesn't matter
    cr = np.zeros_like(u_sign)
    pos = not_z * u_sign > 0
    neg = not_z * u_sign <= 0
    cr[pos] = Rjm[pos]
    cr[neg] = Rjp[neg]
    zero_divide = np.abs(Rj) * cr_max <= np.abs(cr)
    cr[zero_divide] = np.sign(cr[zero_divide]) * np.sign(u_sign[zero_divide]) * cr_max
    cr[~zero_divide] = cr[~zero_divide] / Rj[~zero_divide]
    return cr


def _superbee_fluxlimiter(cr):
    return np.maximum(0.0, np.maximum(np.minimum(1.0, 2 * cr), np.minimum(2.0, cr)))


def second_order_flux_limiter_x(s_center, u_cfl):
    """Get interpolated tracer concentration in X.

    for more info, see
    https://mitgcm.readthedocs.io/en/latest/algorithm/adv-schemes.html#second-order-flux-limiters

    Parameters
    ----------
    s_center: numpy.ndarray
        the tracer field with buffer zone added (lm=2,rm=1), the last dimension being X
    u_cfl: numpy.ndarray
        the u velocity normalized by grid size in X, in s^-1.
    """
    xbuffer = buffer_x_periodic(s_center, 2, 2)
    deltas = np.nan_to_num(np.diff(xbuffer, axis=-1), 0)
    Rjp = deltas[..., 2:]
    Rj = deltas[..., 1:-1]
    Rjm = deltas[..., :-2]

    cr = _slope_ratio(Rjm, Rj, Rjp, u_cfl)
    limiter = _superbee_fluxlimiter(cr)
    swall = (
        np.nan_to_num(xbuffer[..., 1:-2] + xbuffer[..., 2:-1]) * 0.5
        - np.sign(u_cfl) * ((1 - limiter) + u_cfl * limiter) * Rj * 0.5
    )
    return swall


def second_order_flux_limiter_y(s_center, u_cfl):
    """Get interpolated tracer concentration in Y.

    for more info, see
    https://mitgcm.readthedocs.io/en/latest/algorithm/adv-schemes.html#second-order-flux-limiters

    Parameters
    ----------
    s_center: numpy.ndarray
        the tracer field with buffer zone added (lm=2,rm=1), the last dimension being X
    u_cfl: numpy.ndarray
        the u velocity normalized by grid size in y, in s^-1.
    """
    ybuffer = buffer_y_periodic(s_center, 2, 2)
    deltas = np.nan_to_num(np.diff(ybuffer, axis=-2), 0)
    Rjp = deltas[..., 2:, :]
    Rj = deltas[..., 1:-1, :]
    Rjm = deltas[..., :-2, :]

    cr = _slope_ratio(Rjm, Rj, Rjp, u_cfl)
    limiter = _superbee_fluxlimiter(cr)
    swall = (
        np.nan_to_num(ybuffer[..., 1:-2, :] + ybuffer[..., 2:-1, :]) * 0.5
        - np.sign(u_cfl) * ((1 - limiter) + u_cfl * limiter) * Rj * 0.5
    )
    return swall


def second_order_flux_limiter_z(s_center, u_cfl):
    """Get interpolated tracer concentration in Z.

    for more info, see
    https://mitgcm.readthedocs.io/en/latest/algorithm/adv-schemes.html#second-order-flux-limiters

    Parameters
    ----------
    s_center: numpy.ndarray
        the tracer field with buffer zone added (lm=2,rm=1), the last dimension being X
    u_cfl: numpy.ndarray
        the u velocity normalized by grid size in z, in s^-1.
    """
    zbuffer = buffer_z_nearest(s_center, 2, 1)
    deltas = np.nan_to_num(np.diff(zbuffer, axis=-3), 0)
    Rjp = deltas[..., 2:, :, :]
    Rj = deltas[..., 1:-1, :, :]
    Rjm = deltas[..., :-2, :, :]

    cr = _slope_ratio(Rjm, Rj, Rjp, u_cfl, not_z=-1)
    limiter = _superbee_fluxlimiter(cr)
    # swall = np.nan_to_num(zbuffer[...,1:-2,:,:]+zbuffer[...,2:-1,:,:])*0.5- np.sign(u_cfl)**Rj*0.5
    swall = (
        np.nan_to_num(zbuffer[..., 1:-2, :, :] + zbuffer[..., 2:-1, :, :]) * 0.5
        + np.sign(u_cfl) * ((1 - limiter) + u_cfl * limiter) * Rj * 0.5
    )
    return swall


def third_order_upwind_z(s, w):
    """Get interpolated tracer concentration in the vertical.

    for more info, see
    https://mitgcm.readthedocs.io/en/latest/algorithm/adv-schemes.html#third-order-upwind-bias-advection
    This function currently only work when there is no through surface flux.

    Parameters
    ----------
    s: numpy.ndarray
        the tracer field, the first dimension being Z
    w: numpy.ndarray
        the vertical velocity field, the first dimension being Zl
    """
    w[0] = 0
    deltas = np.nan_to_num(s[1:] - s[:-1], 0)
    Rj = np.zeros_like(s)
    Rj[1:] = deltas
    Rjp = np.roll(Rj, -1, axis=0)
    Rjm = np.roll(Rj, 1, axis=0)
    Rjjp = Rjp - Rj
    Rjjm = Rj - Rjm

    ssum = np.zeros_like(s)
    ssum[1:] = s[1:] + s[:-1]
    sz = 0.5 * (ssum - 1 / 6 * (Rjjp + Rjjm) - np.sign(w) * 1 / 6 * (Rjjp - Rjjm))
    return sz


def third_order_DST_x(xbuffer, u_cfl):
    """Get interpolated tracer concentration in X.

    for more info, see
    https://mitgcm.readthedocs.io/en/latest/algorithm/adv-schemes.html#third-order-direct-space-time

    Parameters
    ----------
    xbuffer: numpy.ndarray
        the tracer field with buffer zone added (lm=2,rm=1), the last dimension being X
    u_cfl: numpy.ndarray
        the u velocity normalized by grid size in x, in s^-1.
    """
    lm = 2
    deltas = np.nan_to_num(np.diff(xbuffer, axis=-1), 0)
    Rjp = deltas[..., lm:]
    Rj = deltas[..., lm - 1 : -1]
    Rjm = deltas[..., lm - 2 : -2]

    d0 = (2.0 - u_cfl) * (1.0 - u_cfl) / 6
    d1 = (1.0 - u_cfl * u_cfl) / 6

    sx = 0.5 * (
        (1 + np.sign(u_cfl)) * (xbuffer[..., lm - 1 : -2] + (d0 * Rj + d1 * Rjm))
        + (1 - np.sign(u_cfl)) * (xbuffer[..., lm:-1] - (d0 * Rj + d1 * Rjp))
    )
    return sx


def third_order_DST_y(ybuffer, u_cfl):
    """Get interpolated tracer concentration in Y.

    for more info, see
    https://mitgcm.readthedocs.io/en/latest/algorithm/adv-schemes.html#third-order-direct-space-time

    Parameters
    ----------
    ybuffer: numpy.ndarray
        the tracer field with buffer zone added (lm=2,rm=1), the second last dimension being Y
    u_cfl: numpy.ndarray
        the v velocity normalized by grid size in y, in s^-1.
    """
    lm = 2
    deltas = np.nan_to_num(np.diff(ybuffer, axis=-2), 0)
    Rjp = deltas[..., lm:, :]
    Rj = deltas[..., lm - 1 : -1, :]
    Rjm = deltas[..., lm - 2 : -2, :]

    d0 = (2.0 - u_cfl) * (1.0 - u_cfl) / 6
    d1 = (1.0 - u_cfl * u_cfl) / 6

    sy = 0.5 * (
        (1 + np.sign(u_cfl)) * (ybuffer[..., lm - 1 : -2, :] + (d0 * Rj + d1 * Rjm))
        + (1 - np.sign(u_cfl)) * (ybuffer[..., lm:-1, :] - (d0 * Rj + d1 * Rjp))
    )
    return sy
