import copy

import numpy as np
import xarray as xr

from seaduck.eulerian import Position
from seaduck.runtime_conf import compileable
from seaduck.utils import (
    _time2wall,
    _uleftright_from_udu,
    _which_early,
    parallelpointinpolygon,
)

try:  # pragma: no cover
    import zarr
except ImportError:  # pragma: no cover
    pass

UV_DIC = {"U": 0, "V": 1}
MOVE_DIC = {
    0: np.array([0, 0]),  # delta_y,delta_x
    1: np.array([0, 1]),
    2: np.array([0, 0]),
    3: np.array([1, 0]),
    4: np.array([0, 0]),
    5: np.array([0, 0]),
}


def read_from_ds(particle_ds, oce):
    temp = Position.__new__(Position)
    temp.ocedata = oce
    temp.tp = temp.ocedata.tp

    # it = np.array(particle_ds.it)
    if oce.tp.typ in ["LLC"]:
        temp.face = np.array(particle_ds.fc).astype(int)
    else:
        temp.face = None
    izl = np.array(particle_ds.iz)
    iy = np.array(particle_ds.iy)
    ix = np.array(particle_ds.ix)
    rzl = np.array(particle_ds.rz)
    ry = np.array(particle_ds.ry)
    rx = np.array(particle_ds.rx)

    # temp.it  = it .astype(int)
    temp.izl_lin = izl.astype(int)
    temp.iz = (izl - 1).astype(int)
    temp.iy = iy.astype(int)
    temp.ix = ix.astype(int)
    temp.rzl_lin = rzl
    temp.ry = ry
    temp.rx = rx

    temp.N = len(temp.ix)

    uu = np.array(particle_ds.uu)
    vv = np.array(particle_ds.vv)
    ww = np.array(particle_ds.ww)
    du = np.array(particle_ds.du)
    dv = np.array(particle_ds.dv)
    dw = np.array(particle_ds.dw)

    temp.u = uu
    temp.v = vv
    temp.w = ww
    temp.du = du
    temp.dv = dv
    temp.dw = dw

    temp.lon = np.array(particle_ds.xx)
    temp.lat = np.array(particle_ds.yy)
    temp.dep = np.array(particle_ds.zz)
    temp.t = np.array(particle_ds.tt)

    temp.vs = np.array(particle_ds.vs)

    temp.shapes = list(particle_ds.shapes)

    return temp


def which_wall(pt):
    distl = np.abs(0.5 + pt.rx)
    distr = np.abs(0.5 - pt.rx)
    distd = np.abs(0.5 + pt.ry)
    distt = np.abs(0.5 - pt.ry)
    distdeep = np.abs(pt.rzl_lin)
    distshal = np.abs(1 - pt.rzl_lin)

    distance = np.vstack([distl, distr, distd, distt, distdeep, distshal])
    return np.argmin(distance, axis=0)


def pseudo_motion(pt):
    us = [pt.u, pt.v, pt.w]
    dus = [pt.du, pt.dv, pt.dw]
    xs = [pt.rx, pt.ry, pt.rzl_lin - 1 / 2]
    ts_forward = _time2wall(xs, us, dus, 1e80 * np.ones(pt.N))
    ts_backward = _time2wall(xs, us, dus, -1e80 * np.ones(pt.N))
    tendf, tf = _which_early(1e80, ts_forward)
    tendb, tb = _which_early(-1e80, ts_backward)

    return tendf, tf, tendb, tb


@compileable
def fast_cumsum(shapes):
    return np.cumsum(shapes)


def first_last_neither(shapes, return_neither=True):
    acc = fast_cumsum(shapes)
    last = acc - 1
    first = np.roll(acc, 1)
    first[0] = 0
    if not return_neither:
        return first, last
    else:
        neither = np.array(
            [
                first[i] + j
                for i, length in enumerate(shapes)
                for j in range(1, length - 1)
            ]
        )
        return first, last, neither


def pt_ulist(pt):
    ul, ur = _uleftright_from_udu(pt.u, pt.du, pt.rx)
    vl, vr = _uleftright_from_udu(pt.v, pt.dv, pt.ry)
    wl, wr = _uleftright_from_udu(pt.w, pt.dw, pt.rzl_lin - 0.5)
    return np.array([ul, ur, vl, vr, wl, wr])


def residence_time(pt):
    u_list = pt_ulist(pt)
    return 2 / np.sum(np.abs(u_list), axis=0)


def tres_update(tres0, temp, first, last, fraction_first, fraction_last):
    fracs = np.ones(temp.N)
    # fracs_a = np.ones(temp.N)
    fracs[first + 1] = fraction_first
    # fracs_a[first+1] = fraction_first
    fracs[last] -= fraction_last
    tres = tres0 * fracs

    tres[temp.vs > 6] = 0.0

    return tres


def tres_fraction(temp, first, last, fraction_first, fraction_last):
    tres0 = residence_time(temp)
    tres = tres_update(tres0, temp, first, last, fraction_first, fraction_last)
    return tres


def ind_tend_uv(ind, tp):
    """Return the index of the velocity node.

    The node is not of the same index as the center point.
    iw determines where to return iw
    """
    iw, face, iy, ix = ind
    if iw == 1:
        new_wall, new_ind = tp._ind_tend_V((face, iy, ix), 0)
    elif iw == 0:
        new_wall, new_ind = tp._ind_tend_U((face, iy, ix), 3)
    else:
        raise ValueError("illegal iw")
    niw = UV_DIC[new_wall]
    nfc, niy, nix = new_ind
    return niw, nfc, niy, nix


def deepcopy_inds(temp):
    iz = copy.deepcopy(temp.izl_lin)
    iy = copy.deepcopy(temp.iy)
    ix = copy.deepcopy(temp.ix)
    if temp.face is not None:
        face = copy.deepcopy(temp.face)
        return iz, face, iy, ix
    else:
        return iz, iy, ix


def wall_index(inds, iwall, tp):
    iw = iwall // 2
    iz = copy.deepcopy(inds[0])
    iy = copy.deepcopy(inds[-2])
    ix = copy.deepcopy(inds[-1])

    ind = np.array(inds[1:])
    old_ind = copy.deepcopy(ind)
    naive_move = np.array([MOVE_DIC[i] for i in iwall], dtype=int).T
    ind[-2] += naive_move[0]  # iy
    iy += naive_move[0]
    ind[-1] += naive_move[1]  # ix
    ix += naive_move[1]

    iz[iwall == 4] += 1
    iz -= 1
    if tp.typ in ["LLC"]:
        face = copy.deepcopy(inds[-3])
        illegal = tp.check_illegal(ind, cuvwg="G")
        redo = np.array(np.where(illegal)).T
        for num, loc in enumerate(redo):
            j = loc[0]
            ind = (iw[j],) + tuple(old_ind[:, j])
            new_ind = ind_tend_uv(ind, tp)
            iw[j], face[j], iy[j], ix[j] = new_ind

        return np.array([iw, iz, face, iy, ix]).astype(int)
    else:
        return np.array([iw, iz, ind[-2], ind[-1]]).astype(int)


def redo_index(pt):
    inds = deepcopy_inds(pt)
    tendf, tf, tendb, tb = pseudo_motion(pt)

    funderflow = np.where(tendf == 6)
    bunderflow = np.where(tendb == 6)
    tendf[funderflow] = 0
    tendb[bunderflow] = 0
    vf = wall_index(inds, tendf, pt.ocedata.tp)
    vb = wall_index(inds, tendb, pt.ocedata.tp)
    tim = tf - tb
    frac = -tb / tim
    assert (~np.isnan(tim)).any(), [
        i[np.isnan(tim)] for i in [pt.rx, pt.ry, pt.rzl_lin - 1 / 2]
    ]
    # assert (tim != 0).all(), [i[tim == 0] for i in [pt.rx, pt.ry, pt.rzl_lin - 1 / 2]]
    # at_corner = np.where(tim == 0)
    # frac[at_corner] = 1
    frac = np.nan_to_num(frac, nan=1)
    return vf, vb, frac


def find_ind_frac_tres(neo, oce, region_names=False, region_polys=None, by_type=True):
    temp = read_from_ds(neo, oce)
    temp.shapes = list(temp.shapes)
    if region_names:  # pragma: no cover
        masks = []
        for reg in region_polys:
            mask = parallelpointinpolygon(temp.lon, temp.lat, reg)
            # mask = np.where(mask)[0]
            masks.append(mask)

    first, last, neither = first_last_neither(np.array(temp.shapes))
    if temp.face is not None:
        num_ind = 5
    else:
        num_ind = 4

    if by_type:
        ind1 = np.zeros((num_ind, temp.N), "int16")
        ind2 = np.ones((num_ind, temp.N), "int16")
        frac = np.ones(temp.N)

        if len(neither > 0):
            neithers = temp.subset(neither)
            neither_inds = deepcopy_inds(neithers)
            iwalls = which_wall(neithers)
            ind1[:, neither] = wall_index(neither_inds, iwalls, temp.ocedata.tp)

        firsts = temp.subset(first)
        lasts = temp.subset(last)
        ind1[:, first], ind2[:, first], frac[first] = redo_index(firsts)
        ind1[:, last], ind2[:, last], frac[last] = redo_index(lasts)
    else:
        ind1, ind2, frac = redo_index(temp)

    tres = tres_fraction(temp, first, last, frac[first], frac[last])
    if region_names:
        return ind1, ind2, frac, masks, tres, last, first
    else:
        return ind1, ind2, frac, tres, last, first


def flatten(lstoflst, shapes=None):
    if shapes is None:
        shapes = [len(i) for i in lstoflst]
    suffix = np.cumsum(shapes)
    thething = np.zeros(suffix[-1])
    thething[: suffix[0]] = lstoflst[0]
    for i in range(1, len(lstoflst)):
        thething[suffix[i - 1] : suffix[i]] = lstoflst[i]
    return thething


def particle2xarray(p):
    shapes = [len(i) for i in p.xxlist]
    # it = flatten(p.itlist,shapes = shapes)
    if p.face is not None:
        fc = flatten(p.fclist, shapes=shapes)
    iy = flatten(p.iylist, shapes=shapes)
    iz = flatten(p.izlist, shapes=shapes)
    ix = flatten(p.ixlist, shapes=shapes)
    rx = flatten(p.rxlist, shapes=shapes)
    ry = flatten(p.rylist, shapes=shapes)
    rz = flatten(p.rzlist, shapes=shapes)
    tt = flatten(p.ttlist, shapes=shapes)
    uu = flatten(p.uulist, shapes=shapes)
    vv = flatten(p.vvlist, shapes=shapes)
    ww = flatten(p.wwlist, shapes=shapes)
    du = flatten(p.dulist, shapes=shapes)
    dv = flatten(p.dvlist, shapes=shapes)
    dw = flatten(p.dwlist, shapes=shapes)
    xx = flatten(p.xxlist, shapes=shapes)
    yy = flatten(p.yylist, shapes=shapes)
    zz = flatten(p.zzlist, shapes=shapes)
    vs = flatten(p.vslist, shapes=shapes)

    ds = xr.Dataset(
        coords=dict(shapes=(["shapes"], shapes), nprof=(["nprof"], np.arange(len(xx)))),
        data_vars=dict(
            # it = (['nprof'],it),
            iy=(["nprof"], iy.astype(int)),
            iz=(["nprof"], iz.astype(int)),
            ix=(["nprof"], ix.astype(int)),
            rx=(["nprof"], rx),
            ry=(["nprof"], ry),
            rz=(["nprof"], rz),
            tt=(["nprof"], tt),
            uu=(["nprof"], uu),
            vv=(["nprof"], vv),
            ww=(["nprof"], ww),
            du=(["nprof"], du),
            dv=(["nprof"], dv),
            dw=(["nprof"], dw),
            xx=(["nprof"], xx),
            yy=(["nprof"], yy),
            zz=(["nprof"], zz),
            vs=(["nprof"], vs.astype(int)),
        ),
    )
    if p.face is not None:
        ds["fc"] = xr.DataArray(fc.astype(int), dims="nprof")
    return ds


def dump_to_zarr(
    neo, oce, filename, region_names=False, region_polys=None, preserve_checks=False
):
    if region_names:  # pragma: no cover
        (ind1, ind2, frac, masks, tres, last, first) = find_ind_frac_tres(
            neo, oce, region_names=region_names, region_polys=region_polys
        )
    else:
        ind1, ind2, frac, tres, last, first = find_ind_frac_tres(neo, oce)

    if oce.tp.typ in ["LLC"]:
        neo["face"] = neo["fc"].astype("int16")
        neo["five"] = xr.DataArray(["iw", "iz", "face", "iy", "ix"], dims="five")
    else:
        neo["five"] = xr.DataArray(["iw", "iz", "iy", "ix"], dims="five")
    if region_names:  # pragma: no cover
        for ir, reg in enumerate(region_names):
            neo[reg] = xr.DataArray(masks[ir].astype(bool), dims="nprof")

    neo["ind1"] = xr.DataArray(ind1.astype("int16"), dims=["five", "nprof"])
    neo["ind2"] = xr.DataArray(ind2.astype("int16"), dims=["five", "nprof"])
    neo["frac"] = xr.DataArray(frac, dims="nprof")
    neo["tres"] = xr.DataArray(tres, dims="nprof")

    neo["ix"] = neo["ix"].astype("int16")
    neo["iy"] = neo["iy"].astype("int16")
    neo["iz"] = neo["iz"].astype("int16")
    neo["vs"] = neo["vs"].astype("int16")

    if not preserve_checks:
        neo = neo.drop_vars(["rx", "ry", "rz", "uu", "vv", "ww", "du", "dv", "dw"])
    if "fc" in neo.data_vars:
        neo = neo.drop_vars(["fc"])

    neo.to_zarr(filename, mode="w")
    zarr.consolidate_metadata(filename)


def store_lists(pt, name, region_names=False, region_polys=None, **kwarg):
    neo = particle2xarray(pt)
    dump_to_zarr(
        neo,
        pt.ocedata,
        name,
        region_names=region_names,
        region_polys=region_polys,
        **kwarg
    )


def prefetch_scalar(ds_slc, scalar_names):
    prefetch = {}
    for var in scalar_names:
        # print(var, end = ' ')
        prefetch[var] = np.array(ds_slc[var])
    return prefetch


def read_wall_list(neo, tp, prefetch=None, scalar=True):
    if "face" not in neo.data_vars and "fc" not in neo.data_vars:  # pragma: no cover
        ind = (neo.iz - 1, neo.iy, neo.ix)
        deep_ind = (neo.iz, neo.iy, neo.ix)
        right_ind = tuple(
            [neo.iz - 1]
            + list(
                tp.ind_tend_vec(
                    (neo.iy, neo.ix), np.ones(len(neo.nprof)) * 3, cuvwg="G"
                )
            )
        )
        up_ind = tuple(
            [neo.iz - 1]
            + list(tp.ind_tend_vec((neo.iy, neo.ix), np.ones(len(neo.nprof)) * 0))
        )
        uarray, varray, warray = prefetch
        ur = uarray[right_ind]
        vr = varray[up_ind]
    else:
        ind = (neo.iz - 1, neo.face, neo.iy, neo.ix)
        deep_ind = (neo.iz, neo.face, neo.iy, neo.ix)
        right_ind = tuple(
            [neo.iz - 1]
            + list(
                tp.ind_tend_vec((neo.face, neo.iy, neo.ix), np.ones(len(neo.nprof)) * 3)
            )
        )
        up_ind = tuple(
            [neo.iz - 1]
            + list(
                tp.ind_tend_vec((neo.face, neo.iy, neo.ix), np.ones(len(neo.nprof)) * 0)
            )
        )
        uarray, varray, warray = prefetch

        ur_temp = np.nan_to_num(uarray[right_ind])
        vr_temp = np.nan_to_num(varray[right_ind])
        uu_temp = np.nan_to_num(uarray[up_ind])
        vu_temp = np.nan_to_num(varray[up_ind])
        right_faces = np.vstack([ind[1], right_ind[1]]).T
        up_faces = np.vstack([ind[1], up_ind[1]]).T
        ufromu, ufromv, _, _ = tp.four_matrix_for_uv(right_faces)
        _, _, vfromu, vfromv = tp.four_matrix_for_uv(up_faces)
        if scalar:
            ufromu, ufromv, vfromu, vfromv = (
                np.abs(i) for i in [ufromu, ufromv, vfromu, vfromv]
            )

        ur = ur_temp * ufromu[:, 1] + vr_temp * ufromv[:, 1]
        vr = uu_temp * vfromu[:, 1] + vu_temp * vfromv[:, 1]
    ul = uarray[ind]
    vl = varray[ind]
    wr = warray[ind]
    wl = warray[deep_ind]
    return np.array([np.nan_to_num(i) for i in (ul, ur, vl, vr, wl, wr)])


def crude_convergence(u_list):
    conv = np.array(u_list).T * np.array([1, -1, 1, -1, 1, -1])
    conv = np.sum(conv, axis=-1)
    return conv


def check_particle_data_compat(
    xrpt,
    xrslc,
    tp,
    use_tracer_name=None,
    wall_names=("sx", "sy", "sz"),
    conv_name="divus",
    debug=False,
    allclose_kwarg={},
):
    """Check if you could use Lagrangian budget functionality.

    Parameters
    ----------
    xrpt: xr.Dataset
        A dataset generated by seaduck that contains the location and velocity info.
    xrslc: xr.Dataset
        The ocean model (including budget terms) at a time step.
    tp: sd.Topology
        The topology object for the model
    use_tracer_name: string
        if specified, use cx,cy,cz as wall name
    wall_names: tuple of string
        Name of variables if use_tracer_name is not defined.
    conv_name: string
        The variable from xrslc to be compared against.
    debug: bool
        Whether to return additional debug information.
    allclose_kwarg:
        Keyword arguments for np.allclose.

    Returns
    -------
    OK_or_not: bool
        Is it OK to preceed?
    extra: None or tuple
        Extra information to help with debugging.
    """
    if "iz" not in xrpt.data_vars:  # pragma: no cover
        raise NotImplementedError(
            "This functionality only support 3D simulation at the moment."
        )
    if isinstance(use_tracer_name, str):
        wall_names = tuple(use_tracer_name + i for i in ["x", "y", "z"])
        conv_name = "divu" + use_tracer_name
    elif use_tracer_name is not None:
        raise ValueError("use_tracer_name has to be a string.")
    prefetch = []
    for var in wall_names:
        prefetch.append(np.array(xrslc[var]))
    c_list = read_wall_list(xrpt, tp, prefetch)

    ul, ur = _uleftright_from_udu(
        np.array(xrpt.uu), np.array(xrpt.du), np.array(xrpt.rx)
    )
    vl, vr = _uleftright_from_udu(
        np.array(xrpt.vv), np.array(xrpt.dv), np.array(xrpt.ry)
    )
    wl, wr = _uleftright_from_udu(
        np.array(xrpt.ww), np.array(xrpt.dw), np.array(xrpt.rz) - 0.5
    )
    u_list = np.array([np.array(i) for i in [ul, ur, vl, vr, wl, wr]])

    flux_list = c_list * u_list
    lagrangian_conv = crude_convergence(flux_list)

    if "face" in xrpt.data_vars:
        ind = (xrpt.iz - 1, xrpt.face, xrpt.iy, xrpt.ix)
    else:
        ind = (xrpt.iz - 1, xrpt.iy, xrpt.ix)
    eulerian_conv = np.array(xrslc[conv_name])[ind]
    if debug:
        extra = (u_list, c_list, lagrangian_conv, eulerian_conv)
    else:
        extra = None
    return np.allclose(lagrangian_conv, eulerian_conv, **allclose_kwarg), extra


def prefetch_vector(
    ds_slc, xname="sxprime", yname="syprime", zname="szprime", same_size=True
):
    if same_size:
        return np.array(tuple(np.array(ds_slc[i]) for i in [xname, yname, zname]))
    else:
        xx = np.array(ds_slc[xname])
        yy = np.array(ds_slc[yname])
        zz = np.array(ds_slc[zname])
        shape = (3,) + tuple(
            int(np.max([ar.shape[j] for ar in [xx, yy, zz]]))
            for j in range(len(xx.shape))
        )
        larger = np.empty(shape)
        larger[(0,) + tuple(slice(i) for i in xx.shape)] = xx
        larger[(1,) + tuple(slice(i) for i in yy.shape)] = yy
        larger[(2,) + tuple(slice(i) for i in zz.shape)] = zz
        return larger


def read_prefetched_scalar(ind, scalar_names, prefetch):
    res = {}
    for var in scalar_names:
        res[var] = prefetch[var][ind]
    return res


def lhs_contribution(t, scalar_dic, last, lhs_name="lhs"):
    deltat = np.nan_to_num(np.diff(t))
    deltat[last[:-1]] = 0
    lhs_scalar = scalar_dic[lhs_name][:-1]
    correction = deltat * lhs_scalar
    return correction


def contr_p_relaxed(deltas, tres, step_dic, termlist, p=1, error_prefix=""):
    nds = len(deltas)
    dic = {error_prefix + "error": np.zeros(nds)}

    deno = np.zeros(nds)
    sums = np.zeros(nds)
    for var in termlist:
        deno += step_dic[var][:-1] ** (p + 1)
        sums += step_dic[var][:-1]
    disparity = deltas - sums * tres
    total = np.zeros(nds)
    for var in termlist:
        ratio = step_dic[var][:-1] ** (p + 1) / deno
        dic[var] = step_dic[var][:-1] * tres + ratio * disparity
        total += dic[var]
    final_correction = deltas - total
    dic[error_prefix + "error"] += final_correction
    return dic


def calculate_budget(
    particle_array, data_slice, rhs_list, prefetch_vector_kwarg=dict(), lhs_name="lhs"
):  # pragma: no cover
    """Calculate Lagrangian budget.

    Parameters
    ----------
    particle_array: xr.Dataset
        A dataset generated by seaduck that contains the location and velocity info.
    data_slice: xr.Dataset
        The ocean model (including budget terms) at a time step.
    rhs_list: list
        List of strings for the variable names on the RHS
    prefetch_vector_kwarg: dict
        Keyword arguments for reading wall concentration
    lhs_name: String
        The name of variable that stands for eulerian tendency

    Returns
    -------
    contr_dic: dict of np.ndarray
        The contribution of each RHS term, has same length as number of entries.
    trc_conc: np.ndarray
        Tracer concentration at each point.
    first, last: np.ndarray
        Which indices mark the start and end of the time step for each particle.
    """
    termlist = rhs_list + [lhs_name]
    prefetch_term = prefetch_scalar(data_slice, termlist)
    walls_array = prefetch_vector(data_slice, **prefetch_vector_kwarg)
    ind1 = tuple(particle_array["ind1"].values)
    ind2 = tuple(particle_array["ind2"].values)

    frac = np.array(particle_array.frac)
    shapes = np.array(particle_array.shapes)
    tres = np.array(particle_array.tres)
    tact = np.array(particle_array.tt)

    ind = tuple(
        np.array(i)
        for i in [
            particle_array.iz - 1,
            particle_array.face,
            particle_array.iy,
            particle_array.ix,
        ]
    )

    step_dic = read_prefetched_scalar(ind, termlist, prefetch_term)
    first, last = first_last_neither(shapes, return_neither=False)
    s1 = walls_array[ind1]
    s2 = walls_array[ind2]

    trc_conc = s1 * frac + (1 - frac) * s2
    deltas = np.nan_to_num(np.diff(trc_conc))
    deltas[last[:-1]] = 0
    tres_used = -tres[1:]
    tres_used[last[:-1]] = 0

    correction = lhs_contribution(tact, step_dic, last, lhs_name=lhs_name)
    rhs_contr = deltas - correction

    contr_dic = contr_p_relaxed(rhs_contr, tres_used, step_dic, rhs_list)
    contr_dic[lhs_name] = correction
    return contr_dic, trc_conc, first, last
