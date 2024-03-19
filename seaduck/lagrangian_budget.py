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
    izl = np.array(particle_ds.iz)
    fc = np.array(particle_ds.fc)
    iy = np.array(particle_ds.iy)
    ix = np.array(particle_ds.ix)
    rzl = np.array(particle_ds.rz)
    ry = np.array(particle_ds.ry)
    rx = np.array(particle_ds.rx)

    # temp.it  = it .astype(int)
    temp.izl_lin = izl.astype(int)
    temp.iz = (izl - 1).astype(int)
    temp.face = fc.astype(int)
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


def first_last_neither(shapes, return_neither = True):
    acc = fast_cumsum(shapes)
    last = acc - 1
    first = np.roll(acc, 1)
    first[0] = 0
    if not return_neither:
        return first, last
    else:
        neither = np.array(
            [first[i] + j for i, length in enumerate(shapes) for j in range(1, length - 1)]
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
    # mask = np.logical_and(tres==0, temp.vs<7)
    # assert (tres[temp.vs<7]>0).all(), (tres0[mask],fracs[mask], fracs_a[mask], np.where(mask))

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
    face = copy.deepcopy(temp.face)
    # assert (iz>=1).all(),iz
    return iz, face, iy, ix


def wall_index(inds, iwall, tp):
    iw = iwall // 2
    iz, face, iy, ix = copy.deepcopy(inds)
    # assert (iz>=1).all(),iz

    ind = copy.deepcopy(np.array([face, iy, ix]))
    old_ind = copy.deepcopy(ind)
    naive_move = np.array([MOVE_DIC[i] for i in iwall], dtype=int).T
    iy += naive_move[0]
    ix += naive_move[1]
    ind = np.array([face, iy, ix])
    illegal = tp.check_illegal(ind, cuvwg="C")
    redo = np.array(np.where(illegal)).T
    for num, loc in enumerate(redo):
        j = loc[0]
        ind = (iw[j],) + tuple(old_ind[:, j])
        new_ind = ind_tend_uv(ind, tp)
        iw[j], face[j], iy[j], ix[j] = new_ind

    iz[iwall == 4] += 1
    iz -= 1
    return np.array([iw, iz, face, iy, ix]).astype(int)


def redo_index(pt):
    # assert (pt.izl_lin>=1).all()
    inds = deepcopy_inds(pt)
    iz, face, iy, ix = inds
    # assert (iz>=1).all(),iz
    tendf, tf, tendb, tb = pseudo_motion(pt)

    funderflow = np.where(tendf == 6)
    bunderflow = np.where(tendb == 6)
    tendf[funderflow] = 0
    tendb[bunderflow] = 0
    vf = wall_index(inds, tendf, pt.ocedata.tp)
    vb = wall_index(inds, tendb, pt.ocedata.tp)
    # vf[:,funderflow] = vb[:,funderflow]
    # vb[:,bunderflow] = vf[:,bunderflow]
    tim = tf - tb
    frac = -tb / tim
    assert (~np.isnan(tim)).any(), [
        i[np.isnan(tim)] for i in [pt.rx, pt.ry, pt.rzl_lin - 1 / 2]
    ]
    assert (tim != 0).all(), [i[tim == 0] for i in [pt.rx, pt.ry, pt.rzl_lin - 1 / 2]]
    at_corner = np.where(tim == 0)
    frac[at_corner] = 1
    return vf, vb, frac


def find_ind_frac_tres(neo, oce, region_names=False, region_polys=None):
    temp = read_from_ds(neo, oce)
    temp.shapes = list(temp.shapes)
    if region_names:
        masks = []
        for reg in region_polys:
            mask = parallelpointinpolygon(temp.lon, temp.lat, reg)
            # mask = np.where(mask)[0]
            masks.append(mask)
    first, last, neither = first_last_neither(np.array(temp.shapes))

    ind1 = np.zeros((5, temp.N), "int16")
    ind2 = np.ones((5, temp.N), "int16")
    frac = np.ones(temp.N)

    # ind1[:, wrong_ind] = lookup[:, lookup_ind]

    neithers = temp.subset(neither)
    neither_inds = deepcopy_inds(neithers)
    iwalls = which_wall(neithers)
    ind1[:, neither] = wall_index(neither_inds, iwalls, temp.ocedata.tp)

    firsts = temp.subset(first)
    lasts = temp.subset(last)
    ind1[:, first], ind2[:, first], frac[first] = redo_index(firsts)
    ind1[:, last], ind2[:, last], frac[last] = redo_index(lasts)

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
            fc=(["nprof"], fc),
            iy=(["nprof"], iy),
            iz=(["nprof"], iz),
            ix=(["nprof"], ix),
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
            vs=(["nprof"], vs),
        ),
    )
    return ds


def dump_to_zarr(neo, oce, filename, region_names=False, region_polys=None):
    if region_names:
        (ind1, ind2, frac, masks, tres, last, first) = find_ind_frac_tres(
            neo, oce, region_names=region_names, region_polys=region_polys
        )
    else:
        ind1, ind2, frac, tres, last, first = find_ind_frac_tres(neo, oce)

    neo["five"] = xr.DataArray(["iw", "iz", "face", "iy", "ix"], dims="five")
    if region_names:
        for ir, reg in enumerate(region_names):
            neo[reg] = xr.DataArray(masks[ir].astype(bool), dims="nprof")
        # neo['gulf'] = xr.DataArray(gulf_ind.astype(bool),dims = 'nprof')
        # neo['labr'] = xr.DataArray(labr_ind.astype(bool),dims = 'nprof')
        # neo['gdbk'] = xr.DataArray(gdbk_ind.astype(bool),dims = 'nprof')
        # neo['nace'] = xr.DataArray(nace_ind.astype(bool),dims = 'nprof')
        # neo['egrl'] = xr.DataArray(egrl_ind.astype(bool),dims = 'nprof')
        # region_ind = np.concatenate([gulf_ind,labr_ind,gdbk_ind,nace_ind,egrl_ind]).astype('int32')
        # neo.attrs['region_shape'] = [len(i) for i in [gulf_ind,labr_ind,gdbk_ind,nace_ind,egrl_ind]]
        # if len(region_ind)>0:
        #     neo = neo.assign_coords(region_ind = xr.DataArray(region_ind,dims = 'region_ind'))

    neo["ind1"] = xr.DataArray(ind1.astype("int16"), dims=["five", "nprof"])
    neo["ind2"] = xr.DataArray(ind2.astype("int16"), dims=["five", "nprof"])
    neo["frac"] = xr.DataArray(frac, dims="nprof")
    neo["tres"] = xr.DataArray(tres, dims="nprof")
    # neo['last'] = xr.DataArray(last.astype('int64'), dims = 'shapes')
    # neo['first'] = xr.DataArray(first.astype('int64'), dims = 'shapes')

    neo["face"] = neo["fc"].astype("int16")
    neo["ix"] = neo["ix"].astype("int16")
    neo["iy"] = neo["iy"].astype("int16")
    neo["iz"] = neo["iz"].astype("int16")
    neo["vs"] = neo["vs"].astype("int16")

    neo = neo.drop_vars(["rx", "ry", "rz", "uu", "vv", "ww", "du", "dv", "dw", "fc"])

    neo.to_zarr(filename, mode="w")
    zarr.consolidate_metadata(filename)


def store_lists(pt, name, region_names=False, region_polys=None):
    neo = particle2xarray(pt)
    dump_to_zarr(
        neo, pt.ocedata, name, region_names=region_names, region_polys=region_polys
    )

def prefetch_scalar(ds_slc,scalar_names):
    prefetch = {}
    for var in scalar_names:
        # print(var, end = ' ')
        prefetch[var] = np.array(ds_slc[var])
    return prefetch

def prefetch_vector(ds_slc,
                    xname = 'sxprime',
                    yname = 'syprime',
                    zname = 'szprime'
                   ):
    return np.array(tuple(np.array(ds_slc[i]) for i in [xname,yname,zname]))

def read_prefetched_scalar(ind,scalar_names,prefetch):
    res = {}
    for var in scalar_names:
        res[var] = prefetch[var][ind]
    return res

def lhs_contribution(t, scalar_dic, last, lhs_name = 'lhs'):
    deltat = np.nan_to_num(np.diff(t))
    deltat[last[:-1]] = 0
    lhs_scalar = scalar_dic[lhs_name][:-1]
    correction = deltat*lhs_sum
    return correction

def contr_p_relaxed(deltas, tres, step_dic, termlist, wrong_ind, p = 1):
    nds = len(deltas)
    # if len(wrong_ind)>0:
    #     if wrong_ind[-1] == len(deltas):
    #         wrong_ind = wrong_ind[:-1]
    dic = {'error': np.zeros(nds)}
    # dic['error'][wrong_ind] = deltas[wrong_ind]
    # deltas[wrong_ind] = 0
    # tres[wrong_ind] = 0
    
    deno = np.zeros(nds)
    sums = np.zeros(nds)
    for var in termlist:
        deno += step_dic[var][:-1]**(p+1)
        sums += step_dic[var][:-1]
    disparity = deltas-sums*tres
    total = np.zeros(nds)
    # dic['quality'] = np.nan_to_num(np.abs((disparity/tres)**(p+1)/deno))
    # mask = (dic['quality']<=1).astype(int)
    for var in termlist:
        ratio = step_dic[var][:-1]**(p+1)/deno
        dic[var] = step_dic[var][:-1]*tres+ratio*disparity
        total+=dic[var]
    final_correction = ds-total
    assert np.allclose(final_correction,0)
    dic['error'] += final_correction
    return dic