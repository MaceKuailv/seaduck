import copy
import logging
import warnings

import numpy as np
import xarray as xr

from seaduck.ocedata import RelCoord
from seaduck.smart_read import smart_read


def mask_u_node(maskC, tp):
    """Mask out U-points.

    for MITgcm indexing, U is defined on the left of the cell,
    When the C grid is dry, U are either:
    a. dry;
    b. on the interface, where the cell to the left is wet.
    if b is the case, we need to unmask the udata, because it makes some physical sense.

    Parameters
    ----------
    maskC: numpy.ndarray
        numpy array with the same shape as the model spacial coordinates.
        1 for wet cells (center points), 0 for dry ones.
    tp: Topology object
        The Topology object for the dataset.

    Returns
    -------
    maskU: numpy.ndarray
        numpy array with the same shape as the model spacial coordinates.
        1 for wet U-walls (including interface between wet and dry), 0 for dry ones.
    """
    maskU = copy.deepcopy(maskC)
    indexes = np.array(np.where(maskC == 0)).T
    # find out which points are masked will make the search faster.
    new_ind = tp.ind_tend_vec(indexes.T[1:], np.ones_like(indexes.T[0], int) * 2)
    new_ind = np.vstack([indexes.T[0], new_ind])
    switch = indexes[np.where(maskC[tuple(new_ind)])]
    maskU[tuple(switch.T)] = 1

    return maskU


def mask_v_node(maskC, tp):
    """Mask out v-points.

    for MITgcm indexing, V is defined on the "south" side of the cell,
    When the C grid is dry, V are either:
    a. dry;
    b. on the interface, where the cell to the downside is wet.
    if b is the case, we need to unmask the vdata.

    Parameters
    ----------
    maskC: numpy.ndarray
        numpy array with the same shape as the model spacial coordinates.
        1 for wet cells (center points), 0 for dry ones.
    tp: Topology object
        The Topology object for the dataset.

    Returns
    -------
    + maskV: numpy.ndarray
        numpy array with the same shape as the model spacial coordinates.
        1 for wet W-walls (including interface between wet and dry), 0 for dry ones.
    """
    maskV = copy.deepcopy(maskC)
    indexes = np.array(np.where(maskC == 0)).T
    # find out which points are masked will make the search faster.
    new_ind = tp.ind_tend_vec(indexes.T[1:], np.ones_like(indexes.T[0], int) * 1)
    new_ind = np.vstack([indexes.T[0], new_ind])
    switch = indexes[np.where(maskC[tuple(new_ind)])]
    maskV[tuple(switch.T)] = 1
    return maskV


def mask_w_node(maskC, tp=None):
    # this one does not need tp object
    # if you pass something into it by mistake, it will be ignored.
    """Mask out W-points.

    for MITgcm indexing, W is defined on the top of the cell,
    When the C grid is dry, W are either:
    a. dry;
    b. on the interface, where the cell above is wet.
    if b is the case, we need to unmask the wdata.

    Parameters
    ----------
    maskC: numpy.ndarray
        numpy array with the same shape as the model spacial coordinates.
        1 for wet cells (center points), 0 for dry ones.
    tp: Topology object
        The Topology object for the dataset.

    Returns
    -------
    + maskWvel: numpy.ndarray
        numpy array with the same shape as the model spacial coordinates.
        1 for wet W-walls (including interface between wet and dry), 0 for dry ones.
    """
    maskW = np.zeros_like(maskC)
    maskW[1:] = maskC[:-1]
    maskW = np.logical_or(maskW, maskC).astype(int)
    return maskW


def get_mask_arrays(od):
    """Mask all staggered valocity points.

    A wrapper around mask_u_node, mask_v_node, mask_w_node.
    If there is no maskC in the dataset, just return nothing is masked.

    Parameters
    ----------
    od: OceData object
        The dataset to compute masks on.
    tp: Topology object
        The Topology of the datset

    Returns
    -------
    maskC,maskU,maskV,maskW: numpy.ndarray
        masks at center points, U-walls, V-walls, W-walls respectively.
    """
    tp = od.tp
    keys = od._ds.keys()
    if "maskC" not in keys:
        warnings.warn("no maskC in the dataset, assuming nothing is masked.")
        logging.warning("no maskC in the dataset, assuming nothing is masked.")
        maskC = np.ones_like(od._ds.XC + od._ds.Z)
        # it is inappropriate to fill in the dataset,
        # expecially given that there is no performance boost.
        return maskC, maskC, maskC, maskC
    maskC = np.array(od._ds["maskC"])
    if "Z" not in od._ds["maskC"].dims:
        raise NotImplementedError(
            "2D land mask is not yet supported,"
            "you could potentially get around by adding a psuedo dimension"
            "or you could set knw.ignore_mask = True"
        )
    if "maskU" not in keys:
        logging.info("creating maskU,this is going to be very slow!")
        maskU = mask_u_node(maskC, tp)
        od._ds["maskU"] = od._ds["Z"] + od._ds["XG"]
        od._ds["maskU"].values = maskU
    else:
        maskU = np.array(od._ds["maskU"])
    if "maskV" not in keys:
        logging.info("creating maskV,this is going to be very slow!")
        maskV = mask_v_node(maskC, tp)
        od._ds["maskV"] = od._ds["Z"] + od._ds["YG"]
        od._ds["maskV"].values = maskV
    else:
        maskV = np.array(od._ds["maskV"])
    if "maskWvel" not in keys:
        # there is a maskW with W meaning West in ECCO
        logging.info("creating maskW,this is going to be somewhat slow")
        maskW = mask_w_node(maskC)
        od._ds["maskWvel"] = od._ds["Z"] + od._ds["YC"]
        od._ds["maskWvel"].values = maskW
    else:
        maskW = np.array(od._ds["maskWvel"])
    return maskC, maskU, maskV, maskW


def get_masked(od, ind, cuvwg="C"):
    """Return whether points are masked.

    Return whether the indexes of intersts are masked or not.

    Parameters
    ----------
    od: OceData object
        Dataset to find mask values from.
    ind: tuple of numpy.ndarray
        Indexes of grid points.
    cuvwg: str
        Whether the indexes is for points at center points or on the walls.
        Options are: ['C','U','V','Wvel'].
    """
    if cuvwg not in ["C", "U", "V", "Wvel"]:
        raise NotImplementedError(
            "cuvwg(the kind of grid point) for mask not supported"
        )
    keys = od._ds.keys()
    if "maskC" not in keys:
        warnings.warn("no maskC in the dataset, assuming nothing is masked.")
        return np.ones_like(ind[0])
    elif cuvwg == "C":
        return smart_read(od._ds.maskC, ind)

    name = "mask" + cuvwg
    tp = od.tp
    maskC = np.array(od._ds["maskC"])
    func_dic = {"U": mask_u_node, "V": mask_v_node, "Wvel": mask_w_node}
    rename_dic = {
        "U": lambda x: x if x != "X" else "Xp1",
        "V": lambda x: x if x != "Y" else "Xp1",
        "Wvel": lambda x: x if x != "Z" else "Zl",
    }
    if name not in keys:
        small_mask = func_dic[cuvwg](maskC, tp)
        dims = tuple(map(rename_dic[cuvwg], od._ds.maskC.dims))
        sizes = tuple(len(od._ds[dim]) for dim in dims)
        mask = np.zeros(sizes)
        # indexing sensitive
        old_size = small_mask.shape
        slices = tuple(slice(0, i) for i in old_size)
        mask[slices] = small_mask
        od._ds[name] = xr.DataArray(mask, dims=dims)
        return mask[ind]
    else:
        return smart_read(od._ds[name], ind)


def which_not_stuck(p):
    """Investigate which points are in land mask."""
    ind = []
    if p.izl_lin is not None:
        ind.append(p.izl_lin - 1)
    if p.face is not None:
        ind.append(p.face)
    ind += [p.iy, p.ix]
    ind = tuple(ind)
    return get_masked(p.ocedata, ind).astype(bool)


def abandon_stuck(p):
    """Abandon those stucked in mud."""
    which = which_not_stuck(p)
    vardict = vars(p)
    keys = vardict.keys()
    for i in keys:
        item = vardict[i]
        if isinstance(item, np.ndarray):
            if len(item.shape) == 1:
                setattr(p, i, item[which])
                p.N = len(getattr(p, i))
            else:
                setattr(p, i, item)
        elif isinstance(item, RelCoord):
            setattr(p, i, item.subset(which))
        else:
            setattr(p, i, item)
    return p
