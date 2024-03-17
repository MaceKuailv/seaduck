import numpy as np
import pytest

import seaduck as sd
from seaduck import utils
from seaduck.lagrangian_budget import (
    find_ind_frac_tres,
    flatten,
    ind_tend_uv,
    particle2xarray,
    redo_index,
    store_lists,
)


@pytest.fixture
def custom_pt():
    x = np.linspace(-50, -15, 200)
    y = np.ones_like(x) * 52.0
    z = np.ones_like(x) * (-9)
    t = np.ones_like(x)
    od = sd.OceData(utils.get_dataset("ecco"))
    pt = sd.Particle(
        x=x,
        y=y,
        z=z,
        t=t,
        data=od,
        uname="utrans",
        vname="vtrans",
        wname="wtrans",
        transport=True,
        save_raw=True,
    )
    pt.to_next_stop(t[0] + 1e7)
    return pt


@pytest.fixture
def region_info():
    GULF = np.array(
        [
            [-92.5, 25],
            [-75, 25],
            [-55, 40],
            [-72.5, 40],
        ]
    )
    LABR = np.array([[-72, 63.5], [-60, 50], [-49, 53], [-61, 66.5]])
    GDBK = np.array(
        [
            # GULF[2],
            GULF[3],
            LABR[1],
            LABR[2],
            [-42, 50],
            [-42, 40],
        ]
    )
    NACE = np.array([GDBK[-2], GDBK[-1], [-21, 47], [-21, 57]])
    EGRL = np.array([[-22.5, 72], [-44, 63], [-44, 57], [-22.5, 66]])

    return ["gulf", "labr", "gdbk", "nace", "egrl"], [GULF, LABR, GDBK, NACE, EGRL]


@pytest.mark.parametrize(
    "ind,exp",
    [
        ((0, 2, 45, 45), (0, 2, 45, 46)),
        ((1, 2, 45, 45), (1, 2, 46, 45)),
        ((1, 2, 89, 45), (0, 6, 44, 0)),
        ((0, 5, 45, 89), (1, 7, 0, 44)),
    ],
)
@pytest.mark.parametrize("od", ["ecco"], indirect=True)
def test_ind_tend_uv(ind, exp, od):
    tub = od
    tp = tub.tp
    ans = ind_tend_uv(ind, tp)
    assert exp == ans


def test_ind_tend_uv_error():
    ind = (4, 3, 2, 1)
    with pytest.raises(ValueError):
        ind_tend_uv(ind, None)


def test_redo_index(custom_pt):
    vf, vb, frac = redo_index(custom_pt)
    assert (frac <= 1).all()
    assert (frac >= 0).all()


def test_flatten_list():
    lst = [[0, 1], [1, 1, 1, 1]]
    new = flatten(lst)
    assert (new == np.array([0, 1, 1, 1, 1, 1])).all()
    assert isinstance(new, np.ndarray)


@pytest.mark.parametrize("od", ["ecco"], indirect=True)
def test_ind_frac_find(custom_pt, od):
    particle_datasets = particle2xarray(custom_pt)
    tub = od
    ind1, ind2, frac, tres, last, first = find_ind_frac_tres(particle_datasets, tub)
    assert ind1.shape[0] == 5
    assert (frac != 1).any()
    assert (tres >= 0).all()


def test_store_lists(custom_pt):
    store_lists(custom_pt, "PleaseIgnore_dump.zarr")


def test_store_lists_with_region(custom_pt, region_info):
    region_names, region_polys = region_info
    store_lists(
        custom_pt,
        "PleaseIgnore_dump.zarr",
        region_names=region_names,
        region_polys=region_polys,
    )
