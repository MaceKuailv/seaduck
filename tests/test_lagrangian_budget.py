import numpy as np
import pytest

import seaduck as sd
from seaduck import utils
from seaduck.lagrangian_budget import (
    find_ind_frac_tres,
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


# @pytest.fixture
# def s_list(custom_pt, vec_dict):
#     prfh,is_scalar = vec_dict['sm']
#     return np.array(read_u_list(custom_pt, prefetch = prfh,scalar = is_scalar))


@pytest.mark.parametrize(
    "ind,exp",
    [
        ((0, 2, 45, 45), (0, 2, 45, 46)),
        ((1, 2, 45, 45), (1, 2, 46, 45)),
        ((1, 2, 89, 45), (0, 6, 44, 0)),
        ((0, 5, 45, 89), (1, 7, 0, 44)),
    ],
)
def test_ind_tend_uv(ind, exp):
    tub = sd.OceData(utils.get_dataset("ecco"))
    tp = tub.tp
    ans = ind_tend_uv(ind, tp)
    assert exp == ans


def test_redo_index(custom_pt):
    vf, vb, frac = redo_index(custom_pt)
    assert (frac <= 1).all()
    assert (frac >= 0).all()


def test_ind_frac_find(custom_pt):
    particle_datasets = particle2xarray(custom_pt)
    tub = sd.OceData(utils.get_dataset("ecco"))
    ind1, ind2, frac, tres, last, first = find_ind_frac_tres(particle_datasets, tub)
    assert ind1.shape[0] == 5
    assert (frac != 1).any()
    assert (tres >= 0).all()


@pytest.mark.parametrize(
    "use_region",
    [False],
)
def test_store_lists(custom_pt, use_region):
    store_lists(custom_pt, "PleaseIgnore_dump.zarr", use_region=use_region)
