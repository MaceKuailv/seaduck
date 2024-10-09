import numpy as np
import pytest
import xarray as xr

import seaduck as sd
from seaduck import utils
from seaduck.eulerian_budget import total_div
from seaduck.lagrangian_budget import (
    check_particle_data_compat,
    contr_p_relaxed,
    find_ind_frac_tres,
    first_last_neither,
    flatten,
    ind_tend_uv,
    lhs_contribution,
    particle2xarray,
    prefetch_scalar,
    prefetch_vector,
    read_prefetched_scalar,
    redo_index,
    store_lists,
)


@pytest.fixture
def custom_pt():
    x = np.linspace(-50, -15, 5)
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
def curv_pt():
    od = sd.OceData(utils.get_dataset("curv"))
    curv_p = sd.Particle(
        y=np.array([70.5]),
        x=np.array([-14.0]),
        z=np.array([-10.0]),
        t=np.array([od.ts[0]]),
        data=od,
        uname="U",
        vname="V",
        wname="W",
        save_raw=True,
    )
    curv_p.to_next_stop(od.ts[0] + 1e4)
    return curv_p


@pytest.fixture
def xrslc(grid):
    ds = utils.get_dataset("ecco")
    ds["sx"] = xr.ones_like(ds["utrans"])
    ds["sy"] = xr.ones_like(ds["vtrans"])
    ds["sz"] = xr.ones_like(ds["wtrans"])
    tub = sd.OceData(ds)
    tub["advx"] = (tub["utrans"]).compute()
    tub["advy"] = (tub["vtrans"]).compute()
    tub["advz"] = (tub["wtrans"]).compute()
    tub["advz"][:, 0] = 0
    tub["divus"] = total_div(tub, grid, "advx", "advy", "advz")
    return tub._ds.isel(time=0, Zl=slice(50))


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


@pytest.mark.parametrize("od", ["curv"], indirect=True)
def test_ind_frac_find_noface(curv_pt, od):
    particle_datasets = particle2xarray(curv_pt)
    tub = od
    ind1, ind2, frac, tres, last, first = find_ind_frac_tres(particle_datasets, tub)
    assert ind1.shape[0] == 4
    assert (frac != 1).any()
    assert (tres >= 0).all()


def test_store_lists(custom_pt):
    store_lists(custom_pt, "PleaseIgnore_dump.zarr")


def test_store_lists_with_region(custom_pt, region_info):
    pytest.importorskip("numba")
    region_names, region_polys = region_info
    store_lists(
        custom_pt,
        "PleaseIgnore_dump.zarr",
        region_names=region_names,
        region_polys=region_polys,
    )


def test_first_last_neither():
    shapes = np.random.randint(2, 5, 10)
    first, last, neither = first_last_neither(shapes)
    first1, last1 = first_last_neither(shapes, return_neither=False)
    assert np.allclose(first, first1)
    merged = np.sort(np.concatenate([first, last, neither]))
    sums = np.sum(shapes)
    assert np.allclose(np.arange(sums), merged)


@pytest.mark.parametrize(
    "use_tracer_name,wall_names,conv_name",
    [
        (None, ("sx", "sy", "sz"), "divus"),
        ("s", None, None),
    ],
)
def test_check_particle_data_compat(
    custom_pt, use_tracer_name, wall_names, conv_name, xrslc
):
    xrpt = particle2xarray(custom_pt)
    xrpt["face"] = xrpt["fc"]
    tp = sd.Topology(utils.get_dataset("ecco"))
    assert check_particle_data_compat(
        xrpt,
        xrslc,
        tp,
        use_tracer_name=use_tracer_name,
        wall_names=("sx", "sy", "sz"),
        conv_name="divus",
        debug=False,
        allclose_kwarg={"atol": 1e-11},
    )


def test_prefetch_scalar_and_read(xrslc):
    scalar_name = ["divus", "SALT"]
    prefetch = prefetch_scalar(xrslc, scalar_name)
    res = read_prefetched_scalar((49, 12, 89, 89), scalar_name, prefetch)
    assert isinstance(res, dict)


@pytest.mark.parametrize("same_size", [True, False])
def test_prefetch_vector(xrslc, same_size):
    larger = prefetch_vector(
        xrslc, xname="sx", yname="sy", zname="sz", same_size=same_size
    )
    assert isinstance(larger, np.ndarray)


def test_lhs_contribution():
    lhs = np.random.random(10)
    scalar_dic = {"lhs": lhs}
    last = np.array([8, 10])
    t = np.arange(10)
    ans = lhs_contribution(t, scalar_dic, last)
    assert np.allclose(ans[:3], lhs[:3])
    assert ans[8] == 0


def test_p_relax():
    deltas = np.ones(3) * 3
    tres = np.ones(3)
    step_dic = {"term1": np.array([1, 2, 3, 4]), "term2": np.array([3, 2, 1, 0])}
    termlist = ["term1", "term2"]
    res = contr_p_relaxed(deltas, tres, step_dic, termlist)
    assert np.allclose(0, res["error"])
    assert res["term1"][1] == res["term2"][1]
