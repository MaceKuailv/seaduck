import copy

import numpy as np
import pytest
from scipy.interpolate import interp1d

import seaduck as sd
from seaduck import utils

uknw = sd.lagrangian.uknw
vknw = sd.lagrangian.vknw
wknw = sd.lagrangian.wknw

uuknw = copy.deepcopy(uknw)
vvknw = copy.deepcopy(vknw)
wwknw = copy.deepcopy(wknw)

wrongZknw = sd.KnW()
wrongZknw.vkernel = "something not correct"
wrongTknw = sd.KnW()
wrongTknw.tkernel = "something not correct"


@pytest.fixture
def prefetched():
    od = sd.OceData(utils.get_dataset("ecco"))
    return np.array(od["SALT"][slice(1, 2)])


# use float number to make Particle
@pytest.fixture
def ep():
    od = sd.OceData(utils.get_dataset("ecco"))
    return sd.Particle(x=-37.5, y=10.4586420059204, z=-9.0, t=698155200.0, data=od)


@pytest.mark.parametrize("od", ["aviso"], indirect=True)
def test_tz_not_None(od):
    ap = sd.Position()
    ap.from_latlon(x=-37.5, y=-60.4586420059204, z=-9.0, t=698155200.0, data=od)
    assert isinstance(ap.rel, sd.ocedata.RelCoord)


@pytest.mark.parametrize("y,t", [(10.4586420059204, None), (None, 698155200.0)])
@pytest.mark.parametrize("od", ["ecco"], indirect=True)
def test_xyt_is_None(od, y, t):
    ap = sd.Position()
    ap.from_latlon(x=-37.5, y=y, z=-9.0, t=t, data=od)
    if y is None:
        assert ap.rx is None
    if t is None:
        assert ap.rt is None


@pytest.mark.parametrize(
    "knw,required",
    [
        (uknw, "all"),
        (uknw, "Z"),
        (uknw, ["Z"]),
    ],
)
def test_fatten(ep, knw, required):
    fattened_inds = ep.fatten(knw, required=required)
    assert isinstance(fattened_inds[0], np.ndarray)
    assert len(fattened_inds[0].shape) > 1


@pytest.mark.parametrize("varname,knw", [("WVELMASS", uknw), ("UVELMASS", wknw)])
def test_interp_vertical(ep, varname, knw):
    res = ep.interpolate(varname, knw)
    assert isinstance(res, np.ndarray)
    assert len(res.shape) == 1


def test_interp_with_NoneZ(ep):
    ep.rz = None
    ep.rzl = None
    uuknw.ignore_mask = False
    wwknw.ignore_mask = False
    vvknw.ignore_mask = False

    res = ep.interpolate(["UVELMASS", "WVELMASS", "VVELMASS"], [uuknw, wwknw, vvknw])

    uuknw.ignore_mask = True
    wwknw.ignore_mask = True
    vvknw.ignore_mask = True

    assert isinstance(res, list)


def test_no_face_with_mask(ep):
    vvknw.ignore_mask = False
    uuknw.ignore_mask = False
    (
        output_format,
        main_keys,
        prefetch_dict,
        main_dict,
        hash_index,
        hash_mask,
        hash_read,
        hash_weight,
    ) = ep._register_interpolation_input(("UVELMASS", "VVELMASS"), (uuknw, vvknw))
    index_lookup = ep._fatten_required_index_and_register(hash_index, main_dict)
    transform_lookup = ep._transform_vector_and_register(
        index_lookup, hash_index, main_dict
    )
    neo_transform_lookup = {}
    for key in transform_lookup.keys():
        neo_transform_lookup[key] = None
    mask_dict = ep._mask_value_and_register(
        index_lookup, neo_transform_lookup, hash_mask, hash_index, main_dict
    )
    assert bool(mask_dict)


def test_dict_input(ep):
    registered = ep._register_interpolation_input(
        "SALT",
        {"SALT": uknw},
        prefetched={"SALT": prefetched},
        prefetch_prefix={"SALT": (1, 0, 0, 0, 0)},
    )
    assert len(registered) == 8


@pytest.mark.parametrize(
    "varname,knw,num_prefetched",
    [
        ("SALT", uknw, "one"),
        (("SALT", "SALT"), (uknw, uknw), "two"),
    ],
)
def test_diff_prefetched(ep, prefetched, varname, knw, num_prefetched):
    prefetch = prefetched
    if num_prefetched == "two":
        prefetch = (prefetch, prefetch)
    registered = ep._register_interpolation_input(
        varname, knw, prefetched=prefetch, prefetch_prefix={"SALT": (1, 0, 0, 0, 0)}
    )
    assert len(registered) == 8


@pytest.mark.parametrize(
    "data,x,knw",
    [
        (None, -37.5, uknw),
        (prefetched, -37.5, uknw),
    ],
)
@pytest.mark.parametrize("od", ["ecco"], indirect=True)
def test_wrong_datatype_error(od, data, x, knw):
    with pytest.raises(ValueError):
        the_p = sd.Particle(
            x=x,
            y=np.ones(2) * 10.4586420059204,
            z=np.ones(2) * (-9.0),
            t=np.ones(2) * 698155200.0,
            data=data,
        )
        the_p.fatten(knw)


@pytest.mark.parametrize(
    "x,knw",
    [
        (np.ones((2, 2)), uknw),
        (np.ones(12), uknw),
        (np.array([-37.5, -37.4]), wrongZknw),
        (np.array([-37.5, -37.4]), wrongTknw),
    ],
)
@pytest.mark.parametrize("od", ["ecco"], indirect=True)
def test_init_valueerror(od, x, knw):
    with pytest.raises(ValueError):
        the_p = sd.Particle(
            x=x,
            y=np.ones(2) * 10.4586420059204,
            z=np.ones(2) * (-9.0),
            t=np.ones(2) * 698155200.0,
            data=od,
        )
        the_p.fatten(knw)


@pytest.mark.parametrize(
    "varName,knw,prefetch,prefetch_prefix",
    [
        (1, uknw, None, None),
        ("SALT", [uknw, vknw], None, None),
        (("SALT", "SALT"), (uknw, vknw, wknw), None, None),
        ("SALT", {"s": uknw}, None, None),
        ("SALT", 1, None, None),
        ("SALT", uknw, [1, 1], None),
        ("SALT", uknw, None, [1, 1]),
        ("SALT", uknw, 1, None),
        ("SALT", uknw, None, 1),
    ],
)
def test_interp_register_error(ep, varName, knw, prefetch, prefetch_prefix):
    with pytest.raises(ValueError):
        ep._register_interpolation_input(
            varName, knw, prefetched=prefetch, prefetch_prefix=prefetch_prefix
        )


def test_fatten_none(ep):
    ep.it = None
    ep.izl = None
    ep.iz = None

    assert ep._fatten_v(wknw) is None
    assert ep._fatten_vl(wknw) is None
    assert ep._fatten_t(wknw) is None


def test_partial_flatten():
    thing = np.array((3, 4, 5))
    ind = (thing, thing)
    flattened = sd.eulerian._partial_flatten(ind)
    assert flattened[0].shape == (3, 1)


@pytest.mark.parametrize("od", ["ecco"], indirect=True)
def test_wvel_quant_deepest(od):
    ind = (11, 75, 73)
    face, iy, ix = ind

    np.random.seed(20230625)
    z = np.random.uniform(od.Zl[-1], 0, 5000)
    x = od.XC[ind] * np.ones_like(z)
    y = od.YC[ind] * np.ones_like(z)

    vert_p = sd.Position().from_latlon(x=x, y=y, z=z, data=od)
    assert vert_p.ix[0] == ix, "horizontal index does not match"
    assert vert_p.iy[0] == iy, "horizontal index does not match"
    seaduck_ans = vert_p.interpolate("WVELMASS1", sd.lagrangian.wknw)

    wvel = interp1d(od.Zl, od._ds.WVELMASS1[:, face, iy, ix])
    scipy_ans = wvel(z)

    assert np.allclose(scipy_ans, seaduck_ans)


@pytest.mark.parametrize("od", ["ecco"], indirect=True)
@pytest.mark.parametrize("seed", list(range(5)))
def test_wvel_quant_random_place(od, seed):
    np.random.seed(seed)
    z = np.random.uniform(od.Zl[-1], 0, 50)
    x = np.random.uniform(-180, 180, 1) * np.ones_like(z)
    y = np.random.uniform(-80, 90, 1) * np.ones_like(z)

    vert_p = sd.Position().from_latlon(x=x, y=y, z=z, data=od)
    seaduck_ans = vert_p.interpolate("WVELMASS1", sd.lagrangian.wknw)

    face = vert_p.face[0]
    iy = vert_p.iy[0]
    ix = vert_p.ix[0]

    wvel = interp1d(od.Zl, od._ds.WVELMASS1[:, face, iy, ix])
    scipy_ans = wvel(z)

    assert np.allclose(scipy_ans, seaduck_ans)


@pytest.mark.parametrize("od", ["ecco"], indirect=True)
def test_dw_quant_deepest(od):
    ind = (11, 75, 73)
    face, iy, ix = ind

    np.random.seed(20230625)
    z = np.random.uniform(od.Zl[-1], 0, 5000)
    x = od.XC[ind] * np.ones_like(z)
    y = od.YC[ind] * np.ones_like(z)

    vert_p = sd.Position().from_latlon(x=x, y=y, z=z, data=od)
    assert vert_p.ix[0] == ix, "horizontal index does not match"
    assert vert_p.iy[0] == iy, "horizontal index does not match"
    seaduck_ans = vert_p.interpolate("WVELMASS1", sd.lagrangian.dwknw)

    # dw is a stepwise function.
    small_offset = 1e-12
    dw = -np.diff(np.array(od._ds.WVELMASS1[:, face, iy, ix]))
    zinterp = [0]
    dwinterp = [dw[0]]
    for i, zl in enumerate(od.Zl[1:-1]):
        zinterp.append(zl + small_offset)
        dwinterp.append(dw[i])
        zinterp.append(zl)
        dwinterp.append(dw[i + 1])
    zinterp.append(od.Zl[-1])
    dwinterp.append(dw[-1])
    dwvel = interp1d(zinterp, dwinterp)
    scipy_ans = dwvel(z)

    assert np.allclose(scipy_ans, seaduck_ans)


@pytest.mark.parametrize("od", ["ecco"], indirect=True)
@pytest.mark.parametrize("seed", list(range(7, 12)))
def test_dw_quant_random(od, seed):
    np.random.seed(seed)
    z = np.random.uniform(od.Zl[-1], 0, 50)
    x = np.random.uniform(-180, 180, 1) * np.ones_like(z)
    y = np.random.uniform(-80, 90, 1) * np.ones_like(z)

    vert_p = sd.Position().from_latlon(x=x, y=y, z=z, data=od)
    seaduck_ans = vert_p.interpolate("WVELMASS1", sd.lagrangian.dwknw)

    face = vert_p.face[0]
    iy = vert_p.iy[0]
    ix = vert_p.ix[0]

    # dw is a stepwise function.
    small_offset = 1e-12
    dw = -np.diff(np.array(od._ds.WVELMASS1[:, face, iy, ix]))
    zinterp = [0]
    dwinterp = [dw[0]]
    for i, zl in enumerate(od.Zl[1:-1]):
        zinterp.append(zl + small_offset)
        dwinterp.append(dw[i])
        zinterp.append(zl)
        dwinterp.append(dw[i + 1])
    zinterp.append(od.Zl[-1])
    dwinterp.append(dw[-1])
    dwvel = interp1d(zinterp, dwinterp)
    scipy_ans = dwvel(z)

    assert np.allclose(scipy_ans, seaduck_ans)
