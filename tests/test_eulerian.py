import numpy as np
import xarray as xr
import seaduck as sd
import pytest

# Datadir = "tests/Data/"
# ecco = xr.open_zarr(Datadir + "small_ecco")
# ecco = sd.OceData(ecco)
# avis = xr.open_dataset(Datadir + "aviso_example.nc")
# avis = sd.OceData(avis)

uknw = sd.lagrangian.uknw
vknw = sd.lagrangian.vknw
wknw = sd.lagrangian.wknw
wrongZknw = sd.KnW()
wrongZknw.vkernel = "something not correct"
wrongTknw = sd.KnW()
wrongTknw.tkernel = "something not correct"

prefetched = np.array(ecco["SALT"][slice(1, 2)])

# use float number to make particle
@pytest.fixture
def ep():
    return sd.particle(x=-37.5, y=10.4586420059204, z=-9.0, t=698155200.0, data=ecco)


def test_tz_not_None(avis):
    ap = sd.position()
    ap.from_latlon(x=-37.5, y=10.4586420059204, z=-9.0, t=698155200.0, data=avis)


@pytest.mark.parametrize(
    "knw,required",
    [
        (uknw, "all"),
        (uknw, "Z"),
        (uknw, ["Z"]),
    ],
)
def test_fatten(ep, knw, required):
    ep.fatten(knw, required=required)


@pytest.mark.parametrize("varname,knw", [("WVELMASS", uknw), ("UVELMASS", wknw)])
def test_interp_vertical(ep, varname, knw):
    ep.interpolate(varname, knw)


def test_dict_input(ep):
    ep._register_interpolation_input(
        "SALT",
        {"SALT": uknw},
        prefetched={"SALT": prefetched},
        i_min={"SALT": (1, 0, 0, 0, 0)},
    )


@pytest.mark.parametrize(
    "varname,knw,prefetched",
    [
        ("SALT", uknw, prefetched),
        (("SALT", "SALT"), (uknw, uknw), (prefetched, prefetched)),
    ],
)
def test_diff_prefetched(ep, varname, knw, prefetched):
    ep._register_interpolation_input(
        varname, knw, prefetched=prefetched, i_min={"SALT": (1, 0, 0, 0, 0)}
    )


@pytest.mark.parametrize(
    "data,x,knw",
    [
        (None, -37.5, uknw),
        (prefetched, -37.5, uknw),
        ('ecco', np.ones((2, 2)), uknw),
        ('ecco', np.ones(12), uknw),
        ('ecco', np.array([-37.5, -37.4]), wrongZknw),
        ('ecco', np.array([-37.5, -37.4]), wrongTknw),
    ],
)
def test_init_valueerror(ecco, data, x, knw):
    if isinstance(data,str):
        data = eval(data)
    with pytest.raises(ValueError):
        the_p = sd.particle(
            x=x,
            y=np.ones(2) * 10.4586420059204,
            z=np.ones(2) * (-9.0),
            t=np.ones(2) * 698155200.0,
            data=data,
        )
        the_p.fatten(knw)


@pytest.mark.parametrize(
    "varName,knw,prefetched,i_min",
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
def test_interp_register_error(varName, knw, prefetched, i_min):
    with pytest.raises(ValueError):
        ep._register_interpolation_input(
            varName, knw, prefetched=prefetched, i_min=i_min
        )
