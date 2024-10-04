import numpy as np
import pytest

import seaduck as sd
from seaduck import utils

# Set the number of particles here.
N = 9

# Increase this if you want more in x direction.
skew = 3

# Change the vertical depth of the particles here.
sqrtN = int(np.sqrt(N))

# Change the horizontal range here.
x = np.append(np.linspace(-180, 180, sqrtN * skew), -37.5)
y = np.append(np.linspace(-50, -70, sqrtN // skew), -56.73891)

x, y = np.meshgrid(x, y)
x = x.ravel()
y = y.ravel()
z = None
zz = np.ones_like(x) * (-10.0)

start_time = "1992-02-01"
t = utils.convert_time(start_time) * np.ones_like(x)
end_time = "1992-02-03"
tf = utils.convert_time(end_time)


@pytest.fixture
def p():
    od = sd.OceData(utils.get_dataset("aviso"))
    return sd.Particle(
        x=x,
        y=y,
        z=z,
        t=t,
        data=od,
        # save_raw = True,
        # transport = True,
        uname="u",
        vname="v",
        wname=None,
    )


@pytest.fixture
def ecco_p():
    od = sd.OceData(utils.get_dataset("ecco"))
    return sd.Particle(x=x, y=y, z=zz, t=t, data=od, transport=True)


@pytest.fixture
def kick_back_p():
    x = np.array([-38.594593, -37.512672, -36.42936, -34.08329, -35.06443])
    y = np.array([-77.95619, -77.97306, -77.98856, -77.25903, -76.86412])
    z = np.ones_like(x) * (-0.01)
    t = utils.convert_time(start_time) * np.ones_like(x)
    od = sd.OceData(utils.get_dataset("ecco"))
    return sd.Particle(x=x, y=y, z=z, t=t, data=od, free_surface="kick_back")


normal_stops = np.linspace(t[0], tf, 5)


def test_vol_mode(ecco_p):
    stops, raw = ecco_p.to_list_of_time(normal_stops=[t[0], tf])
    assert sd.get_masks.which_not_stuck(ecco_p).all()


def test_to_list_of_time(p):
    stops, raw = p.to_list_of_time(
        normal_stops=normal_stops, update_stops=[normal_stops[1]]
    )
    with pytest.warns(UserWarning):
        assert sd.get_masks.which_not_stuck(p).all()


def test_subset_update(p):
    np.random.seed(0)
    which = np.random.randint(1, size=p.N, dtype=bool)
    sub = p.subset(which)
    sub.lon += 1
    sub.lat += 1
    p.update_from_subset(sub, which)
    assert isinstance(sub, sd.Particle)
    assert np.allclose(p.ix[which], sub.ix)
    assert np.allclose(p.lon[which], sub.lon)


def test_subset_px_py(ecco_p):
    np.random.seed(1)
    which = np.random.randint(1, size=ecco_p.N, dtype=bool)
    ecco_p.px, ecco_p.py = ecco_p.get_px_py()
    sub = ecco_p.subset(which)
    sub.px, sub.py = sub.get_px_py()
    assert np.allclose(ecco_p.px[:, which], sub.px)
    assert np.allclose(ecco_p.py[:, which], sub.py)


@pytest.mark.parametrize("od", ["curv"], indirect=True)
def test_callback(od):
    curv_p = sd.Particle(
        y=np.array([70.5]),
        x=np.array([-14.0]),
        z=np.array([-10.0]),
        t=np.array([od.ts[0]]),
        data=od,
        uname="U",
        vname="V",
        wname="W",
        callback=lambda pt: pt.lon > -14.01,
    )
    curv_p.to_list_of_time(normal_stops=[od.ts[0], od.ts[-1]], update_stops=[])
    with pytest.warns(UserWarning):
        assert sd.get_masks.which_not_stuck(curv_p).all()


def test_note_taking_error(p):
    with pytest.raises(AttributeError):
        p.note_taking()


def test_no_time_midp_error():
    od = sd.OceData(utils.get_dataset("ecco").drop_vars("time_midp"))
    p = sd.Particle(x=x, y=y, z=zz, t=t, data=od, transport=True)
    delattr(od, "time_midp")
    with pytest.raises(AttributeError):
        p.to_list_of_time(normal_stops=[t[0] + 1])


def test_time_out_of_bound_error(ecco_p):
    with pytest.raises(ValueError):
        ecco_p.to_list_of_time(normal_stops=[0.0, 1.0], update_stops=[])


def test_multidim_uvw_array(ecco_p):
    ecco_p.it[0] += 1
    ecco_p.update_uvw_array()
    assert ecco_p.uarray.shape[0] == 2


def test_update_w_array(ecco_p):
    ecco_p.ocedata._ds["u0"] = ecco_p.ocedata["UVELMASS"].isel(time=0)
    ecco_p.ocedata._ds["v0"] = ecco_p.ocedata["VVELMASS"].isel(time=0)
    ecco_p.ocedata._ds["w0"] = ecco_p.ocedata["WVELMASS"].isel(time=0)
    delattr(ecco_p, "warray")
    ecco_p.uname = "u0"
    ecco_p.vname = "v0"
    ecco_p.wname = "w0"

    ecco_p.update_uvw_array()
    assert len(ecco_p.uarray.shape) == 4
    ecco_p.update_uvw_array()


@pytest.mark.parametrize("od", ["ecco"], indirect=True)
def test_wall_crossing(ecco_p, od):
    od["SN"] = np.array(od["SN"])
    od["CS"] = np.array(od["CS"])
    ecco_p.ocedata.readiness["h"] = "local_cartesian"

    ecco_p._cross_cell_wall_rel()
    assert (~np.isnan(ecco_p.ry)).any()


@pytest.mark.parametrize("od", ["curv"], indirect=True)
def test_wall_crossing_no_face(od):
    od._add_missing_cs_sn()
    od.readiness["h"] = "local_cartesian"
    curv_p = sd.Particle(
        y=np.array([70.5]),
        x=np.array([-14.0]),
        z=np.array([-10.0]),
        t=np.array([od.ts[0]]),
        data=od,
        uname="U",
        vname="V",
        wname="W",
        transport=True,
    )
    curv_p._cross_cell_wall_read()
    assert isinstance(curv_p.cs, np.ndarray)
    curv_p._cross_cell_wall_rel()
    assert (~np.isnan(curv_p.rx)).any()


@pytest.mark.parametrize("od", ["curv"], indirect=True)
def test_get_vol(od):
    curv_p = sd.Particle(
        y=np.array([70.5]),
        x=np.array([-14.0]),
        z=np.array([-10.0]),
        t=np.array([od.ts[0]]),
        data=od,
        uname="U",
        vname="V",
        wname="W",
        transport=True,
    )
    curv_p.get_vol()
    assert np.issubdtype(curv_p.vol.dtype, float)


def test_maxiteration(ecco_p):
    ecco_p.max_iteration = 1
    delattr(ecco_p, "px")
    with pytest.warns(UserWarning):
        ecco_p.to_next_stop(tf)


def test_abandon_ducks(ecco_p):
    N = len(ecco_p.izl_lin)
    ecco_p.izl_lin = (np.ones(N) * 50).astype(int)
    new_p = sd.get_masks.abandon_stuck(ecco_p)
    assert len(new_p.izl_lin) < N


@pytest.mark.parametrize("od", ["ecco"], indirect=True)
@pytest.mark.parametrize("seed", list(range(5)))
def test_reproduce_latlon_oceanparcel(od, seed):
    np.random.seed(seed)
    x = np.random.uniform(-180, 180, 1)
    y = np.random.uniform(-90, 90, 1)
    t = sd.utils.convert_time("1992-02-01")
    z = -5.0
    rand_p = sd.Particle(x=x, y=y, z=z, t=t, data=od)
    rand_p._sync_latlondep_before_cross()
    assert np.allclose(rand_p.lon, x)
    assert np.allclose(rand_p.lat, y)


@pytest.mark.parametrize("od", ["ecco"], indirect=True)
@pytest.mark.parametrize("seed", [678, 15, 7])
def test_get_u_du_quant(seed, od):
    np.random.seed(seed)
    ind = tuple(np.random.randint([13, 89, 89]))
    face, iy, ix = ind
    # without boundary points
    x = od.XC[ind]
    y = od.YC[ind]
    t = tf
    z = np.random.uniform(0, -3000)
    rand_p = sd.Particle(x=x, y=y, z=z, t=t, data=od)
    u = rand_p.u * rand_p.dx
    du = rand_p.du * rand_p.dx
    v = rand_p.v * rand_p.dy
    dv = rand_p.dv * rand_p.dy
    w = rand_p.w * rand_p.dzl_lin
    dw = rand_p.dw * rand_p.dzl_lin

    it = rand_p.it[0]
    iz = rand_p.izl_lin[0] - 1
    u1 = float(od["UVELMASS"][it, iz, face, iy, ix])
    u2 = float(od["UVELMASS"][it, iz, face, iy, ix + 1])
    v1 = float(od["VVELMASS"][it, iz, face, iy, ix])
    v2 = float(od["VVELMASS"][it, iz, face, iy + 1, ix])
    w1 = float(od["WVELMASS"][it, iz + 1, face, iy, ix])
    w2 = float(od["WVELMASS"][it, iz, face, iy, ix])
    rx = rand_p.rx[0]
    ry = rand_p.ry[0]
    rz = rand_p.rzl_lin[0] - 0.5
    ushould = (0.5 + rx) * u2 + (0.5 - rx) * u1
    vshould = (0.5 + ry) * v2 + (0.5 - ry) * v1
    wshould = (0.5 + rz) * w2 + (0.5 - rz) * w1

    assert u1 != 0
    assert np.allclose(u2 - u1, du, atol=1e-18)
    assert np.allclose(v2 - v1, dv, atol=1e-18)
    assert np.allclose(w2 - w1, dw, atol=1e-18)
    assert np.allclose(ushould, u, atol=1e-18)
    assert np.allclose(vshould, v, atol=1e-18)
    assert np.allclose(wshould, w, atol=1e-18)


def test_kick_back(kick_back_p):
    tend = kick_back_p.analytical_step(-1e10)
    kick_back_p.cross_cell_wall(tend)
    assert np.isclose(kick_back_p.rzl_lin, 0.5).any()
