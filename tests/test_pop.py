import numpy as np
import pytest

import seaduck as sd
from seaduck import utils


@pytest.fixture(scope="module")
def pop():
    return utils.get_dataset("pop")


def _face_mean(a, b):
    return 0.5 * (a.fillna(b) + b.fillna(a))


def _add_transport(ds):
    """Volume transport (m^3/s) from the POP salt fluxes, in POP's layout."""
    ds = ds.copy()
    salt = ds["SALT"]
    vol = (ds["TAREA"] * ds["dz"] * 1e-6).transpose("z_t", "nlat", "nlon")
    salt_top = _face_mean(salt, salt.shift(z_t=1)).drop_vars("z_t")
    vol_top = vol.drop_vars("z_t").rename(z_t="z_w_top")
    for flux, name, s_face, v in [
        ("UES", "utrans", _face_mean(salt, salt.roll(nlon=-1)), vol),
        ("VNS", "vtrans", _face_mean(salt, salt.shift(nlat=-1)), vol),
        ("WTS", "wtrans", salt_top.rename(z_t="z_w_top"), vol_top),
    ]:
        ds[name] = (ds[flux] * v / s_face).fillna(0).astype("float32")
        ds[name].attrs = {"grid_loc": ds[flux].attrs["grid_loc"]}
    return ds


@pytest.fixture(scope="module")
def pop_od(pop):
    ds = _add_transport(pop).isel(time=0)
    return sd.OceData(sd.utils.process_pop(ds).load()), ds


def test_process_pop_layout(pop):
    out = sd.utils.process_pop(pop)
    ny, nx = pop["TLONG"].shape
    assert out["XG"].shape == (ny + 1, nx)
    assert out["UES"].dims == ("time", "Z", "Y", "Xp1")
    assert out["VNS"].dims == ("time", "Z", "Yp1", "X")
    assert out["WTS"].dims == ("time", "Zl", "Y", "X")
    # east face of column -1 is the west face of column 0
    np.testing.assert_array_equal(out["UES"][..., 0], pop["UES"][..., -1])
    np.testing.assert_array_equal(out["VNS"][..., 1:, :], pop["VNS"])
    assert (out["VNS"][..., 0, :] == 0).all()
    np.testing.assert_array_equal(out["XG"][1:, 1:], pop["ULONG"][:, :-1])
    # meters, negative upward
    assert (out["Z"] < 0).all() and out["Zl"][0] == 0
    np.testing.assert_allclose(out["drF"].sum(), -out["Zp1"][-1])
    # seconds, middle of the averaging period
    np.testing.assert_allclose(
        out["time"], pop["time_bound"].mean("d2") * 86400.0, rtol=1e-12
    )


def test_process_pop_decoded_time(pop):
    decoded = pop.assign_coords(time=np.array(["2000-01-16"], dtype="datetime64[ns]"))
    with pytest.raises(ValueError):
        sd.utils.process_pop(decoded)


def test_pop_ocedata(pop_od):
    od, _ = pop_od
    assert od.tp.typ == "x_periodic"
    assert od.readiness["h"] == "oceanparcel"
    assert od.readiness["Z"] and od.readiness["Zl"]


def test_x_periodic_wrap(pop_od):
    od, _ = pop_od
    ixmax = od.tp.ixmax
    assert od.tp.ind_tend((10, ixmax), 3) == (10, 0)
    assert od.tp.ind_tend((10, 0), 2) == (10, ixmax)


def _seam_particles(od, n=100, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-40.6, -39.4, n)  # the POP grid seam is at about 320E
    y = rng.uniform(25, 50, n)
    z = rng.uniform(-45, -5, n)
    return sd.Particle(
        x=x,
        y=y,
        z=z,
        t=np.zeros(n),
        data=od,
        uname="utrans",
        vname="vtrans",
        wname="wtrans",
        transport=True,
    )


def test_pop_particle_velocity(pop_od):
    od, ds = pop_od
    p = _seam_particles(od)
    vol = np.array((ds["TAREA"] * ds["dz"] * 1e-6).transpose("z_t", "nlat", "nlon"))
    u_face, v_face = np.array(ds["utrans"]), np.array(ds["vtrans"])
    k, j, i = p.izl_lin - 1, p.iy, p.ix
    assert {0, od.tp.ixmax} <= set(i)
    uw, ue = u_face[k, j, i - 1], u_face[k, j, i]  # i - 1 = -1 wraps around
    vs, vn = v_face[k, j - 1, i], v_face[k, j, i]
    u = ((0.5 - p.rx) * uw + (0.5 + p.rx) * ue) / vol[k, j, i]
    v = ((0.5 - p.ry) * vs + (0.5 + p.ry) * vn) / vol[k, j, i]
    np.testing.assert_allclose(p.u, u, rtol=1e-4, atol=1e-6 * np.abs(u).max())
    np.testing.assert_allclose(p.v, v, rtol=1e-4, atol=1e-6 * np.abs(v).max())


def test_pop_particle_seam(pop_od):
    od, _ = pop_od
    p = _seam_particles(od)
    p = p.subset(np.array(od["maskC"])[p.izl_lin - 1, p.iy, p.ix])
    _, raw = p.to_list_of_time(normal_stops=np.linspace(0, 20 * 86400, 81))
    lon = np.array([r.lon for r in raw])
    lat = np.array([r.lat for r in raw])
    assert np.isfinite(lon).all() and np.isfinite(lat).all()
    dlon = (np.diff(lon, axis=0) + 180) % 360 - 180
    step = np.hypot(dlon * np.cos(np.deg2rad(lat[1:])), np.diff(lat, axis=0))
    assert step.max() * 111e3 < 30e3  # m per 6 hours
