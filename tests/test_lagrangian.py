import seaduck as sd
import numpy as np
import pytest
import xarray as xr

# Set the number of particles here.
N = int(9)

# Increase this if you want more in x direction.
skew = 3

# Change the vertical depth of the particles here.
sqrtN = int(np.sqrt(N))

# Change the horizontal range here.
x = np.linspace(-180, 180, sqrtN * skew)
y = np.linspace(-50, -70, sqrtN // skew)

x, y = np.meshgrid(x, y)
x = x.ravel()
y = y.ravel()
z = None
zz = np.ones_like(x) * (-10.0)

start_time = "1992-02-01"
t = (
    np.array([np.datetime64(start_time) for i in x]) - np.datetime64("1970-01-01")
) / np.timedelta64(1, "s")
tf = (np.datetime64("1992-03-03") - np.datetime64("1970-01-01")) / np.timedelta64(
    1, "s"
)


@pytest.fixture
def p(avis):
    return sd.particle(
        x=x,
        y=y,
        z=z,
        t=t,
        data=avis,
        # save_raw = True,
        # transport = True,
        uname="u",
        vname="v",
        wname=None,
    )


@pytest.fixture
def ecco_p(ecco):
    return sd.particle(x=x, y=y, z=zz, t=t, data=ecco, transport=True)


normal_stops = np.linspace(t[0], tf, 5)


def test_vol_mode(ecco_p):
    stops, raw = ecco_p.to_list_of_time(normal_stops=[t[0], tf])


def test_to_list_of_time(p):
    stops, raw = p.to_list_of_time(
        normal_stops=normal_stops, update_stops=[normal_stops[1]]
    )


def test_analytical_step(p):
    p.analytical_step(10.0)


def test_callback(curv):
    curv_p = sd.particle(
        y=np.array([70.5]),
        x=np.array([-14.0]),
        z=np.array([-10.0]),
        t=np.array([curv.ts[0]]),
        data=curv,
        uname="U",
        vname="V",
        wname="W",
        callback=lambda pt: pt.lon > -14.01,
    )
    curv_p.to_list_of_time(normal_stops=[curv.ts[0], curv.ts[-1]], update_stops=[])


@pytest.mark.parametrize(
    "statement,error",
    [
        ("p.note_taking()", AttributeError),
        ("p.to_list_of_time(normal_stops = [0.0,1.0])", AttributeError),
        (
            "ecco_p.to_list_of_time(normal_stops = [0.0,1.0],update_stops = [])",
            ValueError,
        ),
    ],
)
def test_lagrange_error(statement, error, p, ecco_p):
    with pytest.raises(error):
        eval(statement)


def test_multidim_uvw_array(ecco_p):
    ecco_p.it[0] += 1
    ecco_p.update_uvw_array()


def test_update_w_array(ecco_p, ecco):
    ecco["u0"] = ecco["UVELMASS"].isel(time=0)
    ecco["v0"] = ecco["VVELMASS"].isel(time=0)
    ecco["w0"] = ecco["WVELMASS"].isel(time=0)
    delattr(ecco_p, "warray")
    ecco_p.uname = "u0"
    ecco_p.vname = "v0"
    ecco_p.wname = "w0"

    ecco_p.update_uvw_array()


def test_update_after_cell_change(ecco_p, ecco):
    ecco["SN"] = np.array(ecco["SN"])
    ecco["CS"] = np.array(ecco["CS"])
    ecco_p.ocedata.readiness["h"] = "local_cartesian"

    ecco_p.update_after_cell_change()


def test_update_after_cell_change_no_face(curv):
    curv._add_missing_cs_sn()
    curv.readiness["h"] = "local_cartesian"
    curv_p = sd.particle(
        y=np.array([70.5]),
        x=np.array([-14.0]),
        z=np.array([-10.0]),
        t=np.array([curv.ts[0]]),
        data=curv,
        uname="U",
        vname="V",
        wname="W",
        transport=True,
    )
    curv_p.update_after_cell_change()


def test_get_vol(curv):
    curv_p = sd.particle(
        y=np.array([70.5]),
        x=np.array([-14.0]),
        z=np.array([-10.0]),
        t=np.array([curv.ts[0]]),
        data=curv,
        uname="U",
        vname="V",
        wname="W",
        transport=True,
    )
    curv_p.get_vol()


def test_maxiteration(ecco_p):
    ecco_p.max_iteration = 1
    delattr(ecco_p, "px")
    ecco_p.to_next_stop(tf)
