import numpy as np
import pytest

import seaduck as sd

N = int(1e2)

# Change the vertical depth of the particles here
levels = np.array([-5])
sqrtN = int(np.sqrt(N))

# Change the longitude and latitude positions of the particles here
xx = np.linspace(-19, -9, sqrtN)
yy = np.linspace(63, 57, sqrtN)

# Compute intermediate grid variables
xxx, yyy = np.meshgrid(xx, yy)
x = xxx.ravel()
y = yyy.ravel()
x, z = np.meshgrid(x, levels)
y, z = np.meshgrid(y, levels)
x = x.ravel()
y = y.ravel()
z = z.ravel()

# Change the times here
start_time = "1992-01-17"
t = sd.utils.convert_time(start_time) * np.ones_like(x)

end_time = "1992-02-15"

t_bnds = np.array(
    [
        sd.utils.convert_time(start_time),
        sd.utils.convert_time(end_time),
    ]
)


@pytest.mark.parametrize("od,x,y,z,t", [("ecco", x, y, z, t)], indirect=["od"])
@pytest.mark.parametrize(
    "varList",
    [
        ["ETAN", "maskC"],
        "SALT",
        ("UVELMASS", "VVELMASS"),
        {("UVELMASS", "VVELMASS"): (sd.KnW(), sd.KnW())},
    ],
)
def test_eulerian_oceinterp(od, varList, x, y, z, t):
    sd.OceInterp(od, varList, x, y, z, t)


@pytest.mark.parametrize("od", ["ecco"], indirect=True)
def test_diff_oceinterp(od):
    kernel_kwarg = {
        "hkernel": "dx",
        "h_order": 2,
        "inheritance": [[0, 1, 2, 3, 4, 5, 6, 7, 8]],
        "tkernel": "linear",
    }
    sd.OceInterp(od, ("UVELMASS", "VVELMASS"), x, y, z, t, kernel_kwarg=kernel_kwarg)


@pytest.mark.parametrize(
    "od,varList,x,y,z,t,lagrangian,lagrange_kwarg",
    [
        (
            "ecco",
            ["SALT", "__particle.raw", "__particle.lat", "__particle.lon"],
            x,
            y,
            z,
            t_bnds,
            True,
            {"save_raw": True},
        )
    ],
    indirect=["od"],
)
@pytest.mark.parametrize("return_pt_time", [True, False])
@pytest.mark.filterwarnings("ignore::Warning")
def test_largangian_oceinterp(
    od, varList, x, y, z, t, lagrangian, return_pt_time, lagrange_kwarg
):
    sd.OceInterp(
        od,
        varList,
        x,
        y,
        z,
        t,
        lagrangian=lagrangian,
        return_pt_time=return_pt_time,
        lagrange_kwarg=lagrange_kwarg,
    )


@pytest.mark.parametrize(
    "varList,x,y,z,t,lagrangian,error",
    [
        (
            ["__particle.lat", "__particle.lon"],
            x,
            y,
            z,
            t_bnds[:1],
            True,
            ValueError,
        ),
        (
            ["__particle.lat", "__particle.lon"],
            x,
            y,
            z,
            t,
            False,
            AttributeError,
        ),
        (None, x, y, z, t, False, ValueError),
        ([None], x, y, z, t, False, ValueError),
    ],
)
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
@pytest.mark.filterwarnings("ignore:divide by zero encountered in divide")
@pytest.mark.parametrize("od", ["ecco"], indirect=True)
def test_oceinterp_error(od, varList, x, y, z, t, lagrangian, error):
    with pytest.raises(error):
        sd.OceInterp(od, varList, x, y, z, t, lagrangian=lagrangian)


@pytest.mark.parametrize("t", ["1992-02-01", np.datetime64("1992-02-01")])
@pytest.mark.parametrize("to_array", [True, False])
@pytest.mark.parametrize("od", ["ecco"], indirect=True)
def test_flexible_time_format(od, t, to_array):
    if to_array:
        t = np.array([t for _ in x])
    res = sd.OceInterp(od, "ETAN", x, y, z, t)
    assert ~(np.isnan(res[0]).any())
    assert isinstance(res[0], np.ndarray)
