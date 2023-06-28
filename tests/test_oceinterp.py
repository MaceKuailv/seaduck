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
    "var_list",
    [
        ["ETAN", "maskC"],
        "SALT",
        ("UVELMASS", "VVELMASS"),
        {("UVELMASS", "VVELMASS"): (sd.KnW(), sd.KnW())},
    ],
)
def test_eulerian_oceinterp(od, var_list, x, y, z, t):
    ans = sd.OceInterp(od, var_list, x, y, z, t)
    assert isinstance(ans, list)


@pytest.mark.parametrize("ds", ["ecco"], indirect=True)
def test_xarray_interp(ds):
    ans = sd.OceInterp(ds, "ETAN", x, y, z, t, kernel_list=sd.KnW())
    assert isinstance(ans, list)


@pytest.mark.parametrize("od", ["ecco"], indirect=True)
def test_diff_oceinterp(od):
    kernel_kwarg = {
        "hkernel": "dx",
        "h_order": 2,
        "inheritance": [[0, 1, 2, 3, 4, 5, 6, 7, 8]],
        "tkernel": "linear",
    }
    ans = sd.OceInterp(
        od, ("UVELMASS", "VVELMASS"), x, y, z, t, kernel_kwarg=kernel_kwarg
    )
    assert isinstance(ans[0], tuple)
    assert isinstance(ans[0][0], np.ndarray)


@pytest.mark.parametrize(
    "od,var_list,x,y,z,t,lagrangian,lagrange_kwarg",
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
    od, var_list, x, y, z, t, lagrangian, return_pt_time, lagrange_kwarg
):
    ans = sd.OceInterp(
        od,
        var_list,
        x,
        y,
        z,
        t,
        lagrangian=lagrangian,
        return_pt_time=return_pt_time,
        lagrange_kwarg=lagrange_kwarg,
    )
    if return_pt_time:
        ans = ans[1]
        assert ans[0] is not None
    assert isinstance(ans, list)
    for i, var in enumerate(var_list):
        assert isinstance(ans[i], list), f"{var} not in correct format"


@pytest.mark.parametrize(
    "var_list,x,y,z,t,lagrangian,error",
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
        ("ETAN", x, y, z, 0, False, ValueError),
    ],
)
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
@pytest.mark.filterwarnings("ignore:divide by zero encountered in divide")
@pytest.mark.parametrize("od", ["ecco"], indirect=True)
def test_oceinterp_error(od, var_list, x, y, z, t, lagrangian, error):
    with pytest.raises(error):
        sd.OceInterp(od, var_list, x, y, z, t, lagrangian=lagrangian)


@pytest.mark.parametrize("t", ["1992-02-01", np.datetime64("1992-02-01")])
@pytest.mark.parametrize("to_array", [True, False])
@pytest.mark.parametrize("od", ["ecco"], indirect=True)
def test_flexible_time_format(od, t, to_array):
    if to_array:
        t = np.array([t for _ in x])
    res = sd.OceInterp(od, "ETAN", x, y, z, t)
    assert ~(np.isnan(res[0]).any())
    assert isinstance(res[0], np.ndarray)


@pytest.mark.parametrize(
    "od,var_list",
    [
        (
            "ecco",
            ["__particle.raw", "__particle.lat"],
        )
    ],
    indirect=["od"],
)
def test_lagrangian_warning(od, var_list):
    with pytest.warns(UserWarning):
        ans = sd.OceInterp(
            od,
            var_list,
            x,
            y,
            z,
            t,
            lagrangian=True,
            return_pt_time=False,
            return_in_between=True,
        )
        assert isinstance(ans, list)
