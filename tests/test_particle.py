import numpy as np
import pytest

import seaduck as sd


@pytest.fixture
def one_p():
    od = sd.OceData(sd.utils.get_dataset("ecco"))
    return sd.Particle(
        x=0.0,
        y=0.0,
        z=-10.0,
        t=sd.utils.convert_time("1992-02-01"),
        data=od,
    )


def not_out_of_bound_in_analytical_step(new_p, tf=1e80, tol=1e-4):
    u_list, du_list, pos_list = new_p._extract_velocity_position()
    tf = np.array([tf])
    ts = sd.lagrangian._time2wall(pos_list, u_list, du_list, tf)
    tend, t_event = sd.lagrangian._which_early(tf, ts)
    new_x, new_u = new_p._move_within_cell(t_event, u_list, du_list, pos_list)
    for rr in new_x:
        try:
            assert not np.logical_or(rr > 0.5 + tol, rr < -0.5 - tol).any()
        except AssertionError:
            where = np.where(np.logical_or(rr > 0.5 + tol, rr < -0.5 - tol))[0][0]
            raise ValueError(
                f"Particle way out of bound."
                # f"tend = {tend[where]},"
                f" t_event = {t_event[where]},"
                f" rx = {new_x[0][where]},ry = {new_x[1][where]},rz = {new_x[2][where]}"
                f"start with u = {new_p.u[where]}, du = {new_p.du[where]}, x={new_p.rx[where]}"
                f"start with v = {new_p.v[where]}, dv = {new_p.dv[where]}, y={new_p.ry[where]}"
                f"start with w = {new_p.w[where]}, dw = {new_p.dv[where]}, z={new_p.rzl_lin[where]}"
            )
    return tend[0]


def random_p(one_p, seed):
    np.random.seed(seed)
    (
        one_p.u,
        one_p.du,
        one_p.rx,
        one_p.v,
        one_p.dv,
        one_p.ry,
        one_p.w,
        one_p.dw,
        one_p.rzl_lin,
    ) = (np.array([i]) for i in np.random.uniform(-0.5, 0.5, 9))

    one_p.rzl_lin += 0.5

    one_p.u /= 1e5
    one_p.v /= 1e5
    one_p.w /= 1e5

    one_p.du /= 1e5
    one_p.dv /= 1e5
    one_p.dw /= 1e5

    return one_p


def u_du_from_uwall(ul, ur, rx):
    u = ul * (0.5 - rx) + ur * (rx + 0.5)
    du = ur - ul
    return u, du


def test_reproduce_issue55(one_p):
    one_p.u[:] = 4.695701621346472e-06
    one_p.du[:] = -5.773571643116384e-05
    one_p.rx[:] = -0.009470161238869457
    one_p.v[:] = -1.4674619307268205e-05
    one_p.dv[:] = -2.8246484204894833e-06
    one_p.ry[:] = 0.08627583529220939
    one_p.w[:] = 0.0
    one_p.dw[:] = -2.8246484204894833e-06
    one_p.rzl_lin[:] = 0.5 + 0.3454546420042159
    tend = not_out_of_bound_in_analytical_step(one_p)
    assert tend != 6


def test_exactly_at_corner(one_p):
    # reproduce_issue55vol2
    u = [0.00023098]
    du = [-1.58748431e-05]
    x = [-0.5]
    v = [-4.54496561e-06]
    dv = [1.31782954e-05]
    y = [-0.5]
    w = [0.0]
    dw = [1.31782954e-05]
    z = [0.34545464]

    one_p.u[:] = np.array(u)
    one_p.du[:] = np.array(du)
    one_p.rx[:] = np.array(x)
    one_p.v[:] = np.array(v)
    one_p.dv[:] = np.array(dv)
    one_p.ry[:] = np.array(y)
    one_p.w[:] = np.array(w)
    one_p.dw[:] = np.array(dw)
    one_p.rzl_lin[:] = 0.5 + np.array(z)
    tend = not_out_of_bound_in_analytical_step(one_p)
    assert tend != 6


@pytest.mark.parametrize("seed", [43, 20, 628])
def test_underflow_du(one_p, seed):
    one_p = random_p(one_p, seed)

    one_p.du /= 1e5
    one_p.dv /= 1e5
    one_p.dw /= 1e5

    tend = not_out_of_bound_in_analytical_step(one_p)
    assert tend != 6


@pytest.mark.parametrize("seed", [432, 320, 60288])
def test_underflow_u(one_p, seed):
    one_p = random_p(one_p, seed)

    one_p.u /= 1e5
    one_p.v /= 1e5
    one_p.w /= 1e5

    tend = not_out_of_bound_in_analytical_step(one_p)
    assert tend != 6


@pytest.mark.parametrize("seed", [0])
def test_u_du_uwall_conversion(one_p, seed):
    one_p = random_p(one_p, seed)
    ul, ur = sd.lagrangian._uleftright_from_udu(one_p.u, one_p.du, one_p.rx)
    u, du = u_du_from_uwall(ul, ur, one_p.rx)
    assert np.allclose(u, one_p.u)
    assert np.allclose(du, one_p.du)


@pytest.mark.parametrize("seed", list(range(1999, 2011)))
def test_impossible_wall(one_p, seed):
    not_wall = np.random.randint(0, 2, 6)
    passages = round(np.sum(not_wall))
    where = list(np.where(not_wall)[0])
    uwall_list = np.zeros(6)
    vels = np.random.uniform(-1e-5, 1e-5, passages)
    vels[-1] = -np.sum(vels[:-1])

    uwall_list[not_wall.astype(bool)] = vels
    uwall_list *= np.array([1, -1, 1, -1, 1, -1])
    uwall_list = [np.array([uu]) for uu in uwall_list]

    one_p = random_p(one_p, seed)
    one_p.u, one_p.du = u_du_from_uwall(uwall_list[0], uwall_list[1], one_p.rx)
    one_p.v, one_p.dv = u_du_from_uwall(uwall_list[2], uwall_list[3], one_p.ry)
    one_p.w, one_p.dw = u_du_from_uwall(
        uwall_list[4], uwall_list[5], one_p.rzl_lin - 0.5
    )

    tend = not_out_of_bound_in_analytical_step(one_p)
    if passages > 1:
        assert tend in where, uwall_list


@pytest.mark.parametrize("seed", list(range(2011, 2023)))
def test_impossible_inflow(one_p, seed):
    not_wall = np.random.randint(0, 2, 6)
    passages = round(np.sum(not_wall))
    where = list(np.where(not_wall)[0])
    uwall_list = np.random.random(6)
    uwall_list[~not_wall.astype(bool)] = 1
    vels = np.random.uniform(-1e-5, 1e-5, passages)
    vels[-1] = -np.sum(vels[:-1])

    uwall_list[not_wall.astype(bool)] = vels
    uwall_list *= np.array([1, -1, 1, -1, 1, -1])
    uwall_list = [np.array([uu]) for uu in uwall_list]

    one_p = random_p(one_p, seed)
    one_p.u, one_p.du = u_du_from_uwall(uwall_list[0], uwall_list[1], one_p.rx)
    one_p.v, one_p.dv = u_du_from_uwall(uwall_list[2], uwall_list[3], one_p.ry)
    one_p.w, one_p.dw = u_du_from_uwall(
        uwall_list[4], uwall_list[5], one_p.rzl_lin - 0.5
    )

    tend = not_out_of_bound_in_analytical_step(one_p)
    if passages > 1:
        assert tend in where, uwall_list


@pytest.mark.parametrize("seed", list(range(2023, 2028)))
@pytest.mark.parametrize("tf", [1e80, -1e80])
def test_cross_cell_wall(one_p, seed, tf):
    one_p = random_p(one_p, seed)
    tend = one_p.analytical_step(tf)
    one_p.cross_cell_wall(tend)
    if (abs(one_p.rx) > 1).any():
        where = np.where(abs(one_p.rx) > 1)[0][0]
        raise ValueError(
            f"lon = {one_p.lon[where]}, lat = {one_p.lat[where]}, "
            f"px = {one_p.px.T[where]}, py = {one_p.py.T[where]}"
        )
