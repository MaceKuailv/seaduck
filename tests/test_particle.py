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


def not_out_of_bound_in_analytical_step(new_p, tf, tol=1e-4):
    u_list, du_list, pos_list = new_p._extract_velocity_position()
    ts = sd.lagrangian.time2wall(pos_list, u_list, du_list)
    tend, t_event = sd.lagrangian.which_early(tf, ts)
    new_x, new_u = new_p._move_within_cell(t_event, u_list, du_list, pos_list)
    for rr in new_x:
        try:
            assert not np.logical_or(rr > 0.5 + tol, rr < -0.5 - tol).any()
        except AssertionError:
            where = np.where(np.logical_or(rr > 0.6, rr < -0.6))[0][0]
            raise ValueError(
                f"Particle way out of bound."
                # f"tend = {tend[where]},"
                f" t_event = {t_event[where]},"
                f" rx = {new_x[0][where]},ry = {new_x[1][where]},rz = {new_x[2][where]}"
                f"start with u = {new_p.u[where]}, du = {new_p.du[where]}, x={new_p.rx[where]}"
                f"start with v = {new_p.v[where]}, dv = {new_p.dv[where]}, y={new_p.ry[where]}"
                f"start with w = {new_p.w[where]}, dw = {new_p.dv[where]}, z={new_p.rzl_lin[where]}"
            )


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

    not_out_of_bound_in_analytical_step(one_p, 10799.999999993917)
