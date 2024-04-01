import numpy as np
import pytest

from seaduck.eulerian_budget import (
    buffer_x_periodic,
    buffer_y_periodic,
    buffer_z_nearest_withoutface,
    second_order_flux_limiter_x,
    second_order_flux_limiter_y,
    second_order_flux_limiter_z_withoutface,
    superbee_fluxlimiter,
)


@pytest.fixture
def random_4d():
    np.random.seed(401)
    return np.random.random((3, 4, 5, 4))


def test_superbee():
    cr = np.array([-1, 0.25, 0.5, 1, 2, 100])
    res = superbee_fluxlimiter(cr)
    assert np.allclose(res, np.array([0.0, 0.5, 1.0, 1.0, 2.0, 2.0]))


@pytest.mark.parametrize(["lm", "rm"], [(0, 2), (2, 1), (1, 0)])
def test_buffer_x_periodic(random_4d, lm, rm):
    buffer = buffer_x_periodic(random_4d, lm, rm)
    if rm != 0:
        assert np.allclose(buffer[..., -1], random_4d[..., rm - 1])
    if lm != 0:
        assert np.allclose(buffer[..., 0], random_4d[..., -lm])


@pytest.mark.parametrize(["lm", "rm"], [(0, 2), (2, 1), (1, 0)])
def test_buffer_y_periodic(random_4d, lm, rm):
    buffer = buffer_y_periodic(random_4d, lm, rm)
    if rm != 0:
        assert np.allclose(buffer[..., -1, :], random_4d[..., rm - 1, :])
    if lm != 0:
        assert np.allclose(buffer[..., 0, :], random_4d[..., -lm, :])


@pytest.mark.parametrize(["lm", "rm"], [(0, 2), (2, 1), (1, 0)])
def test_buffer_z_nearest_withoutface(random_4d, lm, rm):
    buffer = buffer_z_nearest_withoutface(random_4d, lm, rm)
    if rm != 0:
        assert np.allclose(buffer[..., -1, :, :], random_4d[..., -1, :, :])
    if lm != 0:
        assert np.allclose(buffer[..., 0, :, :], random_4d[..., 0, :, :])


# The next three tests are not very quantitative,
# but they has been tested with real datasets,
# still there needs to be some improvements.
def test_second_order_flux_limiter_x(random_4d):
    np.random.seed(401)
    u_cfl = np.random.random((3, 4, 5, 5)) * 2 - 1
    ans = second_order_flux_limiter_x(random_4d, u_cfl)
    assert ans.shape == u_cfl.shape
    assert ans.dtype == "float64"


def test_second_order_flux_limiter_y(random_4d):
    np.random.seed(401)
    v_cfl = np.random.random((3, 4, 6, 4)) * 2 - 1
    ans = second_order_flux_limiter_y(random_4d, v_cfl)
    assert ans.shape == v_cfl.shape
    assert ans.dtype == "float64"


def test_second_order_flux_limiter_z_withoutface(random_4d):
    np.random.seed(401)
    w_cfl = np.random.random((3, 4, 5, 4)) * 2 - 1
    ans = second_order_flux_limiter_z_withoutface(random_4d, w_cfl)
    assert ans.shape == w_cfl.shape
    assert ans.dtype == "float64"
