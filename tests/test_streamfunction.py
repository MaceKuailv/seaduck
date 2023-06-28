import numpy as np
import pytest
import xarray as xr

import seaduck as sd


def streamfunction(x, z):
    return np.cos(np.pi * x / 2) * np.cos(np.pi * z / 2)


@pytest.fixture
def overturning_cell():
    N = 100
    M = 50
    x = np.linspace(-1, 1, N + 1)
    y = np.linspace(-0.1, 0.1, 2)
    zl = np.linspace(0, -1000, M)
    zp1 = np.append(zl, -1001)
    xg, yg = np.meshgrid(x, y)
    xv = 0.5 * (xg[:, 1:] + xg[:, :-1])
    yv = 0.5 * (yg[:, 1:] + yg[:, :-1])
    0.5 * (xg[1:] + xg[:-1])
    0.5 * (yg[1:] + yg[:-1])

    xc = 0.5 * (xv[1:] + xv[:-1])
    yc = 0.5 * (yv[1:] + yv[:-1])

    tempx, tempz = np.meshgrid(x, zl)
    strmf = streamfunction(tempx, -tempz / 500 - 1).reshape(len(zl), 1, -1)
    z = 0.5 * (zp1[1:] + zp1[:-1])
    zl = zp1[:-1]
    drf = np.abs(np.diff(zp1))
    u = np.zeros((M, 1, N + 1), float)
    u[:-1] = np.diff(strmf, axis=0)
    w = np.diff(strmf, axis=-1)
    v = np.zeros((M, 2, N), float)
    stream = np.zeros((M, 2, N + 1))
    stream[:] = strmf
    ds = xr.Dataset(
        coords=dict(
            XC=(["Y", "X"], xc),
            YC=(["Y", "X"], yc),
            XG=(["Yp1", "Xp1"], xg),
            YG=(["Yp1", "Xp1"], yg),
            Zl=(["Zl"], zl),
            Z=(["Z"], z),
            drF=(["Z"], drf),
            rA=(["Y", "X"], np.ones_like(xc, float)),
        ),
        data_vars=dict(
            UVELMASS=(["Z", "Y", "Xp1"], u),
            VVELMASS=(["Z", "Yp1", "X"], v),
            WVELMASS=(["Zl", "Y", "X"], w),
            streamfunc=(["Zl", "Yp1", "Xp1"], stream),
        ),
    )
    return sd.OceData(ds)


@pytest.fixture
def gyre():
    M, N = 50, 50
    x = np.linspace(-1, 1, N + 1)
    y = np.linspace(-1, 1, M + 1)
    xg, yg = np.meshgrid(x, y)
    strmf = streamfunction(xg, yg)
    xv = 0.5 * (xg[:, 1:] + xg[:, :-1])
    yv = 0.5 * (yg[:, 1:] + yg[:, :-1])
    0.5 * (xg[1:] + xg[:-1])
    0.5 * (yg[1:] + yg[:-1])

    xc = 0.5 * (xv[1:] + xv[:-1])
    yc = 0.5 * (yv[1:] + yv[:-1])

    u = np.diff(strmf, axis=0)
    v = -np.diff(strmf, axis=1)
    ds = xr.Dataset(
        coords=dict(
            XC=(["Y", "X"], xc),
            YC=(["Y", "X"], yc),
            XG=(["Yp1", "Xp1"], xg),
            YG=(["Yp1", "Xp1"], yg),
            rA=(["Y", "X"], np.ones_like(xc, float)),
        ),
        data_vars=dict(
            UVELMASS=(["Y", "Xp1"], u),
            VVELMASS=(["Yp1", "X"], v),
            streamfunc=(["Yp1", "Xp1"], strmf),
        ),
    )
    return sd.OceData(ds)


@pytest.fixture
def gknw():
    kkk = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    return sd.KnW(kkk, vkernel="linear", tkernel="nearest")


@pytest.mark.filterwarnings("ignore:invalid value encountered")
@pytest.mark.filterwarnings("ignore:divide by zero")
def test_vertical_streamfunc_conservation(overturning_cell, gknw):
    N = 100
    x = np.random.uniform(-1, 1, N)
    z = np.random.uniform(-0.1, -1000, N)
    pt = sd.Particle(
        x=x,
        y=np.zeros_like(x),
        z=z,
        t=np.zeros_like(x),
        data=overturning_cell,
        transport=True,
    )
    with pytest.warns(UserWarning):
        before = pt.interpolate("streamfunc", gknw)
        steps = 40
        stops, ps = pt.to_list_of_time(
            normal_stops=np.linspace(0, -2 * steps * N, steps), update_stops=[]
        )
        after = pt.interpolate("streamfunc", gknw)
        assert np.allclose(after, before, atol=1e-7)


@pytest.mark.filterwarnings("ignore:invalid value encountered")
@pytest.mark.filterwarnings("ignore:divide by zero")
def test_horizontal_streamfunction_conservation(gyre, gknw):
    num_particle = 100
    np.random.seed(2023)
    x = np.random.random(num_particle) * 2 - 1
    y = np.random.random(num_particle) * 2 - 1
    pt = sd.Particle(
        x=x, y=y, z=None, t=np.zeros_like(x), data=gyre, wname=None, transport=True
    )
    gknw.vkernel = "nearest"
    with pytest.warns(UserWarning):
        before = pt.interpolate("streamfunc", gknw)
        steps = 15
        stops, ps = pt.to_list_of_time(
            normal_stops=np.linspace(0, 2 * steps * 50, steps), update_stops=[]
        )
        after = ps[-1].interpolate("streamfunc", gknw)
        assert np.allclose(after, before, atol=1e-7)
