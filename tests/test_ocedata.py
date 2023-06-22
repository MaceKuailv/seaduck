import numpy as np
import pytest
import xarray as xr

import seaduck as sd
from seaduck import utils


@pytest.fixture
def incomplete_data(request):
    ds = utils.get_dataset("curv")
    if request.param == "drop_YG":
        od = sd.OceData(ds)
        ds_out = ds.drop_vars(["YG"])
        od._add_missing_cs_sn()
        ds_out["CS"] = xr.DataArray(od["CS"], dims=ds["XC"].dims)
        ds_out["SN"] = xr.DataArray(od["SN"], dims=ds["XC"].dims)
        return ds_out
    elif request.param == "drop_dyG":
        return ds.drop_vars(["dyG"])
    elif request.param == "drop_time_midp":
        return ds.drop_vars(
            ["time_midp"] + [i for i in ds.data_vars if "time_midp" in ds[i].dims]
        )


@pytest.fixture
def test_create_tree_cartesian(xr_curv):
    sd.utils.create_tree(xr_curv["XC"], xr_curv["YC"], R=None)


@pytest.mark.parametrize(
    "incomplete_data", ["drop_YG", "drop_dyG", "drop_time_midp"], indirect=True
)
def test_incomplete_data(incomplete_data):
    oo = sd.OceData(incomplete_data)
    hrel = oo.find_rel_h(np.array([-14]), np.array([70.5]))
    assert isinstance(hrel, sd.ocedata.HRel)


@pytest.mark.parametrize("ds", ["curv"], indirect=True)
def test_auto_alias(ds):
    with pytest.raises(NotImplementedError):
        sd.OceData(ds, alias="auto")


@pytest.mark.parametrize("ds", ["curv"], indirect=True)
def test_manual_alias(ds):
    an_alias = {
        "dXC": "dxC",
        "dYC": "dyC",
        "dZ": "drC",
        "dXG": "dxG",
        "dYG": "dyG",
        "dZl": "drF",
        "SALT": "S",
    }
    od = sd.OceData(ds, an_alias)
    od["SALT"] = xr.ones_like(od._ds["XC"])
    try:
        od.show_alias()
    except ImportError:
        with pytest.raises(NameError):
            od.show_alias()


@pytest.mark.parametrize("od", ["ecco"], indirect=True)
def test_add_missing_grid(od):
    # TODO
    od._add_missing_grid()


@pytest.mark.parametrize("ds", ["curv"], indirect=True)
def test_not_h_ready(ds):
    temp = ds.drop_vars(["XG", "YG"])
    with pytest.raises(ValueError):
        sd.OceData(temp)


@pytest.mark.parametrize("incomplete_data", ["drop_YG"], indirect=True)
@pytest.mark.parametrize("lat", list(np.linspace(70, 70.618, 3)))
@pytest.mark.parametrize("lon", list(np.linspace(-17, -13.618, 3)))
def test_accurate_reproduce_local_cartesian(incomplete_data, lat, lon):
    tub = sd.OceData(incomplete_data)
    lon = np.array([lon])
    lat = np.array([lat])
    hrel = tub.find_rel_h(lon, lat)
    assert isinstance(hrel, sd.ocedata.HRel)
    face, iy, ix, rx, ry, cs, sn, dx, dy, bx, by = hrel.values()
    new_lon, new_lat = sd.utils.rel2latlon(rx, ry, cs, sn, dx, dy, bx, by)
    assert np.allclose(new_lon, lon)
    assert np.allclose(new_lat, lat)


@pytest.mark.parametrize("od", ["aviso"], indirect=True)
@pytest.mark.parametrize("lat", list(np.linspace(-74.618, -44.0, 3)))
@pytest.mark.parametrize("lon", list(np.linspace(-179.618, 179.618, 4)))
def test_accurate_reproduce_rectilinear(od, lat, lon):
    lon = np.array([lon])
    lat = np.array([lat])
    hrel = od.find_rel_h(lon, lat)
    assert isinstance(hrel, sd.ocedata.HRel)
    face, iy, ix, rx, ry, cs, sn, dx, dy, bx, by = hrel.values()
    new_lon, new_lat = sd.utils.rel2latlon(rx, ry, cs, sn, dx, dy, bx, by)
    assert np.allclose(new_lon, lon)
    assert np.allclose(new_lat, lat)
