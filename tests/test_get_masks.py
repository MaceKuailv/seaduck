import numpy as np
import pytest
import xarray as xr

import seaduck.get_masks as gm
import seaduck.topology as topology
from seaduck import OceData

# TODO: have a dataset that actually has maskC and is also not ECCO in the test datasets


# Datadir = "tests/Data/"


# ecco = xr.open_zarr(Datadir + "small_ecco")
@pytest.fixture
def tp(xr_ecco):
    return topology(xr_ecco)


@pytest.fixture
def masks(ecco, tp):
    return gm.get_masks(ecco, tp)


def test_maskC_contains_others(masks):
    """
    if one of the u,v,w is not defined, this cell must be a solid
    or in other words, the maskC land is contained in other lands.
    """
    maskC, maskU, maskV, maskW = masks
    assert np.logical_or(np.logical_not(maskC), maskU).all()
    assert np.logical_or(np.logical_not(maskC), maskV).all()
    assert np.logical_or(np.logical_not(maskC), maskW).all()


def test_not_the_same(masks):
    maskC, maskU, maskV, maskW = masks
    assert not np.allclose(maskC, maskU)
    assert not np.allclose(maskC, maskV)
    assert not np.allclose(maskC, maskW)


@pytest.mark.parametrize("od", ["rect", "curv"])
def test_without_maskC(od, rect, curv):
    od = eval(od)
    with pytest.warns(Warning):
        maskC, maskU, maskV, maskW = gm.get_masks(od, od.tp)
    assert maskU.all()


@pytest.mark.parametrize("od", ["rect", "curv"])
def test_get_masked_without_maskC(od, rect, curv):
    od = eval(od)
    with pytest.warns(Warning):
        hello = gm.get_masked(od, (1, 1))
    assert hello == 1
