import numpy as np
import pytest

import seaduck as sd
import seaduck.get_masks as gm
from seaduck import utils


# TODO: have a dataset that actually has maskC and is also not ECCO in the test datasets
@pytest.fixture
def masks():
    ds = utils.get_dataset("ecco")
    return gm.get_mask_arrays(sd.OceData(ds))


def test_maskC_contains_others(masks):
    # if one of the u,v,w is not defined, this cell must be a solid
    # or in other words, the maskC land is contained in other lands.
    maskC, maskU, maskV, maskW = masks
    assert np.logical_or(np.logical_not(maskC), maskU).all()
    assert np.logical_or(np.logical_not(maskC), maskV).all()
    assert np.logical_or(np.logical_not(maskC), maskW).all()


def test_not_the_same(masks):
    maskC, maskU, maskV, maskW = masks
    assert not np.allclose(maskC, maskU)
    assert not np.allclose(maskC, maskV)
    assert not np.allclose(maskC, maskW)


@pytest.mark.parametrize("od", ["rect", "curv"], indirect=True)
def test_without_maskC(od):
    with pytest.warns(UserWarning):
        _, maskU, *_ = gm.get_mask_arrays(od)
    assert maskU.all()


@pytest.mark.parametrize("od", ["rect", "curv"], indirect=True)
def test_get_masked_without_maskC(od):
    with pytest.warns(UserWarning):
        hello = gm.get_masked(od, (1, 1))
    assert hello == 1
