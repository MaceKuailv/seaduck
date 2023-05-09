import numpy as np
import pytest
import get_masks as gm
import oceanspy as ospy

# TODO: have a dataset that actually has maskC and is also not ECCO in the test datasets

Datadir = "Data/"
ECCO_url = "{}catalog_ECCO.yaml".format(Datadir)
curv = ospy.open_oceandataset.from_netcdf("{}MITgcm_curv_nc.nc" "".format(Datadir))
rect = ospy.open_oceandataset.from_netcdf("{}MITgcm_rect_nc.nc" "".format(Datadir))
ecco = ospy.open_oceandataset.from_catalog("LLC", ECCO_url)

maskC,maskU,maskV,maskW = gm.get_masks(ecco)

def test_maskC_contains_others():
    '''
    if one of the u,v,w is not defined, this cell must be a solid
    or in other words, the maskC land is contained in other lands. 
    '''
    assert np.logical_or(np.logical_not(maskC),maskU).all()
    assert np.logical_or(np.logical_not(maskC),maskV).all()
    assert np.logical_or(np.logical_not(maskC),maskW).all()
    
def test_not_the_same():
    assert not np.allclose(maskC,maskU)
    assert not np.allclose(maskC,maskV)
    assert not np.allclose(maskC,maskW)
    
@pytest.mark.parametrize('od',[rect,curv])
def test_without_maskC(od):
    with pytest.warns(Warning):
        maskC,maskU,maskV,maskW = gm.get_masks(od)
    assert maskU.all()