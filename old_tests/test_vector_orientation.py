import numpy as np
from topology import topology
import oceanspy as ospy
import pytest

Datadir = "Data/"
ECCO_url = "{}catalog_ECCO.yaml".format(Datadir)
ecco = ospy.open_oceandataset.from_catalog("LLC", ECCO_url)
tp = topology(ecco)

mundane = np.array([[[ 1.,  1.]],[[ 0.,  0.]],[[ 0., -0.]],[[ 1.,  1.]]])

@pytest.mark.parametrize(
    'fface,cis',[
        (np.array([[1,1]]),True),
        (np.array([[1,2]]),True),
        (np.array([[10,2]]),False),
        (np.array([[6,10]]),False)
    ]
)
def test_4_matrix(fface,cis):
    ans = np.array(tp.four_matrix_for_uv(fface))
    if cis:
        assert np.allclose(ans,mundane)
    else:
        assert not np.allclose(ans,mundane)