import numpy as np
import pytest

import seaduck as sd
from seaduck import utils
from seaduck.lagrangian import ind_tend_uv


@pytest.fixture
def custom_pt():
    x = np.linspace(-50, -15, 200)
    y = np.ones_like(x) * 52.0
    z = np.ones_like(x) * (-9)
    t = np.ones_like(x)
    od = sd.OceData(utils.get_dataset("ecco"))
    return sd.Particle(
        x=x,
        y=y,
        z=z,
        t=t,
        data=od,
        uname="utrans",
        vname="vtrans",
        wname="wtrans",
        transport=True,
        save_raw=True,
    )


# @pytest.fixture
# def s_list(custom_pt, vec_dict):
#     prfh,is_scalar = vec_dict['sm']
#     return np.array(read_u_list(custom_pt, prefetch = prfh,scalar = is_scalar))


@pytest.mark.parametrize(
    "ind,exp",
    [
        ((0, 2, 45, 45), (0, 2, 45, 46)),
        ((1, 2, 45, 45), (1, 2, 46, 45)),
        ((1, 2, 89, 45), (0, 6, 44, 0)),
        ((0, 5, 45, 89), (1, 7, 0, 44)),
    ],
)
def test_ind_tend_uv(ind, exp):
    tub = sd.OceData(utils.get_dataset("ecco"))
    tp = tub.tp
    ans = ind_tend_uv(ind, tp)
    assert exp == ans
