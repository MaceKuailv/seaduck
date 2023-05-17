import numpy as np
import pytest
import xarray as xr

import seaduck.kernelNweight as kw
from seaduck.topology import topology

# Datadir = "tests/Data/"
# curv = xr.open_dataset("{}MITgcm_curv_nc.nc" "".format(Datadir))
# rect = xr.open_dataset("{}MITgcm_rect_nc.nc" "".format(Datadir))
# ecco = xr.open_zarr(Datadir + "small_ecco")


@pytest.fixture
def tp(xr_ecco):
    return topology(xr_ecco)


@pytest.fixture
def xr_rect(xr_rect):
    return xr_rect


@pytest.fixture
def xr_curv(xr_curv):
    return xr_curv


@pytest.mark.parametrize("face", [1, 2, 4, 5, 6, 7, 8, 10, 11])
@pytest.mark.parametrize("edge", [0, 1, 2, 3])
def test_get_the_neighbor_face(tp, face, edge):
    nf, ne = tp.get_the_other_edge(face, edge)
    assert nf in range(13)
    assert ne in range(4)


@pytest.mark.parametrize(
    "typ,error",
    [
        ("box", Exception),
        ("x_periodic", Exception),
        ("cubed_sphere", NotImplementedError),
        ("other", NotImplementedError),
    ],
)
@pytest.mark.parametrize(
    "func", ["tpp.get_the_other_edge(0,0)", "tpp.mutual_direction(0,1)"]
)
def test_not_applicable(xr_ecco, typ, func, error):
    tpp = topology(xr_ecco, typ)
    with pytest.raises(error):
        eval(func)


@pytest.mark.parametrize("face,edge", [(0, 1), (3, 1), (9, 3), (12, 3)])
def test_antarctica_error(tp, face, edge):
    with pytest.raises(IndexError):
        nf, ne = tp.get_the_other_edge(face, edge)


def test_mutual_face(tp):
    e1, e2 = tp.mutual_direction(0, 1)
    assert e1 in [0, 1, 2, 3]


@pytest.mark.parametrize(
    "ind,tend,result",
    [
        ((1, 45, 45), 0, (1, 46, 45)),
        ((1, 45, 45), 1, (1, 44, 45)),
        ((1, 45, 45), 2, (1, 45, 44)),
        ((1, 45, 45), 3, (1, 45, 46)),
        ((5, 89, 89), 3, (7, 0, 0)),
        ((5, 89, 89), 0, (6, 0, 89)),
        ((6, 89, 89), 0, (10, 0, 0)),
        ((6, 0, 0), 2, (2, 89, 89)),
    ],
)
def test_llc_ind_tend(tp, ind, tend, result):
    res = tp.ind_tend(ind, tend)
    assert res == result


@pytest.mark.parametrize("ds", ["xr_rect", "xr_curv"])
def test_other_ind_tend(ds, xr_rect, xr_curv):
    ds = eval(ds)
    temp_tp = topology(ds)
    temp_tp.ind_tend((0, 0), 3)


mundane = np.array([[[1.0, 1.0]], [[0.0, 0.0]], [[0.0, -0.0]], [[1.0, 1.0]]])


@pytest.mark.parametrize(
    "fface,cis",
    [
        (np.array([[1, 1]]), True),
        (np.array([[1, 2]]), True),
        (np.array([[10, 2]]), False),
        (np.array([[6, 10]]), False),
    ],
)
def test_4_matrix(tp, fface, cis):
    ans = np.array(tp.four_matrix_for_uv(fface))
    if cis:
        assert np.allclose(ans, mundane)
    else:
        assert not np.allclose(ans, mundane)


@pytest.mark.parametrize(
    "stmt,error",
    [
        ("tp.ind_tend((1,45,45),0,cuvg = 'G')", NotImplementedError),
        ("tp.ind_tend((1,45,45),0,cuvg = 'other')", ValueError),
        ("tp.ind_moves((1,45,45),['left','left'])", ValueError),
    ],
)
def test_other_errors(tp, stmt, error):
    with pytest.raises(error):
        eval(stmt)


def test_ind_moves_with1illegal(tp):
    tp.ind_moves((1, -1, 89), [0, 0])


@pytest.mark.parametrize(
    "ind,tend,ans",
    [
        ((1, 45, 45), 0, (1, 46, 45)),
        ((1, 45, 45), 2, (1, 45, 44)),
        ((1, 0, 0), 2, (12, 89, 0)),
        ((1, 0, 0), 1, (0, 89, 0)),
        ((4, 45, 89), 3, (8, 0, 45)),
    ],
)
def test_ind_tend_v(tp, ind, tend, ans):
    res = tp.ind_tend(ind, tend, cuvg="V")
    assert res == ans


@pytest.mark.parametrize(
    "ans,tend,ind",
    [
        ((1, 45, 45), 1, (1, 46, 45)),
        ((1, 45, 44), 2, (1, 45, 45)),
        ((1, 0, 0), 0, (12, 89, 0)),
        ((4, 45, 89), 1, (8, 0, 45)),
    ],
)
def test_ind_tend_u(tp, ind, tend, ans):
    res = tp.ind_tend(ind, tend, cuvg="U")
    assert res == ans


def test_wall_between(tp):
    # it is a bit hard to think about an example
    # that uses this case from higher level.
    uv, R = tp._find_wall_between((11, 0, 45), (8, 89, 45))
    assert uv == "V"
    assert R == (11, 0, 45)
    # It does not make sense to parametrize now.
    uv, R = tp._find_wall_between((8, 45, 0), (7, 45, 89))
    assert uv == "U"
    assert R == (8, 45, 0)


# def test_fatten_ind_h_ecco():
#     faces = np.array([0, 0])
#     iys = np.array([45, 46])
#     ixs = np.array([45, 46])
#     tp = topology(ecco)
#     nface, niy, nix = kw.fatten_ind_h(faces, iys, ixs, tp)
#     assert nface.dtype == "int"
#     assert nface.shape == (2, 9)


# @pytest.mark.parametrize("od", [rect, curv])
# def test_fatten_ind_h_other(od):
#     faces = None
#     iys = np.array([5, 46])
#     ixs = np.array([5, 6])
#     tp = topology(od)
#     nface, niy, nix = kw.fatten_ind_h(faces, iys, ixs, tp)
#     assert nface is None
#     assert nix.dtype == "int"
#     assert niy.shape == (2, 9)


# def test_fatten_ind_3d_ecco():
#     izs = np.array([9, 10])
#     faces = np.array([0, 0])
#     iys = np.array([45, 46])
#     ixs = np.array([45, 46])
#     tp = topology(ecco)
#     niz, nface, niy, nix = kw.fatten_ind_3d(izs, faces, iys, ixs, tp)
#     assert niz.dtype == "int"
#     assert niz.shape == (2, 18)


# @pytest.mark.parametrize("od", [rect, curv])
# def test_fatten_ind_3d_other(od):
#     izs = np.array([9, 10])
#     faces = None
#     iys = np.array([5, 46])
#     ixs = np.array([5, 46])
#     tp = topology(od)
#     niz, nface, niy, nix = kw.fatten_ind_3d(izs, faces, iys, ixs, tp)
#     assert niz.dtype == "int"
#     assert niz.shape == (2, 18)


# if __name__ == '__main__':
#     test_fatten_ind_3d()
