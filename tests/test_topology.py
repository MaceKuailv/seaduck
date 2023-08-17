import numpy as np
import pytest

from seaduck.topology import Topology, llc_get_uv_mask_from_face, llc_mutual_direction

try:
    import numba

    print(numba)
    # just to avoid the line got deleted by pre-commit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


@pytest.mark.parametrize("face", [1, 2, 4, 5, 6, 7, 8, 10, 11])
@pytest.mark.parametrize("edge", [0, 1, 2, 3])
@pytest.mark.parametrize("tp", ["ecco"], indirect=True)
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
    "func,args",
    [
        ("get_the_other_edge", (0, 0)),
        ("mutual_direction", (0, 1)),
    ],
)
@pytest.mark.parametrize("ds", ["ecco"], indirect=True)
def test_not_applicable(ds, typ, func, args, error):
    tpp = Topology(ds, typ)
    with pytest.raises(error):
        getattr(tpp, func)(*args)


@pytest.mark.parametrize("face,edge", [(0, 1), (3, 1), (9, 3), (12, 3)])
@pytest.mark.parametrize("tp", ["ecco"], indirect=True)
def test_antarctica_error(tp, face, edge):
    with pytest.raises(IndexError):
        tp.get_the_other_edge(face, edge)


@pytest.mark.parametrize("tp", ["ecco"], indirect=True)
def test_mutual_face(tp):
    e1, _ = tp.mutual_direction(0, 1)
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
@pytest.mark.parametrize("tp", ["ecco"], indirect=True)
def test_llc_ind_tend(tp, ind, tend, result):
    res = tp.ind_tend(ind, tend)
    assert res == result


@pytest.mark.parametrize("tend", [0, 1, 2, 3])
@pytest.mark.parametrize("tp", ["aviso", "rect"], indirect=True)
def test_boxish_ind_tend(tp, tend):
    ind = tp.ind_tend((0, 0), tend)
    assert len(ind) == 2


@pytest.mark.parametrize("tend", [0, 3])
@pytest.mark.parametrize("tp", ["aviso", "rect"], indirect=True)
def test_boundary_case(tp, tend):
    out = tp.ind_tend((tp.ixmax, tp.iymax), tend)
    if tp.typ == "box" or tend == 0:
        assert -1 in out

    out = tp.ind_tend((-1, -1), tend)
    assert -1 in out


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
@pytest.mark.parametrize("tp", ["ecco"], indirect=True)
def test_4_matrix(tp, fface, cis):
    ans = np.array(tp.four_matrix_for_uv(fface))
    if cis:
        assert np.allclose(ans, mundane)
    else:
        assert not np.allclose(ans, mundane)


@pytest.mark.parametrize("ds", ["rect"], indirect=True)
def test_unable_to_cereate(ds):
    temp = ds.drop_vars("XC")
    with pytest.raises(KeyError):
        Topology(temp)


@pytest.mark.parametrize("ds", ["aviso"], indirect=True)
def test_create_without_time(ds):
    temp = ds.drop_vars("time")
    tp = Topology(temp)
    assert tp.itmax == 0


@pytest.mark.parametrize(
    "func,args,kwargs,error",
    [
        ("ind_tend", ((1, 45, 45), 0), {"cuvwg": "other"}, ValueError),
        ("ind_moves", ((1, 45, 45), ["left", "left"]), {}, ValueError),
    ],
)
@pytest.mark.parametrize("tp", ["ecco"], indirect=True)
def test_other_errors(tp, func, args, kwargs, error):
    with pytest.raises(error):
        getattr(tp, func)(*args, **kwargs)


@pytest.mark.parametrize("tp", ["ecco"], indirect=True)
def test_ind_moves_with1illegal(tp):
    res = tp.ind_moves((1, -1, 89), [0, 0])
    assert res == (-1, -1, -1)


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
@pytest.mark.parametrize("tp", ["ecco"], indirect=True)
def test_ind_tend_v(tp, ind, tend, ans):
    res = tp.ind_tend(ind, tend, cuvwg="V")
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
@pytest.mark.parametrize("tp", ["ecco"], indirect=True)
def test_ind_tend_u(tp, ind, tend, ans):
    res = tp.ind_tend(ind, tend, cuvwg="U")
    assert res == ans


@pytest.mark.parametrize("tp", ["ecco"], indirect=True)
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


@pytest.mark.parametrize(
    "face,nface,rot",
    [
        (4, 9, True),
        (3, 8, True),
        (9, 4, True),
        (8, 3, True),
        (0, 4, False),
        (1, 3, False),
    ],
)
def test_transitive_mututal_direction(face, nface, rot):
    e1, e2 = llc_mutual_direction(face, nface, transitive=True)
    isrot = (e1 // 2 + e2 // 2) % 2
    assert isrot == rot


@pytest.mark.skipif(
    HAS_NUMBA, reason="Weird numba behavior that needs to be revisited."
)
@pytest.mark.parametrize("transitive,face, nface", [(True, 0, 7), (False, 4, 9)])
def test_mutual_face_error(transitive, face, nface):
    with pytest.raises(ValueError):
        llc_mutual_direction(face, nface, transitive=transitive)


def test_uv_mask():
    faces = np.array([1, 1, 1, 1, 4])
    uu, uv, vu, vv = llc_get_uv_mask_from_face(faces)
    assert (uu == 1).all()


@pytest.mark.parametrize("tp", ["rect"], indirect=True)
def test_uv_mask_error(tp):
    faces = np.array([1, 1, 1, 1, 4])
    with pytest.raises(Exception):
        tp.get_uv_mask_from_face(faces)


@pytest.mark.parametrize("tp", ["ecco"], indirect=True)
def test_wall_between_itself(tp):
    with pytest.raises(IndexError):
        tp._find_wall_between((1, 14, 14), (1, 14, 14))

@pytest.mark.parametrize(
    "ind, tend, exp",
    [
        ((10,89,45),0,('U',(2,44,0))),
        ((7,0,45),1,('U',(5,44,89)))
    ]
)
@pytest.mark.parametrize("tp", ["ecco"], indirect=True)
def test_ind_tend_v(tp,ind,tend,exp):
    ans = tp._ind_tend_V(ind,tend)
    assert ans == exp

@pytest.mark.parametrize(
    "ind, tend, exp",
    [
        ((2,44,0),2,('V',(10,89,45))),
        ((5,44,89),3,('V',(7,0,45)))
    ]
)
@pytest.mark.parametrize("tp", ["ecco"], indirect=True)
def test_ind_tend_u(tp,ind,tend,exp):
    ans = tp._ind_tend_U(ind,tend)
    assert ans == exp
