import numpy as np
import pytest

import seaduck.kernelNweight as kw
import seaduck as sd

sd.rcParam["debug_level"] = "very_high"


@pytest.fixture
def aknw():
    return kw.KnW(
        inheritance=[[0, 1, 2, 3, 4, 5, 6, 7, 8]],
        hkernel="interp",
        vkernel="linear",
        tkernel="dt",
    )


def test_same_size(aknw):
    assert not aknw.same_size(sd.lagrangian.uknw)


def test_equal(aknw):
    assert not aknw == 1


@pytest.mark.parametrize(
    "masked,ans",
    [
        (
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 0, 1, 1, 1, 1],
                    [1, 1, 0, 1, 0, 1, 1, 1, 1],
                    [1, 0, 1, 1, 0, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0, 1, 1, 1, 1],
                ]
            ),
            [[0], [1], [2], [3]],
        ),
        (np.array([[0, 1, 1, 1, 0, 1, 1, 1, 1]]), [[] for i in range(4)]),
    ],
)
def test_cascade(masked, ans):
    pk = kw.find_which_points_for_each_kernel(masked)
    assert ans == pk


@pytest.mark.parametrize(
    "rx,ry", [(np.array([0]), np.array([0])), (np.array([0.5]), np.array([0.08]))]
)
@pytest.mark.parametrize(
    "pk,clause",
    [
        ([[], [], [], []], "np.isnan(np.sum(w))"),
        ([[0], [], [], []], "np.allclose(1,np.sum(w))"),
        ([[], [], [0], []], "np.allclose(1,np.sum(w))"),
    ],
)
def test_cascade_weight(rx, ry, pk, clause):
    w = kw.get_weight_cascade(rx, ry, pk)
    assert eval(clause)


@pytest.mark.parametrize(
    "rx,ry,clause",
    [
        (np.array([0]), np.array([0]), "np.allclose(1,weight[0,0])"),
        (np.array([0.5]), np.array([0.08]), "np.allclose(1,np.sum(weight))"),
    ],
)
@pytest.mark.parametrize(
    "kernel",
    [
        np.array(
            [[0, 0], [0, 1], [0, 2], [0, -1], [0, -2], [-1, 0], [-2, 0], [1, 0], [2, 0]]
        ),
        np.array(
            [
                [0, 0],
                [0, 1],
                [0, -1],
                [-1, 0],
                [1, 0],
            ]
        ),
        np.array(
            [
                [0, 0],
                [0, -1],
                [-1, 0],
                [1, 0],
            ]
        ),
        np.array(
            [
                [0, 0],
                [-1, 0],
                [1, 0],
            ]
        ),
        np.array(
            [
                [0, 0],
                [0, 1],
                [0, -1],
            ]
        ),
        np.array([[0, 0]]),
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
    ],
)
def test_interp_func(kernel, rx, ry, clause):
    func = kw.get_func(kernel)
    weight = func(rx, ry)
    assert weight.shape == (1, len(kernel))
    assert eval(clause)


@pytest.mark.parametrize("ktype", ["dx", "dy"])
@pytest.mark.parametrize("horder", [0, 1])
def test_create_different_square(ktype, horder):
    k = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    kw.get_func(k, hkernel=ktype, h_order=horder)


@pytest.mark.parametrize("hkernel", ["dx", "dy"])
def test_auto_doll(hkernel):
    k = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    kw.auto_doll(k, hkernel=hkernel)


@pytest.mark.parametrize(
    "rx,ry", [(np.array([0]), np.array([0])), (np.array([0.5]), np.array([0.08]))]
)
@pytest.mark.parametrize(
    "kernel",
    [
        np.array(
            [[0, 0], [0, 1], [0, 2], [0, -1], [0, -2], [-1, 0], [-2, 0], [1, 0], [2, 0]]
        ),
        np.array(
            [
                [0, 0],
                [0, 1],
                [0, -1],
                [-1, 0],
                [1, 0],
            ]
        ),
        np.array(
            [
                [0, 0],
                [0, -1],
                [-1, 0],
                [1, 0],
            ]
        ),
        np.array(
            [
                [0, 0],
                [-1, 0],
                [1, 0],
            ]
        ),
    ],
)
@pytest.mark.parametrize("order", [1, 2])
def test_dx(kernel, rx, ry, order):
    func = kw.get_func(kernel, hkernel="dx", h_order=order)
    weight = func(rx, ry)
    assert weight.shape == (1, len(kernel))
    assert np.allclose(0, np.sum(weight))


@pytest.mark.parametrize(
    "rx,ry", [(np.array([0]), np.array([0])), (np.array([0.5]), np.array([0.08]))]
)
@pytest.mark.parametrize(
    "kernel",
    [
        np.array(
            [[0, 0], [0, 1], [0, 2], [0, -1], [0, -2], [-1, 0], [-2, 0], [1, 0], [2, 0]]
        ),
        np.array(
            [
                [0, 0],
                [0, 1],
                [0, -1],
                [-1, 0],
                [1, 0],
            ]
        ),
        np.array([[0, 0], [0, 1], [0, -1]]),
    ],
)
@pytest.mark.parametrize("order", [1, 2])
def test_dy(kernel, rx, ry, order):
    func = kw.get_func(kernel, hkernel="dy", h_order=order)
    weight = func(rx, ry)
    assert weight.shape == (1, len(kernel))
    assert np.allclose(0, np.sum(weight))


@pytest.mark.parametrize("ktype", ["dx", "dy"])
@pytest.mark.parametrize(
    "kernel,order",
    [
        (
            np.array(
                [
                    [0, 0],
                    [0, 1],
                    [0, 2],
                    [0, -1],
                    [0, -2],
                    [-1, 0],
                    [-2, 0],
                    [1, 0],
                    [2, 0],
                ]
            ),
            5,
        ),
        (
            np.array(
                [
                    [0, 0],
                    [0, 1],
                    [0, -1],
                    [-1, 0],
                    [1, 0],
                ]
            ),
            3,
        ),
        (np.array([[0, 0]]), 1),
        (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), 2),
    ],
)
def test_order_too_high_error(kernel, order, ktype):
    with pytest.raises(Exception):
        kw.kernel_weight(kernel, ktype=ktype, order=order)


def test_plot_kernel():
    try:
        import matplotlib

        kw.show_kernels()
    except ImportError:
        with pytest.raises(NameError):
            kw.show_kernels()


def test_dt_and_bottom_scheme(aknw):
    aknw.get_weight(np.array([0.618]), np.array([0.0618]))


@pytest.mark.parametrize("which", ["dx", "dy"])
def test_maxorder_dxdy_square(which):
    kkk = np.array(
        [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [0, 2], [2, 2], [2, 0], [2, 1]]
    )
    xorder = 0
    yorder = 0
    if which == "dx":
        xorder = 2
    else:
        yorder = 2
    func = kw.kernel_weight_s(kkk, xorder, yorder)
    func(np.array([0.0]), np.array([0.0]))
