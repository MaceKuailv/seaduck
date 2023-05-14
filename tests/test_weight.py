import numpy as np
import pytest

import seaduck.kernelNweight as kw


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
    kw.get_weight_cascade(rx, ry, pk)
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
        np.array([
            [0,0],
            [0,1],
            [1,0],
            [1,1]
        ])
    ],
)
def test_interp_func(kernel, rx, ry, clause):
    func = kw.get_func(kernel)
    weight = func(rx, ry)
    assert weight.shape == (1, len(kernel))
    assert eval(clause)

@pytest.mark.parametrize(
    'ktype',['dx','dy']
)
@pytest.mark.parametrize(
    'horder',[0,1]
)
def test_create_different_square(ktype,horder):
    k = np.array([
            [0,0],
            [0,1],
            [1,0],
            [1,1]
        ])
    kw.get_func(k,hkernel= ktype,h_order = horder)

@pytest.mark.parametrize(
    'hkernel',['dx','dy']
)
def test_auto_doll(hkernel):
    k = np.array([
            [0,0],
            [0,1],
            [1,0],
            [1,1]
        ])
    kw.auto_doll(k,hkernel = hkernel)


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
        (
            np.array([
            [0,0],
            [0,1],
            [1,0],
            [1,1]
        ]),2
        )
    ],
)
def test_order_too_high_error(kernel, order, ktype):
    with pytest.raises(Exception):
        kw.kernel_weight(kernel, ktype=ktype, order=order)
