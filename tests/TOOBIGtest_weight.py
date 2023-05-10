import seaduck.kernelNweight as kw 
import pytest
import numpy as np

@pytest.mark.parametrize(
    'masked,ans',[
        (np.array([[1,1,1,1,1,1,1,1,1],
                   [1,1,1,1,0,1,1,1,1],
                   [1,1,0,1,0,1,1,1,1],
                   [1,0,1,1,0,1,1,1,1],
                   [0,1,1,1,0,1,1,1,1],]),[[0],[1],[2],[3]]),
        (np.array([[0,1,1,1,0,1,1,1,1]]),[[] for i in range(4)])]
)
def test_cascade(masked,ans):
    pk = kw.find_which_points_for_each_kernel(masked)
    assert ans == pk

@pytest.mark.parametrize(
    'rx,ry',[
        (np.array([0]),np.array([0])),
        (np.array([0.5]),np.array([.08]))
    ]
)
@pytest.mark.parametrize(
    'pk,clause',[
        ([[],[],[],[]],'np.isnan(np.sum(w))'),
        ([[0],[],[],[]],'np.allclose(1,np.sum(w))'),
        ([[],[],[0],[]],'np.allclose(1,np.sum(w))'),
    ]
)
def test_cascade_weight(rx,ry,pk,clause):
    w = kw.get_weight_cascade(rx,ry,pk)
    assert eval(clause)

@pytest.mark.parametrize(
    'rx,ry,clause',[
        (np.array([0]),np.array([0]),'np.allclose(1,weight[0,0])'),
        (np.array([0.5]),np.array([.08]),'np.allclose(1,np.sum(weight))')
    ]
)
@pytest.mark.parametrize(
    'kernel',[
        np.array([
                [0,0],
                [0,1],
                [0,2],
                [0,-1],
                [0,-2],
                [-1,0],
                [-2,0],
                [1,0],
                [2,0]
        ]),
        np.array([
                [0,0],
                [0,1],
                [0,-1],
                [-1,0],
                [1,0],
        ]),
        np.array([
                [0,0],
                [0,-1],
                [-1,0],
                [1,0],
        ]),
        np.array([
                [0,0],
                [-1,0],
                [1,0],
        ]),
        np.array([
                [0,0],
                [0,1],
                [0,-1],
        ]),
        np.array([
                [0,0]
        ]),
    ]
)
def test_interp_func(kernel,rx,ry,clause):
    func = kw.kernel_weight_x(kernel)
    weight= func(rx,ry)
    assert weight.shape == (1,len(kernel))
    assert eval(clause)
    
@pytest.mark.parametrize(
    'rx,ry',[
        (np.array([0]),np.array([0])),
        (np.array([0.5]),np.array([.08]))
    ]
)
@pytest.mark.parametrize(
    'kernel',[
        np.array([
                [0,0],
                [0,1],
                [0,2],
                [0,-1],
                [0,-2],
                [-1,0],
                [-2,0],
                [1,0],
                [2,0]
        ]),
        np.array([
                [0,0],
                [0,1],
                [0,-1],
                [-1,0],
                [1,0],
        ]),
        np.array([
                [0,0],
                [0,-1],
                [-1,0],
                [1,0],
        ]),
        np.array([
                [0,0],
                [-1,0],
                [1,0],
        ]),
    ]
)
@pytest.mark.parametrize(
    'order',[1,2]
)
def test_dx(kernel,rx,ry,order):
    func = kw.kernel_weight_x(kernel,ktype = 'x',order = order)
    weight= func(rx,ry)
    assert weight.shape == (1,len(kernel))
    assert np.allclose(0,np.sum(weight))
    
@pytest.mark.parametrize(
    'rx,ry',[
        (np.array([0]),np.array([0])),
        (np.array([0.5]),np.array([.08]))
    ]
)
@pytest.mark.parametrize(
    'kernel',[
        np.array([
                [0,0],
                [0,1],
                [0,2],
                [0,-1],
                [0,-2],
                [-1,0],
                [-2,0],
                [1,0],
                [2,0]
        ]),
        np.array([
                [0,0],
                [0,1],
                [0,-1],
                [-1,0],
                [1,0],
        ]),
        np.array([
                [0,0],
                [0,1],
                [0,-1]
        ]),
    ]
)
@pytest.mark.parametrize(
    'order',[1,2]
)
def test_dy(kernel,rx,ry,order):
    func = kw.kernel_weight_x(kernel,ktype = 'y',order = order)
    weight= func(rx,ry)
    assert weight.shape == (1,len(kernel))
    assert np.allclose(0,np.sum(weight))

@pytest.mark.parametrize(
    'ktype',['x','y']
)
@pytest.mark.parametrize(
    'kernel,order',[
        (np.array([
                [0,0],
                [0,1],
                [0,2],
                [0,-1],
                [0,-2],
                [-1,0],
                [-2,0],
                [1,0],
                [2,0]
        ]),5),
        (np.array([
                [0,0],
                [0,1],
                [0,-1],
                [-1,0],
                [1,0],
        ]),3),
        (np.array([
                [0,0]
        ]),1)
    ]
)
def test_order_too_high_error(kernel,order,ktype):
    with pytest.raises(Exception):
        kw.kernel_weight_x(kernel,ktype = ktype,order = order)