from topology import topology
import kernel_and_weight as kw
import numpy as np
import pytest 
import oceanspy as ospy

Datadir = "Data/"
ECCO_url = "{}catalog_ECCO.yaml".format(Datadir)
curv = ospy.open_oceandataset.from_netcdf("{}MITgcm_curv_nc.nc" "".format(Datadir))
rect = ospy.open_oceandataset.from_netcdf("{}MITgcm_rect_nc.nc" "".format(Datadir))
ECCO_url = "{}catalog_ECCO.yaml".format(Datadir)
ecco = ospy.open_oceandataset.from_catalog("LLC", ECCO_url)

@pytest.mark.parametrize(
    'face',[
        1,2,4,5,6,7,8,10,11
    ]
)
@pytest.mark.parametrize(
    'edge',[
        0,1,2,3
    ]
)
def test_get_the_neighbor_face(face,edge):
    tp = topology(ecco)
    nf,ne = tp.get_the_other_edge(face,edge)
    assert nf in range(13)
    assert ne in range(4)

@pytest.mark.parametrize(
    'typ,error',[
        ('box',Exception),
        ('x_periodic',Exception),
        ('cubed_sphere',NotImplementedError),
        ('other',NotImplementedError)
    ]
)
@pytest.mark.parametrize(
    'func',[
        'tp.get_the_other_edge(0,0)',
        'tp.mutual_direction(0,1)'
    ]
)
def test_not_applicable(typ,func,error):
    tp = topology(ecco,typ)
    with pytest.raises(error):
        eval(func)

@pytest.mark.parametrize(
    'face',[
        0,3,9,12
    ]
)
def test_antarctica_error(face):
    tp = topology(ecco)
    with pytest.raises(Exception):
        nf,ne = tp.get_the_other_edge(face,edge)
        
def test_mutual_face():
    tp = topology(ecco)
    e1,e2 = tp.mutual_direction(0,1)
    assert isinstance(e1,int)

@pytest.mark.parametrize(
    'ind,tend,result',[
        ((1,45,45),0,(1,46,45)),
        ((1,45,45),1,(1,44,45)),
        ((1,45,45),2,(1,45,44)),
        ((1,45,45),3,(1,45,46)),
        ((5,89,89),3,(7,0,0)),
        ((5,89,89),0,(6,0,89)),
        ((6,89,89),0,(10,0,0)),
        ((6,0,0),2,(2,89,89))
    ]
)
def test_llc_ind_tend(ind,tend,result):
    tp = topology(ecco)
    res = tp.ind_tend(ind,tend)
    assert res == result
    
def test_fatten_ind_h_ecco():
    faces = np.array([0,0])
    iys = np.array([45,46])
    ixs = np.array([45,46])
    tp = topology(ecco)
    nface,niy,nix = kw.fatten_ind_h(faces,iys,ixs,tp)
    assert nface.dtype == 'int'
    assert nface.shape == (2,9)
    
@pytest.mark.parametrize(
    'od',[rect,curv]
)
def test_fatten_ind_h_other(od):
    faces = None
    iys = np.array([5,46])
    ixs = np.array([5,6])
    tp = topology(od)
    nface,niy,nix = kw.fatten_ind_h(faces,iys,ixs,tp)
    assert nface is None
    assert nix.dtype == 'int'
    assert niy.shape == (2,9)
    
def test_fatten_ind_3d_ecco():
    izs =  np.array([9,10])
    faces = np.array([0,0])
    iys = np.array([45,46])
    ixs = np.array([45,46])
    tp = topology(ecco)
    niz,nface,niy,nix = kw.fatten_ind_3d(izs,faces,iys,ixs,tp)
    assert niz.dtype == 'int'
    assert niz.shape == (2,18)
    
@pytest.mark.parametrize(
    'od',[rect,curv]
)
def test_fatten_ind_3d_other(od):
    izs =  np.array([9,10])
    faces = None
    iys = np.array([5,46])
    ixs = np.array([5,46])
    tp = topology(od)
    niz,nface,niy,nix = kw.fatten_ind_3d(izs,faces,iys,ixs,tp)
    assert niz.dtype == 'int'
    assert niz.shape == (2,18)
    
# if __name__ == '__main__':
#     test_fatten_ind_3d()