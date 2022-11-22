import numpy as np
from numba import njit
from scipy import spatial

@njit
def to_180(x):
    '''
    convert any longitude scale to [-180,180)
    '''
    x = x%360
    return x+(-1)*(x//180)*360

@njit
def spherical2cartesian(Y, X, R=6371.0):
    """
    Convert spherical coordinates to cartesian.
    Parameters
    ----------
    Y: np.array
        Spherical Y coordinate (latitude)
    X: np.array
        Spherical X coordinate (longitude)
    R: scalar
        Earth radius in km
        If None, use geopy default
    Returns
    -------
    x: np.array
        Cartesian x coordinate
    y: np.array
        Cartesian y coordinate
    z: np.array
        Cartesian z coordinate
    """

    # Convert
    Y_rad = np.deg2rad(Y)
    X_rad = np.deg2rad(X)
    x = R * np.cos(Y_rad) * np.cos(X_rad)
    y = R * np.cos(Y_rad) * np.sin(X_rad)
    z = R * np.sin(Y_rad)

    return x, y, z

@njit
def find_ind_z(array, value):
    '''
    find the nearest level that is lower than the given level
    '''
    array = np.asarray(array)
    idx = np.argmin(np.abs(array - value))
    if array[idx]>value:
    #z is special because it does not make much sense to interpolate beyond the two layers
        idx+=1
    idx = int(idx)
    return idx,array[idx]

@njit
def find_ind_t(array, value):
    '''
    find the latest time that is before the given time
    '''
    array = np.asarray(array)
    idx = np.argmin(np.abs(array - value))
    if array[idx]>value:
        idx-=1
    idx = int(idx)
    return idx,array[idx]

@njit
def find_ind_nearest(array,value):
    '''
    just find the nearest
    '''
    array = np.asarray(array)
    idx = np.argmin(np.abs(array - value))
    idx = int(idx)
    return idx,array[idx]
    

deg2m = 6271e3*np.pi/180
def find_ind_h(Xs,Ys,tree,h_shape):
    '''
    use ckd tree to find the indexes,
    2-index case can be thinked about as having only 1 face,
    we don't support that yet. but i think it would be easy.
    '''
    x,y,z = spherical2cartesian(Ys,Xs)
    _,index1d = tree.query(
                np.array([x,y,z]).T
            )
    if len(h_shape) == 3:
        faces,iys,ixs = np.unravel_index((index1d), h_shape)    
    elif len(h_shape) ==2:
        faces = None
        iys,ixs = np.unravel_index((index1d), h_shape)  
    return faces,iys,ixs

@njit
def find_rel_nearest(value,ts):
    its = np.zeros_like(value)
    rts = np.ones_like(value)*0.0#the way to create zeros with float32 type
    dts = np.ones_like(value)*0.0
    bts = np.ones_like(value)*0.0
    
    DT = np.zeros(len(ts)+1)
    DT[1:-1] = ts[1:] - ts[:-1]
    DT[0] = DT[1]
    DT[-1] = DT[-2]
    for i in range(len(value)):
        t = value[i]
        it,bt = find_ind_nearest(ts,t)
        delta_t = t-bt
        if delta_t*DT[i]>0:   
            Delta_t = DT[it+1]
        else:
            Delta_t = DT[it]
        rt = delta_t/abs(Delta_t)
        its[i] = it 
        rts[i] = rt
        dts[i] = abs(Delta_t)
        bts[i] = bt
    return its,rts,dts,bts

@njit
def find_rel_z(depth,some_z,some_dz):
    '''
    iz = the index
    rz  = how_much_higher_than_node/cell_size
    dz = cell_size
    '''
    izs = np.zeros_like(depth)
    rzs = np.ones_like(depth)*0.0#the way to create zeros with float32 type
    dzs = np.ones_like(depth)*0.0
    bzs = np.ones_like(depth)*0.0
    for i,d in enumerate(depth):
        iz,bz = find_ind_z(some_z,d)
        izs[i] = iz 
        bzs[i] = bz
#         try:
        delta_z = d-bz
#         except IndexError:
#             raise IndexError('the point is too deep')
        Delta_z = some_dz[iz]
        dzs[i] = Delta_z
        rzs[i] = delta_z/Delta_z
    return izs,rzs,dzs,bzs

@njit
def find_rel_time(time,ts):
    '''
    it = the index
    rt  = how_much_later_than_the_closest_time/time_interval
    dt = time_interval
    '''
    its = np.zeros(time.shape)
    rts = np.ones(time.shape)*0.0
    dts = np.ones(time.shape)*0.0
    bts = np.ones(time.shape)*0.0
    
    for i,t in enumerate(time):
        it,bt = find_ind_t(ts,t)
        delta_t = t-bt
        Delta_t = ts[it+1]-ts[it]
        rt = delta_t/Delta_t
        its[i] = it 
        rts[i] = rt
        dts[i] = Delta_t
        bts[i] = bt
    return its,rts,dts,bts

@njit
def read_h_with_face(some_x,some_y,some_dx,some_dy,CS,SN,faces,iys,ixs):
    '''
    read find_rel_h for more info,
    
    '''
    n = len(ixs)
    dx = np.ones_like(ixs)*0.0
    dy = np.ones_like(ixs)*0.0
    bx = np.ones_like(ixs)*0.0
    by = np.ones_like(ixs)*0.0
    cs = np.ones_like(ixs)*0.0
    sn = np.ones_like(ixs)*0.0
    for i in range(n):
        
        bx[i] = some_x[faces[i],iys[i],ixs[i]]
        by[i] = some_y[faces[i],iys[i],ixs[i]]

        cs[i] = CS[faces[i],iys[i],ixs[i]]
        sn[i] = SN[faces[i],iys[i],ixs[i]]
        
        dx[i] = some_dx[faces[i],iys[i],ixs[i]]
        dy[i] = some_dy[faces[i],iys[i],ixs[i]]
    
    return cs,sn,dx,dy,bx,by

@njit
def read_h_without_face(some_x,some_y,some_dx,some_dy,CS,SN,iys,ixs):
    '''
    read find_rel_h for more info,
    
    '''
    n = len(ixs)
    dx = np.ones_like(ixs)*0.0
    dy = np.ones_like(ixs)*0.0
    bx = np.ones_like(ixs)*0.0
    by = np.ones_like(ixs)*0.0
    cs = np.ones_like(ixs)*0.0
    sn = np.ones_like(ixs)*0.0
    for i in range(n):
        
        bx[i] = some_x[iys[i],ixs[i]]
        by[i] = some_y[iys[i],ixs[i]]
        
        cs[i] = CS[iys[i],ixs[i]]
        sn[i] = SN[iys[i],ixs[i]]
        
        dx[i] = some_dx[iys[i],ixs[i]]
        dy[i] = some_dy[iys[i],ixs[i]]
        
    
    return cs,sn,dx,dy,bx,by

@njit
def find_rx_ry_naive(x,y,bx,by,cs,sn,dx,dy):
    dlon = to_180(x - bx)
    dlat = to_180(y - by)
    rx = (dlon*np.cos(by*np.pi/180)*cs+dlat*sn)*deg2m/dx
    ry = (dlat*cs-dlon*sn*np.cos(by*np.pi/180))*deg2m/dy
    return rx,ry

def find_rel_h_naive(Xs,Ys,some_x,some_y,some_dx,some_dy,CS,SN,tree):
    '''
    very similar to find_rel_time/v
    rx,ry,dx,dy are defined the same way
    for example
    rx = "how much to the right of the node"/"size of the cell in left-right direction"
    dx = "size of the cell in left-right direction"
    
    cs,sn is just the cos and sin of the grid orientation.
    It will come in handy when we transfer vectors.
    '''
    h_shape = some_x.shape
    faces,iys,ixs = find_ind_h(Xs,
                               Ys,
                               tree,
                               h_shape
                              )
    if faces is not None:
        cs,sn,dx,dy,bx,by = read_h_with_face(  some_x,
                                               some_y,
                                               some_dx,
                                               some_dy,
                                               CS,
                                               SN,
                                               faces,
                                               iys,
                                               ixs)
    else:
        cs,sn,dx,dy,bx,by = read_h_without_face(some_x,
                                               some_y,
                                               some_dx,
                                               some_dy,
                                               CS,
                                               SN,
                                               iys,
                                               ixs)
    rx,ry = find_rx_ry_naive(Xs,Ys,bx,by,cs,sn,dx,dy)
    return faces,iys,ixs,rx,ry,cs,sn,dx,dy,bx,by

def find_rel_h_oceanparcel(x,y,some_x,some_y,some_dx,some_dy,CS,SN,XG,YG,tree,tp):
    h_shape = some_x.shape
    faces,iys,ixs = find_ind_h(x,
                               y,
                               tree,
                               h_shape
                              )
    if faces is not None:
        cs,sn,dx,dy,bx,by = read_h_with_face(  some_x,
                                               some_y,
                                               some_dx,
                                               some_dy,
                                               CS,
                                               SN,
                                               faces,
                                               iys,
                                               ixs)
        px,py = find_px_py(XG,YG,tp,faces,
                                               iys,
                                               ixs)
    else:
        cs,sn,dx,dy,bx,by = read_h_without_face(some_x,
                                               some_y,
                                               some_dx,
                                               some_dy,
                                               CS,
                                               SN,
                                               iys,
                                               ixs)
        px,py = find_px_py(XG,YG,tp,iys,
                                               ixs)
    rx,ry = find_rx_ry_oceanparcel(x,y,px,py)
    return faces,iys,ixs,rx,ry,cs,sn,dx,dy,bx,by
    

def find_cs_sn(thetaA,phiA,thetaB,phiB):
    '''
    theta is the angle 
    between the meridian crossing point A
    and the geodesic connecting A and B
    
    this function return cos and sin of theta
    '''
    # O being north pole
    AO = np.pi/2 - thetaA
    BO = np.pi/2 - thetaB
    dphi = phiB-phiA
    # Spherical law of cosine on AOB
    cos_AB = np.cos(BO)*np.cos(AO)+np.sin(BO)*np.sin(AO)*np.cos(dphi)
    sin_AB = np.sqrt(1-cos_AB**2)
    # spherical law of sine on triangle AOB
    SN = np.sin(BO)*np.sin(dphi)/sin_AB
    CS = np.sign(thetaB-thetaA)*np.sqrt(1-SN**2)
    return CS,SN

def find_px_py(XG,YG,tp,*ind,gridoffset = -1):
    N = len(ind[0])
    ind1 = tuple(i for i in tp.ind_tend_vec(ind,np.ones(N)*3,gridoffset = gridoffset))
    ind2 = tuple(i for i in tp.ind_tend_vec(ind1,np.zeros(N),gridoffset = gridoffset))
    ind3 = tuple(i for i in tp.ind_tend_vec(ind, np.zeros(N),gridoffset = gridoffset))
    
    x0 = XG[ind]
    x1 = XG[ind1]
    x2 = XG[ind2]
    x3 = XG[ind3]
    
    y0 = YG[ind]
    y1 = YG[ind1]
    y2 = YG[ind2]
    y3 = YG[ind3]
    
    px = np.vstack([x0,x1,x2,x3]).astype('float64')
    py = np.vstack([y0,y1,y2,y3]).astype('float64')
    
    return px,py

@njit
def find_rx_ry_oceanparcel(x,y,px,py):
    rx = np.ones_like(x)*0.0
    ry = np.ones_like(y)*0.0
    x0 = px[0]
    
    x = to_180(x-x0)
    px = to_180(px-x0)
    
    invA = np.array([[1., 0., 0., 0.],
                         [-1., 1., 0., 0.],
                         [-1., 0., 0., 1.],
                         [1., -1., 1., -1.]])
    a = np.dot(invA,px)
    b = np.dot(invA,py)
    
    aa = a[3]*b[2] - a[2]*b[3]
    bb = a[3]*b[0] - a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + x*b[3] - y*a[3]
    cc = a[1]*b[0] - a[0]*b[1] + x*b[1] - y*a[1]
    
    det2 = bb*bb-4*aa*cc
    
    order1 = np.abs(aa)<1e-12
    order2 = np.logical_and(~order1,det2>=0)
#     nans   = np.logical_and(~order1,det2< 0)
    
#     ry[order1] = -(cc/bb)[order1]
    ry = -(cc/bb) # if it is supposed to be nan, just try linear solve. 
    ry[order2] = ((-bb+np.sqrt(det2))/(2*aa))[order2]
#     ry[nans  ] = np.nan
    
    rot_rectilinear = np.abs(a[1]+a[3]*ry) < 1e-12
    rx[rot_rectilinear ] = ((y-py[0])/(py[1]-py[0]) + (y-py[3])/(py[2]-py[3]))[rot_rectilinear] * .5
    rx[~rot_rectilinear] = ((x-a[0]-a[2]*ry) / (a[1]+a[3]*ry))[~rot_rectilinear]
        
    return rx-1/2,ry-1/2

def weight_f_node(rx,ry):
    return np.vstack([(0.5-rx)*(0.5-ry),
                      (0.5+rx)*(0.5-ry),
                      (0.5+rx)*(0.5+ry),
                      (0.5-rx)*(0.5+ry)]).T