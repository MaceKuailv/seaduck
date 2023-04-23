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

def NoneIn(lst):
    '''
    See if there is a None in the iterable object. Return a Boolean.
    '''
    ans = False
    for i in lst:
        if i is None:
            ans = True
            break
    return ans

@njit
def spherical2cartesian(Y, X, R=6371.0):
    """
    Convert spherical coordinates to cartesian.

    Parameters:
    ------------
    Y: np.array
        Spherical Y coordinate (latitude)
    X: np.array
        Spherical X coordinate (longitude)
    R: scalar
        Earth radius in km
        If None, use geopy default
    
    Returns:
    ---------
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
    find the index of the nearest level that is lower than the given level
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
    find the index of the latest time that is before the given time
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
    Find the index of the nearest value in the array to the given value. 
    '''
    array = np.asarray(array)
    idx = np.argmin(np.abs(array - value))
    idx = int(idx)
    return idx,array[idx]

@njit
def find_ind_periodic(array,value,peri):
    '''
    Find the index of the nearest value in the array to the given value, where the values are periodic. 
    '''
    array = np.asarray(array)
    idx = np.argmin(np.abs((array - value)%peri))
    idx = int(idx)
    return idx,array[idx]
    

deg2m = 6271e3*np.pi/180
def find_ind_h(Xs,Ys,tree,h_shape):
    '''
    use ckd tree to find the horizontal indexes,
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
    '''
    Find the rel-coords based on the find_ind_nearest method. 
    '''
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
def find_rel_periodic(value,ts,peri):
    '''
    Find the rel-coords based on the find_ind_periodic method. 
    '''
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
        it,bt = find_ind_periodic(ts,t,peri)
        delta_t = (t-bt)%peri
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
    find the rel-coords of the vertical coords

    Paramters:
    -----------
    depth: numpy.ndarray
        1D array for the depth of interest in meters. More negative means deeper. 
    some_z: numpy.ndarray
        The depth of reference depth.
    some_dz: numpy.ndarray
        dz_i = abs(z_{i+1}- z_i)

    Returns:
    ---------
    iz: numpy.ndarray
        Indexes of the reference z level
    rz: numpy.ndarray
        Non-dimensional distance to the reference z level
    dz: numpy.ndarray
        distance between the reference z level and the next one. 
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
    find the rel-coords of the temporal coords

    Paramters:
    -----------
    time: numpy.ndarray
        1D array for the time since 1970-01-01 in seconds. 
    ts: numpy.ndarray
        The time of model time steps also in seconds. 

    Returns:
    ---------
    it: numpy.ndarray
        Indexes of the reference t level
    rt: numpy.ndarray
        Non-dimensional distance to the reference t level
    dt: numpy.ndarray
        distance between the reference t level and the next one. 
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
def _read_h_with_face(some_x,some_y,some_dx,some_dy,CS,SN,faces,iys,ixs):
    '''
    read the grid coords when there is a face dimension to it. 
    '''
    n = len(ixs)
    
    bx = np.ones_like(ixs)*0.0
    by = np.ones_like(ixs)*0.0
    for i in range(n):
        bx[i] = some_x[faces[i],iys[i],ixs[i]]
        by[i] = some_y[faces[i],iys[i],ixs[i]]
        
    if CS is not None and SN is not None:
        cs = np.ones_like(ixs)*0.0
        sn = np.ones_like(ixs)*0.0
        for i in range(n):
            cs[i] = CS[faces[i],iys[i],ixs[i]]
            sn[i] = SN[faces[i],iys[i],ixs[i]]
    else:
        cs = None
        sn = None
        
    if some_dx is not None and some_dy is not None:
        dx = np.ones_like(ixs)*0.0
        dy = np.ones_like(ixs)*0.0
        for i in range(n):
            dx[i] = some_dx[faces[i],iys[i],ixs[i]]
            dy[i] = some_dy[faces[i],iys[i],ixs[i]]
    else:
        dx = None
        dy = None
    
    return cs,sn,dx,dy,bx,by

@njit
def _read_h_without_face(some_x,some_y,some_dx,some_dy,CS,SN,iys,ixs):
    '''
    read _read_h_with_face for more info.
    
    '''
    # TODO ADD test if those are Nones.
    n = len(ixs)
    if some_dx is not None and some_dy is not None:
        dx = np.ones_like(ixs)*0.0
        dy = np.ones_like(ixs)*0.0
        for i in range(n):
            dx[i] = some_dx[iys[i],ixs[i]]
            dy[i] = some_dy[iys[i],ixs[i]]
    else:
        dx = None
        dy = None
        
    if CS is not None and SN is not None:
        cs = np.ones_like(ixs)*0.0
        sn = np.ones_like(ixs)*0.0
        for i in range(n):
            cs[i] = CS[iys[i],ixs[i]]
            sn[i] = SN[iys[i],ixs[i]]
    else:
        cs = None
        sn = None
            
    bx = np.ones_like(ixs)*0.0
    by = np.ones_like(ixs)*0.0
    for i in range(n):
        bx[i] = some_x[iys[i],ixs[i]]
        by[i] = some_y[iys[i],ixs[i]]
        
    return cs,sn,dx,dy,bx,by

@njit
def find_rx_ry_naive(x,y,bx,by,cs,sn,dx,dy):
    '''
    Find the non-dimensional coords using the local cartesian scheme
    '''
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
    if NoneIn([Xs,Ys,some_x,some_y,some_dx,some_dy,CS,SN,tree]):
        raise ValueError('Some of the required variables are missing')
    h_shape = some_x.shape
    faces,iys,ixs = find_ind_h(Xs,
                               Ys,
                               tree,
                               h_shape
                              )
    if faces is not None:
        cs,sn,dx,dy,bx,by = _read_h_with_face(  some_x,
                                               some_y,
                                               some_dx,
                                               some_dy,
                                               CS,
                                               SN,
                                               faces,
                                               iys,
                                               ixs)
    else:
        cs,sn,dx,dy,bx,by = _read_h_without_face(some_x,
                                               some_y,
                                               some_dx,
                                               some_dy,
                                               CS,
                                               SN,
                                               iys,
                                               ixs)
    rx,ry = find_rx_ry_naive(Xs,Ys,bx,by,cs,sn,dx,dy)
    return faces,iys,ixs,rx,ry,cs,sn,dx,dy,bx,by

def find_rel_h_rectilinear(x,y,lon,lat):
    '''
    Find the rel-coords using the rectilinear scheme
    '''
    ratio = 6371e3*np.pi/180
    ix,rx,dx,bx = find_rel_periodic(x,lon,360.)
    iy,ry,dy,by = find_rel_periodic(y,lat,360.)
    dx = np.cos(y*np.pi/180)*ratio*dx
    dy = ratio*dy
    face = None
    cs = np.ones_like(x)
    sn = np.zeros_like(x)
    return face,iy,ix,rx,ry,cs,sn,dx,dy,bx,by
    

def find_rel_h_oceanparcel(x,y,some_x,some_y,some_dx,some_dy,CS,SN,XG,YG,tree,tp):
    '''
    Find the rel-coords using the rectilinear scheme
    '''
    if NoneIn([x,y,some_x,some_y,XG,YG,tree]):
        raise ValueError('Some of the required variables are missing')
    h_shape = some_x.shape
    faces,iys,ixs = find_ind_h(x,
                               y,
                               tree,
                               h_shape
                              )
    if faces is not None:
        cs,sn,dx,dy,bx,by = _read_h_with_face(  some_x,
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
        cs,sn,dx,dy,bx,by = _read_h_without_face(some_x,
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
    

# def find_cs_sn(thetaA,phiA,thetaB,phiB):
#     '''
#     theta is the angle 
#     between the meridian crossing point A
#     and the geodesic connecting A and B
    
#     this function return cos and sin of theta
#     '''
#     # O being north pole
#     AO = np.pi/2 - thetaAf
#     BO = np.pi/2 - thetaB
#     dphi = phiB-phiA
#     # Spherical law of cosine on AOB
#     cos_AB = np.cos(BO)*np.cos(AO)+np.sin(BO)*np.sin(AO)*np.cos(dphi)
#     sin_AB = np.sqrt(1-cos_AB**2)
#     # spherical law of sine on triangle AOB
#     SN = np.sin(BO)*np.sin(dphi)/sin_AB
#     CS = np.sign(thetaB-thetaA)*np.sqrt(1-SN**2)
#     return CS,SN

def find_px_py(XG,YG,tp,*ind,gridoffset = -1):
    '''
    Find the nearest 4 corner points. This is used in oceanparcel interpolation scheme. 
    '''
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
    '''
    find the non-dimensional horizontal distance using the oceanparcel scheme. 
    '''
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
    '''
    assign weight based on the non-dimensional coords to the four corner points. 
    '''
    return np.vstack([(0.5-rx)*(0.5-ry),
                      (0.5+rx)*(0.5-ry),
                      (0.5+rx)*(0.5+ry),
                      (0.5-rx)*(0.5+ry)]).T