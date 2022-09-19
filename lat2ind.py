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
def find_rel_h_with_face(Xs,Ys,some_x,some_y,some_dx,some_dy,CS,SN,faces,iys,ixs):
    '''
    read find_rel_h for more info,
    
    '''
    n = len(Xs)
    rx = np.ones_like(Xs)*0.0
    ry = np.ones_like(Ys)*0.0
    dx = np.ones_like(Xs)*0.0
    dy = np.ones_like(Ys)*0.0
    bx = np.ones_like(Xs)*0.0
    by = np.ones_like(Ys)*0.0
    cs = np.ones_like(Xs)*0.0
    sn = np.ones_like(Ys)*0.0
    for i in range(n):
        base_lon = some_x[faces[i],iys[i],ixs[i]]
        base_lat = some_y[faces[i],iys[i],ixs[i]]
        
        bx[i] = base_lon
        by[i] = base_lat

        cs[i] = CS[faces[i],iys[i],ixs[i]]
        sn[i] = SN[faces[i],iys[i],ixs[i]]

        Delta_x = some_dx[faces[i],iys[i],ixs[i]]
        Delta_y = some_dy[faces[i],iys[i],ixs[i]]
        
        dlon = to_180(Xs[i] - base_lon)
        dlat = to_180(Ys[i] - base_lat)
        
        dx[i] = Delta_x
        dy[i] = Delta_y

        rx[i] = (dlon*np.cos(base_lat*np.pi/180)*cs[i]+dlat*sn[i])*deg2m/Delta_x
        ry[i] = (dlat*cs[i]-dlon*sn[i]*np.cos(base_lat*np.pi/180))*deg2m/Delta_y
    
    return rx,ry,cs,sn,dx,dy,bx,by

@njit
def find_rel_h_without_face(Xs,Ys,some_x,some_y,some_dx,some_dy,CS,SN,iys,ixs):
    '''
    read find_rel_h for more info,
    
    '''
    n = len(Xs)
    rx = np.ones_like(Xs)*0.0
    ry = np.ones_like(Ys)*0.0
    dx = np.ones_like(Xs)*0.0
    dy = np.ones_like(Ys)*0.0
    bx = np.ones_like(Xs)*0.0
    by = np.ones_like(Ys)*0.0
    cs = np.ones_like(Xs)*0.0
    sn = np.ones_like(Ys)*0.0
    for i in range(n):
        base_lon = some_x[iys[i],ixs[i]]
        base_lat = some_y[iys[i],ixs[i]]
        
        bx[i] = base_lon
        by[i] = base_lat

        cs[i] = CS[iys[i],ixs[i]]
        sn[i] = SN[iys[i],ixs[i]]

        Delta_x = some_dx[iys[i],ixs[i]]
        Delta_y = some_dy[iys[i],ixs[i]]
        
        dlon = to_180(Xs[i] - base_lon)
        dlat = to_180(Ys[i] - base_lat)
        
        dx[i] = Delta_x
        dy[i] = Delta_y

        rx[i] = (dlon*np.cos(base_lat*np.pi/180)*cs[i]+dlat*sn[i])*deg2m/Delta_x
        ry[i] = (dlat*cs[i]-dlon*sn[i]*np.cos(base_lat*np.pi/180))*deg2m/Delta_y
    
    return rx,ry,cs,sn,dx,dy,bx,by

def find_rel_h(Xs,Ys,some_x,some_y,some_dx,some_dy,CS,SN,tree):
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
        rx,ry,cs,sn,dx,dy,bx,by = find_rel_h_with_face(Xs,
                                               Ys,
                                               some_x,
                                               some_y,
                                               some_dx,
                                               some_dy,
                                               CS,
                                               SN,
                                               faces,
                                               iys,
                                               ixs)
    else:
        rx,ry,cs,sn,dx,dy,bx,by = find_rel_h_without_face(Xs,
                                               Ys,
                                               some_x,
                                               some_y,
                                               some_dx,
                                               some_dy,
                                               CS,
                                               SN,
                                               iys,
                                               ixs)
    return faces,iys,ixs,rx,ry,cs,sn,dx,dy,bx,by