import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import copy

# from OceInterp.kernel_and_weight import kernel_weight,get_weight_cascade
from seaduck.topology import topology
from seaduck.utils import get_combination
from seaduck.RuntimeConf import rcParam

# default kernel for interpolation.
default_kernel = np.array([
    [0,0],
    [0,1],
    [0,2],
    [0,-1],
    [0,-2],
    [-1,0],
    [-2,0],
    [1,0],
    [2,0]
])
default_inheritance = [
    [0,1,2,3,4,5,6,7,8],
    [0,1,2,3,5,7,8],
    [0,1,3,5,7],
    [0]
]
weight_func = dict()

default_kernels = [np.array([default_kernel[i] for i in doll]) for doll in default_inheritance]

# It just tell you what the kernels look like
def show_kernels(kernels = default_kernels):
    '''
    Plot a small scatter plot of the shape of a list of kernel

    **Parameters:**
    
    + kernels: list of numpy.ndarray
        Each of the element is a (n,2) shaped array, where n is the number of element in the kernel. 
    '''
    for i,k in enumerate(kernels):
        x,y = k.T
        plt.plot(x+0.1*i,y+0.1*i,'+')
    
def _translate_to_tendency(k):
    '''
    A kernel looks like 
    np.array([
    [x1,y1],
    [x2,y2],....
    ])
    where [xn,yn] represent a coordinate relative to [0,0]
    which is just the nearest neighbor.
    this function return how you could move from [0,0] to k = [x,y]
    If you need to go to [0,2], you need to move up twice.
    it will return [0,0] or more explicitly ['up','up']
    [0,0] will produce a empty array.

    **Parameters:**
    
    + k: numpy.ndarray
        A (n,2)-shaped array, where n is the number of element in the kernel. 
    '''
    tend = []
    x,y = k 
    if y>0:
        for j in range(y):
            tend.append(0)#up
    else:
        for j in range(-y):
            tend.append(1)#down
    if x<0:
        for j in range(-x):
            tend.append(2)#left
    else:
        for j in range(x):
            tend.append(3)#right
    return tend

def fatten_ind_h(faces,iys,ixs,tp,kernel=default_kernel):
    '''
    faces,iys,ixs is now 1d arrays of size n. 
    We are applying a kernel of size m.
    This is going to return a n * m array of indexes.
    each row represen all the node needed for interpolation of a single point.
    "h" represent we are only doing it on the horizontal plane

    **Parameters:**
    
    + faces: numpy.ndarray or None
        The index of faces that the points are on. None if there is no face dimension. 
    + iys,ixs: numpy.ndarray or None
        1D array of indexes on the points' horizontal position
    '''
    kernel_tends =  [_translate_to_tendency(k) for k in kernel]
    m = len(kernel_tends)
    n = len(iys)
    
    # the arrays we are going to return 
    if faces is not None:
        n_faces = np.zeros((n,m))
        n_faces.T[:] = faces
    n_iys = np.zeros((n,m))
    n_ixs = np.zeros((n,m))
    
    # first try to fatten it naively(fast and vectorized)
    for i,node in enumerate(kernel):
        x_disp,y_disp = node
        n_iys[:,i] = iys+y_disp
        n_ixs[:,i] = ixs+x_disp
    if faces is not None:
        illegal = tp.check_illegal((n_faces,n_iys,n_ixs))
    else:
        illegal = tp.check_illegal((n_iys,n_ixs))
        
    redo = np.array(np.where(illegal)).T
    for num,loc in enumerate(redo):
        j,i = loc
        if faces is not None:
            ind = (faces[j],iys[j],ixs[j])
        else:
            ind = (iys[j],ixs[j])
        # everyone start from the [0,0] node
        moves = kernel_tends[i]
        # moves is a list of operations to get to a single point
        #[2,2] means move to the left and then move to the left again.
        n_ind = tp.ind_moves(ind,moves)
        if faces is not None:
            n_faces[j,i],n_iys[j,i],n_ixs[j,i] = n_ind
        else:
            n_iys[j,i],n_ixs[j,i] = n_ind
    if faces is not None:
        return n_faces.astype('int'),n_iys.astype('int'),n_ixs.astype('int')
    else:
        return None,n_iys.astype('int'),n_ixs.astype('int')

# def fatten_ind_3d(iz,faces,iy,ix,tp,kernel=default_kernel):
#     '''
#     fatten the kernel some more in the vertical direcion,
#     (1-rz)*this_layer+rz*that_layer
#     essentially, there are only two points in the vertical kernel.
#     more generally it should be kronecker product, I find it kind of 
#     unnecessary.
#     '''
#     ffaces,fiy,fix = fatten_ind_h(faces,iy,ix,tp,kernel)
#     n,m = fiy.shape
#     fiz = np.zeros_like(fiy)
#     fiz = np.concatenate((fiz,fiz),axis = 1)
#     for i in range(len(fiz)):
#         fiz[i,:m] = iz[i]
#         if iz[i]!=0:
#             fiz[i,m:] = iz[i]-1
#         # the commented lines below are not necessary,
#         # but it make it easier to explain what we are doing.
#         # if when we are considering the top level,
#         # we just do a 2D interpolation.
#         # top_layer = (1-rz)*top_layer+rz*top_layer
#         # it will be more obvious after you read the function of weight
# #         else:
# #             fiz[i,m:] = iz[i]
#     if faces is None:
#         ffaces = None
#     else:
#         ffaces = np.concatenate((ffaces,ffaces),axis = 1)
#     fiy = np.concatenate((fiy,fiy),axis = 1)
#     fix = np.concatenate((fix,fix),axis = 1)
#     return fiz,ffaces,fiy,fix

def fatten_linear_dim(iz,ind,maximum = None,minimum = None,kernel_type = 'linear'):
    '''
    this function linearly fattened the index in t or z dimension

    **Parameters:**
    
    + iz: np.ndarray
        1D array of particle indexs in a linear dimension, including depth, 
        time and horizontal dimensions if there is no face dimension
    + ind: tuple of np.ndarray
        Index arrays that are already fattened in other directions
    + maximum: int or None
        None if the neighboring cell has index 1 larger than iz. 
        If the value of iz == maximum, the neighboring point is just maximum. 
    + minimum: int or None
        None if the neighboring cell has index 1 smaller than iz. 
    + kernel_type: 'linear', 'dz' or 'nearest'
        Whether to fatten the index using the nearest one point or two points. 
    '''
    if maximum and minimum:
        raise Exception('either interpolate the node with'
                        'larger index (provide maximum) or lower index(provide )')
    ori_shape = ind[-1].shape
    n_ind = []
    if kernel_type in ['linear','dz']:
        new_shape = list(ori_shape)
        new_shape.append(2)
        added_dim = np.zeros(new_shape[::-1])
        added_dim[0] = iz
        if minimum is not None:
            added_dim[1] = np.maximum(minimum,iz-1)
        elif maximum is not None:
            added_dim[1] = np.minimum(maximum,iz+1)
        else:
            added_dim[1] = iz-1
        n_ind.append(added_dim.T.astype(int))
        
        for idim in ind:
            if idim is not None:
                n_ind.append(np.stack((idim,idim),axis = -1))
            else:
                n_ind.append(None)

    elif kernel_type == 'nearest':
        new_shape = list(ori_shape)
        new_shape.append(1)
        added_dim = np.zeros(new_shape)
        added_dim.T[:] = iz
        n_ind.append(added_dim.astype(int))
        for idim in ind:
            if idim is not None:
                n_ind.append(idim.reshape(new_shape))
            else:
                n_ind.append(None)
    else:
        raise Exception('kernel_type not recognized. should be either linear, dz, or nearest')
    return tuple(n_ind)
    
# def fatten_ind_4d(it,iz,face,iy,ix,tp,
#                hkernel=default_kernel,
#                zkernel='linear',
#                tkernel='linear',
#               ):
#     # perform horizontal fattening 
#     hface,hiy,hix = fatten_ind_h(face,iy,ix,tp,hkernel)
    
#     # perform vertical
#     n,m = hiy.shape
#     if zkernel in ['linear','dz']:
#         vhiy = np.stack((hiy,hiy),axis = -1)
#         vhix = np.stack((hix,hix),axis = -1)
#         if face is not None:
#             vhface = np.stack((hface,hface),axis = -1)
#         vhiz = np.zeros((2,m,n))
#         vhiz[0] = iz
#         vhiz[1] = (abs(iz-1)+(iz-1))/2 #relu function
#         vhiz = vhiz.T
#     elif zkernel == 'nearest':
#         vhiy = hiy.reshape((n,m,1))
#         vhix = hix.reshape((n,m,1))
#         vhiz = np.zeros((n,m,1))
#         vhiz.T[:] = iz
#     else:
#         raise Exception('zkernel not recognized. should be either linear, dz, or nearest')

#     # perform temperal
#     n,m,p = vhiy.shape
#     if tkernel in ['linear','dt']:
#         tvhiy = np.stack((vhiy,vhiy),axis = -1)
#         tvhix = np.stack((vhix,vhix),axis = -1)
#         tvhiz = np.stack((vhiz,vhiz),axis = -1)
#         if face is not None:
#             tvhface = np.stack((vhface,vhface),axis = -1)
#         tvhit = np.zeros((2,p,m,n))
#         tvhit[0] = it
#         tvhit[1] = np.minimum(tp.itmax,it+1)
#         tvhit = tvhit.T
#     elif tkernel == 'nearest':
#         tvhiy = vhiy.reshape(n,m,p,1)
#         tvhix = vhix.reshape(n,m,p,1)
#         tvhiz = vhiz.reshape(n,m,p,1)
#         tvhit = np.zeros((n,m,p,1))
#         tvhit.T[:] = it
#     else:
#         raise Exception('tkernel not recognized. should be either linear, dt, or nearest')
        
#     if face is None:
#         tvhface = None
#     return tvhit.astype(int),tvhiz.astype(int),tvhface.astype(int),tvhiy.astype(int),tvhix.astype(int)

def kernel_weight_x(kernel,ktype = 'interp',order = 0):
    '''
    return the function that calculate the interpolation/derivative weight 
    given a cross-shaped (that's where x is coming from) Lagrangian kernel.

    **Parameters:**
    
    + kernel: np.ndarray
        2D array with shape (n,2), where n is the number of nodes. 
        It has to be shaped like a cross
    + ktype: str
        "interp" (default): Use both x y direction for interpolation, implies that order = 0
        "x": Use only x direction for interpolation/derivative
        "y": Use only y direction for interpolation/derivative
    + order: int
        The order of derivatives. Zero for interpolation. 
    
    **Returns:**
    
    + func(rx,ry): compilable function 
        function to calculate the hotizontal interpolation/derivative weight
    '''
    xs = np.array(list(set(kernel.T[0]))).astype(float)
    ys = np.array(list(set(kernel.T[1]))).astype(float)
    
    # if you the kernel is a line rather than a cross
    if len(xs) == 1:
        ktype = 'y'
    elif len(ys)==1:
        ktype = 'x'
    
    """
    If you don't want to know what is going on under the hood.
    it's totally fine.
    
    all of the following is a bit hard to understand.
    The k th (k>=0) derivative of the lagrangian polynomial is 
          \Sigma_{i\neq j} \Pi_{i<m-1-k} (x-x_i)
    w_j= ----------------------------------------
          \Pi_{i\neq j} (x_j - x_i)
    
    for example: if the points are [-1,0,1] for point 0
    k = 0: w = (x-1)(x+1)/(0-1)(0+1)
    k = 1: w = [(x+1)+(x-1)]/(0-1)(0+1)
    
    for a cross shape kernel:
    f(rx,ry) = f_x(rx) + f_y(ry) - f(0,0)
    
    The following equation is just that.
    """
    
    x_poly = []
    y_poly = []
    if ktype == 'interp':
        for ax in xs:
            x_poly.append(get_combination([i for i in xs if i!=ax],len(xs)-1))
        for ay in ys:
            y_poly.append(get_combination([i for i in ys if i!=ay],len(ys)-1))
    if ktype == 'x':
        for ax in xs:
            x_poly.append(
                get_combination([i for i in xs if i!=ax],len(xs)-1-order))
        y_poly=[[[]]]
    if ktype == 'y':
        x_poly = [[[]]]
        for ay in ys:
            y_poly.append(
                get_combination([i for i in ys if i!=ay],len(ys)-1-order))
    x_poly = np.array(x_poly).astype(float)
    y_poly = np.array(y_poly).astype(float)
    @njit
    def the_interp_func(rx,ry):
        nonlocal kernel,xs,ys,x_poly,y_poly
        n = len(rx)
        m = len(kernel)
        weight = np.ones((n,m))*0.0
        for i,(x,y) in enumerate(kernel):
            if x!=0:
                ix = np.where(xs==x)[0][0]
                poly = x_poly[ix]
                for term in poly:
                    another = np.ones(n)
                    for other in term:
                        another*=(rx-other)
                    weight[:,i]+=another
                    for other in xs:
                        if other!=x:
                            weight[:,i]/=(x-other)
            if y!=0:
                iy = np.where(ys==y)[0][0]
                poly = y_poly[iy]
                for term in poly:
                    another = np.ones(n)*1.0
                    for other in term:
                        another*=(ry-other)
                    weight[:,i]+=another
                    for other in ys:
                        if other!=y:
                            weight[:,i]/=(y-other)
            elif x**2+y**2==0:
                xthing = np.zeros(n)*0.0
                ix = np.where(xs==0)[0][0]
                poly = x_poly[ix]
                for term in poly:
                    another = np.ones(n)*1.0
                    for other in term:
                        another*=(rx-other)
                    xthing+=another
                    for other in xs:
                        if other!=x:
                            xthing/=(x-other)
                
                ything = np.zeros(n)*0.0
                iy = np.where(ys==y)[0][0]
                poly = y_poly[iy]
                for term in poly:
                    another = np.ones(n)
                    for other in term:
                        another*=(ry-other)
                    ything+=another
                    for other in ys:
                        if other!=y:
                            ything/=(y-other)
                weight[:,i]=xthing+ything-1
        return weight
    @njit
    def the_x_func(rx,ry):
        nonlocal kernel,xs,ys,x_poly,order
        n = len(rx)
        m = len(kernel)
        weight = np.ones((n,m))*0.0
        for i,(x,y) in enumerate(kernel):
            if y==0:
                ix = np.where(xs==x)[0][0]
                poly = x_poly[ix]
                for term in poly:
                    another = np.ones(n)*1.0
                    for other in term:
                        another*=(rx-other)
                    
                    weight[:,i]+=another
                for other in xs:
                    if other!=x:
                        weight[:,i]/=(x-other)
        return weight
    @njit
    def the_x_maxorder_func(rx,ry):
        nonlocal kernel,xs,ys,order
        n = len(rx)
        m = len(kernel)
        common = 1
        for i in range(1,order):
            common*=i
        weight = np.ones((n,m))*float(common)
        for i,(x,y) in enumerate(kernel):
            if y==0:
                for other in xs:
                    if other!=x:
                        weight[:,i]/=(x-other)
            else:
                weight[:,i] = 0.0
        return weight
    @njit
    def the_y_func(rx,ry):
        nonlocal kernel,xs,ys,y_poly,order
        n = len(rx)
        m = len(kernel)
        weight = np.ones((n,m))*0.0
        for i,(x,y) in enumerate(kernel):
            if x==0:
                iy = np.where(ys==y)[0][0]
                poly = y_poly[iy]
                for term in poly:
                    another = np.ones(n)*1.0
                    for other in term:
                        another*=(ry-other)
                    
                    weight[:,i]+=another
                for other in ys:
                    if other!=y:
                        weight[:,i]/=(y-other)
        return weight
    @njit
    def the_y_maxorder_func(rx,ry):
        nonlocal kernel,xs,ys,order
        n = len(rx)
        m = len(kernel)
        common = 1
        for i in range(1,order):
            common*=i
        weight = np.ones((n,m))*float(common)
        for i,(x,y) in enumerate(kernel):
            if x==0:
                for other in ys:
                    if other!=y:
                        weight[:,i]/=(y-other)
            else:
                weight[:,i] = 0.0
        return weight
    if ktype == 'interp':
        return the_interp_func
    if ktype =='x':
        max_order = len(xs)-1
        if order<max_order:
            return the_x_func
        elif order == max_order:
            return the_x_maxorder_func
        else:
            raise Exception('Kernel is too small for this derivative')
    if ktype =='y':
        max_order = len(ys)-1
        if order<max_order:
            return the_y_func
        elif order == max_order:
            return the_y_maxorder_func
        else:
            raise Exception('Kernel is too small for this derivative')
# we can define the default interpolation functions here, 
# so if we are using it over and over, we don't have to compile it.
# and it really takes a lot of time to compile. 

def kernel_weight_s(kernel,xorder = 0,yorder = 0):
    '''
    return the function that calculate the interpolation/derivative weight 
    given a rectangle-shaped (that's where x is coming from) Lagrangian kernel.

    **Parameters:**
    
    + kernel: np.ndarray
        2D array with shape (n,2), where n is the number of nodes. 
        It has to be shaped like a rectangle
    + xorder: int
        The order of derivatives in the x direction. Zero for interpolation. 
    + yorder: int
        The order of derivatives in the y direction. Zero for interpolation. 
    
    **Returns:**
    
    + func(rx,ry): compilable function
        function to calculate the hotizontal interpolation/derivative weight
    '''
    xs = np.array(list(set(kernel.T[0]))).astype(float)
    ys = np.array(list(set(kernel.T[1]))).astype(float)
    xmaxorder = False
    ymaxorder = False
    if xorder<len(xs)-1:
        pass
    elif xorder == len(xs)-1:
        xmaxorder = True
    else:
        raise Exception('Kernel is too small for this derivative')
        
    if yorder<len(ys)-1:
        pass
    elif yorder == len(ys)-1:
        ymaxorder = True
    else:
        raise Exception('Kernel is too small for this derivative')

    x_poly = []
    y_poly = []
    for ax in xs:
        x_poly.append(
            get_combination([i for i in xs if i!=ax],len(xs)-1-xorder))
    for ay in ys:
        y_poly.append(
            get_combination([i for i in ys if i!=ay],len(ys)-1-yorder))
    x_poly = np.array(x_poly).astype(float)
    y_poly = np.array(y_poly).astype(float)
    @njit
    def the_square_func(rx,ry):
        nonlocal kernel,xs,ys,y_poly,x_poly,xorder,yorder
        n = len(rx)
        mx = len(xs)
        my = len(ys)
        m = len(kernel)
        yweight = np.ones((n,my))
        xweight = np.ones((n,mx))
        weight = np.ones((n,m))*0.0

        if ymaxorder:
            common = 1
            for i in range(1,yorder):
                common*=i
            yweight *=float(common)
        else:
            yweight *= 0.0
        for i,y in enumerate(ys):
            if not ymaxorder:
                iy = np.where(ys==y)[0][0]
                poly = y_poly[iy]
                for term in poly:
                    another = np.ones(n)*1.0
                    for other in term:
                        another*=(ry-other)

                    yweight[:,i]+=another
            for other in ys:
                if other!=y:
                    yweight[:,i]/=(y-other)


        if xmaxorder:
            common = 1
            for i in range(1,xorder):
                common*=i
            xweight *=float(common)
        else:
            xweight *= 0.0
        for i,x in enumerate(xs):
            if not xmaxorder:
                ix = np.where(xs==x)[0][0]
                poly = x_poly[ix]
                for term in poly:
                    another = np.ones(n)*1.0
                    for other in term:
                        another*=(rx-other)

                    xweight[:,i]+=another
            for other in xs:
                if other!=x:
                    xweight[:,i]/=(x-other)

        for i,(x,y) in enumerate(kernel):
            iy = np.where(ys==y)[0][0]
            ix = np.where(xs==x)[0][0]
            weight[:,i] = yweight[:,iy]*xweight[:,ix]

        return weight
    return the_square_func

def kernel_weight(kernel,ktype = 'interp',order = 0):
    '''
    A wrapper around kernel_weight_x and kernel_weight_s.
    Return the function that calculate the interpolation/derivative weight 
    of a  Lagrangian kernel.

    **Parameters:**
    
    + kernel: np.ndarray
        2D array with shape (n,2), where n is the number of nodes. 
        It need to either shape like a rectangle or a cross
    + ktype: str
        "interp" (default): Use both x y direction for interpolation, implies that order = 0
        "dx": Use only x direction for interpolation/derivative
        "dy": Use only y direction for interpolation/derivative
    + order: int
        The order of derivatives. Zero for interpolation. 
    
    **Returns:**
    
    + func(rx,ry): compilable function
        function to calculate the hotizontal interpolation/derivative weight
    '''
    mx = len(set(kernel[:,0]))
    my = len(set(kernel[:,1]))
    if len(kernel) == mx+my-1:
        if 'd' in ktype:
            ktype = ktype[1:]
        return kernel_weight_x(kernel,ktype = ktype,order = order)
    elif len(kernel) == mx*my:# mx*my == mx+my-1 only when mx==1 or my ==1
        if ktype == 'interp':
            return kernel_weight_s(kernel,xorder = 0,yorder = 0)
        elif ktype == 'dx':
            return kernel_weight_s(kernel,xorder = order,yorder = 0)
        elif ktype == 'dy':
            return kernel_weight_s(kernel,xorder = 0,yorder = order)

default_interp_funcs = [kernel_weight_x(a_kernel) for a_kernel in default_kernels]

def find_which_points_for_each_kernel(masked,russian_doll = 'default'):
    '''
    masked is going to be a n*m array,
    where n is the number of points of interest.
    m is the size of the largest kernel.
    
    russian_doll defines the shape of smaller kernels.
    say
    russian_doll = [
    [0,1,2,3,4],
    [0,1],
    [0]
    ]
    it means that the largest kernel have all 5 nodes
    the second kernel only contain the first and second node,
    and the last one only have the nearest neighbor. 
    
    if a row of matrix looks like [1,1,1,1,1],
    the index of the row will be in the first element(list) of the return variable.
    
    if a row of matrix looks like [1,1,1,1,0],
    although it fits both 2nd and 3rd kernel, 2nd has priority, so the index will
    be in the 2nd element of the return pk.
    
    if a row looks like [0,0,0,0,0],
    none of the kernel can fit it, so the index will not be in the return
    '''
    if russian_doll == 'default':
        russian_doll = default_inheritance
    already_wet = []
    for i,doll in enumerate(russian_doll):
        wet_1d = masked[:,np.array(doll)].all(axis = 1)
        already_wet.append(np.where(wet_1d==True)[0])
    point_for_each_kernel = [list(already_wet[0])]
    for i in range(1,len(russian_doll)):
        point_for_each_kernel.append(
            list(np.setdiff1d(already_wet[i],already_wet[i-1]))
        )
    return point_for_each_kernel

def get_weight_cascade(rx,ry,pk,
                       kernel_large = default_kernel,
                       russian_doll = default_inheritance,
                       funcs = default_interp_funcs):
    weight = np.zeros((len(rx),len(kernel_large)))
    weight[:,0] = np.nan
    '''
    apply the corresponding functions that was figured out in 
    find_which_points_for_each_kernel

    **Parameters:**
    
    + rx,ry: numpy.ndarray
        1D array with length N of non-dimensional relative horizontal position to the nearest node
    + kernel_large: numpy.ndarray
        A numpy kernel of shape (M,2) that contains all the kernels needed.
    + russian_doll: list of list(s)
        The inheritance sequence when some of the node is masked.
    + funcs: list of compileable functions
        The weight function of each kernel in the inheritance sequence. 

    **Returns:**
    
    + weight: numpy.ndarray
        The horizontal weight of interpolation/derivative for the points with shape (N,M)

    '''
    for i in range(len(pk)):
        if len(pk[i]) == 0:
            continue
        sub_rx = rx[pk[i]]
        sub_ry = ry[pk[i]]
    #     slim_weight = interp_func[i](sub_rx,sub_ry)
        sub_weight = np.zeros((len(pk[i]),len(kernel_large)))
        sub_weight[:,np.array(russian_doll[i])] = funcs[i](sub_rx,sub_ry)
        weight[pk[i]] = sub_weight
    return weight

# def find_which_points_for_2layer_kernel(masked,russian_doll = default_russian_doll):
#     # extend the find_which_points_for_each_kernel to the z dimension
#     n,m = masked.shape
#     m = m//2
#     pk1 = find_which_points_for_each_kernel(masked[:,:m],russian_doll)
#     pk2 = find_which_points_for_each_kernel(masked[:,m:],russian_doll)
#     return pk1,pk2

# def get_weight_2layer(rx,ry,rz,pk1,pk2,bc = 'no_flux',
#                       kernel_large = default_kernel,
#                        russian_doll = default_russian_doll,
#                        funcs = default_interp_funcs):
#     n = len(rx)
#     m = len(kernel_large)
#     weight = np.zeros((n,2*m))
#     # here the weight is multiplied by rz, which is just the weight associated with z location
#     # essentially a 2-point interpolation
#     weight[:,:m] = (get_weight_cascade(rx,ry,pk1,kernel_large,russian_doll,funcs).T*(1-rz)).T
#     weight[:,m:] = (get_weight_cascade(rx,ry,pk2,kernel_large,russian_doll,funcs).T*rz).T
#     if bc == 'no_flux':
#         # for salt the natural bottom bc is no_flux,
#         # between the lowest wet node and the dry node right under it.
#         # the gradient shoule be zero, and a nearest neighbor interpolation should be used.
#         # rather than returning nan
#         for i in range(n):
#             if np.isnan(weight[i,m]) and not np.isnan(weight[i,0]):
#                 weight[i,m] = 0
#     return weight

def find_pk_4d(masked,russian_doll = default_inheritance):
    '''
    find the masking condition for 4D space time. 
    '''
    maskedT = masked.T
    ind_shape = maskedT.shape
    tz = []
    for i in range(ind_shape[0]):
        z = []
        for j in range(ind_shape[1]):
            z.append(find_which_points_for_each_kernel(maskedT[i,j].T,russian_doll))
        tz.append(z)
    return tz

def get_weight_4d(rx,ry,rz,rt,
                  pk4d,
                  hkernel = default_kernel,
                  russian_doll = default_inheritance,
                  funcs = default_interp_funcs,
                  tkernel = 'linear',#'dt','nearest'
                  zkernel = 'linear',#'dz','nearest'
                  bottom_scheme = 'no flux'# None
                 ):
    '''
    Return the weight of values given particle rel-coords

    **Parameters:**
    
    + rx,ry,rz,rt: numpy.ndarray
        1D array of non-dimensional particle positions of shape (N)
    + pk4d: list
        A mapping on which points should use which horizontal kernel.
    + hkernel:
        A horizontal numpy kernel that contains all the horizontal kernels needed.
    + russian_doll: list of list(s)
        The inheritance sequence when some of the node is masked.
    + funcs: list of compileable functions
        The weight function of each kernel in the inheritance sequence. 
    + tkernel: str
        What kind of operation to do in the temporal dimension: 
        'linear', 'nearest' interpolation, or 'dt'
    + zkernel: str
        What kind of operation to do in the vertical: 
        'linear', 'nearest' interpolation, or 'dz'
    + bottom_scheme: str
        Whether to assume there is a ghost point with same value at the bottom boundary.
        Choose None for vertical flux, 'no flux' for most other cases. 

    **Returns:**

    + weight: numpy.ndarray
        The weight of interpolation/derivative for the points with shape (N,M), 
        M is the num of node in the largest kernel. 
    '''
    nt = len(pk4d)
    nz = len(pk4d[0])
    
    if tkernel == 'linear':
        rp = copy.deepcopy(rt)
        tweight = [(1-rp).reshape((len(rp),1,1)),rp.reshape((len(rp),1,1))]
    elif tkernel == 'dt':
        tweight = [-1,1]
    elif tkernel == 'nearest':
        tweight = [1,0]

    if zkernel == 'linear':
        rp = copy.deepcopy(rz)
        zweight = [(1-rp).reshape((len(rp),1)),rp.reshape((len(rp),1))]
    elif zkernel == 'dz':
        zweight = [-1,1]
    elif zkernel == 'nearest':
        zweight = [1,0]

    weight = np.zeros((len(rx),len(hkernel),nz,nt))
    for jt in range(nt):
        for jz in range(nz):
            weight[:,:,jz,jt] =   get_weight_cascade(rx,ry,
                                                      pk4d[jt][jz],
                                                      kernel_large = hkernel,
                                                      russian_doll = russian_doll,
                                                      funcs = funcs
                                                     )
    for jt in range(nt):
        if (zkernel == 'linear') and (bottom_scheme == 'no flux'):
            # whereever the bottom layer is masked, replace it with a ghost point above it
            secondlayermasked = np.isnan(weight[:,:,0,jt]).any(axis = 1)
            # setting the value at this level zero
            weight[secondlayermasked,:,0,jt] = 0
            shouldbemasked = np.logical_and(secondlayermasked,rz<1/2)
            weight[shouldbemasked,:,1,jt] = 0
            # setting the vertical weight of the above value to 1
            zweight[1][secondlayermasked] = 1
        for jz in range(nz):
            weight[:,:,jz,jt] *= zweight[jz]
        weight[:,:,:,jt]*=tweight[jt]
#         break

    return weight

def kash(kernel):#hash kernel
    '''
    Hash a horizontal kernel. Return the hash value. 

    **Parameters:**
    
    + kernel: numpy.ndarray
        A horizontal kernel
    '''
    temp_lst = [(i,j) for (i,j) in kernel]
    return hash(tuple(temp_lst))

def get_func(kernel,hkernel = 'interp',h_order = 0):
    '''
    Similar to the kernel_weight function,
    the only difference is that this function can 
    read existing functions from a global dictionary,
    and can register to the dictionary when new ones are created. 
    '''
    global weight_func
    ker_ind = kash(kernel)
    layer_1 = weight_func.get(ker_ind)
    if layer_1 is None:
        weight_func[ker_ind] = dict()
    layer_1 = weight_func[ker_ind]
    
    layer_2 = layer_1.get(hkernel)
    if layer_2 is None:
        layer_1[hkernel] = dict()
    layer_2 = layer_1[hkernel]
    
    layer_3 = layer_2.get(h_order)
    if layer_3 is None:
        if rcParam['debug_level'] == 'very_high':
            print('Creating new weight function, the first time is going to be slow')
        layer_2[h_order] = kernel_weight(kernel,ktype = hkernel,order = h_order)
    layer_3 = layer_2[h_order]
    
    return layer_3

def auto_doll(kernel,hkernel = 'interp'):
    '''
    Find a natural inheritance pattern given one horizontal kernel
    '''
    if hkernel == 'interp':
        doll = [[i for i in range(len(kernel))]]
    elif hkernel == 'dx':
        doll = [[i for i in range(len(kernel)) if kernel[i][1]==0]]
    elif hkernel == 'dy':
        doll = [[i for i in range(len(kernel)) if kernel[i][0]==0]]
    doll[0] = sorted(doll[0],key = lambda i:max(abs(kernel[i]+np.array([0.01,0.00618]))))
    last = doll[-1]
    lask = np.array([kernel[i] for i in last])
    dist = round(max(np.max(abs(lask),axis = 1)))
    for radius in range(dist-1,-1,-1):
        new = [i for i in last if max(abs(kernel[i]))<=radius]
        if new != last:
            last = new
            doll.append(last)
    return doll      

class KnW(object):
    '''
    A class that describes anything about the interpolation/derivative kernel to be used. 

    **Parameters:**
    
    + kernel: numpy.ndarray
        The largest horizontal kernel to be used
    + inheritance: list
        The inheritance sequence of the kernels
    + hkernel: str
        What to do in the horizontal direction
        'interp', 'dx', or 'dy'?
    + tkernel: str
        What kind of operation to do in the temporal dimension: 
        'linear', 'nearest' interpolation, or 'dt'
    + zkernel: str
        What kind of operation to do in the vertical: 
        'linear', 'nearest' interpolation, or 'dz'
    + h_order: int
        How many derivative to take in the horizontal direction. Zero for pure interpolation
    + ignore_mask: bool
        Whether to diregard the masking of the dataset. You can select True if there is no 
        inheritance, or if performance is a big concern. 
    '''
    def __init__(self,kernel = default_kernel,
                 inheritance = 'auto',#None, or list of lists
                 hkernel = 'interp',# dx,dy
                 vkernel = "nearest",# linear,dz
                 tkernel = "nearest",# linear,dt
                 h_order = 0,# depend on hkernel type
                 ignore_mask = False,
                ):
        ksort = np.abs(kernel+np.array([0.01,0.00618])).sum(axis = 1).argsort()
        ksort_inv = ksort.argsort()
        
        if inheritance is not None and ignore_mask and rcParam['debug_level']=='very_high':
            print('Warning:overwriting the inheritance object to None, because we ignore masking')
        
        if inheritance == 'auto':
            inheritance = auto_doll(kernel,hkernel = hkernel)
        elif inheritance is None:# does not apply cascade
            inheritance = [[i for i in range(len(kernel))]]
        elif isinstance(inheritance,list):
            pass
        else:
            raise Exception('Unknown type of inherirance')
            
        self.kernel = kernel[ksort]
        self.inheritance = [sorted([ksort_inv[i] for i in heir]) for heir in inheritance]
        self.hkernel = hkernel
        self.vkernel = vkernel
        self.tkernel = tkernel
        self.h_order = h_order
        self.ignore_mask = ignore_mask
        
        self.kernels = [np.array([self.kernel[i] for i in doll]) for doll in self.inheritance]
        self.hfuncs = [
            get_func(kernel = a_kernel,
                     hkernel = self.hkernel,
                     h_order = self.h_order) 
            for a_kernel in self.kernels]
    def same_hsize(self,other):
        '''
        return True if 2 KnW object has the same horizontal size
        '''
        type_same = isinstance(other, type(self))
        if not type_same:
            raise TypeError('the argument is not a KnW object')
        return (self.kernel == other.kernel).all()
    
    def same_size(self,other):
        '''
        return True if 2 KnW object has the same 4D size
        '''
        only_size = {
            'dz':2,
            'linear':2,
            'dt':2,
            'nearest':1
        }
        hsize_same = self.same_hsize(other)
        vsize_same = (only_size[self.vkernel] == only_size[other.vkernel])
        tsize_same = (only_size[self.tkernel] == only_size[other.tkernel])
        return hsize_same and vsize_same and tsize_same
        
        
    def __eq__(self,other):
        type_same = isinstance(other, type(self))
        if not type_same:
            return False
        shpe_same = ((self.kernel == other.kernel).all() and self.inheritance == other.inheritance)
        diff_same = (
            (self.hkernel == other.hkernel) and
            (self.vkernel == other.vkernel) and
            (self.tkernel == other.tkernel)
        )
        return type_same and shpe_same and diff_same
    
    def __hash__(self):
        return hash((kash(self.kernel),
                     tuple(tuple(i for i in heir) for heir in self.inheritance),
                     self.ignore_mask,
                     self.h_order,
                     self.hkernel,
                     self.vkernel,
                     self.tkernel))
    
    # def hash_largest(self):
    #     return hash((kash(self.kernel),
    #                  self.h_order,
    #                  self.hkernel,
    #                  self.vkernel,
    #                  self.tkernel))
    
    def size_hash(self):
        '''
        produce a hash value simply based on the 4D size of the KnW object
        '''
        only_size = {
            'dz':2,
            'linear':2,
            'dt':2,
            'nearest':1
        }
        return hash((kash(self.kernel),
                     only_size[self.vkernel],
                     only_size[self.tkernel]))
    
    def get_weight(self,rx,ry,rz=0,rt=0,
                      pk4d = None,# All using the largest 
                      bottom_scheme = 'no flux'# None
                     ):
        '''
        Return the weight of values given particle rel-coords

        **Parameters:**
        
        + rx,ry,rz,rt: numpy.ndarray
            1D array of non-dimensional particle positions
        + pk4d: list
            A mapping on which points should use which kernel.
        + bottom_scheme: str
            Whether to assume there is a ghost point with same value at the bottom boundary.
            Choose None for vertical flux, 'no flux' for most other cases. 

        **Returns:**
        
        + weight: numpy.ndarray
            The weight of interpolation/derivative for the points with shape (N,M), 
            M is the num of node in the largest kernel. 
        '''
        if self.vkernel in ['linear','dz']:
            nz = 2
        else:
            nz = 1
        if self.tkernel in ['linear','dt']:
            nt = 2
        else:
            nt = 1
        
        weight = np.zeros((len(rx),len(self.kernel),nz,nt))
        if isinstance(rz,(int,float,complex)) and self.vkernel!='nearest':
            rz = np.array([rz for i in range(len(rx))])
        if isinstance(rt,(int,float,complex)) and self.tkernel!='nearest':
            rt = np.array([rt for i in range(len(rx))])

        if self.tkernel == 'linear':
            rp = copy.deepcopy(rt)
            tweight = [(1-rp).reshape((len(rp),1,1)),rp.reshape((len(rp),1,1))]
        elif self.tkernel == 'dt':
            tweight = [-1,1]
        elif self.tkernel == 'nearest':
            tweight = [1]

        if self.vkernel == 'linear':
            rp = copy.deepcopy(rz)
            zweight = [(1-rp).reshape((len(rp),1)),rp.reshape((len(rp),1))]
        elif self.vkernel == 'dz':
            zweight = [-1,1]
        elif self.vkernel == 'nearest':
            zweight = [1]
            
        if pk4d is None:
#             pk4d = [
#                 [
#                     [list(range(len(rx)))]# all points are in the same catagory
#                 for i in range(nz)]# Every layer is the same
#             for j in range(nt)]# 
            for jt in range(nt):
                for jz in range(nz):
                    weight[:,self.inheritance[0],jz,jt] =   self.hfuncs[0](rx,ry)
        else:
            if nt != len(pk4d) or nz != len(pk4d[0]):
                raise ValueError('The kernel and the input pk4d does not match')


            for jt in range(nt):
                for jz in range(nz):
                    weight[:,:,jz,jt] =   get_weight_cascade(rx,ry,
                                                              pk4d[jt][jz],
                                                              kernel_large = self.kernel,
                                                              russian_doll = self.inheritance,
                                                              funcs = self.hfuncs
                                                             )
        for jt in range(nt):
            if (self.vkernel == 'linear') and (bottom_scheme == 'no flux'):
                # whereever the bottom layer is masked, replace it with a ghost point above it
                secondlayermasked = np.isnan(weight[:,:,0,jt]).any(axis = 1)
                # setting the value at this level zero
                weight[secondlayermasked,:,0,jt] = 0
                shouldbemasked = np.logical_and(secondlayermasked,rz<1/2)
                weight[shouldbemasked,:,1,jt] = 0
                # setting the vertical weight of the above value to 1
                zweight[1][secondlayermasked] = 1
            for jz in range(nz):
                weight[:,:,jz,jt] *= zweight[jz]
            weight[:,:,:,jt]*=tweight[jt]

        return weight