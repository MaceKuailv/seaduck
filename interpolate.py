import numpy as np
from utils import find_rel_4d,grid2array,local_to_latlon,find_rel_2d,find_rel_time,find_rel_z
import utils as _u
from kernel_and_weight import (fatten_ind_4d,
                               find_pk_4d,
                               get_weight_4d,
                               kernel_weight_x,
                               fatten_ind_h,
                               fatten_linear_dim,
                               find_pk_4d,
                               get_weight_4d)
from get_masks import get_masks,get_masked
from topology import topology
from smart_read import smart_read as sread

weight_func = dict()

kernel_large = np.array([[ 0,  0],
       [ 0,  1],
       [ 0, -1],
       [-1,  0],
       [ 1,  0],
       [ 0,  2],
       [ 0, -2],
       [-2,  0],
       [ 2,  0]])
u_doll = [
    [0,1,2,3,4,5,6,7,8],
    [0,1,2,3,4,5,8],
    [0,1,2,3,4],
    [0,4],
    [0]
]
v_doll = [
    [0,1,2,3,4,5,6,7,8],
    [0,1,2,3,4,5,8],
    [0,1,2,3,4],
    [0,1],
    [0]
]

def kash(kernel):#hash kernel
    temp_lst = [(i,j) for (i,j) in kernel]
    return hash(tuple(temp_lst))

def get_func(kernel = kernel_large,ktype = 'interp',h_order = 0):
    global weight_func
    ker_ind = kash(kernel)
    layer_1 = weight_func.get(ker_ind)
    if layer_1 is None:
        weight_func[ker_ind] = dict()
    layer_1 = weight_func[ker_ind]
    
    layer_2 = layer_1.get(ktype)
    if layer_2 is None:
        layer_1[ktype] = dict()
    layer_2 = layer_1[ktype]
    
    layer_3 = layer_2.get(h_order)
    if layer_3 is None:
        print('Creating new weight function, it is going to be slow')
        layer_2[h_order] = kernel_weight_x(kernel,ktype = ktype,order = h_order)
    layer_3 = layer_2[h_order]
    
    return layer_3

def auto_doll(kernel,ktype = 'interp',h_order = 0):
    if ktype == 'interp':
        doll = [[i for i in range(len(kernel))]]
        last = doll[-1]
        lask = np.array([kernel[i] for i in last])
        dist = round(max(np.max(abs(lask),axis = 1)))
        for radius in range(dist-1,-1,-1):
            new = [i for i in last if max(abs(kernel[i]))<=radius]
            if new != last:
                last = new
                doll.append(last)
    elif ktype == 'dx':
        doll = [[i for i in range(len(kernel)) if kernel[i][1]==0]]
        last = doll[-1]
        lask = np.array([kernel[i] for i in last])
        dist = round(max(np.max(abs(lask),axis = 1)))
        for radius in range(dist-1,-1,-1):
            new = [i for i in last if max(abs(kernel[i]))<=radius]
            if new != last:
                last = new
                doll.append(last)
    elif ktype == 'dy':
        doll = [[i for i in range(len(kernel)) if kernel[i][0]==0]]
        last = doll[-1]
        lask = np.array([kernel[i] for i in last])
        dist = round(max(np.max(abs(lask),axis = 1)))
        for radius in range(dist-1,-1,-1):
            new = [i for i in last if max(abs(kernel[i]))<=radius]
            if new != last:
                last = new
                doll.append(last)
    return doll      

def auto_udoll(kernel):
    doll = auto_doll(kernel)
    right = np.logical_and(kernel[:,0] == 1,kernel[:,1]==0)
    if not right.any():
        raise Exception('kernel does not contain [1,0], therefore udoll cannot be created')
    right = np.where(right)[0][0]
    doll.insert(-1,[0,right])
    return doll

def auto_vdoll(kernel):
    doll = auto_doll(kernel)
    right = np.logical_and(kernel[:,0] == 0,kernel[:,1]==1)
    if not right.any():
        raise Exception('kernel does not contain [0,1], therefore vdoll cannot be created')
    right = np.where(right)[0][0]
    doll.insert(-1,[0,right])
    return doll

def interpolate(od,varList,
                x,y,z,t,
                ktype = 'interp',#'dx,dy,'
                order = 0,
                kernel = None,
                doll = 'auto',
                tkernel = 'linear',#'dt','nearest'
                zkernel = 'linear',#'dz','nearest'
                bottom_scheme = 'no flux'# None
                 ):
    ## TODO: implement interpolating Z
    N = len(x)
    
    if kernel is None:
        kernel = kernel_large
    if ktype in ['interp','dz','dt']:
        h_order = 0
    if ktype == 'dz':
        zkernel = dz
    if ktype == 'dt':
        tkernel = 'dt'
    kernel = kernel[abs(kernel).sum(axis = 1).argsort()]
    
    if isinstance(varList,str):
        varList = [varList]
    dims_need = []
    for name in varList:
        if isinstance(name,str):
            for dim in od._ds[name].dims:
                dims_need.append(dim)
        elif isinstance(name,list):
            for subname in name:
                for dim in od._ds[subname].dims:
                    dims_need.append(dim)
    dims_need = set(dims_need)
    
    tp = topology(od)
    
    if 'X' in dims_need:
        face,iy,ix,rx,ry,cs,sn,dx,dy = find_rel_2d(x,y)
        hface,hiy,hix = fatten_ind_h(face,iy,ix,tp,kernel = kernel)
    if 'time' in dims_need:
        it,rt,dt = find_rel_time(t,_u.ts)
        it = it.astype(int)
    if 'Z' in dims_need:
        iz,rz,dz = find_rel_z(z,_u.Z,_u.dZ)
        iz = iz.astype(int)
    if 'Zl' in dims_need:
        izl,rzl,dzl = find_rel_z(z,_u.Zl,_u.dZl)
        izl = izl.astype(int)

    if doll == 'auto':
        doll = auto_doll(kernel,ktype = ktype,h_order = h_order)
    elif doll is None:# does not apply cascade
        doll = [[i for i in range(len(kernel))]]
    elif isinstance(doll,list):
        pass
    else:
        raise Exception('Unknown type of doll')
        
    kernels = [np.array([kernel[i] for i in dol]) for dol in doll]
    funcs = [get_func(kernel = a_kernel,ktype = ktype,h_order = h_order) for a_kernel in kernels]
    returns = []
    for name in varList:
        tkernel = 'linear'
        zkernel = 'linear'
        #TODO: implement case with no horizontal direction
        ind4d = (hface,hiy,hix)

        if 'Z' in od._ds[name].dims:
            ind4d = fatten_linear_dim(iz ,ind4d,minimum = 0)
        elif 'Zl' in od._ds[name].dims:
            ind4d = fatten_linear_dim(izl,ind4d,minimum = 0)
        else:
            zkernel = 'nearest'
            ind4d = fatten_linear_dim(np.zeros_like(ix),ind4d,minimum = 0,kernel_type = zkernel)

        if 'time' in od._ds[name].dims:
            ind4d = fatten_linear_dim(it,ind4d,maximum = tp.itmax)
        else:
            tkernel = 'nearest'
            ind4d = fatten_linear_dim(np.zeros_like(ix),ind4d,maximum = tp.itmax,kernel_type = tkernel)
        dic_ind = {
                'time':ind4d[0],
                'Z':ind4d[1],
                'Zl':ind4d[1],
                'face':ind4d[2],
                'Yp1':ind4d[3],
                'Y':ind4d[3],
                'Xp1':ind4d[4],
                'X':ind4d[4]
            }
        try:
            rz
        except NameError:
            rz = 0

        try:
            rt
        except NameError:
            rt = 0
            
        if isinstance(name,str):
            if 'Zl' in od._ds[name].dims:
                mask = get_masked(od,tuple([i for i in ind4d[1:] if i is not None]),gridtype = 'Wvel')
                this_bottom_scheme = None
            else:
                mask = get_masked(od,tuple([i for i in ind4d[1:] if i is not None]))
                this_bottom_scheme = bottom_scheme
            pk4d = find_pk_4d(mask,russian_doll = doll)
            weight = get_weight_4d(rx,ry,rz,rt,pk4d,
                          hkernel = kernel,
                          russian_doll = doll,
                          funcs = funcs,
                          tkernel = tkernel,
                          zkernel = zkernel,
                          bottom_scheme = this_bottom_scheme
                         )
            
            n_s = sread(od._ds[name],[dic_ind[var] for var in od._ds[name].dims])
            
            s = np.einsum('nijk,nijk->n',n_s,weight)
            returns.append(s)
            
        elif len(name) == 2:
            uname,vname = name
            
            umask = get_masked(od,tuple([i for i in ind4d[1:] if i is not None]),gridtype = 'U')
            vmask = get_masked(od,tuple([i for i in ind4d[1:] if i is not None]),gridtype = 'V')
            n_u = sread(od._ds[uname],[dic_ind[var] for var in od._ds[uname].dims])
            n_v = sread(od._ds[vname],[dic_ind[var] for var in od._ds[vname].dims])
            if ind4d[2] is not None:
#                 hface = ind4d[2][:,:,0,0]
                UfromUvel,UfromVvel,VfromUvel, VfromVvel = tp.four_matrix_for_uv(hface)
                
                temp_n_u = np.einsum('nijk,ni->nijk',n_u,UfromUvel)+np.einsum('nijk,ni->nijk',n_v,UfromVvel)
                temp_n_v = np.einsum('nijk,ni->nijk',n_u,VfromUvel)+np.einsum('nijk,ni->nijk',n_v,VfromVvel)
                
                n_u = temp_n_u
                n_v = temp_n_v
                
                umask = np.round(np.einsum('nijk,ni->nijk',umask,UfromUvel)+
                                 np.einsum('nijk,ni->nijk',vmask,UfromVvel))
                vmask = np.round(np.einsum('nijk,ni->nijk',umask,VfromUvel)+
                                 np.einsum('nijk,ni->nijk',vmask,VfromVvel))
            udoll = doll
            vdoll = doll
            ufuncs = funcs
            vfuncs = funcs
#             udoll = u_doll
#             vdoll = v_doll
#             ukernels = [np.array([kernel[i] for i in dol]) for dol in u_doll]
#             ufuncs = [get_func(kernel = a_kernel,ktype = ktype,h_order = h_order) for a_kernel in ukernels]
#             vkernels = [np.array([kernel[i] for i in dol]) for dol in v_doll]
#             vfuncs = [get_func(kernel = a_kernel,ktype = ktype,h_order = h_order) for a_kernel in vkernels]
            
            upk4d = find_pk_4d(umask,russian_doll = udoll)
            vpk4d = find_pk_4d(vmask,russian_doll = vdoll)
            uweight = get_weight_4d(rx+1/2,ry,rz,rt,upk4d,
                      hkernel = kernel,
                      russian_doll = udoll,
                      funcs = ufuncs,
                      tkernel = tkernel,
                      zkernel = zkernel
                     )
            vweight = get_weight_4d(rx,ry+1/2,rz,rt,vpk4d,
                      hkernel = kernel,
                      russian_doll = vdoll,
                      funcs = vfuncs,
                      tkernel = tkernel,
                      zkernel = zkernel
                     )
            u = np.einsum('nijk,nijk->n',n_u,uweight)
            v = np.einsum('nijk,nijk->n',n_v,vweight)
            u,v = local_to_latlon(u,v,cs,st at 0x7f691d90a850>n)
            returns.append([u,v])
    return returns