import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import copy

from OceInterp.kernel_and_weight import kernel_weight,get_weight_cascade
from OceInterp.topology import topology
from OceInterp.utils import get_combination
from OceInterp.RuntimeConf import rcParam

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

def kash(kernel):#hash kernel
    temp_lst = [(i,j) for (i,j) in kernel]
    return hash(tuple(temp_lst))

def get_func(kernel,hkernel = 'interp',h_order = 0):
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
        
        if inheritance is not None and ignore_mask:
            print('Warning:overwriting the inheritance oject to None, because we ignore masking')
        
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
        type_same = isinstance(other, type(self))
        if not type_same:
            raise TypeError('the argument is not a KnW object')
        return (self.kernel == other.kernel).all()
    
    def same_size(self,other):
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
                     self.ignore_mask
                     self.h_order,
                     self.hkernel,
                     self.vkernel,
                     self.tkernel))
    
    def hash_largest(self):
        return hash((kash(self.kernel),
                     self.h_order,
                     self.hkernel,
                     self.vkernel,
                     self.tkernel))
    
    def size_hash(self):
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
                    weight[:,:,jz,jt] =   self.hfuncs[0](rx,ry)
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