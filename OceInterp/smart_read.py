import numpy as np
import xarray as xr
from collections import OrderedDict as orderdic

def smart_read(da,ind):
    '''
    Read from a xarray.DataArray given tuple indexes, and try to do it fast.

    **Parameters:**
    
    + da: xarray.DataArray
        DataArray to read from
    + ind: tuple of numpy.ndarray
        The indexes of points of interest, each element does not need to be 1D

    **Returns:**
    
    + values: numpy.ndarray
        The values of the points of interest. Has the same shape as the elements in ind. 
    '''
#     print('read called')
    the_shape = ind[0].shape
    ind = tuple([i.ravel() for i in ind])
    memory_chunk = 3
    xarray_more_efficient = 100
    if da.chunks is None:
        npck = np.array(da)
        return npck[ind].reshape(the_shape)
    if np.prod([len(i) for i in da.chunks])<=memory_chunk:# if the number of chunks is small don't bother 
        npck = np.array(da)
        return npck[ind].reshape(the_shape)
    cksz = orderdic(da.chunksizes)
    keys = list(cksz.keys())
    n = len(ind[0])
    result = np.zeros(n)
    
    if len(keys)!=len(ind):
        raise Exception('index does not match the number of dimensions')
    new_dic = dict()
    # typically what happens is that the first a few indexes are chunked
    # here we figure out what is the last dimension chunked.
    for i in range(len(cksz)-1,-1,-1):
        if len(cksz[keys[i]])>1:
            last = i
            break
    
    ckbl = np.zeros((n,i+1)).astype(int)
    # register each each dimension and the chunk they are in
    for i,k in enumerate(keys[:i+1]):
        ix = ind[i]
        suffix = np.cumsum(cksz[k])
        new_dic[i] = suffix
        ckbl[:,i] = np.searchsorted(suffix,ix,side = 'right')
    # this is the time limiting step for localized long query.
    ckus,inverse = np.unique(ckbl,axis = 0,return_inverse = True)
    # ckus is the individual chunks used
    if len(ckus) <=xarray_more_efficient:
#         print('use smart')
        for i,k in enumerate(ckus):
            ind_str = []
            pre = []
            which = (inverse == i)
            for j,p in enumerate(k):
                sf = new_dic[j][p]# the upperbound of index
                pr = sf-cksz[keys[j]][p]# the lower bound of index
                ind_str.append(f'{pr}:{sf}')
                pre.append(pr)
            prs = np.zeros(len(keys)).astype(int)
            prs[:last+1] = pre
            npck = eval(f'np.array(da[{",".join(ind_str)}])')
            subind = tuple([ind[dim][which]-prs[dim] for dim in range(len(ind))])
            result[which] = npck[subind]
        return result.reshape(the_shape)
    else:
#         print('use xarray')
        xrind = tuple([xr.DataArray(dim, dims=["x"]) for dim in ind])
        return np.array(da[xrind]).reshape(the_shape)