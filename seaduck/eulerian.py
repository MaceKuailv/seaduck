from seaduck.OceData import OceData
from seaduck.kernelNweight import KnW,_translate_to_tendency,find_pk_4d
# from OceInterp.kernel_and_weight import _translate_to_tendency,find_pk_4d
from seaduck.smart_read import smart_read as sread
from seaduck.get_masks import get_masked
from seaduck.utils import local_to_latlon,get_key_by_value,_general_len,local_to_latlon,to_180,get_combination,find_px_py,weight_f_node

import warnings
import numpy as np
import copy

# def to_180(x):
#     '''
#     convert any longitude scale to [-180,180)
#     '''
#     x = x%360
#     return x+(-1)*(x//180)*360

# def local_to_latlon(u,v,cs,sn):
#     '''convert local vector to north-east '''
#     uu = u*cs-v*sn
#     vv = u*sn+v*cs
#     return uu,vv

# def get_combination(lst,select):
#     '''
#     Iteratively find all the combination that
#     has (select) amount of elements
#     and every element belongs to lst
#     This is the same as the one in itertools, 
#     but I didn't know at the time.
#     '''
#     n = len(lst)
#     if select==1:
#         return [[num] for num in lst]
#     else:
#         the_lst = []
#         for i,num in enumerate(lst):
#             sub_lst = get_combination(lst[i+1:],select-1)
#             for com in sub_lst:
#                 com.append(num)
# #             print(sub_lst)
#             the_lst+=sub_lst
#         return the_lst
    
def _ind_broadcast(x,ind):
    '''
    Perform a "cartesian product" on two fattened dimensions 

    **Parameters:**
    
    + x: numpy.ndarray
        A new dimension
    + ind: tuple
        Existing indexes
    '''
    n = x.shape[0]
    if len(x.shape) ==1:
        x = x.reshape((n,1))
    xsp = x.shape
    ysp = ind[0].shape
    final_shape = [n]+list(ysp[1:])+list(xsp[1:])
    
    R = [np.zeros(final_shape,int) for i in range(len(ind)+1)]
    
    dims = len(final_shape)
    ydim = len(ysp)-1
    trsp = list(range(1,1+ydim))+[0]+list(range(1+ydim,dims))
    inv  = np.argsort(trsp)
    R[0] = R[0].transpose(trsp)
    R[0][:] = x
    R[0] = R[0].transpose(inv)
    
    for i in range(1,len(ind)+1):
        R[i] = R[i].T
        R[i][:] = ind[i-1].T
        R[i] = R[i].T
    return R

def _partial_flatten(ind):
    '''
    Converting a high dimensional index set to a 2D one
    '''
    if isinstance(ind,tuple):
        shape = ind[0].shape
        
        # num_neighbor = 1
        # for i in range(1,len(shape)):
        #     num_neighbor*=shape[i]
        R = []
        for i in range(len(ind)):
            R.append(ind[i].reshape(shape[0],-1))
        return tuple(R)
    elif isinstance(ind,np.ndarray):
        shape = ind.shape
        
        num_neighbor = 1
        for i in range(1,len(shape)):
            num_neighbor*=shape[i]
        return ind.reshape(shape[0],num_neighbor)
    
def _in_required(name,required):
    '''
    see if a name is in required. 
    '''
    if required == 'all':
        return True
    else:
        return name in required
    
def _ind_for_mask(ind,dims):
    '''
    If dims does not include a vertical dimension, assume to be 0.
    If dims has a temporal dimension, take it away. 
    Return the index for masking. 
    '''
    ind_for_mask = [ind[i] for i in range(len(ind)) if dims[i] not in ['time']]
    if 'Z' not in dims and 'Zl' not in dims:
        ind_for_mask.insert(0,np.zeros_like(ind[0]))
    return tuple(ind_for_mask)

def _subtract_i_min(ind,i_min):
    '''
    Subtract the index prefix from the actual index. 
    This is used when one is reading from a prefetched subset of the data.
    '''
    temp_ind = []
    for i in range(len(i_min)):
        temp_ind.append(ind[i]-i_min[i])
    return tuple(temp_ind)

class position():
    '''
    The position object that performs the interpolation.
    Create a empty one by default. 
    To actually do interpolation, use from_latlon method to tell the ducks where they are. 
    '''
#     self.ind_h_dict = {}
    def from_latlon(self,x = None,y = None,z = None,t = None,data = None):
        '''
        Use the methods from the ocedata to transform 
        from lat-lon-dep-time coords to rel-coords
        store the output in the position object. 

        **Parameters:**
        
        + x,y,z,t: numpy.ndarray
            1D array of the lat-lon-dep-time coords
        + data: OceData object
            The field where the positions are defined on. 
        '''
        if data is None:
            try:
                self.ocedata
            except AttributeError:
                raise ValueError('data not provided')
        else:
            self.ocedata = data
        self.tp = self.ocedata.tp
        length = [_general_len(i) for i in [x,y,z,t]]
        self.N = max(length)
        if any([i!= self.N for i in length if i>1]):
            raise ValueError('Shapes of input coordinates are not compatible')
        
        if isinstance(x,float):
            x = np.array([1.0])*x
        if isinstance(y,float):
            y = np.array([1.0])*y
        if isinstance(z,float):
            z = np.array([1.0])*z
        if isinstance(z,float):
            t = np.array([1.0])*t
        
        for thing in [x,y,z,t]:
            if len(x.shape)>1:
                raise ValueError('Input need to be 1D numpy arrays')
        if (x is not None) and (y is not None):
            self.lon = x
            self.lat = y
            (
                 self.face,
                 self.iy,
                 self.ix,
                 self.rx,
                 self.ry,
                 self.cs,
                 self.sn,
                 self.dx,
                 self.dy,
                 self.bx,
                 self.by
            ) = self.ocedata.find_rel_h(x,y)
        else:
            self.lon  = None
            self.lat  = None
            self.face = None
            self.iy   = None
            self.ix   = None
            self.rx   = None
            self.ry   = None
            self.cs   = None
            self.sn   = None
            self.dx   = None
            self.dy   = None
            self.bx   = None
            self.by   = None
        if (z is not None):
            self.dep = z
            if (self.ocedata.readiness['Z']):
                (
                    self.iz,
                    self.rz,
                    self.dz,
                    self.bz 
                ) = self.ocedata.find_rel_v(z)
            else:
                (
                    self.iz,
                    self.rz,
                    self.dz,
                    self.bz,
                ) = [None for i in range(4)]
            if (self.ocedata.readiness['Zl']):
                (
                    self.izl,
                    self.rzl,
                    self.dzl,
                    self.bzl 
                ) = self.ocedata.find_rel_vl(z)
            else:
                (
                    self.izl,
                    self.rzl,
                    self.dzl,
                    self.bzl,
                ) = [None for i in range(4)]
        else:
            (
                self.iz,
                self.rz,
                self.dz,
                self.bz,
                self.izl,
                self.rzl,
                self.dzl,
                self.bzl,
                self.dep
            ) = [None for i in range(9)]
            
        if (t is not None):
            self.t = t
            if self.ocedata.readiness['time']:
                (
                    self.it,
                    self.rt,
                    self.dt,
                    self.bt 
                ) = self.ocedata.find_rel_t(t)
            else:
                (
                    self.it,
                    self.rt,
                    self.dt,
                    self.bt,
                ) = [None for i in range(4)]
        else:
            (
                self.it,
                self.rt,
                self.dt,
                self.bt,
                self.t
            ) = [None for i in range(5)]
        return self
    
    def subset(self,which):
        '''
        Create a subset of the position object

        **Parameters:**
        
        + which: numpy.ndarray
            Define which points survive the subset operation.
            It be an array of either boolean or int.
            The selection is similar to that of selecting from a 1D numpy array.

        **Returns:**
        
        + the_subset: position object
            The selected positions. 
        '''
        p = position()
        keys = self.__dict__.keys()
        for i in keys:
            item = self.__dict__[i]
            if isinstance(item,np.ndarray):
                if len(item.shape) ==1:
                    p.__dict__[i] = item[which]
                    p.N = len(p.__dict__[i])
                else:
                    p.__dict__[i] = item
            else:
                p.__dict__[i] = item
        # p.N = max([_general_len(i) for i in p.__dict__.values()])
        return p
        
    def fatten_h(self,knw,ind_moves_kwarg = {}):
        '''
        Fatten means to find the neighboring points of the points of interest based on the kernel.
        faces,iys,ixs are 1d arrays of size n. 
        We are applying a kernel of size m.
        This is going to return a n * m array of indexes. 
        A very slim vector is now a matrix, and hence the name. 
        each row represen all the node needed for interpolation of a single point.
        "h" represent we are only doing it on the horizontal plane

        **Parameters:**
        
        + knw: KnW object
            The kernel used to find neighboring points.
        + ind_moves_kward: dict
            Key word argument to put into ind_moves method of the topology object. 
            Read topology.ind_moves for more detail. 
        '''
#         self.ind_h_dict
        kernel = knw.kernel
        kernel_tends =  [_translate_to_tendency(k) for k in kernel]
        m = len(kernel_tends)
        n = len(self.iy)
        tp = self.ocedata.tp

        # the arrays we are going to return 
        if self.face is not None:
            n_faces = np.zeros((n,m))
            n_faces.T[:] = self.face
        n_iys = np.zeros((n,m))
        n_ixs = np.zeros((n,m))

        # first try to fatten it naively(fast and vectorized)
        for i,node in enumerate(kernel):
            x_disp,y_disp = node
            n_iys[:,i] = self.iy+y_disp
            n_ixs[:,i] = self.ix+x_disp
        if self.face is not None:
            illegal = tp.check_illegal((n_faces,n_iys,n_ixs))
        else:
            illegal = tp.check_illegal((n_iys,n_ixs))

        redo = np.array(np.where(illegal)).T
        for num,loc in enumerate(redo):
            j,i = loc
            if self.face is not None:
                ind = (self.face[j],self.iy[j],self.ix[j])
            else:
                ind = (self.iy[j],self.ix[j])
            # everyone start from the [0,0] node
            moves = kernel_tends[i]
            # moves is a list of operations to get to a single point
            #[2,2] means move to the left and then move to the left again.
            n_ind = tp.ind_moves(ind,moves,**ind_moves_kwarg)
            if self.face is not None:
                n_faces[j,i],n_iys[j,i],n_ixs[j,i] = n_ind
            else:
                n_iys[j,i],n_ixs[j,i] = n_ind
        if self.face is not None:
            return n_faces.astype('int'),n_iys.astype('int'),n_ixs.astype('int')
        else:
            return None,n_iys.astype('int'),n_ixs.astype('int')
        
    def fatten_v(self,knw):
        '''
        Finding the neighboring center grid points in the vertical direction.

        **Parameters:**
        
        + knw: KnW object
            The kernel used to find neighboring points.
        '''
        if self.iz is None:
            return None
        if knw.vkernel == 'nearest':
            return copy.deepcopy(self.iz.astype(int))
        elif knw.vkernel in ['dz','linear']:
            try:
                self.iz_lin
            except AttributeError:
                (
                    self.iz_lin,
                    self.rz_lin,
                    self.dz_bin,
                    self.bz_lin
                ) = self.ocedata.find_rel_v_lin(self.dep)
            return np.vstack([self.iz_lin.astype(int),self.iz_lin.astype(int)-1]).T
        else:
            raise Exception('vkernel not supported')
            
            
    def fatten_vl(self,knw):
        '''
        Finding the neighboring staggered grid points in the vertical direction.

        **Parameters:**
        
        + knw: KnW object
            The kernel used to find neighboring points.
        '''
        if self.izl is None:
            return None
        if knw.vkernel == 'nearest':
            return copy.deepcopy(self.izl.astype(int))
        elif knw.vkernel in ['dz','linear']:
            try:
                self.izl_lin
            except AttributeError:
                (
                    self.izl_lin,
                    self.rzl_lin,
                    self.dzl_bin,
                    self.bzl_lin
                ) = self.ocedata.find_rel_vl_lin(self.dep)
            return np.vstack([self.izl_lin.astype(int),
                              self.izl_lin.astype(int)-1]).T
        else:
            raise Exception('vkernel not supported')
            
    def fatten_t(self,knw):
        '''
        Finding the neighboring center grid points in the temporal dimension.

        **Parameters:**
        
        + knw: KnW object
            The kernel used to find neighboring points.
        '''
        if self.it is None:
            return None
        if knw.tkernel == 'nearest':
            return copy.deepcopy(self.it.astype(int))
        elif knw.tkernel in ['dt','linear']:
            try:
                self.it_lin
            except AttributeError:
                (
                    self.it_lin,
                    self.rt_lin,
                    self.dt_bin,
                    self.bt_lin
                ) = self.ocedata.find_rel_t_lin(self.t)
            return np.vstack([self.it_lin.astype(int),self.it_lin.astype(int)+1]).T
        else:
            raise Exception('vkernel not supported')
    
    def fatten(self,knw,fourD = False,required = 'all',ind_moves_kwarg = {}):
        '''
        Finding the neighboring center grid points in all 4 dimensions.

        **Parameters:**
        
        + knw: KnW object
            The kernel used to find neighboring points.
        + fourD: Boolean
            When we are doing nearest neighbor interpolation on some of the dimensions, 
            with fourD = True, this will create dimensions with length 1, and will squeeze 
            the dimension if fourD = False
        + required: str, iterable of str
            Which dims is needed in the process
        + ind_moves_kward: dict
            Key word argument to put into ind_moves method of the topology object. 
            Read topology.ind_moves for more detail.
        '''
        if required!='all' and isinstance(required,str):
            required = tuple([required])
        if required =='all' or isinstance(required,tuple):
            pass
        else:
            required = tuple(required)
        
        #TODO: register the kernel shape
        if _in_required('X',required) or _in_required('Y',required) or _in_required('face',required):
            ffc,fiy,fix = self.fatten_h(knw,ind_moves_kwarg = ind_moves_kwarg)
            if ffc is not None:
                R = (ffc,fiy,fix)
                keys = ['face','Y','X']
            else:
                R = (fiy,fix)
                keys = ['Y','X']
        else:
            R = (np.zeros(self.N))
            keys = ['place_holder']
            
        if _in_required('Z',required):
            fiz = self.fatten_v(knw)
            if fiz is not None:
                R = _ind_broadcast(fiz,R)
                keys.insert(0,'Z')
        elif _in_required('Zl',required):
            fizl = self.fatten_vl(knw)
            if fizl is not None:
                R = _ind_broadcast(fizl,R)
                keys.insert(0,'Zl')
        elif fourD:
            R = [np.expand_dims(R[i],axis = -1) for i in range(len(R))]
            
        if _in_required('time',required):
            fit = self.fatten_t(knw)
            if fit is not None:
                R = _ind_broadcast(fit,R)
                keys.insert(0,'time')
        elif fourD:
            R = [np.expand_dims(R[i],axis = -1) for i in range(len(R))]
        R = dict(zip(keys,R))
        if required == 'all':
            required = [i for i in keys if i!='place_holder']
        return tuple([R[i] for i in required])
    
    def get_px_py(self):
        '''
        Get the nearest 4 corner points of the given point. 
        Used for oceanparcel style horizontal interpolation.
        '''
        if self.face is not None:
            return find_px_py(self.ocedata.XG,
                              self.ocedata.YG,
                              self.ocedata.tp,
                              self.face,
                              self.iy,self.ix
                             )
        else:
            return find_px_py(self.ocedata.XG,
                              self.ocedata.YG,
                              self.ocedata.tp,
                              self.iy,self.ix
                             )
    def get_f_node_weight(self):
        '''
        The weight for the corner points interpolation
        '''
        return weight_f_node(self.rx,self.ry)
    def get_lon_lat(self):
        '''
        Return the lat-lon value based on relative coordinate.
        This method only work if the dataset has readiness['h'] == 'oceanparcel'
        '''
        px,py = self.get_px_py()
        w = self.get_f_node_weight()
        lon = np.einsum('nj,nj->n',w,px.T)
        lat = np.einsum('nj,nj->n',w,py.T)
        return lon,lat
            
    
    def _get_needed(self,varName,knw,required = None,prefetched = None,**kwarg):
        '''An internal testing function'''
        if required is None:
            required = self.ocedata._ds[varName].dims
        ind = self.fatten(knw,required = required,**kwarg)
        if len(ind)!= len(self.ocedata._ds[varName].dims):
            raise Exception("""dimension mismatch.
                            Please check if the position objects have all the dimensions needed""")
        if prefetched is None:
            return sread(self.ocedata[varName],ind)
        else:
            return prefetched[ind]
    
    def _get_masked(self,knw,gridtype = 'C',**kwarg):
        '''An internal testing function'''
        ind = self.fatten(knw,fourD = True,**kwarg)
        if self.it is not None:
            ind = ind[1:]
        if len(ind)!=len(self.ocedata._ds['maskC'].dims):
            raise Exception("""dimension mismatch.
                            Please check if the position objects have all the dimensions needed""")
        return get_masked(self.ocedata,ind,gridtype = gridtype)
    
    def _find_pk4d(self,knw,gridtype = 'C'):
        '''An internal testing function'''
        masked = self._get_masked(knw,gridtype = gridtype)
        pk4d = find_pk_4d(masked,russian_doll = knw.inheritance)
        return pk4d
    
    def _register_interpolation_input(self,varName,knw,
                    prefetched = None,i_min = None):
        '''
        Part of the interpolation function.
        Register the input of the interpolation function.
        For the input format, go to interpolation for more details. 

        **Returns:**
        
        + output_format: dict
            Record information about the original varName input. 
            Provide the formatting information for output, 
            such that it matches the input in a direct fashion.
        + main_keys: list
            A list that defines each interpolation to be performed. 
            The combination of variable name and kernel uniquely define such an operation.
        + prefetch_dict: dict
            Lookup the prefetched the data and its index prefix using main_key
        + main_dict: dict
            Lookup the raw information using main_key
        + hash_index: dict
            Lookup the token that uniquely define its indexing operation using main_key. 
            Different main_key could share the same token. 
        + hash_mask: dict
            Lookup the token that uniquely define its masking operation using main_key. 
            Different main_key could share the same token. 
        + hash_read: dict
            Lookup the token that uniquely define its reading of the data using main_key. 
            Different main_key could share the same token. 
        + hash_weight: dict
            Lookup the token that uniquely define its computation of the weight using main_key. 
            Different main_key could share the same token. 
        '''
        # prefetch_dict = {var:(prefetched,i_min)}
        # main_dict = {var:(var,kernel)}
        # hash_index = {var:hash(cuvg,kernel_size)}
        # hash_read  = {var:hash(var,kernel_size)}
        # hash_weight= {var:hash(kernel,cuvg)}
        output_format = {}
        if isinstance(varName,str) or isinstance(varName,tuple):
            varName = [varName]
            output_format['single'] = True
        elif isinstance(varName,list):
            output_format['single'] = False
        else:
            raise ValueError('varName needs to be string, tuple, or a list of the above.')
        Nvar = len(varName)
        
        if isinstance(knw,KnW):
            knw = [knw for i in range(Nvar)]
        if isinstance(knw,tuple):
            if len(knw)!=2:
                raise ValueError('When knw is a tuple, we assume it to be kernels for a horizontal vector,'
                                 'thus, it has to have 2 elements')
            knw = [knw for i in range(Nvar)]
        elif isinstance(knw,list):
            if len(knw)!=Nvar:
                raise ValueError('Mismatch between the number of kernels and variables')
        elif isinstance(knw,dict):
            temp = []
            for var in varName:
                a_knw = knw.get(var)
                if (a_knw is None) or not(isinstance(a_knw,KnW)):
                    raise ValueError(f'Variable {var} does not have a proper corresponding kernel')
                else:
                    temp.append(a_knw)
            knw = temp
        else:
            raise ValueError('knw needs to be a KnW object, or list/dictionaries containing that ')
            
        if isinstance(prefetched,np.ndarray):
            prefetched = [prefetched for i in range(Nvar)]
        elif isinstance(prefetched,tuple):
            prefetched = [prefetched for i in range(Nvar)]
        elif prefetched is None:
            prefetched = [prefetched for i in range(Nvar)]
        elif isinstance(prefetched,list):
            if len(prefetched)!=Nvar:
                raise ValueError('Mismatch between the number of prefetched arrays and variables')
        elif isinstance(prefetched,dict):
            prefetched = [prefetched.get(var) for var in varName]
        else:
            raise ValueError('prefetched needs to be a numpy array/tuple pair of numpy array,'
                             ' or list/dictionaries containing that')
            
        if isinstance(i_min,tuple):
            i_min = [i_min for i in range(Nvar)]
        elif i_min is None:
            i_min = [None for i in range(Nvar)]
        elif isinstance(i_min,list):
            if len(i_min)!=Nvar:
                raise ValueError('Mismatch between the number of prefetched arrays prefix i_min and variables')
        elif isinstance(i_min,dict):
            i_min = [i_min.get(var) for var in varName]
        else:
            raise ValueError('prefetched prefix i_min needs to be a tuple, or list/dictionaries containing that ')
            
            
        output_format['ori_list'] = copy.deepcopy(list(zip(varName,knw)))
        new_varName = []
        new_prefetched = []
        new_knw = []
        new_i_min = []
        for i,var in enumerate(varName):
            if isinstance(var,str):
                new_varName.append(var)
                new_prefetched.append(prefetched[i])
                new_knw.append(knw[i])
                new_i_min.append(i_min[i])
            elif isinstance(var,tuple):
                if self.face is None:
                    for j in range(len(var)):
                        new_varName.append(var[j])
                        new_prefetched.append(prefetched[i][j])
                        new_knw.append(knw[i][j])
                        new_i_min.append(i_min[i])
                else:
                    new_varName.append(var)
                    new_prefetched.append(prefetched[i])
                    new_knw.append(knw[i])
                    new_i_min.append(i_min[i])
            elif var is None:
                pass
            else:
                raise ValueError('varName needs to be string, tuple, or a list of the above.')
        
        
        # new_prefetched = []
        # new_knw = []
        # new_i_min = []
        # for i in range(len(knw)):
        #     if output_format['ori_list'][i] is None:
        #         pass
        #     elif isinstance(output_format['ori_list'][i],tuple):
        #         if self.face is None:
        #             for j in range(len(output_format['ori_list'][i])):
        #                 new_prefetched.append(prefetched[i][j])
        #                 new_knw.append(knw[i][j])
        #                 new_i_min.append(i_min[i])
        #         else:
        #             new_prefetched.append(prefetched[i])
        #             new_knw.append(knw[i])
        #             new_i_min.append(i_min[i])
        #     elif isinstance(output_format['ori_list'][i],str):
        #         new_prefetched.append(prefetched[i])
        #         new_knw.append(knw[i])
        #         new_i_min.append(i_min[i])
        prefetched = new_prefetched
        knw = new_knw
        i_min = new_i_min
        varName = new_varName
        output_format['final_varName'] = list(zip(varName,knw))
            
            
        kernel_size_hash = []
        kernel_hash = []
        mask_ignore = []
        for kkk in knw:
            if isinstance(kkk,KnW):
                kernel_size_hash.append(kkk.size_hash())
                kernel_hash.append(hash(kkk))
                mask_ignore.append(kkk.ignore_mask)
            elif isinstance(kkk,tuple):
                if len(kkk)!=2:
                    raise ValueError('When knw is a tuple, we assume it to be kernels for a horizontal vector,'
                                     'thus, it has to have 2 elements')
                uknw,vknw = kkk
                # if not uknw.same_size(vknw):
                #     raise Exception('u,v kernel needs to have same size'
                #                     'to navigate the complex grid orientation.'
                #                     'use a kernel that include both of the uv kernels'
                #                    )
                kernel_size_hash.append(uknw.size_hash())
                kernel_hash.append(hash((uknw,vknw)))
                mask_ignore.append(uknw.ignore_mask or vknw.ignore_mask)
        dims = []
        for var in varName:
            if isinstance(var,str):
                dims.append(self.ocedata[var].dims)
            elif isinstance(var,tuple):
                temp = []
                for vvv in var:
                    temp.append(self.ocedata[vvv].dims)
                dims.append(tuple(temp))
        
        main_keys = list(zip(varName,kernel_hash))
        prefetch_dict = dict(zip(main_keys,zip(prefetched,i_min)))
        main_dict     = dict(zip(main_keys,zip(varName,dims,knw)))
        hash_index    = dict(zip(main_keys,[hash(i) for i in zip(dims,kernel_size_hash)]))
        hash_mask     = dict(zip(main_keys,[hash(i) for i in zip(dims,mask_ignore,kernel_size_hash)]))
        hash_read     = dict(zip(main_keys,[hash(i) for i in zip(varName,kernel_size_hash)]))
        hash_weight   = dict(zip(main_keys,[hash(i) for i in zip(dims,kernel_hash)]))
        return output_format,main_keys,prefetch_dict,main_dict,hash_index,hash_mask,hash_read,hash_weight
        
    
    def _fatten_required_index_and_register(self,hash_index,main_dict):
        '''
        Perform the fatten process for each unique token. Register the result as a dict.

        **Parameters:**
        
        + hash_index: dict
            See _register_interpolation_input
        + main_dict: dict
            See _register_interpolation_input

        **Returns:**
        
        + index_lookup: dict
            A dictionary to lookup fatten results. 
            The keys are the token in hash_index.
            The values are corresponding results. 
        '''
        hsh = np.unique(list(hash_index.values()))
        index_lookup = {}
        for hs in hsh:
            main_key = get_key_by_value(hash_index,hs)
            varName,dims,knw = main_dict[main_key]
            if isinstance(varName,str):
                old_dims = dims
            elif isinstance(varName,tuple):
                old_dims = dims[0]
            dims = []
            for i in old_dims:
                if i in ['Xp1','Yp1']:
                    dims.append(i[:1])
                else:
                    dims.append(i)
            dims = tuple(dims)
            if isinstance(varName,str):
                ind = self.fatten(knw,required = dims,fourD = True)
                index_lookup[hs] = ind
            elif isinstance(varName,tuple):
                uknw,vknw = knw
                uind = self.fatten(uknw,required = dims,fourD = True,ind_moves_kwarg = {'cuvg':'U'})
                vind = self.fatten(vknw,required = dims,fourD = True,ind_moves_kwarg = {'cuvg':'V'})
                index_lookup[hs] = (uind,vind)
            
        return index_lookup
    
    def _transform_vector_and_register(self,index_lookup,hash_index,main_dict):
        '''
        Perform the vector transformation. 
        This is to say that sometimes u velocity becomes v velocity for datasets with face topology.
          Register the result as a dict.

        **Parameters:**
        
        + index_lookup: dict
            See _fatten_required_index_and_register
        + hash_index: dict
            See _register_interpolation_input
        + main_dict: dict
            See _register_interpolation_input

        **Returns:**
        
        + transform_lookup: dict
            A dictionary to lookup transformation results. 
            The keys are the token in hash_index.
            The values are corresponding results. 
        '''
        hsh = np.unique(list(hash_index.values()))
        transform_lookup = {}
        if self.face is None:
            for hs in hsh:
                transform_lookup[hs] = None
            return transform_lookup
        for hs in hsh:
            main_key = get_key_by_value(hash_index,hs)
            varName,dims,knw = main_dict[main_key]
            if isinstance(varName,str):
                transform_lookup[hs] = None
            elif isinstance(varName,tuple):
                uind,vind = index_lookup[hs]
                uind_dic = dict(zip(dims[0],uind))
                vind_dic = dict(zip(dims[1],vind))
                # This only matters when dim == 'face', no need to think about 'Xp1'
                (UfromUvel,
                 UfromVvel,
                 _,
                 _        ) = self.ocedata.tp.four_matrix_for_uv(uind_dic['face'][:,:,0,0])

                (_,
                 _,
                 VfromUvel,
                 VfromVvel) = self.ocedata.tp.four_matrix_for_uv(vind_dic['face'][:,:,0,0])
                
                UfromUvel = np.round(UfromUvel)
                UfromVvel = np.round(UfromVvel)
                VfromUvel = np.round(VfromUvel)
                VfromVvel = np.round(VfromVvel)
                
                bool_ufromu = np.abs(UfromUvel).astype(bool)
                bool_ufromv = np.abs(UfromVvel).astype(bool)
                bool_vfromu = np.abs(VfromUvel).astype(bool)
                bool_vfromv = np.abs(VfromVvel).astype(bool)
                
                indufromu = tuple([idid[bool_ufromu] for idid in uind])
                indufromv = tuple([idid[bool_ufromv] for idid in uind])
                indvfromu = tuple([idid[bool_vfromu] for idid in vind])
                indvfromv = tuple([idid[bool_vfromv] for idid in vind])
                
                transform_lookup[hs] = (
                    (UfromUvel,UfromVvel,VfromUvel,VfromVvel),
                    (bool_ufromu,bool_ufromv,bool_vfromu,bool_vfromv),
                    (indufromu,indufromv,indvfromu,indvfromv)
                )
            else:
                raise ValueError(f'unsupported dims: {dims}')
        # modify the index_lookup
        return transform_lookup
    
    def _mask_value_and_register(self,index_lookup,transform_lookup,hash_mask,hash_index,main_dict):
        '''
        Perform the masking process and register in a dictionary. 

        **Parameters:**
        
        + index_lookup: dict
            See _fatten_required_index_and_register
        + transform_lookup: dict
            See _transform_vector_and_lookup 
        + hash_mask: dict
            See _register_interpolation_input
        + hash_index: dict
            See _register_interpolation_input
        + main_dict: dict
            See _register_interpolation_input

        **Returns:**
        
        + mask_lookup: dict
            A dictionary to lookup masking results. 
            The keys are the token in hash_mask.
            The values are corresponding results. 
        '''
        hsh = np.unique(list(hash_mask.values()))
        mask_lookup = {}
        for hs in hsh:
            main_key = get_key_by_value(hash_mask,hs)
            varName,dims,knw = main_dict[main_key]
            hsind = hash_index[main_key]
            longDims = ''.join([str(a_thing) for a_thing in dims])
            if isinstance(knw,KnW):
                ignore_mask = knw.ignore_mask
            elif isinstance(knw,tuple):
                ignore_mask = (knw[0].ignore_mask) or (knw[1].ignore_mask)
            
            if ignore_mask or ('X' not in longDims) or ('Y' not in longDims):
                mask_lookup[hs] = None
            elif isinstance(varName,str):
                ind = index_lookup[hsind]
                ind_for_mask = _ind_for_mask(ind,dims)
                if 'Zl' in dims:
                    cuvw = 'Wvel'
                elif 'Z' in dims:
                    if 'Xp1' in dims and 'Yp1' in dims:
                        raise NotImplementedError('The masking of corner points are open to '
                                                  'interpretations thus not implemented, '
                                                  'let knw.ignore_mask =True to go around')
                    elif 'Xp1' in dims:
                        cuvw =  'U'
                    elif 'Yp1' in dims:
                        cuvw = 'V'
                    else:
                        cuvw = 'C'
                else:
                    cuvw = 'C'
                masked = get_masked(self.ocedata,ind_for_mask,gridtype = cuvw)
                mask_lookup[hs] = masked
            elif isinstance(varName,tuple):
                to_unzip = transform_lookup[hsind]
                uind,vind = index_lookup[hsind]
                if to_unzip is None:
                    uind_for_mask = _ind_for_mask(uind,dims[0])
                    vind_for_mask = _ind_for_mask(vind,dims[1])
                    umask = get_masked(self.ocedata,uind_for_mask,gridtype = 'U')
                    vmask = get_masked(self.ocedata,vind_for_mask,gridtype = 'V')
                else:
                    (
                        _,
                        (bool_ufromu,bool_ufromv,bool_vfromu,bool_vfromv),
                        (indufromu,indufromv,indvfromu,indvfromv)
                    ) = to_unzip
                    umask = np.full(uind[0].shape,np.nan)
                    vmask = np.full(vind[0].shape,np.nan)
                    
                    umask[bool_ufromu] = get_masked(self.ocedata,_ind_for_mask(indufromu,dims[0]),gridtype = 'U')
                    umask[bool_ufromv] = get_masked(self.ocedata,_ind_for_mask(indufromv,dims[1]),gridtype = 'V')
                    vmask[bool_vfromu] = get_masked(self.ocedata,_ind_for_mask(indvfromu,dims[0]),gridtype = 'U')
                    vmask[bool_vfromv] = get_masked(self.ocedata,_ind_for_mask(indvfromv,dims[1]),gridtype = 'V')
                mask_lookup[hs] = (umask,vmask)
        return mask_lookup
    
    def _read_data_and_register(self,index_lookup,transform_lookup,hash_read,hash_index,main_dict,prefetch_dict):
        '''
        Read the data and register them as dict. 

        **Parameters:**
        
        + index_lookup: dict
            See _fatten_required_index_and_register
        + transform_lookup: dict
            See _transform_vector_and_lookup 
        + hash_read: dict
            See _register_interpolation_input
        + hash_index: dict
            See _register_interpolation_input
        + main_dict: dict
            See _register_interpolation_input
        + prefetch_dict: dict
            See _register_interpolation_input

        **Returns:**
        
        + read_lookup: dict
            A dictionary to lookup data reading results. 
            The keys are the token in hash_read.
            The values are corresponding results. 
        '''
        hsh = np.unique(list(hash_read.values()))
        data_lookup = {}
        for hs in hsh:
            main_key = get_key_by_value(hash_read,hs)
            hsind = hash_index[main_key]
            varName,dims,knw = main_dict[main_key]
            prefetched,i_min = prefetch_dict[main_key]
            if isinstance(varName,str):
                ind = index_lookup[hsind]
                if prefetched is not None:
                    if i_min is None:
                        raise ValueError('please pass value of the prefix of prefetched dataset, '
                                         'even if the prefix is zero')
                    temp_ind = _subtract_i_min(ind,i_min)
                    needed = np.nan_to_num(prefetched[temp_ind])
                else:
                    needed = np.nan_to_num(sread(self.ocedata[varName],ind))
                data_lookup[hs] = needed
            elif isinstance(varName,tuple):
                uname,vname = varName
                uind,vind = index_lookup[hsind]
                (
                    (UfromUvel,UfromVvel,VfromUvel,VfromVvel),
                    (bool_ufromu,bool_ufromv,bool_vfromu,bool_vfromv),
                    (indufromu,indufromv,indvfromu,indvfromv)
                ) = transform_lookup[hsind]
                temp_n_u = np.full(uind[0].shape,np.nan)
                temp_n_v = np.full(vind[0].shape,np.nan)
                if prefetched is not None:
                    upre,vpre = prefetched
                    ufromu = np.nan_to_num(upre[_subtract_i_min(indufromu,i_min)])
                    ufromv = np.nan_to_num(vpre[_subtract_i_min(indufromv,i_min)])
                    vfromu = np.nan_to_num(upre[_subtract_i_min(indvfromu,i_min)])
                    vfromv = np.nan_to_num(vpre[_subtract_i_min(indvfromv,i_min)])
                else:
                    ufromu = np.nan_to_num(sread(self.ocedata[uname],indufromu))
                    ufromv = np.nan_to_num(sread(self.ocedata[vname],indufromv))
                    vfromu = np.nan_to_num(sread(self.ocedata[uname],indvfromu))
                    vfromv = np.nan_to_num(sread(self.ocedata[vname],indvfromv))
                temp_n_u[bool_ufromu] = ufromu#0#
                temp_n_u[bool_ufromv] = ufromv#1#
                temp_n_v[bool_vfromu] = vfromu#0#
                temp_n_v[bool_vfromv] = vfromv#1#
                # dont_break = np.isclose(temp_n_v,2).any()
                # if not dont_break:
                #     print(bool_vfromu)
                #     raise Exception('what is going on')
                # else:
                #     print('somehow it passed')
                
                n_u = (np.einsum('nijk,ni->nijk',temp_n_u,UfromUvel)
                      +np.einsum('nijk,ni->nijk',temp_n_u,UfromVvel))
                n_v = (np.einsum('nijk,ni->nijk',temp_n_v,VfromUvel)
                      +np.einsum('nijk,ni->nijk',temp_n_v,VfromVvel))
                data_lookup[hs] = (n_u,n_v)
                
        return data_lookup
    
    def _compute_weight_and_register(self,mask_lookup,hash_weight,hash_mask,main_dict):
        '''
        Read the data and register them as dict. 

        **Parameters:**
        
        + mask_lookup: dict
            See _mask_value_and_register
        + hash_weight: dict
            See _register_interpolation_input
        + hash_mask: dict
            See _register_interpolation_input
        + main_dict: dict
            See _register_interpolation_input

        **Returns:**
        
        + weight_lookup: dict
            A dictionary to lookup the weights computed. 
            The keys are the token in hash_weight.
            The values are corresponding results. 
        '''
        hsh = np.unique(list(hash_weight.values()))
        weight_lookup = {}
        for hs in hsh:
            main_key = get_key_by_value(hash_weight,hs)
            hsmsk = hash_mask[main_key]
            varName,dims,knw = main_dict[main_key]
            masked = mask_lookup[hsmsk]
            if isinstance(varName,tuple):
                ori_dims = dims
                dims = ori_dims[0]
                ori_knw = knw
                knw = ori_knw[0]
                # Assuming the two kernels have the same 
                # vertical dimensions, which is reasonable.
            
            # shared part for 'vertical direction'
            this_bottom_scheme = 'no_flux'
            if 'Z' in dims:
                if self.rz is not None:
                    if knw.vkernel == 'nearest':
                        rz = self.rz
                    else:
                        rz = self.rz_lin
                else:
                    rz = 0
            elif 'Zl' in dims:
                this_bottom_scheme = None
                if self.rz is not None:
                    if knw.vkernel == 'nearest':
                        rz = self.rzl
                    else:
                        rz = self.rzl_lin
                else:
                    rz = 0
            else:
                rz = 0
            if self.rt is not None:
                if knw.tkernel == 'nearest':
                    rt = self.rt
                else:
                    rt = self.rt_lin
            else:
                rt = 0
                
            if isinstance(varName,str):
                if 'Xp1' in dims:
                    rx = self.rx+0.5
                else:
                    rx = self.rx
                if 'Yp1' in dims:
                    ry = self.ry+0.5
                else:
                    ry = self.ry
                if masked is None:
                    pk4d = None
                else:
                    pk4d = find_pk_4d(masked,russian_doll = knw.inheritance)
                weight = knw.get_weight(rx = rx,ry = ry,
                                    rz = rz,rt = rt,
                                    pk4d = pk4d,
                                    bottom_scheme = this_bottom_scheme)
                weight_lookup[hs] = weight
            elif isinstance(varName,tuple):
                uknw,vknw = ori_knw
                if masked is None:
                    upk4d = None
                    vpk4d = None
                else:
                    umask,vmask = masked
                    upk4d = find_pk_4d(umask,russian_doll = uknw.inheritance)
                    vpk4d = find_pk_4d(vmask,russian_doll = vknw.inheritance)
                uweight = uknw.get_weight(self.rx+1/2,self.ry,rz = rz,rt = rt,pk4d = upk4d)
                vweight = vknw.get_weight(self.rx,self.ry+1/2,rz = rz,rt = rt,pk4d = vpk4d)
                weight_lookup[hs] = (uweight,vweight)
        return weight_lookup
    
    def interpolate(self,varName,knw,
                    vec_transform = True,
                    prefetched = None,i_min = None):
        '''
        This is the method that does the actual interpolation/derivative.
        It is a combination of the following methods:
        _register_interpolation_input,
        _fatten_required_index_and_register,
        _transform_vector_and_register,
        _read_data_and_register,
        _mask_value_and_register,
        _compute_weight_and_registe,

        **Parameters:**
        
        + varName: list, str, tuple
            The variables to interpolate. Tuples are used for horizontal vectors. 
            Put str and list in a list if you have multiple things to interpolate. 
            This input also defines the format of the output. 
        + knw: KnW object, tuple, list, dict
            The kernel object used for the operation. 
            Put them in the same order as varName. 
            Some level of automatic broadcasting is also supported. 
        + vec_transform: Boolean
            Whether to project the vector field to the local zonal/meridional direction. 
        + prefetched: numpy.ndarray, tuple, list, dict, None
            The prefetched array for the data, this will effectively overwrite varName.
            Put them in the same order as varName. 
            Some level of automatic broadcasting is also supported.
        + i_min: tuple, list, dict, None
            The prefix of the prefetched array. 
            Put them in the same order as varName. 
            Some level of automatic broadcasting is also supported.

        **Returns:**
        
        + R: list, numpy.array, tuple
            The interpolation/derivative output in the same format as varName. 
        '''
        R = []
        (
            output_format,
            main_keys,
            prefetch_dict,
            main_dict,
            hash_index,
            hash_mask,
            hash_read,
            hash_weight
        ) = self._register_interpolation_input(varName,knw,
                    prefetched = prefetched,i_min = i_min)
        index_lookup = self._fatten_required_index_and_register(hash_index,
                                                                main_dict)
        transform_lookup = self._transform_vector_and_register(index_lookup,
                                                               hash_index,
                                                               main_dict)
        data_lookup = self._read_data_and_register(index_lookup,transform_lookup,
                                                   hash_read,hash_index,
                                                   main_dict,prefetch_dict)
        mask_lookup = self._mask_value_and_register(index_lookup,transform_lookup,
                                                    hash_mask,hash_index,
                                                    main_dict)
        weight_lookup = self._compute_weight_and_register(mask_lookup,
                                                          hash_weight,hash_mask,
                                                          main_dict)
        # index_list = []
        for key in main_keys:
            varName,dims,knw = main_dict[key]
            if isinstance(varName,str):
                needed = data_lookup[hash_read[key]]
                weight = weight_lookup[hash_weight[key]]
                needed = _partial_flatten(needed)
                weight = _partial_flatten(weight)
                R.append(np.einsum('nj,nj->n',needed,weight))
                # index_list.append((index_lookup[hash_index[key]],
                #                    transform_lookup[hash_index[key]],
                #                    data_lookup[hash_read[key]]))
            elif isinstance(varName,tuple):
                n_u,n_v = data_lookup[hash_read[key]]
                uweight,vweight = weight_lookup[hash_weight[key]]
                u = np.einsum('nijk,nijk->n',n_u,uweight)
                v = np.einsum('nijk,nijk->n',n_v,vweight)
                if vec_transform:
                    u,v = local_to_latlon(u,v,self.cs,self.sn)
                R.append((u,v))
                # index_list.append((index_lookup[hash_index[key]],
                #                    transform_lookup[hash_index[key]],
                #                    data_lookup[hash_read[key]]))
            else:
                raise ValueError(f'unexpected varName: {varName}')
                
        final_dict = dict(zip(output_format['final_varName'],R))
        ori_list = output_format['ori_list']
        output = []
        # print(ori_list,R,final_dict.keys())
        for key in ori_list:
            var,knw = key
            
            if var is None:
                output.append(None)
            elif isinstance(var, tuple):
                if self.face is None:
                    temp_key = [(var[i],knw[i]) for i in range(len(var))]
                    values = tuple(final_dict.get(k) for k in temp_key)
                    output.append(values)
                else:
                    output.append(final_dict.get(key))
            else:
                output.append(final_dict.get(key))
        if output_format['single']:
            output = output[0]
        return output
    
#     def interpolate(self,varName,knw,
#                     vec_transform = True,
#                     prefetched = None,i_min = None):
#         # implement shortcut u,v,w
#         # TODO: fix the very subtle bug that cause velocity component parallel to face connection
#         # TODO: add function to interpolate multiple variable at once. 
#         if prefetched is not None:
#             # TODO: I could have a warning about prefetch
#             # overwriting varName.
#             # But should I?
#             pass
#         if isinstance(varName,str):
#             old_dims = self.ocedata._ds[varName].dims
#             dims = []
#             for i in old_dims:
#                 if i in ['Xp1','Yp1']:
#                     dims.append(i[:1])
#                 else:
#                     dims.append(i)
#             dims = tuple(dims)
#             if 'Xp1' in old_dims:
#                 rx = self.rx+0.5
#             else:
#                 rx = self.rx
#             if 'Yp1' in old_dims:
#                 ry = self.ry+0.5
#             else:
#                 ry = self.ry
#             ind = self.fatten(knw,required = dims,fourD = True)
#             ind_dic = dict(zip(dims,ind))
#             if prefetched is not None:
#                 temp_ind = []
#                 if i_min is None:
#                     raise ValueError('please pass value for the prefix of prefetched dataset, even if the prefix is zero')
#                 for i,dim in enumerate(dims):
#                     a_ind = ind[i]
#                     a_ind-= i_min[i]
#                     temp_ind.append(a_ind)
#                 temp_ind = tuple(temp_ind)
#                 needed = np.nan_to_num(prefetched[temp_ind])
#             else:
#                 needed = np.nan_to_num(sread(self.ocedata[varName],ind))
            
#             if 'Z' in dims:
#                 if self.rz is not None:
#                     if knw.vkernel == 'nearest':
#                         rz = self.rz
#                     else:
#                         rz = self.rz_lin
#                 else:
#                     rz = 0
#             elif 'Zl' in dims:
#                 if self.rz is not None:
#                     if knw.vkernel == 'nearest':
#                         rz = self.rzl
#                     else:
#                         rz = self.rzl_lin
#                 else:
#                     rz = 0
#             else:
#                 rz = 0

#             if self.rt is not None:
#                 if knw.tkernel == 'nearest':
#                     rt = self.rt
#                 else:
#                     rt = self.rt_lin
#             else:
#                 rt = 0
            
#             if not ('X' in dims and 'Y' in dims):
#                 # if it does not have a horizontal dimension, then we don't have to mask
#                 masked = np.ones_like(ind[0])
#             else:
#                 if 'Zl' in dims:
#                     # something like wvel
#                     ind_for_mask = tuple([ind[i] for i in range(len(ind)) if dims[i] not in ['time']])
#                     masked = get_masked(self.ocedata,ind_for_mask,gridtype = 'Wvel')
#                     this_bottom_scheme = None
#                 elif 'Z' in dims:
#                     # something like salt
#                     ind_for_mask = tuple([ind[i] for i in range(len(ind)) if dims[i] not in ['time']])
#                     masked = get_masked(self.ocedata,ind_for_mask,gridtype = 'C')
#                     this_bottom_scheme = 'no_flux'
#                 else:
#                     # something like etan
#                     ind_for_mask = [ind[i] for i in range(len(ind)) if dims[i] not in ['time']]
#                     ind_for_mask.insert(0,np.zeros_like(ind[0]))
#                     ind_for_mask = ind_for_mask
#                     masked = get_masked(self.ocedata,ind_for_mask,gridtype = 'C')
#                     this_bottom_scheme = 'no_flux'
                    
#             pk4d = find_pk_4d(masked,russian_doll = knw.inheritance)

#             weight = knw.get_weight(rx = rx,ry = ry,
#                                     rz = rz,rt = rt,
#                                     pk4d = pk4d,
#                                     bottom_scheme = this_bottom_scheme)

#             needed = partial_flatten(needed)
#             weight = partial_flatten(weight)

#             R = np.einsum('nj,nj->n',needed,weight)
#             return R
# # vector case is here
#         elif isinstance(varName,list) or isinstance(varName,tuple):
#             if len(varName)!=2:
#                 raise Exception('list varName can only have length 2, representing horizontal vectors')
#             uname,vname = varName
#             uknw,vknw = knw
            
#             if prefetched is not None:
#                 upre,vpre = prefetched
#             else:
#                 upre = None
#                 vpre = None
                
#             if self.face is None:
#                 # treat them as scalar then. 
#                 u = self.interpolate(uname,uknw,
#                     prefetched = upre,i_min = i_min)
#                 v = self.interpolate(vname,vknw,
#                     prefetched = vpre,i_min = i_min)
#             else:
#                 if not uknw.same_size(vknw):
#                     raise Exception('u,v kernel needs to have same size'
#                                     'to navigate the complex grid orientation.'
#                                     'use a kernel that include both of the uv kernels'
#                                    )

#                 old_dims = self.ocedata._ds[uname].dims
#                 dims = []
#                 for i in old_dims:
#                     if i in ['Xp1','Yp1']:
#                         dims.append(i[:1])
#                     else:
#                         dims.append(i)
#                 dims = tuple(dims)
#                 uind = self.fatten(uknw,required = dims,fourD = True,ind_moves_kwarg = {'cuvg':'U'})
#                 vind = self.fatten(uknw,required = dims,fourD = True,ind_moves_kwarg = {'cuvg':'V'})
#                 uind_dic = dict(zip(dims,uind))
#                 vind_dic = dict(zip(dims,vind))

#                 if prefetched is not None:
#                     temp_uind = []
#                     temp_vind = []
#                     for i,dim in enumerate(dims):
#                         a_uind = uind[i]
#                         a_uind-= i_min[i]
#                         temp_uind.append(a_uind)
                        
#                         a_vind = vind[i]
#                         a_vind-= i_min[i]
#                         temp_vind.append(a_vind)
#                     temp_uind = tuple(temp_uind)
#                     temp_vind = tuple(temp_vind)
                    
#                     n_ufromu = np.nan_to_num(upre[temp_uind])
#                     n_ufromv = np.nan_to_num(vpre[temp_uind])
#                     n_vfromu = np.nan_to_num(upre[temp_vind])
#                     n_vfromv = np.nan_to_num(vpre[temp_vind])
#                 else:  
#                     # n_u = np.nan_to_num(sread(self.ocedata[uname],ind))
#                     # n_v = np.nan_to_num(sread(self.ocedata[vname],ind))
#                     n_ufromu = np.nan_to_num(sread(self.ocedata[uname],uind))
#                     n_ufromv = np.nan_to_num(sread(self.ocedata[vname],uind))
#                     n_vfromu = np.nan_to_num(sread(self.ocedata[uname],vind))
#                     n_vfromv = np.nan_to_num(sread(self.ocedata[vname],vind))
#     #             np.nan_to_num(n_u,copy = False)
#     #             np.nan_to_num(n_v,copy = False)


#                 if 'Z' in dims:
#                     if self.rz is not None:
#                         if uknw.vkernel == 'nearest':
#                             rz = self.rz
#                         else:
#                             rz = self.rz_lin
#                     else:
#                         rz = 0
#                 elif 'Zl' in dims:
#                     if self.rz is not None:
#                         if uknw.vkernel == 'nearest':
#                             rz = self.rzl
#                         else:
#                             rz = self.rzl_lin
#                     else:
#                         rz = 0
#                 else:
#                     rz = 0

#                 if self.rt is not None:
#                     if uknw.tkernel == 'nearest':
#                         rt = self.rt
#                     else:
#                         rt = self.rt_lin
#                 else:
#                     rt = 0

#                 if not ('X' in dims and 'Y' in dims):
#                     # if it does not have a horizontal dimension, then we don't have to mask
#                     umask = np.ones_like(ind[0])
#                     vmask = np.ones_like(ind[0])
#                 else:
#                     # In this part, I am using uind for masking, which is not correct, this is kind of temporary, 
#                     # because I am ging to overhaul the whole function afterwards
#                     if 'Zl' in dims:
#                         warnings.warn('the vertical value of vector is between cells, may result in wrong masking')
#                         ind_for_mask = tuple([uind[i] for i in range(len(uind)) if dims[i] not in ['time']])
#                         this_bottom_scheme = None
#                     elif 'Z' in dims:
#                         # something like salt
#                         ind_for_mask = tuple([uind[i] for i in range(len(uind)) if dims[i] not in ['time']])
#                         this_bottom_scheme = 'no_flux'
#                     else:
#                         # something like 
#                         ind_for_mask = [uind[i] for i in range(len(uind)) if dims[i] not in ['time']]
#                         ind_for_mask.insert(0,np.zeros_like(ind[0]))
#                         ind_for_mask = ind_for_mask
#                         this_bottom_scheme = 'no_flux'

#                     umask = get_masked(self.ocedata,ind_for_mask,gridtype = 'U')
#                     vmask = get_masked(self.ocedata,ind_for_mask,gridtype = 'V')
#                 if self.face is not None:
#     #                 hface = ind4d[2][:,:,0,0]
#                     (UfromUvel,
#                      UfromVvel,
#                      _,
#                      _        ) = self.ocedata.tp.four_matrix_for_uv(uind_dic['face'][:,:,0,0])
        
#                     (_,
#                      _,
#                      VfromUvel,
#                      VfromVvel) = self.ocedata.tp.four_matrix_for_uv(vind_dic['face'][:,:,0,0])


#                     temp_n_u = (np.einsum('nijk,ni->nijk',n_ufromu,UfromUvel)
#                                +np.einsum('nijk,ni->nijk',n_ufromv,UfromVvel))
#                     temp_n_v = (np.einsum('nijk,ni->nijk',n_vfromu,VfromUvel)
#                                +np.einsum('nijk,ni->nijk',n_vfromv,VfromVvel))

#                     n_u = temp_n_u
#                     n_v = temp_n_v
                    
#                     # again this part is not accurate
#                     temp_umask = np.round(np.einsum('nijk,ni->nijk',umask,UfromUvel)+
#                                      np.einsum('nijk,ni->nijk',vmask,UfromVvel))
#                     temp_vmask = np.round(np.einsum('nijk,ni->nijk',umask,VfromUvel)+
#                                      np.einsum('nijk,ni->nijk',vmask,VfromVvel))

#                     umask = temp_umask
#                     vmask = temp_vmask
#                 else:
#                     n_u = n_ufromu
#                     n_v = n_vfromv

#                 upk4d = find_pk_4d(umask,russian_doll = uknw.inheritance)
#                 vpk4d = find_pk_4d(vmask,russian_doll = vknw.inheritance)
#                 uweight = uknw.get_weight(self.rx+1/2,self.ry,rz = rz,rt = rt,pk4d = upk4d)
#                 vweight = vknw.get_weight(self.rx,self.ry+1/2,rz = rz,rt = rt,pk4d = vpk4d)

#     #             n_u    = partial_flatten(n_u   )
#     #             uweight = partial_flatten(uweight)
#     #             n_v    = partial_flatten(n_v   )
#     #             vweight = partial_flatten(veight)
#                 u = np.einsum('nijk,nijk->n',n_u,uweight)
#                 v = np.einsum('nijk,nijk->n',n_v,vweight)

#             if vec_transform:
#                 u,v = local_to_latlon(u,v,self.cs,self.sn)
#             return u,v
#         else:
#             raise Exception('varList type not supported.')
            