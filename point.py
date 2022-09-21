from OceData import OceData
from kernelNweight import KnW
import numpy as np
from kernel_and_weight import translate_to_tendency,find_pk_4d
from smart_read import smart_read as sread
import copy
from get_masks import get_masked
from utils import local_to_latlon

def to_180(x):
    '''
    convert any longitude scale to [-180,180)
    '''
    x = x%360
    return x+(-1)*(x//180)*360

def local_to_latlon(u,v,cs,sn):
    '''convert local vector to north-east '''
    uu = u*cs-v*sn
    vv = u*sn+v*cs
    return uu,vv

def get_combination(lst,select):
    '''
    Iteratively find all the combination that
    has (select) amount of elements
    and every element belongs to lst
    This is the same as the one in itertools, 
    but I didn't know at the time.
    '''
    n = len(lst)
    if select==1:
        return [[num] for num in lst]
    else:
        the_lst = []
        for i,num in enumerate(lst):
            sub_lst = get_combination(lst[i+1:],select-1)
            for com in sub_lst:
                com.append(num)
#             print(sub_lst)
            the_lst+=sub_lst
        return the_lst
    
def ind_broadcast(x,ind):
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

def partial_flatten(ind):
    if isinstance(ind,tuple):
        shape = ind[0].shape
        
        num_neighbor = 1
        for i in range(1,len(shape)):
            num_neighbor*=shape[i]
        R = []
        for i in range(len(ind)):
            R.append(ind[i].reshape(shape[0],num_neighbor))
        return tuple(R)
    elif isinstance(ind,np.ndarray):
        shape = ind.shape
        
        num_neighbor = 1
        for i in range(1,len(shape)):
            num_neighbor*=shape[i]
        return ind.reshape(shape[0],num_neighbor)

class point():
#     self.ind_h_dict = {}
    def from_latlon(self,**kwarg):
        try:
            self.ocedata
        except AttributeError:
            if 'data' not in kwarg.keys():
                raise Exception('data not provided')
            self.ocedata = kwarg['data']
        self.tp = self.ocedata.tp
        if 'x' in kwarg.keys() and 'y' in kwarg.keys():
            self.lon = kwarg['x']
            self.lat = kwarg['y']
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
            ) = self.ocedata.find_rel_h(kwarg['x'],kwarg['y'])
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
        if 'z' in kwarg.keys():
            (
                self.iz,
                self.rz,
                self.dz,
                self.bz 
            ) = self.ocedata.find_rel_v(kwarg['z'])
            self.dep = kwarg['z']
        else:
            (
                self.iz,
                self.rz,
                self.dz,
                self.bz,
                self.dep
            ) = [None for i in range(5)]
            
        if 't' in kwarg.keys():
            (
                self.it,
                self.rt,
                self.dt,
                self.bt 
            ) = self.ocedata.find_rel_t(kwarg['t'])
            self.tim = kwarg['t']
        else:
            (
                self.it,
                self.rt,
                self.dt,
                self.bt,
                self.tim
            ) = [None for i in range(5)]
        return self
             
    def fatten_h(self,knw):
        '''
        faces,iys,ixs is now 1d arrays of size n. 
        We are applying a kernel of size m.
        This is going to return a n * m array of indexes.
        each row represen all the node needed for interpolation of a single point.
        "h" represent we are only doing it on the horizontal plane
        '''
#         self.ind_h_dict
        kernel = knw.kernel
        kernel_tends =  [translate_to_tendency(k) for k in kernel]
        m = len(kernel_tends)
        n = len(self.iy)
        tp = self.tp

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
            if faces is not None:
                ind = (self.face[j],self.iy[j],self.ix[j])
            else:
                ind = (self.iy[j],self.ix[j])
            # everyone start from the [0,0] node
            moves = kernel_tends[i]
            # moves is a list of operations to get to a single point
            #[2,2] means move to the left and then move to the left again.
            n_ind = tp.ind_moves(ind,moves)
            if faces is not None:
                n_faces[j,i],n_iys[j,i],n_ixs[j,i] = n_ind
            else:
                n_iys[j,i],n_ixs[j,i] = n_ind
        if self.face is not None:
            return n_faces.astype('int'),n_iys.astype('int'),n_ixs.astype('int')
        else:
            return None,n_iys.astype('int'),n_ixs.astype('int')
        
    def fatten_v(self,knw):
        if self.iz is None:
            return None
        if knw.vkernel == 'nearest':
            return copy.deepcopy(self.iz.astype(int))
        elif knw.vkernel in ['dz','interp']:
            return np.vstack([self.iz.astype(int),self.iz.astype(int)-1]).T
        else:
            raise Exception('vkernel not supported')
            
    def fatten_t(self,knw):
        if self.it is None:
            return None
        if knw.tkernel == 'nearest':
            return copy.deepcopy(self.it.astype(int))
        elif knw.tkernel in ['dt','interp']:
            return np.vstack([self.it.astype(int),self.it.astype(int)+1]).T
        else:
            raise Exception('vkernel not supported')
    
    def fatten(self,knw,fourD = False,dic = False):
        #TODO: register the kernel shape
        ffc,fiy,fix = self.fatten_h(knw)
        fiz = self.fatten_v(knw)
        fit = self.fatten_t(knw)
        if ffc is not None:
            R = (ffc,fiy,fix)
        else:
            R = (fiy,fix)
            
        if fiz is not None:
            R = ind_broadcast(fiz,R)
        elif fourD:
            R = [np.expand_dims(R[i],axis = -1) for i in range(len(R))]
            
        if fit is not None:
            R = ind_broadcast(fit,R)
        elif fourD:
            R = [np.expand_dims(R[i],axis = -1) for i in range(len(R))]
        
        if dic:
            if ffc is not None:
                keys = ['face','Y','X']
            else:
                keys = ['Y','X']

            if fiz is not None:
                keys.insert(0,'Z')
                
            if fit is not None:
                keys.insert(0,'time')

            R = dict(zip(keys,R))
        return R
    
    def get_needed(self,varName,knw,**kwarg):
        ind = self.fatten(knw,**kwarg)
        if len(ind)!= len(self.ocedata[varName].dims):
            raise Exception("""dimension mismatch.
                            Please check if the point objects have all the dimensions needed""")
        return sread(self.ocedata[varName],ind)
    
    def get_masked(self,knw,gridtype = 'C'):
        ind = self.fatten(knw,fourD = True)
        if self.it is not None:
            ind = ind[1:]
        if len(ind)!=len(self.ocedata['maskC'].dims):
            raise Exception("""dimension mismatch.
                            Please check if the point objects have all the dimensions needed""")
        return get_masked(self.ocedata,ind,gridtype = gridtype)
    
    def find_pk4d(self,knw,gridtype = 'C'):
        masked = self.get_masked(knw,gridtype = 'C')
        pk4d = find_pk_4d(masked,russian_doll = knw.inheritance)
        return pk4d
    
    def interpolate(self,varName,knw):
        if self.rz is not None:
            rz = self.rz
        else:
            rz = 0
            
        if self.rt is not None:
            rt = self.rt
        else:
            rt = 0
        if isinstance(varName,str):
            needed = self.get_needed(varName,knw)
            pk4d = self.find_pk4d(knw)

            weight = knw.get_weight(self.rx,self.ry,rz = rz,rt = rt,pk4d = pk4d)

            needed = partial_flatten(needed)
            weight = partial_flatten(weight)

            R = np.einsum('nj,nj->n',needed,weight)
            return R
        elif isinstance(varName,list):
            if len(varName)!=2:
                raise Exception('list varName can only have length 2, representing horizontal vectors')
            uname,vname = varName
            uknw,vknw = knw
            if not uknw.same_size(vknw):
                raise Exception('u,v kernel needs to have same size')
            umask = self.get_masked(uknw,gridtype = 'U')
            vmask = self.get_masked(vknw,gridtype = 'V')
            n_u = self.get_needed(uname,uknw,fourD = True)
            n_v = self.get_needed(vname,vknw,fourD = True)
            ind = self.fatten(uknw,dic = True,fourD = True)
            if self.face is not None:
#                 hface = ind4d[2][:,:,0,0]
                (UfromUvel,
                 UfromVvel,
                 VfromUvel,
                 VfromVvel) = self.ocedata.tp.four_matrix_for_uv(ind['face'][0,0])
                
                temp_n_u = np.einsum('nijk,ni->nijk',n_u,UfromUvel)+np.einsum('nijk,ni->nijk',n_v,UfromVvel)
                temp_n_v = np.einsum('nijk,ni->nijk',n_u,VfromUvel)+np.einsum('nijk,ni->nijk',n_v,VfromVvel)
                
                n_u = temp_n_u
                n_v = temp_n_v
                
                umask = np.round(np.einsum('nijk,ni->nijk',umask,UfromUvel)+
                                 np.einsum('nijk,ni->nijk',vmask,UfromVvel))
                vmask = np.round(np.einsum('nijk,ni->nijk',umask,VfromUvel)+
                                 np.einsum('nijk,ni->nijk',vmask,VfromVvel))
                
            upk4d = find_pk_4d(umask,russian_doll = uknw.inheritance)
            vpk4d = find_pk_4d(vmask,russian_doll = vknw.inheritance)
            uweight = uknw.get_weight(self.rx+1/2,self.ry,rz = rz,rt = rt,pk4d = upk4d)
            vweight = vknw.get_weight(self.rx,self.ry+1/2,rz = rz,rt = rt,pk4d = upk4d)
            
#             n_u    = partial_flatten(n_u   )
#             uweight = partial_flatten(uweight)
#             n_v    = partial_flatten(n_v   )
#             vweight = partial_flatten(veight)
            u = np.einsum('nijk,nijk->n',n_u,uweight)
            v = np.einsum('nijk,nijk->n',n_v,vweight)
            u,v = local_to_latlon(u,v,self.cs,self.sn)
            return u,v