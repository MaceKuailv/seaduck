from OceData import OceData
from kernelNweight import KnW
import numpy as np
from kernel_and_weight import translate_to_tendency

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
    xsp = x.shape
    ysp = ind[0].shape
    n = x.shape[0]
    final_shape = [n]+list(ysp[1:])+list(xsp[1:])
    
    R = [np.zeros(final_shape) for i in range(len(ind)+1)]
    
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
            return copy.deepcopy(self.iz)
        elif knw.vkernel in ['dz','interp']:
            return np.vstack([self.iz,self.iz-1]).T
        else:
            raise Exception('vkernel not supported')
            
    def fatten_t(self,knw):
        if self.it is None:
            return None
        if knw.tkernel == 'nearest':
            return copy.deepcopy(self.it)
        elif knw.tkernel in ['dt','interp']:
            return np.vstack([self.it,self.it+1]).T
        else:
            raise Exception('vkernel not supported')
    
    def fatten(self,knw):
        ffc,fiy,fix = self.fatten_h(knw)
        fiz = self.fatten_v(knw)
        fit = self.fatten_t(knw)
        if ffc is not None:
            R = (ffc,fiy,fix)
        else:
            R = (fiy,fix)
            
        if fiz is not None:
            R = ind_broadcast(fiz,R)
            
        if fit is not None:
            R = ind_broadcast(fit,R)
            
        return R