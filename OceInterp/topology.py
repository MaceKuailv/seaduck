# import pandas as pd
import numpy as np
from numba import njit
import copy
from OceInterp.RuntimeConf import rcParam

# If you have encountered a NotImplementedError and come to this file,
# I suggest you read the ***class topology*** near the bottom of this file.

# llc_face_connect = pd.read_csv('llc_face_connect.csv')
# llc_face_connect = llc_face_connect.drop(columns = 'Unnamed: 0',axis = 1).fillna(42).astype(int)
tends = [0,1,2,3]#up,down,left,right #list(llc_face_connect.columns)

# llc_face_connect = np.array(llc_face_connect)
llc_face_connect = np.array([[ 1, 42, 12,  3],
       [ 2,  0, 11,  4],
       [ 6,  1, 10,  5],
       [ 4, 42,  0,  9],
       [ 5,  3,  1,  8],
       [ 6,  4,  2,  7],
       [10,  5,  2,  7],
       [10,  5,  6,  8],
       [11,  4,  7,  9],
       [12,  3,  8, 42],
       [ 2,  7,  6, 11],
       [ 1,  8, 10, 12],
       [ 0,  9, 11, 42]])

directions = np.array([np.pi/2,-np.pi/2,np.pi,0])

@njit
def llc_mutual_direction(face,nface):
    '''
    0,1,2,3 stands for up, down, left, right
    given 2 faces, the returns are 
    1. the 2nd face is to which direction of the 1st face
    2. the 1st face is to which direction of the 2nd face
    '''
    edge_n = np.where(llc_face_connect[face] == nface)
    nedge_n = np.where(llc_face_connect[nface] == face)
    return edge_n[0][0],nedge_n[0][0]

@njit
def llc_get_the_other_edge(face,edge):
    '''
    The (edge) side of the (face) is connected to
    the (nedge) side of the (nface)
    '''
    face_connect = llc_face_connect
    nface = face_connect[face,edge]
    if nface ==42:
        raise IndexError('Reaching the edge where the face is not connected to any other face')
    nedge_n = np.where(face_connect[nface] == face)
    return nface,nedge_n[0][0]

@njit
def box_ind_tend(ind,tend,iymax,ixmax):
    iy,ix = ind
    if tend == 0:
        iy+=1
    elif tend == 1:
        iy -=1
    elif tend == 2:
        ix-=1
    elif tend == 3:
        ix +=1
    # it would be better to raise an error here.
    if (iy>iymax) or (iy<0):
        return (-1,-1)
    if (iy>iymax) or (iy<0):
        return (-1,-1)
    return (iy,ix)
@njit
def x_per_ind_tend(ind,tend,iymax,ixmax):
    iy,ix = ind
    if tend == 0:
        iy+=1
    elif tend == 1:
        iy -=1
    elif tend == 2:
        ix-=1
    elif tend == 3:
        ix +=1
    if (iy>iymax) or (iy<0):
        return (-1,-1)
    if ix>ixmax:
        return (iy,ix-ixmax)
    if ix<0:
        return (iy,ixmax+ix+1)
    return (iy,ix)

@njit
def llc_ind_tend(ind,tendency,iymax,ixmax,gridoffset = 0):
    '''
    ind is a tuple that is face,iy,ix,
    tendency again is up, down, left, right represented by 0,1,2,3
    return the next cell.
    Essentially, just try all the possibilities. 
    use gridoffset when you are dealing with f-node, 
    -1 for MITgcm, 1 for NEMO.
    '''
#     iymax = 89
#     ixmax = 89
    face,iy,ix = ind
    if tendency == 3:
        if ix!=ixmax:
            ix+=1
        else:
            nface,nedge = llc_get_the_other_edge(face,3) 
            if nedge == 1:
                face,iy,ix = [nface,0,ixmax-iy]
                if gridoffset ==0:
                    pass
                elif gridoffset==-1:
                    face,iy,ix = llc_ind_tend((face,iy,ix),3,iymax,ixmax)
                elif gridoffset == 1:
                    face,iy,ix = llc_ind_tend((face,iy,ix),2,iymax,ixmax)
                else:
                    raise ValueError('gridoffset must be -1,1 or 1')
            elif nedge == 0:
                face,iy,ix = [nface,iymax,iy]
            elif nedge == 2:
                face,iy,ix = [nface,iy,0]
            elif nedge == 3:
                face,iy,ix = [nface,iymax-iy,ixmax]
    if tendency == 2:
        if ix!=0:
            ix-=1
        else:
            nface,nedge = llc_get_the_other_edge(face,2) 
            if nedge == 1:
                face,iy,ix = [nface,0,iy]
            elif nedge == 0:
                face,iy,ix = [nface,iymax,ixmax-iy]
                if gridoffset ==0:
                    pass
                elif gridoffset==-1:
                    face,iy,ix = llc_ind_tend((face,iy,ix),3,iymax,ixmax)
                elif gridoffset == 1:
                    face,iy,ix = llc_ind_tend((face,iy,ix),2,iymax,ixmax)
                else:
                    raise ValueError('gridoffset must be -1,1 or 1')
            elif nedge == 2:
                face,iy,ix = [nface,iymax-iy,0]
            elif nedge == 3:
                face,iy,ix = [nface,iy,ixmax]
    if tendency == 0:
        if iy!=iymax:
            iy+=1
        else:
            nface,nedge = llc_get_the_other_edge(face,0) 
            if nedge == 1:
                face,iy,ix = [nface,0,ix]
            elif nedge == 0:
                face,iy,ix = [nface,iymax,ixmax-ix]
            elif nedge == 2:
                face,iy,ix = [nface,iymax-ix,0]
                if gridoffset ==0:
                    pass
                elif gridoffset==-1:
                    face,iy,ix = llc_ind_tend((face,iy,ix),0,iymax,ixmax)
                elif gridoffset == 1:
                    face,iy,ix = llc_ind_tend((face,iy,ix),1,iymax,ixmax)
                else:
                    raise ValueError('gridoffset must be -1,1 or 1')
            elif nedge == 3:
                face,iy,ix = [nface,ix,ixmax]
    if tendency == 1:
        if iy!=0:
            iy-=1
        else:
            nface,nedge = llc_get_the_other_edge(face,1) 
            if nedge == 1:
                face,iy,ix = [nface,0,ixmax - ix]
            elif nedge == 0:
                face,iy,ix = [nface,iymax,ix]
            elif nedge == 2:
                face,iy,ix = [nface,ix,0]
            elif nedge == 3:
                face,iy,ix = [nface,iymax-ix,ixmax]
                if gridoffset ==0:
                    pass
                elif gridoffset==-1:
                    face,iy,ix = llc_ind_tend((face,iy,ix),0,iymax,ixmax)
                elif gridoffset == 1:
                    face,iy,ix = llc_ind_tend((face,iy,ix),1,iymax,ixmax)
                else:
                    raise ValueError('gridoffset must be -1,1 or 1')
    return(face,iy,ix)

@njit
def llc_get_uv_mask_from_face(faces):
    # we are considering a row from the fatten_face
    # faces is essentially which face each node of the kernel is on. 
    n = len(faces)# it should have been m to be more consistent with other code
    UfromUvel = np.ones(n)
    UfromVvel = np.zeros(n)
    VfromUvel = np.zeros(n)
    VfromVvel = np.ones(n)
    # if all of the node are on the same face, we don't have to convert anything
    if np.abs(np.ones(n)*faces[0]-faces).max()<1e-5:
        return UfromUvel,UfromVvel,VfromUvel, VfromVvel
    else:
        for i in range(1,n):
            if faces[i]==faces[0]:
                continue
            # if the face is not the same, we need to do something
            else:
                # get how much the new face is rotated from the old face
                edge,nedge = llc_mutual_direction(faces[0],faces[i])
                rot = np.pi-directions[edge]+directions[nedge]
                # you can think of this as a rotation matrix
                UfromUvel[i] = np.cos(rot)
                UfromVvel[i] = np.sin(rot)
                VfromUvel[i] = -np.sin(rot)
                VfromVvel[i] = np.cos(rot)
        return UfromUvel,UfromVvel,VfromUvel, VfromVvel

class topology():
    def __init__(self,od,typ = None):
        try:
            h_shape = od['XC'].shape
        except KeyError:
            try:
                h_shape = (int(od['lat'].shape[0]),int(od['lon'].shape[0]))
            except KeyError:
                raise KeyError("Either XC or lat/lon is needed to create the topology object")
        self.h_shape = h_shape
        try:
            self.itmax = len(od['time'])-1
        except KeyError:
            self.itmax = 0
        try:
            self.izmax = len(od['Z'])-1
        except KeyError:
            self.izmax = 0
            
        if typ:
            self.typ = typ
        elif typ is None:
            if len(h_shape) == 3:
                self.num_face,self.iymax,self.ixmax = h_shape
                self.iymax -=1
                self.ixmax -=1
                if self.num_face ==13:
                    self.typ = 'LLC'
                    # we can potentially generate the face connection in runtime
                    # say, put the csv file on cloud
                elif self.num_face ==6:
                    self.typ = 'cubed_sphere'
            elif len(h_shape) == 2:
                self.iymax,self.ixmax = h_shape
                self.iymax -=1
                self.ixmax -=1
                try:
                    lon_right = float(od['XC'][0,self.ixmax])
                    lon_left  = float(od['XC'][0,  0  ])
                except KeyError:
                    lon_right = float(od['lon'][self.ixmax])
                    lon_left  = float(od['lon'][0])
                left_to_right = (lon_right - lon_left)%360
                right_to_left = (lon_left - lon_right)%360
                if left_to_right >50*right_to_left:
                    self.typ = 'x_periodic'
                else:
                    self.typ = 'box'
                    
    def get_the_other_edge(self,face,edge):
        '''
        The (edge) side of the (face) is connected to
        the (nedge) side of the (nface)
        '''
        if self.typ =='LLC':
            return llc_get_the_other_edge(face,edge)
        elif self.typ in ['x_periodic','box']:
            raise Exception('It makes no sense to tinker with face_connection when there is only one face')
        else:
            raise NotImplementedError
    def mutual_direction(self,face,nface):
        '''
        0,1,2,3 stands for up, down, left, right
        given 2 faces, the returns are 
        1. the 2nd face is to which direction of the 1st face
        2. the 1st face is to which direction of the 2nd face
        '''
        if self.typ =='LLC':
            return llc_mutual_direction(face,nface)
        elif self.typ in ['x_periodic','box']:
            raise Exception('It makes no sense to tinker with face_connection when there is only one face')
        else:
            raise NotImplementedError
            
    def ind_tend(self,ind,tend,**kwarg):
        '''
        ind is a tuple that is (face,)iy,ix,
        tendency again is up, down, left, right represented by 0,1,2,3
        return the next cell.
        '''
        if -1 in ind:# meaning invalid point
            return tuple([-1 for i in ind])
#         if tend not in [0,1,2,3]:
#             raise Exception('Illegal move. Must be 0,1,2,3')
        if self.typ == 'LLC':
            return llc_ind_tend(ind,tend,self.iymax,self.ixmax,**kwarg)
        elif self.typ == 'x_periodic':
            return x_per_ind_tend(ind,tend,self.iymax,self.ixmax,**kwarg)
        elif self.typ == 'box':
            return box_ind_tend(ind,tend,self.iymax,self.ixmax,**kwarg)
        else:
            raise NotImplementedError
    def ind_moves(self,ind,moves,**kwarg):
        '''
        moves being a list of directions (0,1,2,3),
        ind being the starting index,
        return the index after moving in the directions in the list
        '''
        if self.check_illegal(ind):
            return tuple([-1 for i in ind])# the origin is invalid
        if not set(moves).issubset({0,1,2,3}):
            raise Exception('Illegal move. Must be 0,1,2,3')
        if self.typ in ['LLC','cubed_sphere']:
            face,iy,ix = ind
            for k in range(len(moves)):
                move = moves[k]
                ind =  self.ind_tend(ind,move,**kwarg)
                if ind[0]!=face:# if the face has changed
                    '''
                    there are times where the the kernel lies between
                    2 faces that define 'left' differently. That's why 
                    when that happens we need to correct the direction
                    you want to move the indexes.
                    '''
                    edge,nedge = self.mutual_direction(face,ind[0])
                    rot = (np.pi-directions[edge]+directions[nedge])%(np.pi*2)
                    if np.isclose(rot,0):
                        pass
                    elif np.isclose(rot,np.pi/2):
                        moves[k+1:] = [[2,3,1,0][move] for move in moves[k+1:]]
                    elif np.isclose(rot,3*np.pi/2):
                        moves[k+1:] = [[3,2,0,1][move] for move in moves[k+1:]]
                    face = ind[0]
                    # if the old face is on the left of the new face, 
                    # the particle should be heading right
        elif self.typ in ['x_periodic','box']:
            for move in moves:
                ind = self.ind_tend(ind,move)
        return ind
    def check_illegal(self,ind):
        '''
        A vectorized check to see whether the index is legal,
        index can be a tuple of numpy ndarrays.
        no negative index is permitted for sanity reason. 
        '''
        if isinstance(ind[0],int):# for single item
            result = False
            for i,z in enumerate(ind):
                max_pos = self.h_shape[i]
                if not (0<=z<=max_pos-1):
                    result = True
            return result
        else:# for numpy ndarray
            result = np.zeros_like(ind[0])
            result = False # make it cleaner
            for i,z in enumerate(ind):
                max_pos = self.h_shape[i]
                result = np.logical_or(np.logical_or((0>z),(z>max_pos-1)),result)
            return result
    def ind_tend_vec(self,inds,tend,**kwarg):
        inds = np.array(inds)
        old_inds = copy.deepcopy(inds)
        move_dic = {
            0:np.array([1,0]),# delta_y,delta_x
            1:np.array([-1,0]),
            2:np.array([0,-1]),
            3:np.array([0,1])
        }
        naive_move = np.array([move_dic[i] for i in tend]).T.astype(int)
        inds[-2:]+=naive_move
        illegal = self.check_illegal(inds)
        redo = np.array(np.where(illegal)).T
        particle_on_edge = False
        for num,loc in enumerate(redo):
            j = loc[0]
            ind = tuple(old_inds[:,j])
            try:
                n_ind = self.ind_tend(ind,int(tend[j]),**kwarg)
            except:
                particle_on_edge = True
                n_ind = ind
            inds[:,j] = np.array(n_ind).ravel()
        if particle_on_edge and rcParam['debug_level'] == 'very_high':
            print('Warning:Some points are on the edge')
        for i in range(len(inds)):
            inds[i] = inds[i].astype(int)
        return inds
        
    def get_uv_mask_from_face(self,faces):
        if self.typ =='LLC':
            return llc_get_uv_mask_from_face(faces)
        elif self.typ in ['x_periodic','box']:
            raise Exception('It makes no sense to tinker with face_connection when there is only one face')
        else:
            raise NotImplementedError
            
    def four_matrix_for_uv(self,fface):
        # apply get_uv_mask for the n*m matrix
        UfromUvel,UfromVvel,VfromUvel, VfromVvel = [np.zeros(fface.shape) for i in range(4)]
        for i in range(fface.shape[0]):
            UfromUvel[i],UfromVvel[i],VfromUvel[i], VfromVvel[i] = self.get_uv_mask_from_face(fface[i])
        return UfromUvel,UfromVvel,VfromUvel, VfromVvel