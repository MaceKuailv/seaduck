import numpy as np
import copy
from numba import njit

from OceInterp.kernelNweight import KnW
from OceInterp.eulerian import position
from OceInterp.lat2ind import find_rel_time,find_rx_ry_oceanparcel

deg2m = 6271e3*np.pi/180

@njit
def rel2latlon(rx,ry,rzl,cs,sn,dx,dy,dzl,dt,bx,by,bzl):
    temp_x = rx*dx/deg2m
    temp_y = ry*dy/deg2m
    dlon = (temp_x*cs-temp_y*sn)/np.cos(by*np.pi/180)
    dlat = (temp_x*sn+temp_y*cs)
    lon = dlon+bx
    lat = dlat+by
    dep = bzl+dzl*rzl
    return lon,lat,dep

@njit
def to_180(x):
    '''
    convert any longitude scale to [-180,180)
    '''
    x = x%360
    return x+(-1)*(x//180)*360

@njit
def increment(t,u,du):
    return u/du*(np.exp(du*t)-1)

def stationary(t,u,du,x0):
    incr = increment(t,u,du)
    nans = np.isnan(incr)
    incr[nans] = (u*t)[nans]
    return incr+x0

@njit
def stationary_time(u,du,x0):
    tl = np.log(1-du/u*(0.5+x0))/du
    tr = np.log(1+du/u*(0.5-x0))/du
    no_gradient = du==0
    if no_gradient.any():
        tl[no_gradient] = (-x0[no_gradient]-0.5)/u[no_gradient]
        tr[no_gradient] = (0.5-x0[no_gradient])/u[no_gradient]
    return tl,tr

def time2wall(xs,us,dus):
    ts = []
    for i in range(3):
        tl,tr = stationary_time(us[i],dus[i],xs[i])
        ts.append(tl)
        ts.append(tr)
    return ts

def which_early(tf,ts):
    ts.append(np.ones(len(ts[0]))*tf)#float or array both ok
    t_directed = np.array(ts)*np.sign(tf)
    t_directed[np.isnan(t_directed)] = np.inf
    t_directed[t_directed<=0] = np.inf
    tend = t_directed.argmin(axis = 0)
    the_t = np.array([ts[te][i] for i,te in enumerate(tend)])
    return tend,the_t

uvkernel = np.array([
    [0,0],
    [1,0],
    [0,1]
])
ukernel = np.array([
    [0,0],
    [1,0]
])
vkernel = np.array([
    [0,0],
    [0,1]
])
wkernel = np.array([
    [0,0]
])
udoll = [[0,1]]
vdoll = [[0,2]]
wdoll = [[0]]
ktype = 'interp'
h_order = 0
wknw = KnW(kernel =  wkernel,inheritance = None,vkernel = 'linear')
uknw = KnW(kernel = uvkernel,inheritance = udoll)
vknw = KnW(kernel = uvkernel,inheritance = vdoll)
dwknw = KnW(kernel =  wkernel,inheritance = None,vkernel = 'dz')
duknw = KnW(kernel = uvkernel,inheritance = udoll,hkernel = 'dx',h_order = 1)
dvknw = KnW(kernel = uvkernel,inheritance = vdoll,hkernel = 'dy',h_order = 1)

class particle(position):
    def __init__(self,
                memory_limit = 1e7,# 10MB
                uname = 'UVELMASS',
                vname = 'VVELMASS',
                wname = 'WVELMASS',
                 dont_fly = True,
                 save_raw = False,
                 transport = False,
                 stop_criterion = None,
                **kwarg
                ):
        self.from_latlon(**kwarg)
        
        (
            self.izl_lin,
            self.rzl_lin,
            self.dzl_lin,
            self.bzl_lin
        ) = self.ocedata.find_rel_vl_lin(self.dep)
        
        self.uname = uname
        self.vname = vname
        self.wname = wname
        
        #  user defined function to stop integration. 
        self.stop_criterion = stop_criterion
        
        # whether u,v,w is in m^3/s or m/s
        self.transport = transport
        if self.transport:
            try:
                self.ocedata['vol']
            except KeyError:
                self.ocedata['vol'] = np.array(self.ocedata._ds['drF']*self.ocedata._ds['rA'])
        
        # whether or not setting the w at the surface
        # just to prevent particles taking off
        self.dont_fly = dont_fly
        if dont_fly:
            if wname is not None:
                try:
                    self.ocedata[wname].loc[dict(Zl = 0)] = 0
                except KeyError:
                    pass
        self.too_large = self.ocedata._ds['XC'].nbytes>memory_limit
        
        if self.too_large:
            pass
        else:
            self.update_uvw_array()
        (
            self.u,
            self.v,
            self.w,
            self.du,
            self.dv,
            self.dw,
            self.Vol
        ) = [np.zeros(self.N).astype(float) for i in range(7)]
        if self.transport==True:
            self.get_vol()
        self.fillna()
        
        self.save_raw = save_raw
        if self.save_raw:
            self.itlist = [[] for i in range(self.N)]
            self.fclist = [[] for i in range(self.N)]
            self.iylist = [[] for i in range(self.N)]
            self.izlist = [[] for i in range(self.N)]
            self.ixlist = [[] for i in range(self.N)]
            self.rxlist = [[] for i in range(self.N)]
            self.rylist = [[] for i in range(self.N)]
            self.rzlist = [[] for i in range(self.N)]
            self.ttlist = [[] for i in range(self.N)]
            self.uulist = [[] for i in range(self.N)]
            self.vvlist = [[] for i in range(self.N)]
            self.wwlist = [[] for i in range(self.N)]
            self.dulist = [[] for i in range(self.N)]
            self.dvlist = [[] for i in range(self.N)]
            self.dwlist = [[] for i in range(self.N)]
            self.xxlist = [[] for i in range(self.N)]
            self.yylist = [[] for i in range(self.N)]
            self.zzlist = [[] for i in range(self.N)]
            
    def update_uvw_array(self
                        ):
        uname = self.uname
        vname = self.vname
        wname = self.wname
        if 'time' not in self.ocedata[uname].dims:
            try:
                self.uarray
                self.varray
                self.warray
            except AttributeError:
                self.uarray = np.array(self.ocedata[uname])
                self.varray = np.array(self.ocedata[vname])
                if self.wname is not None:
                    self.warray = np.array(self.ocedata[wname])
                    if self.dont_fly:
                        # I think it's fine
                        self.warray[0] = 0.0
        else:
            self.itmin = int(np.min(self.it))
            self.itmax = int(np.max(self.it))
            if self.itmax!=self.itmin:
                self.uarray = np.array(self.ocedata[uname][self.itmin:self.itmax+1])
                self.varray = np.array(self.ocedata[vname][self.itmin:self.itmax+1])
                self.warray = np.array(self.ocedata[wname][self.itmin:self.itmax+1])
            else:
                self.uarray = np.array(self.ocedata[uname][[self.itmin]])
                self.varray = np.array(self.ocedata[vname][[self.itmin]])
                self.warray = np.array(self.ocedata[wname][[self.itmin]])
            if self.dont_fly:
                # I think it's fine
                self.warray[:,0] = 0.0
            
    def get_vol(self,which = None):
        if which is None:
            which = np.ones(self.N).astype(bool)
        sub = self.subset(which)
        if self.face is not None:
            Vol = self.ocedata['vol'][sub.iz,sub.face,sub.iy,sub.ix]
        else:
            Vol = self.ocedata['vol'][sub.iz,sub.iy,sub.ix]
        self.Vol[which] = Vol
        
    def get_u_du(self,which = None):
        if which is None:
            which = np.ones(self.N).astype(bool)
        if self.too_large:
            if self.wname is not None:
                w     = self.subset(which).interpolate(self.wname,wknw)
                dw    = self.subset(which).interpolate(self.wname,dwknw)
            else:
                w  = np.zeros(self.subset(which).N,float)
                dw = np.zeros(self.subset(which).N,float)
            self.iz = self.izl_lin-1
            u,v   = self.subset(which).interpolate([self.uname,self.vname],
                                                   [uknw,vknw  ],
                                                   vec_transform = False
                                                  )
            du,dv = self.subset(which).interpolate([self.uname,self.vname],
                                                   [duknw,dvknw],
                                                   vec_transform = False
                                                  )
        else:
            if 'time' not in self.ocedata[self.uname].dims:
                ifirst = 0
            else:
                ifirst = self.itmin

            if self.wname is not None:
                i_min = [0 for i in self.warray.shape]
                i_min[0] = ifirst
                w     = self.subset(which).interpolate(self.wname,
                                                       wknw ,
                                                       prefetched = self.warray,
                                                       i_min = i_min)
                dw    = self.subset(which).interpolate(self.wname,
                                                       dwknw,
                                                       prefetched = self.warray,
                                                       i_min = i_min)
            else:
                w = np.zeros(self.subset(which).N,float)
                dw = np.zeros(self.subset(which).N,float)
            
            self.iz = self.izl_lin-1
            i_min = [0 for i in self.uarray.shape]
            i_min[0] = ifirst
            u,v   = self.subset(which).interpolate([self.uname,self.vname],
                                    [uknw,vknw],vec_transform = False,
                                    prefetched = [self.uarray,self.varray],
                                    i_min = i_min,
                                   )
            du,dv = self.subset(which).interpolate([self.uname,self.vname],
                                    [duknw,dvknw],vec_transform = False,
                                    prefetched = [self.uarray,self.varray],
                                    i_min = i_min,
                                   )
#             ow     = self.subset(which).interpolate(self.wname,wknw)
#             odw    = self.subset(which).interpolate(self.wname,dwknw)
#             self.iz = self.izl_lin-1
#             ou,ov   = self.subset(which).interpolate([self.uname,self.vname],
#                                                    [uknw,vknw  ],
#                                                    vec_transform = False
#                                                   )
#             odu,odv = self.subset(which).interpolate([self.uname,self.vname],
#                                                    [duknw,dvknw],
#                                                    vec_transform = False
#                                                   )
#             _wmatch = (np.nan_to_num( ow)==np.nan_to_num(w )).all()
#             dwmatch = (np.nan_to_num(odw)==np.nan_to_num(dw)).all()
#             _vmatch = (np.nan_to_num( ov)==np.nan_to_num(v )).all()
#             dvmatch = (np.nan_to_num(odv)==np.nan_to_num(dv)).all()
#             _umatch = (np.nan_to_num( ou)==np.nan_to_num(u )).all()
#             dumatch = (np.nan_to_num(odu)==np.nan_to_num(du)).all()
#             if _wmatch and dwmatch and _vmatch and dvmatch and _umatch and dumatch:
#                 pass
#             else:
#                 if not _wmatch:
#                     print('w mismatch')
#                 if not dwmatch:
#                     print('dw mismatch')
#                 if not _vmatch:
#                     print('v mismatch')
#                 if not dvmatch:
#                     print('dv mismatch')
#                 if not _umatch:
#                     print('u mismatch')
#                 if not dumatch:
#                     print('du mismatch')
#                 print(self.it[0])
#                 print(self.itmin,self.itmax)
#                 self.w  = (w)
#                 self.ow = (ow)
#                 print((np.nan_to_num( ou)==np.nan_to_num(u )).all())
#                 print((np.nan_to_num(self.ou)==np.nan_to_num(self.u )).all())
#                 raise Exception('two schemes mismatch')
        
        if not self.transport:
        
            self.u [which] =  u/self.dx[which]
            self.v [which] =  v/self.dy[which]
            self.w [which] =  w/self.dzl_lin[which]
            self.du[which] = du/self.dx[which]
            self.dv[which] = dv/self.dy[which]
            self.dw[which] = dw/self.dzl_lin[which]
            
        else:
            self.u [which] =  u/self.Vol[which]
            self.v [which] =  v/self.Vol[which]
            self.w [which] =  w/self.Vol[which]
            self.du[which] = du/self.Vol[which]
            self.dv[which] = dv/self.Vol[which]
            self.dw[which] = dw/self.Vol[which]
        
        self.fillna()
        
#     def get_u_du(self,which = None):
#         if which is None:
#             which = np.ones(self.N).astype(bool)
#         if self.face is None:
#             _,uiy,uix = fatten_ind_h(self.face,self.iy[which],self.ix[which],self.tp,kernel = ukernel)
#             _,viy,vix = fatten_ind_h(self.face,self.iy[which],self.ix[which],self.tp,kernel = vkernel)
#             _,wiy,wix = fatten_ind_h(self.face,self.iy[which],self.ix[which],self.tp,kernel = wkernel)
        
#             uind4d = (uiy,uix)
#             vind4d = (viy,vix)
#             wind4d = (wiy,wix)
#         else:
#             uface,uiy,uix = fatten_ind_h(self.face[which],self.iy[which],self.ix[which],self.tp,kernel = ukernel)
#             vface,viy,vix = fatten_ind_h(self.face[which],self.iy[which],self.ix[which],self.tp,kernel = vkernel)
#             wface,wiy,wix = fatten_ind_h(self.face[which],self.iy[which],self.ix[which],self.tp,kernel = wkernel)
        
#             uind4d = (uface,uiy,uix)
#             vind4d = (vface,viy,vix)
#             wind4d = (wface,wiy,wix)

#         uind4d = fatten_linear_dim(self.izl[which]-1,uind4d,minimum = 0,kernel_type = self.zkernel)
#         vind4d = fatten_linear_dim(self.izl[which]-1,vind4d,minimum = 0,kernel_type = self.zkernel)
#         wind4d = fatten_linear_dim(self.izl[which]  ,wind4d,minimum = 0,kernel_type = 'linear')
        
#         if self.too_large:
#             uind4d = fatten_linear_dim(self.it[which],
#                                        uind4d,maximum = self.tp.itmax,
#                                        kernel_type = self.tkernel)
#             vind4d = fatten_linear_dim(self.it[which],
#                                        vind4d,maximum = self.tp.itmax,
#                                        kernel_type = self.tkernel)
#             wind4d = fatten_linear_dim(self.it[which],
#                                        wind4d,maximum = self.tp.itmax,
#                                        kernel_type = self.tkernel)
#         else:
#             uind4d = fatten_linear_dim(self.it[which]-self.itmin,
#                                        uind4d,maximum = self.tp.itmax,
#                                        kernel_type = self.tkernel)
#             vind4d = fatten_linear_dim(self.it[which]-self.itmin,
#                                        vind4d,maximum = self.tp.itmax,
#                                        kernel_type = self.tkernel)
#             wind4d = fatten_linear_dim(self.it[which]-self.itmin,
#                                        wind4d,maximum = self.tp.itmax,
#                                        kernel_type = self.tkernel)
# #         self.wind4d = wind4d
#         self.uind4d = uind4d
#         umask = get_masked(self.od,tuple([i for i in uind4d[1:] if i is not None]),gridtype = 'U')
#         vmask = get_masked(self.od,tuple([i for i in uind4d[1:] if i is not None]),gridtype = 'V')
#         wmask = get_masked(self.od,tuple([i for i in wind4d[1:] if i is not None]),gridtype = 'Wvel')
        
#         # it would be better to make a global variable
#         if self.too_large:
#             n_u = sread(self.od._ds[self.uname],uind4d)
#             n_v = sread(self.od._ds[self.vname],vind4d)
#             n_w = sread(self.od._ds[self.wname],wind4d)
#         else:
#             n_u = np.nan_to_num(self.uarray[uind4d])
#             n_v = np.nan_to_num(self.varray[vind4d])
#             n_w = np.nan_to_num(self.warray[wind4d])
            
#         if self.face is not None:

#             UfromUvel,UfromVvel,VfromUvel, VfromVvel = self.tp.four_matrix_for_uv(uface)

#             temp_n_u = np.einsum('nijk,ni->nijk',n_u,UfromUvel)+np.einsum('nijk,ni->nijk',n_v,UfromVvel)
#             temp_n_v = np.einsum('nijk,ni->nijk',n_u,VfromUvel)+np.einsum('nijk,ni->nijk',n_v,VfromVvel)

#             n_u = temp_n_u
#             n_v = temp_n_v

#             temp_umask = np.round(np.einsum('nijk,ni->nijk',umask,UfromUvel)+
#                              np.einsum('nijk,ni->nijk',vmask,UfromVvel))
#             temp_vmask = np.round(np.einsum('nijk,ni->nijk',umask,VfromUvel)+
#                              np.einsum('nijk,ni->nijk',vmask,VfromVvel))

#             umask = temp_umask
#             vmask = temp_vmask

#         upk4d = find_pk_4d(umask,russian_doll = udoll)
#         vpk4d = find_pk_4d(vmask,russian_doll = vdoll)
#         wpk4d = find_pk_4d(wmask,russian_doll = wdoll)
        
#         rx,ry,rz,rzl,rt = (
#             self.rx[which],
#             self.ry[which],
#             self.rz[which],
#             self.rzl[which],
#             self.rt[which]
#         )

#         uweight = get_weight_4d(rx+1/2,ry,rz,rt,upk4d,
#                   hkernel = ukernel,
#                   russian_doll = udoll,
#                   funcs = ufuncs,
#                   tkernel = self.tkernel,
#                   zkernel = self.zkernel
#                  )
#         duweight = get_weight_4d(rx+1/2,ry,rz,rt,upk4d,
#                   hkernel = ukernel,
#                   russian_doll = udoll,
#                   funcs = dufuncs,
#                   tkernel = self.tkernel,
#                   zkernel = self.zkernel
#                  )
#         vweight = get_weight_4d(rx,ry+1/2,rz,rt,vpk4d,
#                   hkernel = vkernel,
#                   russian_doll = vdoll,
#                   funcs = vfuncs,
#                   tkernel = self.tkernel,
#                   zkernel = self.zkernel
#                  )
#         dvweight = get_weight_4d(rx,ry+1/2,rz,rt,vpk4d,
#                   hkernel = vkernel,
#                   russian_doll = vdoll,
#                   funcs = dvfuncs,
#                   tkernel = self.tkernel,
#                   zkernel = self.zkernel
#                  )
#         wweight = get_weight_4d(rx,ry,rzl,rt,wpk4d,
#                   hkernel = wkernel,
#                   russian_doll = wdoll,
#                   funcs = wfuncs,
#                   tkernel = self.tkernel,
#                   zkernel = 'linear',
#                   bottom_scheme = None
#                  )
#         dwweight = get_weight_4d(rx,ry,rzl,rt,wpk4d,
#                   hkernel = wkernel,
#                   russian_doll = wdoll,
#                   funcs = wfuncs,
#                   tkernel = self.tkernel,
#                   zkernel = 'dz'
#                  )
#         np.nan_to_num( uweight,copy = False)
#         np.nan_to_num(duweight,copy = False)
#         np.nan_to_num( vweight,copy = False)
#         np.nan_to_num(dvweight,copy = False)
#         np.nan_to_num( wweight,copy = False)
#         np.nan_to_num(dwweight,copy = False)
        
#         self.u [which] = np.einsum('nijk,nijk->n',n_u, uweight)/self.dx[which]
#         self.v [which] = np.einsum('nijk,nijk->n',n_v, vweight)/self.dy[which]
#         self.w [which] = np.einsum('nijk,nijk->n',n_w, wweight)/self.dzl[which]
#         self.du[which] = np.einsum('nijk,nijk->n',n_u,duweight)/self.dx[which]
#         self.dv[which] = np.einsum('nijk,nijk->n',n_v,dvweight)/self.dy[which]
#         self.dw[which] = np.einsum('nijk,nijk->n',n_w,dwweight)/self.dzl[which]
        
#         self.w = np.zeros_like(self.u)
#         self.dw = np.zeros_like(self.u)

    def fillna(self):
#         np.nan_to_num(self.rx,copy = False)
#         np.nan_to_num(self.ry,copy = False)
#         np.nan_to_num(self.rz,copy = False)
        np.nan_to_num(self.u ,copy = False)
        np.nan_to_num(self.v ,copy = False)
        np.nan_to_num(self.w ,copy = False)
        np.nan_to_num(self.du,copy = False)
        np.nan_to_num(self.dv,copy = False)
        np.nan_to_num(self.dw,copy = False)
        
    def note_taking(self,which = None):
        if which is None:
            which = np.ones(self.N).astype(bool)
        where = np.where(which)[0]
        try:
            self.ttlist
        except AttributeError:
            raise AttributeError('This is not a particle_rawlist object')
        for i in where:
            if self.face is not None:
                self.fclist[i].append(self.face[i])
            self.itlist[i].append(self.it[i])
            self.iylist[i].append(self.iy[i])
            self.izlist[i].append(self.izl_lin[i])
            self.ixlist[i].append(self.ix[i])
            self.rxlist[i].append(self.rx[i])
            self.rylist[i].append(self.ry[i])
            self.rzlist[i].append(self.rzl_lin[i])
            self.ttlist[i].append(self.t[i])
            self.uulist[i].append(self.u[i])
            self.vvlist[i].append(self.v[i])
            self.wwlist[i].append(self.w[i])
            self.dulist[i].append(self.du[i])
            self.dvlist[i].append(self.dv[i])
            self.dwlist[i].append(self.dw[i])
            self.xxlist[i].append(self.lon[i])
            self.yylist[i].append(self.lat[i])
            self.zzlist[i].append(self.dep[i])
            
    def empty_lists(self):
        
        self.itlist = [[] for i in range(self.N)]
        self.fclist = [[] for i in range(self.N)]
        self.iylist = [[] for i in range(self.N)]
        self.izlist = [[] for i in range(self.N)]
        self.ixlist = [[] for i in range(self.N)]
        self.rxlist = [[] for i in range(self.N)]
        self.rylist = [[] for i in range(self.N)]
        self.rzlist = [[] for i in range(self.N)]
        self.ttlist = [[] for i in range(self.N)]
        self.uulist = [[] for i in range(self.N)]
        self.vvlist = [[] for i in range(self.N)]
        self.wwlist = [[] for i in range(self.N)]
        self.dulist = [[] for i in range(self.N)]
        self.dvlist = [[] for i in range(self.N)]
        self.dwlist = [[] for i in range(self.N)]
        self.xxlist = [[] for i in range(self.N)]
        self.yylist = [[] for i in range(self.N)]
        self.zzlist = [[] for i in range(self.N)]
        
    def out_of_bound(self):
        x_out = np.logical_or(self.rx >0.5,self.rx < -0.5)
        y_out = np.logical_or(self.ry >0.5,self.ry < -0.5)
        z_out = np.logical_or(self.rzl_lin>1  ,self.rzl_lin< 0   )
        return np.logical_or(np.logical_or(x_out,y_out),z_out)

    
    def trim(self,verbose = False,tol = 1e-6):
        # tol = 1e-6 # about 10 m horizontal
        xmax = np.nanmax(self.rx)
        xmin = np.nanmin(self.rx)
        ymax = np.nanmax(self.ry)
        ymin = np.nanmin(self.ry)
        zmax = np.nanmax(self.rzl_lin)
        zmin = np.nanmin(self.rzl_lin)
        if xmax>=0.5-tol:
            where = self.rx>=0.5-tol
            cdx = (0.5-tol)-self.rx[where]
            self.rx[where]+=cdx
            self.u[where] += self.du[where]*cdx
            if verbose:
                print(f'converting {xmax} to 0.5')
        if xmin<=-0.5+tol:
            where = self.rx<=-0.5+tol
            cdx = (-0.5+tol)-self.rx[where]
            self.rx[where]+=cdx
            self.u[where] += self.du[where]*cdx
            if verbose:
                print(f'converting {xmin} to -0.5')
        if ymax>=0.5-tol:
            where = self.ry>=0.5-tol
            cdx = (0.5-tol)-self.ry[where]
            self.ry[where]+=cdx
            self.v[where] += self.dv[where]*cdx
            if verbose:
                print(f'converting {ymax} to 0.5')
        if ymin<=-0.5+tol:
            where = self.ry<=-0.5+tol
            cdx = (-0.5+tol)-self.ry[where]
            self.ry[where]+=cdx
            self.v[where] += self.dv[where]*cdx
            if verbose:
                print(f'converting {ymin} to -0.5')
        if zmax>=1.-tol:
            where = self.rzl_lin>=1.-tol
            cdx = (1.-tol)-self.rzl_lin[where]
            self.rzl_lin[where]+=cdx
            self.w[where] += self.dw[where]*cdx
            if verbose:
                print(f'converting {zmax} to 1')
        if zmin<=-0.+tol:
            where = self.rzl_lin<=-0.+tol
            cdx = (-0.+tol)-self.rzl_lin[where]
            self.rzl_lin[where]+=cdx
            self.w[where] += self.dw[where]*cdx
            if verbose:
                print(f'converting {zmin} to 0')
    
    def contract(self):
        max_time = 1e3
        out = self.out_of_bound()
        # out = np.logical_and(out,u!=0)
        xs = [self.rx[out],self.ry[out],self.rzl_lin[out]-1/2]
        us = [self.u[out],self.v[out],self.w[out]]
        dus= [self.du[out],self.dv[out],self.dw[out]]
        tmin = -np.ones_like(self.rx[out])*np.inf
        tmax = np.ones_like(self.rx[out])*np.inf
        for i in range(3):
            tl,tr = stationary_time(us[i],dus[i],xs[i])
            np.nan_to_num(tl,copy = False)
            np.nan_to_num(tr,copy = False)
            tmin = np.maximum(tmin,np.minimum(tl,tr))
            tmax = np.minimum(tmax,np.maximum(tl,tr))
        dead = tmin>tmax
        
        contract_time = (tmin+tmax)/2
        contract_time = np.maximum(-max_time,contract_time)
        contract_time = np.maximum(max_time,contract_time)

        np.nan_to_num(contract_time,copy = False,posinf = 0,neginf = 0)
        
        con_x = []
        for i in range(3):
            con_x.append(stationary(contract_time,us[i],dus[i],0))
            
        cdx= np.nan_to_num(con_x[0])
        cdy= np.nan_to_num(con_x[1])
        cdz= np.nan_to_num(con_x[2])
        
        self.rx[out] += cdx
        self.ry[out] += cdy
        self.rzl_lin[out]+= cdz
        
        self.u[out]+=cdx*self.du[out]
        self.v[out]+=cdy*self.dv[out]
        self.w[out]+=cdz*self.dw[out]
        
        self.t[out] += contract_time
        
    def update_after_cell_change(self):
        self.iz,self.rz,self.dz,self.bz = self.ocedata.find_rel_v(self.dep)
        if self.face is not None:
            self.bx,self.by,self.bzl_lin = (
                self.ocedata.XC[self.face,self.iy,self.ix],
                self.ocedata.YC[self.face,self.iy,self.ix],
                self.ocedata.Zl[self.izl_lin]
            )
            self.cs,self.sn = (
                self.ocedata.CS[self.face,self.iy,self.ix],
                self.ocedata.SN[self.face,self.iy,self.ix]
            )
            self.dx,self.dy,self.dz,self.dzl_lin = (
                self.ocedata.dX[self.face,self.iy,self.ix],
                self.ocedata.dY[self.face,self.iy,self.ix],
                self.ocedata.dZ[self.iz],
                self.ocedata.dZl[self.izl_lin]
            )
        else:
            self.bx,self.by,self.bzl_lin = (
                self.ocedata.XC[self.iy,self.ix],
                self.ocedata.YC[self.iy,self.ix],
                self.ocedata.Zl[self.izl_lin]
            )
            self.cs,self.sn = (
                self.ocedata.CS[self.iy,self.ix],
                self.ocedata.SN[self.iy,self.ix]
            )
            self.dx,self.dy,self.dz,self.dzl_lin = (
                self.ocedata.dX[self.iy,self.ix],
                self.ocedata.dY[self.iy,self.ix],
                self.ocedata.dZ[self.iz],
                self.ocedata.dZl[self.izl_lin]
            )
        
        try:
            self.px,self.py = self.get_px_py()
            self.rx,self.ry = find_rx_ry_oceanparcel(self.lon,self.lat,self.px,self.py)
            if np.isnan(self.rx).any() or np.isnan(self.ry).any():
                whereNan = np.logical_or(np.isnan(self.rx),np.isnan(self.ry))
                print(self.lon[whereNan],self.lat[whereNan])
                print(self.px[:,whereNan],self.py[:,whereNan])
                print(self.ix[whereNan],self.iy[whereNan],self.iz[whereNan],self.face[whereNan])
                raise Exception('no tolerant for NaN!')
        except AttributeError:
#         if True:
            dlon = to_180(self.lon - self.bx)
            dlat = to_180(self.lat - self.by)
            self.rx = (dlon*np.cos(self.by*np.pi/180)*self.cs+dlat*self.sn)*deg2m/self.dx
            self.ry = (dlat*self.cs-dlon*self.sn*np.cos(self.by*np.pi/180))*deg2m/self.dy
        self.rzl_lin= (self.dep - self.bzl_lin)/self.dzl_lin
    
    def analytical_step(self,tf,which = None):
        
        if which is None:
            which = np.ones(self.N).astype(bool)
        if isinstance(tf,float):
            tf = np.array([tf for i in range(self.N)])
        
        tf = tf[which]
        
        if self.out_of_bound().any():
            raise Exception('this step should always be after trim...')

        xs = [self.rx[which],self.ry[which],self.rzl_lin[which]-1/2]
        us = [self.u[which],self.v[which],self.w[which]]
        dus= [self.du[which],self.dv[which],self.dw[which]]
        
        ts = time2wall(xs,us,dus)
            
        tend,the_t = which_early(tf,ts)
        self.t[which] +=the_t
        
        new_x = []
        new_u = []
        for i in range(3):
            x_move = stationary(the_t,us[i],dus[i],0)
            new_u.append(us[i]+dus[i]*x_move)
            new_x.append(x_move+xs[i])
            
        self.rx[which],self.ry[which],self.rzl_lin[which] = new_x
        self.u[which],self.v[which],self.w[which] = new_u
        self.rzl_lin[which] +=1/2
        try:
            px,py = self.px,self.py
            w = self.get_f_node_weight()
            self.lon = np.einsum('nj,nj->n',w,px.T)
            self.lat = np.einsum('nj,nj->n',w,py.T)
            self.dep = self.bzl_lin+self.dzl_lin*self.rzl_lin
        except AttributeError:
            self.lon,self.lat,self.dep = rel2latlon(self.rx,self.ry,self.rzl_lin,
                                                       self.cs,self.sn,
                                                         self.dx,self.dy,self.dzl_lin,
                                           self.dt,self.bx,self.by,self.bzl_lin)
        if self.save_raw:
            # record the moment just before crossing the wall
            # or the moment reaching destination.
            self.note_taking(which)
        type1 = tend<=3
        translate = {
            0:2,#left
            1:3,#right
            2:1,#down
            3:0 #up
        }
        trans_tend = np.array([translate[i] for i in tend[type1]])
        if self.face is not None:
            tface,tiy,tix,tiz = (
                self.face[which].astype(int),
                self.iy[which].astype(int),
                self.ix[which].astype(int),
                self.izl_lin[which].astype(int)
            )
            tface[type1],tiy[type1],tix[type1] = self.tp.ind_tend_vec(
                (tface[type1],tiy[type1],tix[type1]),
                trans_tend)
        else:
            tiy,tix,tiz = (
                self.iy[which].astype(int),
                self.ix[which].astype(int),
                self.izl_lin[which].astype(int)
            )
            tiy[type1],tix[type1] = self.tp.ind_tend_vec(
                (tiy[type1],tix[type1]),
                trans_tend)
        type2 = tend==4
        tiz[type2]+=1
        type3 = tend==5
        tiz[type3]-=1
        
        # investigate stuck
#         now_masked = maskc[tiz-1,tface,tiy,tix]==0
#         if now_masked.any():
#             wrong_ind = (np.where(now_masked))[0]
#             print(wrong_ind)
#             print((tiz-1)[wrong_ind],tface[wrong_ind],tiy[wrong_ind],tix[wrong_ind])
#             print('rx',[xs[i][wrong_ind] for i in range(3)])
#             print('u',[us[i][wrong_ind] for i in range(3)])
#             print('du',[dus[i][wrong_ind] for i in range(3)])
#             print(tend[wrong_ind])
#             print(t_directed[:,wrong_ind])
#             print('stuck!')
#             raise Exception('ahhhhh!')
        if self.face is not None:
            self.face[which],self.iy[which],self.ix[which],self.izl_lin[which] = tface,tiy,tix,tiz
        else:
            self.iy[which],self.ix[which],self.izl_lin[which] = tiy,tix,tiz
    
    def deepcopy(self):
        p = position()
        p.ocedata = self.ocedata
        p.N = self.N
        keys = self.__dict__.keys()
        for i in keys:
            item = self.__dict__[i]
            if isinstance(item,np.ndarray):
                if len(item.shape) ==1:
                    p.__dict__[i] = copy.deepcopy(item)
                else:
                    pass
            elif isinstance(item,list):
                p.__dict__[i] = copy.deepcopy(item)
            else:
                pass
        return p
        
    def to_next_stop(self,t1):
        tol = 0.5
        tf = t1 - self.t
        todo = abs(tf)>tol
        if self.stop_criterion is not None:
            todo = np.logical_and(todo,self.stop_criterion(self))
        trim_tol = 1e-6
        for i in range(200):
            if i > 50:
                trim_tol = 1e-2
            elif i > 30:
                trim_tol = 1e-3
            elif i > 20:
                trim_tol = 1e-4
            elif i > 10:
                trim_tol = 1e-5
            self.trim(tol = trim_tol)
            print(sum(todo),'left',end = ' ')
            self.analytical_step(tf,todo)
            self.update_after_cell_change()
            if self.transport==True:
                self.get_vol()
            self.get_u_du(todo)
            tf = t1 - self.t
            todo = abs(tf)>tol
            if self.stop_criterion is not None:
                todo = np.logical_and(todo,self.stop_criterion(self))
            if sum(todo) == 0:
                break
            if self.save_raw:
                # record those who cross the wall
                self.note_taking(todo)
#             self.contract()
        if i ==199:
            print('maximum iteration count reached')
        self.t = np.ones(self.N)*t1
        self.it,self.rt,self.dt,self.bt = self.ocedata.find_rel_t(self.t)
        self.it,_,_,_ = find_rel_time(self.t,self.ocedata.time_midp)
        self.it += 1
        
    def to_list_of_time(self,normal_stops,update_stops = 'default',return_in_between  =True):
        t_min = np.minimum(np.min(normal_stops),self.t[0])
        t_max = np.maximum(np.max(normal_stops),self.t[0])
        
        if 'time' not in self.ocedata[self.uname].dims:
            pass
        else:
            data_tmin = self.ocedata.ts.min()
            data_tmax = self.ocedata.ts.max()
            if t_min<data_tmin or t_max>data_tmax:
                raise Exception(f'time range not within bound({data_tmin},{data_tmax})')
            if update_stops == 'default':
                update_stops = self.ocedata.time_midp[np.logical_and(t_min<self.ocedata.time_midp,
                                                             self.ocedata.time_midp<t_max)]
        temp = (list(zip(normal_stops,np.zeros_like(normal_stops)))+
                list(zip(update_stops,np.ones_like(update_stops))))
        temp.sort(key = lambda x:abs(x[0]-self.t[0]))
        stops,update = list(zip(*temp))
#         return stops,update
        self.get_u_du()
        R = []
        for i,tl in enumerate(stops):
            print()
            print(np.datetime64(round(tl),'s'))
            if self.save_raw:
                # save the very start of everything. 
                self.note_taking()
            self.to_next_stop(tl)
            if update[i]:
                if not self.too_large:
                    self.update_uvw_array()
                self.get_u_du()
                if return_in_between:
                    R.append(self.deepcopy())
            else:
                R.append(self.deepcopy())
            if self.save_raw:
                self.empty_lists()
        return stops,R