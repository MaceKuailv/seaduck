import xarray as xr
import numpy as np
import pandas as pd

from OceInterp.topology import topology
from OceInterp.utils import create_tree
from OceInterp.lat2ind import *

no_alias = {
    'XC':'XC',
    'YC':'YC',
    'Z':'Z',
    'Zl':'Zl',
    'time':'time',
    'dXC':'dxC',
    'dYC':'dyC',
    'dZ':'drC',
    'XG':'XG',
    'YG':'YG',
    'dXG':'dxG',
    'dYG':'dyG',
    'dZl':'drF',
    'CS':'CS',
    'SN':'SN',
}

class OceData(object):
    def __init__(self,data,
                 alias = None,
                 memory_limit = 1e7):
        self._ds = data
        self.tp = topology(data)
        if alias == None:
            self.alias = no_alias
        elif alias == 'auto':
            raise NotImplementedError('auto alias not yet implemented')
        elif isinstance(alias,dict):
            self.alias = alias
            
        try:
            self.too_large = self._ds['XC'].nbytes>memory_limit
        except KeyError:
            self.too_large = False
        readiness,missing = self.check_readiness()
        self.readiness = readiness
        if readiness:
            self.grid2array()
        else:
            raise(f'''
            use add_missing_variables or set_alias to create {missing},
            then call OceData.grid2array.
            ''')

    def __setitem__(self, key, item):
        if isinstance(item,xr.DataArray):
            if key in self.alias.keys():
                self._ds[self.alias[key]] = item
            else:
                self._ds[key] = item
        else:
            self.__dict__[key] = item

    def __getitem__(self, key):
        if key in self.__dict__.keys():
            return self.__dict__[key]
        else:
            if key in self.alias.keys():
                return self._ds[self.alias[key]]
            else:
                return self._ds[key]
    
    def check_readiness(self):
        # TODO: make the check more detailed
        varnames = list(self._ds.data_vars)+list(self._ds.coords)
        readiness = {}
        missing = []
        if all([i in varnames for i in ['XC','YC','XG','YG']]):
            readiness['h'] = 'oceanparcel'
            # could be a curvilinear grid that can use oceanparcel style
        elif all([i in varnames for i in ['XC','YC','dxG','dyG','CS','SN']]):
            readiness['h'] = 'local_cartesian'
            # could be a curvilinear grid that can use local cartesian style
        elif all([i in varnames for i in ['lon','lat']]):
            ratio = 6371e3*np.pi/180
            self.dlon = np.gradient(self['lon'])*ratio
            self.dlat = np.gradient(self['lat'])*ratio
            readiness['h'] = 'rectilinear'
            # corresponding to a rectilinear dataset
        else:
            readiness['h'] = False
            # readiness['overall'] = False
            missing.append('''
            For the most basic use case,
            {XC,YC,XG,YG} or {XC,YC,dxG,dyG,CS,SN} for curvilinear grid;
            or {lon, lat} for rectilinear grid.
            ''')
            return False,missing
        for _ in ['time','Z','Zl']:
            readiness[_] = (_ in varnames) and (len(self[_])>1)
            
        return readiness,missing
        
    def add_missing_grid(self):
        # TODO:
        '''
        we need to add at least the following variables here,
        XC,YC,dxG,dyG,Z,Zl,
        '''
        pass
    def show_alias(self):
        return pd.DataFrame.from_dict(self.alias,orient = 'index',columns = ['original name'])
    
    def missing_cs_sn(self):
        try:
            self['CS']
            self['SN']
        except AttributeError:
            xc = np.deg2rad(np.array(self['XC']))
            yc = np.deg2rad(np.array(self['YC']))
            cs = np.zeros_like(xc)
            sn = np.zeros_like(xc)
            cs[0],sn[0] = find_cs_sn(
                yc[0],xc[0],
                yc[1],xc[1]
            )
            cs[-1],sn[-1] = find_cs_sn(
                yc[-2],xc[-2],
                yc[-1],xc[-1]
            )
            cs[1:-1],sn[1:-1] = find_cs_sn(
                yc[:-2],xc[:-2],
                yc[2:],xc[2:]
            )
            # it makes no sense to turn it into DataArray again when you already have in memory
            # and you know where this data is defined. 
            self['CS'] = cs
            self['SN'] = sn
            
    def hgrid2array(self):
        way = self.readiness['h']
        if self.too_large:
            print("Loading grid into memory, it's a large dataset please be patient")
            
        if way == 'oceanparcel':
            for var in ['XC','YC','XG','YG']:
                self[var] = np.array(self[var]).astype('float32')
            for var in ['rA','CS','SN']:
                try:
                    self[var] = np.array(self[var]).astype('float32')
                except:
                    print(f'no {var} in dataset, skip')
                    self[var] = None
            try:
                self.dX = np.array(self['dXG']).astype('float32')
                self.dY = np.array(self['dYG']).astype('float32')
            except:
                self.dX = None
                self.dY = None
            if self.too_large:
                print('numpy arrays of grid loaded into memory')
            self.tree = create_tree(self.XC,self.YC)  
            if self.too_large:
                print('cKD created')
                    
        if way == 'local_cartesian':
            for var in ['XC','YC','CS','SN']:
                self[var] = np.array(self[var]).astype('float32')
            self.dX = np.array(self['dXG']).astype('float32')
            self.dY = np.array(self['dYG']).astype('float32')
        
            if not self.too_large:
                for var in ['XG','YG','dXC','dYC','rA']:
                    try:
                        self[var] = np.array(self[var]).astype('float32')
                    except:
                        print(f'no {var} in dataset, skip')
                        self[var] = None
            if self.too_large:
                print('numpy arrays of grid loaded into memory')
            self.tree = create_tree(self.XC,self.YC)  
            if self.too_large:
                print('cKD created')
                        
        if way == 'rectilinear':
            self.lon = np.array(self['lon']).astype('float32')
            self.lat = np.array(self['lat']).astype('float32')
            
    def vgrid2array(self):
        self.Z = np.array(self['Z']).astype('float32')
        self.dZ = np.array(self['dZ']).astype('float32')
        
    def vlgrid2array(self):
        self.Zl = np.array(self['Zl']).astype('float32')
        self.dZl = np.array(self['dZl']).astype('float32')
        
        # special treatment for dZl
        self.dZl = np.roll(self.dZl,1)
        self.dZl[0] = 1e-10
        
    def tgrid2array(self):
        self.t_base = 0
        self.ts = np.array(self['time'])
        self.ts = (self.ts).astype(float)/1e9
        try:
            self.time_midp = np.array(self['time_midp'])
            self.time_midp = (self.time_midp).astype(float)/1e9
        except:
            self.time_midp = (self.ts[1:]+self.ts[:-1])/2
        
    def grid2array(self):
        if self.readiness['h']:
            self.hgrid2array()
        if self.readiness['Z']:
            self.vgrid2array()
        if self.readiness['Zl']:
            self.vlgrid2array()
        if self.readiness['time']:
            self.tgrid2array()
        
    def find_rel_h(self,x,y):
        # give find_rel_h a new cover
        if self.readiness['h'] == 'oceanparcel':
            faces,iys,ixs,rx,ry,cs,sn,dx,dy,bx,by = find_rel_h_oceanparcel(x,y,
                                                                           self.XC,self.YC,
                                                                           self.dX,self.dY,
                                                                           self.CS,self.SN,
                                                                           self.XG,self.YG,
                                                                           self.tree,self.tp)
        elif self.readiness['h'] == 'local_cartesian':
            faces,iys,ixs,rx,ry,cs,sn,dx,dy,bx,by = find_rel_h_naive(x,y,
                                                     self.XC,self.YC,
                                                     self.dX,self.dY,
                                                     self.CS,self.SN,
                                                     self.tree)
        elif self.readiness['h'] == 'rectilinear':
            faces,iys,ixs,rx,ry,cs,sn,dx,dy,bx,by = find_rel_h_rectilinear(x,y,
                                                                           self.lon,self.lat
                                                                          )
        return faces,iys,ixs,rx,ry,cs,sn,dx,dy,bx,by
    
    def find_rel_vl(self,t):
        iz,rz,dz,bz = find_rel_nearest(t,self.Zl)
        return iz.astype(int),rz,dz,bz
    
    def find_rel_vl_lin(self,z):
        iz,rz,dz,bz = find_rel_z(z,self.Zl,self.dZl)
        return iz.astype(int),rz,dz,bz 
    
    def find_rel_v(self,z):
        iz,rz,dz,bz = find_rel_z(z,self.Zl,self.dZl)
        return (iz-1).astype(int),rz-0.5,dz,bz 
    
    def find_rel_v_lin(self,z):
        iz,rz,dz,bz = find_rel_z(z,self.Z,self.dZ)
        return iz.astype(int),rz,dz,bz 
    
    def find_rel_t(self,t):
        it,rt,dt,bt = find_rel_nearest(t,self.ts)
        return it.astype(int),rt,dt,bt
    
    def find_rel_t_lin(self,t):
        it,rt,dt,bt = find_rel_time(t,self.ts)
        return it.astype(int),rt,dt,bt
    
    def find_rel_3d(self,x,y,z):
        faces,iys,ixs,rx,ry,cs,sn,dx,dy,bx,by = find_rel_h(x,y,
                                                     self.XC,self.YC,
                                                     self.dX,self.dY,
                                                     self.CS,self.SN,
                                                     self.tree)
        iz,rz,dz,bz = find_rel_z(z,self.Zl,self.dZl)
        iz = iz.astype(int)
        return iz.astype(int),faces,iys,ixs,rx,ry,rz,cs,sn,dx,dy,dz,bx,by,bz
    
    def find_rel_4d(self,x,y,z,t):
        faces,iys,ixs,rx,ry,cs,sn,dx,dy,bx,by = find_rel_h(x,y,
                                                     self.XC,self.YC,
                                                     self.dX,self.dY,
                                                     self.CS,self.SN,
                                                     self.tree)
        iz,rz,dz,bz = find_rel_z(z,self.Zl,self.dZl)
        iz = iz.astype(int)
        it,rt,dt,bt = find_rel_time(t,self.ts)
        it = it.astype(int)
        return it.astype(int),iz.astype(int),faces,iys,ixs,rx,ry,rz,rt,cs,sn,dx,dy,dz,dt,bx,by,bz,bt
    
    def find_rel(self,*arg):
        if len(arg) ==2:
            return self.find_rel_2d(*arg)
        if len(arg) ==3:
            return self.find_rel_3d(*arg)
        if len(arg) ==4:
            return self.find_rel_4d(*arg)