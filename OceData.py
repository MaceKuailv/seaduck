import xarray as xr
import numpy as np
import pandas as pd
from topology import topology
from utils import create_tree
from lat2ind import *

no_alias = {
    'XC':'XC',
    'YC':'YC',
    'Z':'Z',
    'Zl':'Zl',
    'time':'time',
    'dX':'dxC',
    'dY':'dyC',
    'dZ':'drC',
    'XG':'XG',
    'YG':'YG',
    'dXG':'dxG',
    'dYG':'dyG',
    'dZl':'drF',
    'CS':'CS',
    'SN':'SN',
}

class OceData():
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
            
        self.too_large = self._ds['XC'].nbytes>memory_limit
        ready,missing = self.check_readiness()
        if ready:
            self.grid2array()
        else:
            print(f'''
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
        # TODO:
        return True,[]
    def add_missing_grid(self):
        # TODO:
        '''
        we need to add at least the following variables here,
        XC,YC,dxG,dyG,Z,Zl,
        '''
        pass
    def show_alias(self):
        return pd.DataFrame.from_dict(self.alias,orient = 'index',columns = ['original name'])
            
    def grid2array(self,all_of_them = False):
        if self.too_large:
            print("Loading grid into memory, it's a large dataset please be patient")
        self.Z = np.array(self['Z'])
        self.dZ = np.array(self['dZ'])
        self.Zl = np.array(self['Zl'])
        self.dZl = np.array(self['dZl'])
        
        # special treatment for dZl
        self.dZl = np.roll(self.dZl,1)
        self.dZl[0] = 1e-10

        self.dX = np.array(self['dX']).astype('float32')
        self.dY = np.array(self['dY']).astype('float32')

        self.XC = np.array(self['XC']).astype('float32')
        self.YC = np.array(self['YC']).astype('float32')

        self.CS = np.array(self['CS']).astype('float32')
        self.SN = np.array(self['SN']).astype('float32')
        self.ts = np.array(self['time'])
        self.ts = (self.ts-self.ts[0]).astype(float)/1e9
        
        # add optional ones here
        for var in ['XG','YG','dXG','dYG']:
            try:
                self[var] = np.array(self[var]).astype('float32')
            except:
                print(f'no {var} in dataset, skip')
        
        if self.too_large:
            print('numpy arrays of grid loaded into memory')
        self.tree = create_tree(self.XC,self.YC)  
        if self.too_large:
            print('cKD created')

    def find_rel_h(self,x,y):
        # give find_rel_h a new cover
        faces,iys,ixs,rx,ry,cs,sn,dx,dy,bx,by = find_rel_h(x,y,
                                                     self.XC,self.YC,
                                                     self.dX,self.dY,
                                                     self.CS,self.SN,
                                                     self.tree)
        return faces,iys,ixs,rx,ry,cs,sn,dx,dy,bx,by
    
    def find_rel_vl(self,z):
        iz,rz,dz,bz = find_rel_z(z,self.Zl,self.dZl)
        return iz.astype(int),rz,dz,bz 
    
    def find_rel_v(self,z):
        iz,rz,dz,bz = find_rel_z(z,self.Zl,self.dZl)
        return (iz-1).astype(int),rz-0.5,dz,bz 
    
    def find_rel_t(self,t):
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