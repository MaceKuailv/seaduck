from seaduck.lagrangian import particle,uknw,vknw
from seaduck.eulerian import position
from seaduck.OceData import OceData
from seaduck.kernelNweight import KnW

import numpy as np
import warnings

lagrange_token = '__particle.'

def OceInterp(od,varList,x,y,z,t,
              kernelList = None,
              lagrangian = False,
              lagrange_kwarg = {},
              update_stops = 'default',
              return_in_between  =True,
              return_pt_time = True,
              **kernel_kwarg):
    '''
    **This is the centerpiece function of the package, through which you can access almost all of its functionality.**

    **Parameters:**

    + od: OceInterp.OceData object or xarray.Dataset (limited support for netCDF Dataset)
        The dataset to work on.
    + varList: str or list
        A list of variable or pair of variables.
    + kernelList: OceInterp.KnW or list of OceInterp.KnW objects
        Indicates which kernel to use for each interpolation.
    + x, y, z: numpy.ndarray
        The location of the particles, where x and y are in degrees, and z is in meters (deeper locations are represented by more negative values).
    + t: numpy.ndarray
        In the Eulerian scheme, this represents the time of interpolation. In the Lagrangian scheme, it represents the time needed for output. 
    + lagrangian: bool
        Specifies whether the interpolation is done in the Eulerian or Lagrangian scheme.
    + lagrange_kwarg: dict
        Keyword arguments passed into the OceInterp.lagrangian.particle object.
    + update_stops: None, 'default', or iterable of float
        Specifies the time to update the prefetch velocity.
    + return_in_between: bool
        In Lagrangian mode, this returns the interpolation not only at time t, but also at every point in time when the speed is updated.
    + return_pt_time: bool
        Specifies whether to return the time of all the steps.
    '''
    if not isinstance(od,OceData):
        od = OceData(od)

    if isinstance(varList,dict):
        kernelList = list(varList.values())
        varList    = list(varList.keys())
        print(f"result will be in the order of {varList}")
    elif isinstance(varList,str):
        varList = [varList]
    elif isinstance(varList,tuple):
        varList = [varList]
    elif isinstance(varList,list):
        pass
    else:
        raise ValueError("varList type not recognized.")

    if isinstance(kernelList,list):
        pass
    elif kernelList is None:
        kernelList = []
        the_kernel = KnW(**kernel_kwarg)
        for i in varList:
            if isinstance(i,str):
                kernelList.append(the_kernel)
            elif isinstance(i,tuple):
                if kernel_kwarg != dict():
                    kernelList.append((the_kernel,the_kernel))
                else:
                    kernelList.append((uknw,vknw))
            else:
                raise ValueError("varList need to be made up of string or tuples")
    if not lagrangian:
        pt = position()
        pt.from_latlon(x = x,y=y,z=z,t=t,data = od)
        for i,var in enumerate(varList):
            if lagrange_token in var:
                raise AttributeError('__particle variables is only available for Lagrangian particles')
        R = pt.interpolate(varList,kernelList)
        return R
            
    else:
        try:
            assert len(t)>1
        except AssertionError:
            raise Exception('There needs to be at least two time steps to run the lagrangian particle')
        t_start = t[0]
        t_nec = t[1:]
        pt = particle(x = x,y=y,z=z,t=np.ones_like(x)*t_start,data = od,**lagrange_kwarg)
        stops,raw = pt.to_list_of_time(t_nec,
                                       update_stops = update_stops,
                                       return_in_between = return_in_between)
        R = []
        for i,var in enumerate(varList):
            if var == lagrange_token+'raw':
                R.append(raw)
            elif lagrange_token in var:
                sublist = []
                for snap in raw:
                    sublist.append(snap.__dict__[var[len(lagrange_token):]])
                R.append(sublist)
            else:
                sublist = []
                for snap in raw:
                    sublist.append(snap.interpolate(var,kernelList[i]))
                R.append(sublist)
                
        if return_pt_time:
            return stops,R
        else:
            if return_in_between:
                warnings.warn('Some of the returns is not on the times you specified.')
            return R