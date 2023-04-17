from OceInterp.lagrangian import particle,uknw,vknw
from OceInterp.eulerian import position
from OceInterp.OceData import OceData
from OceInterp.kernelNweight import KnW

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
    The center piece function of the package, from here 
    you can access almost all the functionality of the 
    package.
    -------
    od: OceInterp.OceData object or xarray.Dataset (hopefully also netcdf)
        the dataset to work on
    varList: str or lst
        (list of) variable/pair of variables
    kernelList: OceInterp.KnW/list of OceInterp.KnW object
        which kernel to use for each interpolation
    x,y,z: numpy.ndarray
        location of the particles, x,y are in degrees, and 
        z are in meters, the deeper, the more negative. 
    t: numpy.ndarray
        In Eulerian scheme, the time of interpolation.
        In Lagrangian scheme, the time needed for output. 
    lagrangian: bool
        whether the interpolation is done in Eulerian or 
        Lagarangian scheme
    lagrange_kwarg: dict
        keyward argument passed into the OceInterp.lagrangia
        n.particle object.
    update_stops: None, 'defalut', iterable of float
        time to update the prefetch the velocity.
    return_in_between: bool
        in lagrangian mode, return the interpolation not only
        at t, but also at whenever the speed is updated.
    return_pt_time: bool
        whether to return time of all the steps.
    '''
    if not isinstance(od,OceData):
        od = OceData(od)

    if isinstance(varList,dict):
        kernelList = list(varList.values())
        varList    = list(varList.keys())
        print(f"result will be in the order of {varList}")
    elif isinstance(varList,str):
        varList = [varList]
    elif isinstance(varList,list):
        pass
    else:
        raise Exception("varList type not recognized.")

    if isinstance(kernelList,list):
        pass
    elif kernelList is None:
        kernelList = []
        the_kernel = KnW(**kernel_kwarg)
        for i in varList:
            if isinstance(i,str):
                kernelList.append(the_kernel)
            elif isinstance(i,list):
                if kernel_kwarg != dict():
                    kernelList.append([the_kernel,the_kernel])
                else:
                    kernelList.append([uknw,vknw])
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