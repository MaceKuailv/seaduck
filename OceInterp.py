from lagrange import particle,uknw,vknw
from point import point
from OceData import OceData
from kernelNweight import KnW

def OceInterp(od,varList,x,y,z,t,kernelList = None,lagrangian = False,lagrange_kwarg = {},**kwarg):
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
        the_kernel = KnW(**kwarg)
        for i in varList:
            if isinstance(i,str):
                kernelList.append(the_kernel)
            elif isinstance(i,list):
                if kwarg != dict():
                    kernelList.append([the_kernel,the_kernel])
                else:
                    kernelList.append([uknw,vknw])
    if not lagrangian:
        pt = point()
        pt.from_latlon(x = x,y=y,z=z,t=t,data = od)
        R = []
        for i,var in enumerate(varList):
            if "__particle" in var:
                raise AttributeError('__particle variables is only available for Lagrangian particles')
            R.append(pt.interpolate(var,kernelList[i]))
        return R
            
    else:
        try:
            assert len(t)>1
        except:
            raise Exception('There needs to be at least two time steps to run the lagrangian particle')
        t_start = t[0]
        t_nec = t[1:]
        pt = particle(x = x,y=y,z=z,t=np.ones_like(x)*t_start,data = od,**lagrange_kwarg)
        
        