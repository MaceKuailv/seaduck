from lagrange import particle
from point import point
from OceData import OceData

def OceInterp(od,varList,x,y,z,t,kernelList = 'default',lagarangian = False,**kwarg):
    if not lagrangian:
        if not isinstance(od,OceData):
            od = OceData(od)
        pt = point()
        pt.from_latlon(x = x,y=y,z=z,t=t,data = od)
        
        