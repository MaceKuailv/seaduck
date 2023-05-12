import numpy as np
import xarray as xr
import seaduck as sd
import pytest

Datadir = "tests/Data/"
ecco = xr.open_zarr(Datadir + "small_ecco")

N = int(1e2)

# Change the vertical depth of the particles here
levels = np.array([-5])
sqrtN = int(np.sqrt(N))

# Change the longitude and latitude positions of the particles here
xx = np.linspace(-19, -9, sqrtN)
yy = np.linspace(63, 57, sqrtN)

# Compute intermediate grid variables
xxx, yyy = np.meshgrid(xx, yy)
x = xxx.ravel()
y = yyy.ravel()
x, z = np.meshgrid(x, levels)
y, z = np.meshgrid(y, levels)
x = x.ravel()
y = y.ravel()
z = z.ravel()

# Change the times here
start_time = "1992-01-17"
t = (
    np.array([np.datetime64(start_time) for i in x]) - np.datetime64("1970-01-01")
) / np.timedelta64(1, "s")

end_time = "1992-02-15"

t_bnds = np.array(
    [
        np.datetime64(start_time) - np.datetime64("1970-01-01"),
        np.datetime64(end_time) - np.datetime64("1970-01-01"),
    ]
) / np.timedelta64(1, "s")

@pytest.mark.parametrize(
    'od,x,y,z,t',
    [(ecco, x, y, z, t)]
)
@pytest.mark.parametrize(
    'varList',
    [
        ["ETAN", "maskC"], 
        "SALT",
        ("UVELMASS", "VVELMASS"), 
        {("UVELMASS", "VVELMASS"):(sd.KnW(),sd.KnW())},
    ]
)
def test_eulerian_oceinterp(od,
                            varList,
                            x,y,z,t,
                            ):
    R = sd.OceInterp(od,varList,x,y,z,t)
    
@pytest.mark.parametrize(
    'od,varList,x,y,z,t,kernel_kwarg',
    [(ecco,("UVELMASS", "VVELMASS"), x, y, z, t,dict(hkernel="dx", h_order=2, inheritance=[[0, 1, 2, 3, 4, 5, 6, 7, 8]], tkernel="linear"))]
)
def test_diff_oceinterp(od,
                            varList,
                            x,y,z,t,
                            kernel_kwarg
                            ):
    R = sd.OceInterp(od,varList,x,y,z,t,kernel_kwarg = kernel_kwarg)
    
@pytest.mark.parametrize(
    'od,varList,x,y,z,t,lagrangian,lagrange_kwarg',
    [
        (
            ecco,
            ["SALT", "__particle.raw", "__particle.lat", "__particle.lon"],
            x,
            y,
            z,
            t_bnds,
            True,
            {"save_raw": True},
        )
    ]
)
@pytest.mark.parametrize(
    'return_pt_time',
    [True,False]
)
@pytest.mark.filterwarnings("ignore::Warning")
def test_largangian_oceinterp(od,
                            varList,
                            x,y,z,t,
                            lagrangian,
                            return_pt_time,
                            lagrange_kwarg):
    R = sd.OceInterp(od,varList,x,y,z,t,
                     lagrangian = lagrangian,
                     return_pt_time = return_pt_time,
                     lagrange_kwarg = lagrange_kwarg)

@pytest.mark.parametrize(
    'od,varList,x,y,z,t,lagrangian,error',
    [
        (ecco,["__particle.lat", "__particle.lon"],x,y,z,t_bnds[:1],True,ValueError),
        (ecco,["__particle.lat", "__particle.lon"],x,y,z,t,False,AttributeError),
        (ecco,None,x,y,z,t,False,ValueError),
        (ecco,[None],x,y,z,t,False,ValueError),
    ]
)
@pytest.mark.filterwarnings("ignore::Warning")
def test_oceinterp_error(od,varList,x,y,z,t,lagrangian,error):
    with pytest.raises(error):
        R = sd.OceInterp(od,varList,x,y,z,t,
                     lagrangian = lagrangian)