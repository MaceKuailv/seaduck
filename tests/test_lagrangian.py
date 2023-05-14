import seaduck as sd
import numpy as np
import pytest
import xarray as xr

Datadir = "tests/Data/"
aviso = xr.open_dataset(Datadir+'aviso_example.nc')
ecco = xr.open_zarr(Datadir + "small_ecco")
curv = xr.open_dataset(Datadir+'MITgcm_curv_nc.nc')

oce = sd.OceData(aviso)
eco = sd.OceData(ecco)
cuv = sd.OceData(curv)

# Set the number of particles here.
N = int(9)

# Increase this if you want more in x direction.
skew = 3

# Change the vertical depth of the particles here.
sqrtN = int(np.sqrt(N))

# Change the horizontal range here.
x = np.linspace(-180, 180, sqrtN * skew)
y = np.linspace(-50, -70, sqrtN // skew)

x, y = np.meshgrid(x, y)
x = x.ravel()
y = y.ravel()
z = None
zz = np.ones_like(x)*(-10.0)

start_time = "1992-02-01"
t = (
    np.array([np.datetime64(start_time) for i in x]) - np.datetime64("1970-01-01")
) / np.timedelta64(1, "s")
tf = (np.datetime64("1992-03-03") - np.datetime64("1970-01-01")) / np.timedelta64(
    1, "s"
)

p = sd.particle(
    x=x,
    y=y,
    z=z,
    t=t,
    data=oce,
    # save_raw = True,
    # transport = True,
    uname="u",
    vname="v",
    wname=None,
)
ecco_p = sd.particle(x = x,y=y,z=zz,t=t,data=eco,transport = True)

normal_stops = np.linspace(t[0], tf, 5)

def test_vol_mode():
#     ecco_p = sd.particle(x = x,y=y,z=zz,t=t,data=eco,transport = True)
    stops, raw = ecco_p.to_list_of_time(normal_stops=[t[0],tf])

def test_to_list_of_time():
    stops, raw = p.to_list_of_time(normal_stops=normal_stops, update_stops=[normal_stops[1]])
    
def test_analytical_step():
    p.analytical_step(10.0)
    
def test_callback():
    curv_p = sd.particle(
        y = np.array([70.5]), 
        x = np.array([-14.]),
        z = np.array([-10.]),
        t = np.array([1832320850.0]),
        data = cuv,
        uname = 'U',
        vname = 'V',
        wname = 'W',
        callback = lambda pt: pt.lon>-14.01
    )
    curv_p.to_list_of_time(normal_stops=[1832320850.0,1832320880.0],update_stops = [])
    
@pytest.mark.parametrize(
    'statement,error',
    [
        ('p.note_taking()',AttributeError),
        ('p.to_list_of_time(normal_stops = [0.0,1.0])',AttributeError),
        ('ecco_p.to_list_of_time(normal_stops = [0.0,1.0],update_stops = [])',ValueError)
    ]
)
def test_lagrange_error(statement,error):
    with pytest.raises(error):
        eval(statement)